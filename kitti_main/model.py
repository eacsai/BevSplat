"""``BevSplatKITTI`` — the Stage-1 model for KITTI main experiment.

This is a focused rewrite of the Stage-1 branch of
``models/models_kitti_nips.py``. The Stage-0 (self-supervised rotation
pretraining) branch lives only in the legacy file; the Stage-1
checkpoints already include the rotation-pre-head weights so this class
can load them with ``strict=False`` and run end-to-end.

The forward pass mirrors lines 495-655 of ``models_kitti_nips.py``
exactly. Submodule names (``SatFeatureNet``, ``feat_gaussian_encoder``,
``dino_feat``, ``dpt``, ``FeatureForT``, ``TransRefine``, ``coe_R``,
``coe_T``) are preserved so existing ``.pth`` files load cleanly.

Pipeline (Stage 1):

    grd_img (B,3,256,1024) ─► resize ► (B,3,64,256) ─► DINOv2 ─► DPT ─► (grd_feat, grd_conf)
                                                       ▲
              grd_depth (B,H,W)  pre-computed *_grd_depth.pt
                                                       ▼
                              GaussianFeatEncoder ─► Np=3 per-pixel Gaussians (means, covs, opacities, feats, confs)
                                                       │
                       rotation pre-head (no_grad) ────► predicted heading θ
                                                       │
                            render_projections ────► BEV feature map (B, 32, 128, 128)

    sat_map (B,3,512,512) ─► resize ► (B,3,128,128) ─► DINOv2 ─► DPT ─► (sat_feat, sat_conf)

    crop BEV feat to (A - 60/mpp) per level → cross-correlate with sat_feat.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import data_utils as utils
from jacobian import grid_sample
from gaussian.encoder_feat_nips import GaussianFeatEncoder
from models.VGGW import VGGUnet
from models.dino_fit import DINO
from models.dpt_single import DPT
from models.swin_transformer import TransOptimizerG2SP_V1
from vis_gaussian_feat import render_projections

from .config import BEV_HEIGHT_M, BEV_RES, BEV_WIDTH_M, GRD_H, GRD_W, RAW_GRD_H, RAW_GRD_W, SAT_H, SAT_W


class BevSplatKITTI(nn.Module):
    """Stage-1 BevSplat model for KITTI.

    Args
    ----
    args
        Namespace carrying ``level`` (e.g. ``"1"``), ``channels``
        (``"32_16_4"``), ``share`` (0/1), ``rotation_range``,
        ``shift_range_lat``, ``shift_range_lon``, ``N_iters``,
        ``ConfGrd``, ``ConfSat``. The legacy ``Model`` class consumes
        the same Namespace; we accept it as-is so the trainer can pass
        through whatever the user set on the command line.
    device
        Target device for module buffers and the precomputed view masks.
    """

    def __init__(self, args, device=None):
        super().__init__()
        self.args = args
        self.device = device

        self.level = sorted(int(item) for item in args.level.split("_"))
        self.N_iters = args.N_iters
        self.channels = [int(item) for item in args.channels.split("_")]
        self.gs_channels = [32, 16, 4]

        # Stage-0 / rotation-pre-head modules. SatFeatureNet is also used in
        # Stage 1 to extract features for the heading estimator when
        # ``args.rotation_range > 0`` (under torch.no_grad()).
        self.SatFeatureNet = VGGUnet(self.level, self.gs_channels)
        self.TransRefine = TransOptimizerG2SP_V1(self.gs_channels)
        self.coe_R = nn.Parameter(torch.tensor(-5.0))  # Stage-0 loss scale; unused here.
        self.coe_T = nn.Parameter(torch.tensor(-3.0))  # Same.

        # Stage-1 head: DINOv2 backbone + DPT outputting (feat, confidence).
        self.dino_feat = DINO()
        self.dpt = DPT(self.dino_feat.feat_dim)

        # Per-pixel Gaussian primitive encoder (predicts offsets, opacities,
        # rotations, scales; feature/confidence come from DPT).
        self.feat_gaussian_encoder = GaussianFeatEncoder()

        # Translation refinement head — instantiated only to preserve ckpt
        # key names; not actually used in the Stage-1 forward.
        if args.share:
            self.FeatureForT = VGGUnet(self.level, self.gs_channels)
        else:
            self.GrdFeatureForT = VGGUnet(self.level, self.gs_channels)
            self.SatFeatureForT = VGGUnet(self.level, self.gs_channels)

        # Pyramid meter-per-pixel — level 3 is full resolution, level 0 is /8.
        self.meters_per_pixel = {}
        base_mpp = utils.get_meter_per_pixel()
        for lvl in range(4):
            self.meters_per_pixel[lvl] = base_mpp * (2 ** (3 - lvl))

        # Precompute the BEV → ground-camera visibility mask per level.
        # Uses the KITTI reference intrinsic; the mask only flags which BEV
        # pixels project inside the original 256x1024 image plane.
        self._register_view_masks()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _register_view_masks(self):
        """Compute ``self.masks[level]`` for each pyramid level.

        Mirrors ``Model._register_view_masks`` (the loop in
        ``models_kitti_nips.py:87-106``). The reference intrinsic comes
        from the KITTI left color camera; the values are the same as the
        legacy code so the masks are bit-identical.
        """
        ori_camera_k = torch.tensor(
            [[
                [582.9802, 0.0000, 496.2420],
                [0.0000, 482.7076, 125.0034],
                [0.0000, 0.0000, 1.0000],
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        self.masks = {}
        for level in range(4):
            A = 512 / 2 ** (3 - level)
            XYZ_1 = self.sat2world(A)
            B = 1
            zero = torch.zeros([B, 1], device=self.device)
            uv, mask = self.World2GrdImgPixCoordinates(
                zero, zero, zero, XYZ_1, ori_camera_k, RAW_GRD_H, RAW_GRD_W, RAW_GRD_H, RAW_GRD_W
            )
            self.masks[level] = mask[:, :, :, 0]

    # ------------------------------------------------------------------
    # Geometry primitives
    # ------------------------------------------------------------------

    def sat2world(self, satmap_sidelength):
        """Build a ``[A, A, 4]`` grid of world-frame homogeneous points.

        Verbatim from ``models_kitti_nips.py:168-198``.
        """
        i = j = torch.arange(0, int(satmap_sidelength)).cuda()
        ii, jj = torch.meshgrid(i, j)
        uv = torch.stack([jj, ii], dim=-1).float()
        u0 = v0 = int(satmap_sidelength) // 2
        uv_center = uv - torch.tensor([u0, v0]).cuda()

        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / int(satmap_sidelength)
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()
        Aff_sat2real = meter_per_pixel * R

        XZ = torch.einsum("ij, hwj -> hwi", Aff_sat2real, uv_center)
        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        return torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)

    def World2GrdImgPixCoordinates(
        self,
        ori_shift_u,
        ori_shift_v,
        ori_heading,
        XYZ_1,
        ori_camera_k,
        grd_H,
        grd_W,
        ori_grdH,
        ori_grdW,
    ):
        """Project world-frame BEV pixels into the camera image plane.

        Returns ``(uv [B,H,W,2], mask [B,H,W,1])`` where ``mask`` is 1 for
        BEV pixels that land inside ``[0, grd_W) x [0, grd_H)``.

        Verbatim from ``models_kitti_nips.py:115-166`` — used only to
        compute ``self.masks`` at init time and inside the rotation
        pre-head.
        """
        B = ori_heading.shape[0]
        shift_u_meters = self.args.shift_range_lon * ori_shift_u
        shift_v_meters = self.args.shift_range_lat * ori_shift_v
        heading = ori_heading * self.args.rotation_range / 180 * np.pi

        cos = torch.cos(-heading)
        sin = torch.sin(-heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1).view(B, 3, 3)

        camera_height = utils.get_camera_height()
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1).unsqueeze(-1)

        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1, :] * grd_W / ori_grdW
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        P = camera_k @ torch.cat([R, T], dim=-1)

        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.full_like(uv1[:, :, :, 2:], 1e-6))
        uv = uv1[:, :, :, :2] / uv1_last

        mask = (
            torch.greater(uv1_last, torch.full_like(uv1[:, :, :, 2:], 1e-6))
            * torch.greater_equal(uv[:, :, :, 0:1], torch.zeros_like(uv[:, :, :, 0:1]))
            * torch.less(uv[:, :, :, 0:1], torch.full_like(uv[:, :, :, 0:1], grd_W))
            * torch.greater_equal(uv[:, :, :, 1:2], torch.zeros_like(uv[:, :, :, 1:2]))
            * torch.less(uv[:, :, :, 1:2], torch.full_like(uv[:, :, :, 1:2], grd_H))
        )
        return uv * mask, mask

    def project_grd_to_map(
        self, grd_f, grd_c, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH, ori_grdW
    ):
        """Sample ground feature map at the BEV grid (used by rotation pre-head).

        Verbatim from ``models_kitti_nips.py:332-360``.
        """
        XYZ_1 = self.sat2world(satmap_sidelength)
        uv, mask = self.World2GrdImgPixCoordinates(
            shift_u, shift_v, heading, XYZ_1, camera_k, grd_f.shape[-2], grd_f.shape[-1], ori_grdH, ori_grdW
        )
        grd_f_trans, _ = grid_sample(grd_f, uv, None)
        grd_f_trans = grd_f_trans * mask[:, None, :, :, 0]
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
            grd_c_trans = grd_c_trans * mask[:, None, :, :, 0]
        else:
            grd_c_trans = None
        return grd_f_trans, grd_c_trans, uv, mask

    # ------------------------------------------------------------------
    # Rotation pre-head (Stage-0 weights, frozen at Stage-1 eval/training)
    # ------------------------------------------------------------------

    def _trans_update(self, shift_u, shift_v, heading, grd_feat_proj, sat_feat, level):
        """Single iteration of the translation/rotation refiner.

        Verbatim from ``models_kitti_nips.py:200-227``.
        """
        B = shift_u.shape[0]
        grd_feat_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, torch.full_like(grd_feat_norm, 1e-6))
        grd_feat_proj = grd_feat_proj / grd_feat_norm[:, None, None, None]
        delta = self.TransRefine(grd_feat_proj, sat_feat, level)

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        heading_new = heading + delta[:, 2:3]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)
        return shift_u_new, shift_v_new, heading_new

    def _predict_heading(self, sat_map, grd_img_left, left_camera_k, ori_grdH, ori_grdW):
        """Estimate the heading correction θ via the Stage-0 rotation pre-head.

        Implements the body of ``models_kitti_nips.py:548-562``. Runs under
        ``torch.no_grad()`` and is only called when
        ``args.rotation_range > 0``.
        """
        B = sat_map.shape[0]
        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=self.device)
        shift_v = torch.zeros_like(shift_u)
        heading = torch.zeros_like(shift_u)

        sat_feat_dict, _ = self.SatFeatureNet(sat_map)
        grd_feat_dict, _ = self.SatFeatureNet(grd_img_left)

        shift_us_all, shift_vs_all, headings_all = [], [], []
        for _ in range(self.N_iters):
            shift_us, shift_vs, headings = [], [], []
            for level in self.level:
                sat_feat = sat_feat_dict[level]
                grd_feat = grd_feat_dict[level]
                A = sat_feat.shape[-1]
                overhead_feat, _, _, _ = self.project_grd_to_map(
                    grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW
                )
                shift_u, shift_v, heading = self._trans_update(
                    shift_u, shift_v, heading, overhead_feat, sat_feat, level
                )
                shift_us.append(shift_u[:, 0])
                shift_vs.append(shift_v[:, 0])
                headings.append(heading[:, 0])
            shift_us_all.append(torch.stack(shift_us, dim=1))
            shift_vs_all.append(torch.stack(shift_vs, dim=1))
            headings_all.append(torch.stack(headings, dim=1))

        shift_lats = torch.stack(shift_vs_all, dim=1)
        shift_lons = torch.stack(shift_us_all, dim=1)
        thetas = torch.stack(headings_all, dim=1)
        return shift_lats, shift_lons, thetas

    # ------------------------------------------------------------------
    # Stage-1 forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sat_align_cam,        # unused (Stage 0 only); kept in signature for dataloader compat
        sat_map,
        grd_img_left,
        grd_depth,
        grd_ori,              # unused; kept for dataloader compat
        left_camera_k,
        gt_heading=None,
        gt_shift_u=None,
        gt_shift_v=None,
        train=False,
        loop=None,
        save_dir=None,
    ):
        """Stage-1 forward — produces the dicts consumed by ``losses.py``.

        Returns
        -------
        sat_feat_dict, sat_conf_dict, g2s_feat_dict, g2s_conf_dict, mask_dict,
        shift_lats, shift_lons, thetas, render_loss
        """
        del sat_align_cam, grd_ori, gt_shift_u, gt_shift_v, train, loop, save_dir
        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        # DINOv2 + DPT consume the FULL-resolution images. The DPT head
        # downsamples 4× internally, so the resulting (grd_feat, grd_conf)
        # land at (64, 256) for the ground branch and (128, 128) for the
        # satellite branch — which is exactly the resolution the
        # GaussianFeatEncoder needs to align its per-pixel features.
        with torch.no_grad():
            sat_feats = self.dino_feat(sat_map)
            grd_feats = self.dino_feat(grd_img_left)
            if isinstance(sat_feats, (tuple, list)):
                sat_feats = [_f.detach() for _f in sat_feats]
            if isinstance(grd_feats, (tuple, list)):
                grd_feats = [_f.detach() for _f in grd_feats]

        sat_feat, sat_conf = self.dpt(sat_feats)
        grd_feat, grd_conf = self.dpt(grd_feats)

        level = self.level[0]  # main experiment uses a single level
        sat_feat_dict = {level: sat_feat}
        sat_conf_dict = {level: sat_conf}

        # Resized ground image used by the encoder's per-pixel Gaussian
        # initialization. Must match DPT output spatial dims (64, 256).
        grd_resized = F.interpolate(grd_img_left, size=(GRD_H, GRD_W), mode="bilinear", align_corners=False)

        # --- Build per-pixel Gaussians -------------------------------------
        grd_view = grd_resized.unsqueeze(1)
        camera_k = left_camera_k.clone()
        camera_k[:, :1, :] = camera_k[:, :1, :] / grd_depth.shape[2]
        camera_k[:, 1:2, :] = camera_k[:, 1:2, :] / grd_depth.shape[1]
        camera_k = camera_k.unsqueeze(1)
        extrinsics = (
            torch.eye(4, device=grd_view.device)
            .unsqueeze(0)
            .repeat(B, 1, 1)
            .unsqueeze(1)
        )
        grd_gaussian = self.feat_gaussian_encoder(
            grd_view,
            grd_depth,
            grd_feat[:, None],
            grd_conf[:, None],
            camera_k,
            extrinsics,
        )

        # --- Heading estimation (optional) --------------------------------
        with torch.no_grad():
            if self.args.rotation_range > 0:
                shift_lats, shift_lons, thetas = self._predict_heading(
                    sat_map, grd_img_left, left_camera_k, ori_grdH, ori_grdW
                )
                heading = thetas[:, -1, -1:].detach()
            else:
                heading = torch.zeros([B, 1], dtype=torch.float32, device=sat_map.device)
                thetas = heading.unsqueeze(1)
                shift_lats = None
                shift_lons = None

        # --- BEV splat ----------------------------------------------------
        _, bev_feat, bev_conf = render_projections(
            grd_gaussian,
            BEV_RES,
            heading=heading,
            width=BEV_WIDTH_M,
            height=BEV_HEIGHT_M,
        )

        # Mask of rendered (non-zero feature) BEV pixels, used by losses.py
        # when args.ConfGrd == 0. Shape: [B, H, W, 1].
        mask = (bev_feat != 0).any(dim=1, keepdim=True).permute(0, 2, 3, 1)
        mask_dict = {level: mask}

        # --- Crop to the valid pose-search window per pyramid level -------
        g2s_feat_dict = {}
        g2s_conf_dict = {}
        for lvl in self.level:
            meter_per_pixel = self.meters_per_pixel[lvl]
            A = sat_feat_dict[lvl].shape[-1]
            # 20 is the legacy hardcoded ``shift_range_lat=20``; we keep the
            # exact crop size so existing checkpoints behave identically.
            crop_H = int(A - 20 * 3 / meter_per_pixel)
            crop_W = int(A - 20 * 3 / meter_per_pixel)
            g2s_feat_dict[lvl] = TF.center_crop(bev_feat, [crop_H, crop_W])
            g2s_conf_dict[lvl] = TF.center_crop(bev_conf, [crop_H, crop_W])

        render_loss = torch.tensor(0.0, device=sat_map.device)
        return (
            sat_feat_dict,
            sat_conf_dict,
            g2s_feat_dict,
            g2s_conf_dict,
            mask_dict,
            shift_lats,
            shift_lons,
            thetas,
            render_loss,
        )
