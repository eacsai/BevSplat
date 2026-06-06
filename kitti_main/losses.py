"""Loss functions and correlation primitives for the KITTI main experiment.

All math is copied verbatim from ``models/models_kitti_nips.py``:

    batch_wise_cross_corr                    L573  ->  used in training
    weak_supervise_loss                      L766  ->  L_Weakly when GPS_error_coe == 0
    Weakly_supervised_loss_w_GPS_error       L785  ->  L_Weakly + L_GPS pair (paper Eq. 1)
    corr_for_translation                     L902  ->  used at eval time

The shapes that matter:

    sat_feat_dict[level]    [B, C, A, A]                A = 128 for level=1
    sat_conf_dict[level]    [B, 1, A, A]
    g2s_feat_dict[level]    [B, C, crop_H, crop_W]      cropped to A - 60/mpp
    g2s_conf_dict[level]    [B, 1, crop_H, crop_W]
    corr_maps[level]        [B, B, H, W]                M sat tiles vs N grd queries (training)
    corr (eval)             [B, H, W]                   one query per sat tile

Convention: ``corr = 2 - 2 * cosine_similarity``, so smaller is more
similar. ``Peak(P) = corr.min()`` and the InfoNCE-style softplus on
``pos - dis`` pushes the GT peak below all negatives.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Training-time correlation: every grd query vs every sat tile in the batch
# ---------------------------------------------------------------------------

def batch_wise_cross_corr(
    sat_feat_dict,
    sat_conf_dict,
    g2s_feat_dict,
    g2s_conf_dict,
    args,
    masks=None,
):
    """Cosine-distance correlation, batch-wise.

    For each pyramid level, produces a ``[M, N, H, W]`` map where ``M`` is
    the number of satellite tiles in the batch and ``N`` is the number of
    ground queries. ``corr[m, n, h, w]`` is small when the ground BEV
    feature ``n`` aligns with the satellite tile ``m`` at sliding-window
    position ``(h, w)``. Diagonal entries (``m == n``) are the GT pairs.

    The ``args.ConfGrd`` / ``args.ConfSat`` knobs weight the numerator and
    the denominator by per-pixel confidence maps. Default config uses
    ConfGrd=1, ConfSat=0.
    """

    levels = sorted(int(item) for item in args.level.split("_"))
    corr_maps = {}
    for level in levels:
        sat_feat = sat_feat_dict[level]
        sat_conf = sat_conf_dict[level]
        g2s_feat = g2s_feat_dict[level]
        g2s_conf = g2s_conf_dict[level]

        B, C, crop_H, crop_W = g2s_feat.shape

        if args.ConfGrd > 0:
            if args.ConfSat > 0:
                # Both sides confidence-weighted.
                signal = (sat_feat * sat_conf.pow(2)).repeat(1, B, 1, 1)
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                denominator_sat = []
                sat_feat_conf_pow = (sat_feat * sat_conf).pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(B):
                    denom_sat = torch.sum(
                        F.conv2d(sat_feat_conf_pow[i, :, None, :, :], g2s_conf_pow),
                        dim=0,
                    )
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))

                denominator_grd = []
                sat_conf_pow = sat_conf.pow(2)
                g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
                for i in range(B):
                    denom_grd = torch.sum(
                        F.conv2d(
                            sat_conf_pow[i : i + 1, :, :, :].repeat(1, C, 1, 1),
                            g2s_feat_conf_pow,
                        ),
                        dim=1,
                    )
                    denominator_grd.append(denom_grd)
                denominator_grd = torch.sqrt(torch.stack(denominator_grd, dim=0))

            else:
                # Ground-side confidence only (the default).
                signal = sat_feat.repeat(1, B, 1, 1)
                kernel = g2s_feat * g2s_conf.pow(2)
                corr = F.conv2d(signal, kernel, groups=B)

                denominator_sat = []
                sat_feat_pow = sat_feat.pow(2)
                g2s_conf_pow = g2s_conf.pow(2)
                for i in range(B):
                    denom_sat = torch.sum(
                        F.conv2d(sat_feat_pow[i, :, None, :, :], g2s_conf_pow),
                        dim=0,
                    )
                    denominator_sat.append(denom_sat)
                denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))

                denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1)
                shape = denominator_sat.shape
                denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

        else:
            # No confidence weighting; use a binary BEV mask to limit the L2 norm
            # to the valid (rendered) region of the BEV plane.
            mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()
            signal = sat_feat.repeat(1, B, 1, 1)
            kernel = g2s_feat
            corr = F.conv2d(signal, kernel, groups=B)

            l2_norm_kernel = mask.repeat(1, C, 1, 1)
            sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)
            denominator_sat = torch.sqrt(sat_feat_squared_sum + 1e-8)

            denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)
            shape = denominator_sat.shape
            denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])

        denominator = torch.maximum(denominator_sat * denominator_grd, torch.full_like(denominator_sat, 1e-6))
        corr_maps[level] = 2 - 2 * corr / denominator

    return corr_maps


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def weak_supervise_loss(corr_maps):
    """``L_Weakly`` from paper Eq. (1) — InfoNCE-style softplus on peak distances.

    ``Peak(P_i)`` is the minimum of corr_map ``i``. For each sample we push
    the diagonal (positive) peak below every off-diagonal (negative) peak.
    """
    losses = []
    for _, corr in corr_maps:
        M, N, _, _ = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]   # [M, N]
        pos = torch.diagonal(dis)                             # [M]
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N - 1))
        losses.append(loss)
    return torch.mean(torch.stack(losses, dim=0))


def Weakly_supervised_loss_w_GPS_error(
    corr_maps,
    gt_shift_u,
    gt_shift_v,
    gt_heading,
    args,
    meter_per_pixels,
    GPS_error: float = 5,
):
    """Returns ``(L_Weakly, L_GPS)`` (paper Eq. 1 components).

    L_GPS penalizes the gap between the global peak ``Peak(P_pos)`` and the
    best peak inside a GPS-error-radius patch around the noisy GT location.
    Set ``args.GPS_error_coe = 0`` to ignore L_GPS at training time (λ₁=0
    setting in Table 1).
    """

    matching_losses = []
    GPS_error_losses = []

    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    gt_delta_x = -gt_shift_u[:, 0] * args.shift_range_lon
    gt_delta_y = -gt_shift_v[:, 0] * args.shift_range_lat
    gt_delta_x_rot = -gt_delta_x * cos + gt_delta_y * sin
    gt_delta_y_rot = gt_delta_x * sin + gt_delta_y * cos

    levels = [int(item) for item in args.level.split("_")]
    for level in levels:
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N

        # --- L_Weakly: softplus(Peak(P_neg) - Peak(P_pos)) -------------
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis)
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N - 1))
        matching_losses.append(loss)

        # --- L_GPS: |Peak(P_pos) - Peak inside GPS radius| -------------
        meter_per_pixel = meter_per_pixels[level]
        w = torch.round(W / 2 - 0.5 + gt_delta_x_rot / meter_per_pixel).long()
        h = torch.round(H / 2 - 0.5 + gt_delta_y_rot / meter_per_pixel).long()
        radius = int(np.ceil(GPS_error / meter_per_pixel))

        GPS_dis = []
        for b in range(M):
            start_h = torch.max(torch.tensor(0).long(), h[b] - radius)
            end_h = torch.min(torch.tensor(corr.shape[2]).long(), h[b] + radius)
            start_w = torch.max(torch.tensor(0).long(), w[b] - radius)
            end_w = torch.min(torch.tensor(corr.shape[3]).long(), w[b] + radius)
            GPS_dis.append(torch.min(corr[b, b, start_h:end_h, start_w:end_w]))
        GPS_error_losses.append(torch.abs(torch.stack(GPS_dis) - pos))

    return (
        torch.mean(torch.stack(matching_losses, dim=0)),
        torch.mean(torch.stack(GPS_error_losses, dim=0)),
    )


# ---------------------------------------------------------------------------
# Inference-time correlation: each query vs its own satellite tile
# ---------------------------------------------------------------------------

def corr_for_translation(
    sat_feat_dict,
    sat_conf_dict,
    g2s_feat_dict,
    g2s_conf_dict,
    args,
    meter_per_pixels,
    gt_heading,
    masks=None,
):
    """Eval-time correlation — returns ``(pred_u, pred_v, corr)``.

    ``pred_u`` and ``pred_v`` are the argmax-decoded shifts in METERS in
    the rotated camera frame (longitudinal / lateral after applying the
    GT heading, used to project the result back onto north/east axes).

    ``corr`` is the cropped ``[B, H, W]`` correlation map clipped to the
    valid pose-search window (``shift_range_lat * 3`` / meter_per_pixel).
    """

    level = max(int(item) for item in args.level.split("_"))
    meter_per_pixel = meter_per_pixels[level]

    sat_feat = sat_feat_dict[level]
    sat_conf = sat_conf_dict[level]
    g2s_feat = g2s_feat_dict[level]
    g2s_conf = g2s_conf_dict[level]

    B, C, crop_H, crop_W = g2s_feat.shape
    A = sat_feat.shape[2]

    if args.ConfGrd > 0:
        if args.ConfSat > 0:
            signal = (sat_feat * sat_conf.pow(2)).reshape(1, -1, A, A)
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]

            sat_feat_conf_pow = (sat_feat * sat_conf).pow(2).transpose(0, 1)
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_conf_pow, g2s_conf_pow, groups=B).transpose(0, 1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

            sat_conf_pow = sat_conf.pow(2).repeat(1, C, 1, 1).reshape(1, -1, A, A)
            g2s_feat_conf_pow = (g2s_feat * g2s_conf).pow(2)
            denominator_grd = F.conv2d(sat_conf_pow, g2s_feat_conf_pow, groups=B)[0]
            denominator_grd = torch.sqrt(denominator_grd)

        else:
            signal = sat_feat.reshape(1, -1, A, A)
            kernel = g2s_feat * g2s_conf.pow(2)
            corr = F.conv2d(signal, kernel, groups=B)[0]

            sat_feat_pow = sat_feat.pow(2).transpose(0, 1)
            g2s_conf_pow = g2s_conf.pow(2)
            denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)
            denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))

            denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1)
            shape = denominator_sat.shape
            denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

    else:
        signal = sat_feat.reshape(1, -1, A, A)
        kernel = g2s_feat
        corr = F.conv2d(signal, kernel, groups=B)[0]

        mask = TF.center_crop(masks[level].permute(0, 3, 1, 2), [crop_H, crop_W]).float()
        l2_norm_kernel = mask.repeat(1, C, 1, 1)
        sat_feat_squared_sum = F.conv2d(signal.pow(2), l2_norm_kernel, stride=1, padding=0, groups=B)[0]
        denominator_sat = torch.maximum(
            torch.sqrt(sat_feat_squared_sum + 1e-8),
            torch.full_like(sat_feat_squared_sum, 1e-6),
        )

        denom_grd = torch.linalg.norm(g2s_feat.reshape(B, -1), dim=-1)
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])

    denominator = torch.maximum(denominator_sat * denominator_grd, torch.full_like(denominator_sat, 1e-6))
    corr = corr / denominator

    corr_H = int(args.shift_range_lat * 3 / meter_per_pixel)
    corr_W = int(args.shift_range_lon * 3 / meter_per_pixel)
    corr = TF.center_crop(corr[:, None], [corr_H, corr_W])[:, 0]

    B, corr_H, corr_W = corr.shape
    max_index = torch.argmax(corr.reshape(B, -1), dim=1)

    if getattr(args, "visualize", 0):
        # Keep the legacy visualization escape hatch: return raw pixel offsets.
        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * np.power(2, 3 - level)
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * np.power(2, 3 - level)
        return pred_u, pred_v, corr

    # Meter-space decoding rotated back into north/east axes via gt_heading.
    pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel
    pred_v = -(max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel

    cos = torch.cos(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    sin = torch.sin(gt_heading[:, 0] * args.rotation_range / 180 * np.pi)
    pred_u1 = pred_u * cos + pred_v * sin
    pred_v1 = -pred_u * sin + pred_v * cos
    return pred_u1, pred_v1, corr
