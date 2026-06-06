"""KITTI evaluation — one function replacing legacy ``test1`` / ``test2``.

Both legacy functions (``train_KITTI_weak_nips.py:265`` and ``:575``) were
near-duplicates: same metric set, different file names, with a 130-line
matplotlib block bolted onto ``test1``. This module collapses them into a
single ``evaluate(loader, split_name, ...)`` call and reports the columns
in the same order as Table 1 of the paper.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io as scio
import torch

from .config import ANGLE_THRESHOLDS_DEG, DIST_THRESHOLDS_M
from .losses import corr_for_translation


@torch.no_grad()
def evaluate(
    net,
    loader,
    args,
    save_path: str | Path,
    epoch: int,
    split_name: str,
    *,
    device: torch.device | str = "cuda",
    dist_thresholds: Sequence[int] = DIST_THRESHOLDS_M,
    angle_thresholds: Sequence[int] = ANGLE_THRESHOLDS_DEG,
):
    """Evaluate a model on one KITTI test split.

    Parameters
    ----------
    net
        ``BevSplatKITTI`` (or the legacy ``Model``) instance.
    loader
        DataLoader from ``kitti_main.data.load_test1`` or ``load_test2``.
    args
        Namespace forwarded to ``corr_for_translation`` (needs
        ``level``, ``ConfGrd``, ``ConfSat``, ``shift_range_lat``,
        ``shift_range_lon``, ``rotation_range``).
    save_path
        Directory where ``<split>_results.txt`` and ``<split>_result.mat``
        are written.
    epoch
        Epoch number — purely informational, recorded in the txt header.
    split_name
        Label written into filenames and stdout (e.g. ``"test1"`` for
        Same-Area, ``"test2"`` for Cross-Area).

    Returns
    -------
    dict
        Flat metric dict; keys match the Table 1 columns.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    net.eval()

    pred_lons, pred_lats, pred_oriens = [], [], []
    gt_lons, gt_lats, gt_oriens = [], [], []

    n_batches = len(loader)
    print(f"[{split_name}] batch_size={args.batch_size}, num_batches={n_batches}")
    t0 = time.time()
    for i, batch in enumerate(loader):
        (
            sat_align_cam,
            sat_map,
            left_camera_k,
            grd_left_imgs,
            grd_left_imgs_ori,
            gt_shift_u,
            gt_shift_v,
            gt_heading,
            grd_depth,
        ) = (item.to(device) for item in batch[:9])

        sat_feat_d, sat_conf_d, g2s_feat_d, g2s_conf_d, mask_d, _, _, thetas, _ = net(
            sat_align_cam,
            sat_map,
            grd_left_imgs,
            grd_depth,
            grd_left_imgs_ori,
            left_camera_k,
            gt_heading,
        )

        pred_u, pred_v, _ = corr_for_translation(
            sat_feat_d,
            sat_conf_d,
            g2s_feat_d,
            g2s_conf_d,
            args,
            net.meters_per_pixel,
            gt_heading=gt_heading,
            masks=mask_d,
        )
        pred_orien = thetas[:, -1, -1]

        pred_lons.append(pred_u.cpu().numpy())
        pred_lats.append(pred_v.cpu().numpy())
        pred_oriens.append(pred_orien.cpu().numpy() * args.rotation_range)
        gt_lons.append(gt_shift_u[:, 0].cpu().numpy() * args.shift_range_lon)
        gt_lats.append(gt_shift_v[:, 0].cpu().numpy() * args.shift_range_lat)
        gt_oriens.append(gt_heading[:, 0].cpu().numpy() * args.rotation_range)

        if i % 20 == 0:
            print(f"[{split_name}] batch {i}/{n_batches}")

    duration_per_image = (time.time() - t0) / max(n_batches * args.batch_size, 1)

    pred_lons = np.concatenate(pred_lons)
    pred_lats = np.concatenate(pred_lats)
    pred_oriens = np.concatenate(pred_oriens)
    gt_lons = np.concatenate(gt_lons)
    gt_lats = np.concatenate(gt_lats)
    gt_oriens = np.concatenate(gt_oriens)

    scio.savemat(
        save_path / f"{split_name}_result.mat",
        {
            "gt_lons": gt_lons,
            "gt_lats": gt_lats,
            "gt_oriens": gt_oriens,
            "pred_lons": pred_lons,
            "pred_lats": pred_lats,
            "pred_oriens": pred_oriens,
        },
    )

    metrics = _compute_metrics(
        pred_lons, pred_lats, pred_oriens, gt_lons, gt_lats, gt_oriens,
        dist_thresholds=dist_thresholds, angle_thresholds=angle_thresholds,
    )

    _report(save_path / f"{split_name}_results.txt", metrics, split_name, epoch, duration_per_image,
            dist_thresholds, angle_thresholds)

    net.train()
    return metrics


def _compute_metrics(
    pred_lons,
    pred_lats,
    pred_oriens,
    gt_lons,
    gt_lats,
    gt_oriens,
    *,
    dist_thresholds,
    angle_thresholds,
):
    """Compute the full Table 1 metric set."""
    distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)
    diff_lats = np.abs(pred_lats - gt_lats)
    diff_lons = np.abs(pred_lons - gt_lons)
    angle_diff = np.remainder(np.abs(pred_oriens - gt_oriens), 360)
    angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)

    metrics = {
        "loc_mean_m": float(np.mean(distance)),
        "loc_median_m": float(np.median(distance)),
        "lat_mean_m": float(np.mean(diff_lats)),
        "lat_median_m": float(np.median(diff_lats)),
        "lon_mean_m": float(np.mean(diff_lons)),
        "lon_median_m": float(np.median(diff_lons)),
        "angle_mean_deg": float(np.mean(angle_diff)),
        "angle_median_deg": float(np.median(angle_diff)),
    }
    for t in dist_thresholds:
        metrics[f"lat_d={t}m_%"] = float(np.mean(diff_lats < t) * 100)
        metrics[f"lon_d={t}m_%"] = float(np.mean(diff_lons < t) * 100)
        metrics[f"dist_d={t}m_%"] = float(np.mean(distance < t) * 100)
    for t in angle_thresholds:
        metrics[f"angle_t={t}deg_%"] = float(np.mean(angle_diff < t) * 100)
    return metrics


def _report(
    out_path: Path,
    metrics: dict,
    split_name: str,
    epoch: int,
    duration_per_image: float,
    dist_thresholds: Sequence[int],
    angle_thresholds: Sequence[int],
):
    """Write the metrics in the same order as Table 1 columns of the paper."""
    lines = [
        "====================================",
        f"  {split_name}   EPOCH: {epoch}",
        f"  Time per image (s): {duration_per_image:.4f}",
        "------------------------------------",
        f"  Localization mean  (m): {metrics['loc_mean_m']:.3f}",
        f"  Localization median(m): {metrics['loc_median_m']:.3f}",
        f"  Lateral mean  (m): {metrics['lat_mean_m']:.3f}",
        f"  Lateral median(m): {metrics['lat_median_m']:.3f}",
        f"  Longitudinal mean  (m): {metrics['lon_mean_m']:.3f}",
        f"  Longitudinal median(m): {metrics['lon_median_m']:.3f}",
        f"  Azimuth mean  (deg): {metrics['angle_mean_deg']:.3f}",
        f"  Azimuth median(deg): {metrics['angle_median_deg']:.3f}",
        "------------------------------------",
    ]
    for t in dist_thresholds:
        lines.append(f"  Lateral d={t}m  : {metrics[f'lat_d={t}m_%']:.2f} %")
    for t in dist_thresholds:
        lines.append(f"  Longitudinal d={t}m: {metrics[f'lon_d={t}m_%']:.2f} %")
    for t in dist_thresholds:
        lines.append(f"  Localization within d={t}m: {metrics[f'dist_d={t}m_%']:.2f} %")
    for t in angle_thresholds:
        lines.append(f"  Azimuth theta={t}deg: {metrics[f'angle_t={t}deg_%']:.2f} %")
    lines.append("====================================")

    body = "\n".join(lines) + "\n"
    print(body)
    with out_path.open("a") as f:
        f.write(body)
