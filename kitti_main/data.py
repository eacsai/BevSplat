"""Thin wrappers around ``dataLoader.KITTI_dataset``.

The legacy ``load_train_data`` / ``load_test1_data`` / ``load_test2_data``
helpers do all the heavy lifting (transforms, file lists, distance-aware
batch sampling for weakly-supervised training). We expose them with
sensible defaults and document the shape of the batch they return.

A batch is a tuple of length 10:

    0  sat_align_cam    [B, 3, 512, 512]   satellite tile, north-aligned to vehicle heading
    1  sat_map          [B, 3, 512, 512]   satellite tile with the random shift+rot applied
    2  left_camera_k    [B, 3, 3]          KITTI left-camera intrinsics (rescaled to 256x1024)
    3  grd_left_imgs    [B, 3, 256, 1024]  ground perspective image (sky masked)
    4  grd_left_imgs_ori[B, 3, 256, 1024]  ground image without sky masking (unused by Stage 1)
    5  gt_shift_u       [B, 1]             GT longitudinal shift, normalized to [-1, 1]
    6  gt_shift_v       [B, 1]             GT lateral shift, normalized to [-1, 1]
    7  gt_heading       [B, 1]             GT heading offset, normalized to [-1, 1]
    8  grd_depth        [B, H, W]          pre-computed ground depth (*_grd_depth.pt files)
    9  file_name        list[str]          KITTI relative file path

Train batches additionally carry a 10th element ``GPS`` (Tensor[B, 2]) but
this is only used by the weakly-supervised distance batch sampler.
"""

from dataLoader.KITTI_dataset import (
    load_test1_data as _load_test1,
    load_test2_data as _load_test2,
    load_train_data as _load_train,
)

from .config import ROTATION_RANGE, SHIFT_RANGE_LAT, SHIFT_RANGE_LON


def load_train(
    batch_size: int,
    *,
    shift_range_lat: float = SHIFT_RANGE_LAT,
    shift_range_lon: float = SHIFT_RANGE_LON,
    rotation_range: float = ROTATION_RANGE,
    data_amount: float = 1.0,
):
    """Weakly-supervised KITTI training loader (uses DistanceBatchSampler)."""
    return _load_train(
        batch_size,
        shift_range_lat=shift_range_lat,
        shift_range_lon=shift_range_lon,
        rotation_range=rotation_range,
        weak_supervise=True,
        train_noisy=False,
        stage=1,
        data_amount=data_amount,
    )


def load_test1(
    batch_size: int,
    *,
    shift_range_lat: float = SHIFT_RANGE_LAT,
    shift_range_lon: float = SHIFT_RANGE_LON,
    rotation_range: float = ROTATION_RANGE,
):
    """KITTI ``test1`` split — corresponds to Same-Area in Table 1."""
    return _load_test1(batch_size, shift_range_lat, shift_range_lon, rotation_range)


def load_test2(
    batch_size: int,
    *,
    shift_range_lat: float = SHIFT_RANGE_LAT,
    shift_range_lon: float = SHIFT_RANGE_LON,
    rotation_range: float = ROTATION_RANGE,
):
    """KITTI ``test2`` split — corresponds to Cross-Area in Table 1."""
    return _load_test2(batch_size, shift_range_lat, shift_range_lon, rotation_range)
