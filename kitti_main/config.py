"""Centralized constants for the KITTI main experiment.

Every magic number that was previously scattered across
``train_KITTI_weak_nips.py`` and ``models/models_kitti_nips.py`` lives here.
Values come from the *existing code*, not the paper. Where the paper and
code disagree, a comment notes which value we are using.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Filesystem
# ----------------------------------------------------------------------

# KITTI dataset root. Set in dataLoader/KITTI_dataset.py:21.
KITTI_ROOT = Path("/data/dataset/KITTI")

# Where checkpoints land. Mirrors train_KITTI_weak_nips.py:943.
CKPT_ROOT = Path("/data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF")

# Stage-0 (rotation pre-head) checkpoint used to initialize Stage-1 training
# when --rotation_range > 0. Source: train_KITTI_weak_nips.py:1028.
STAGE0_INIT_CKPT = (
    CKPT_ROOT / "Stage0"
    / "lat20.0m_lon20.0m_rot10.0_Nit1_TransV1_geo_Level1_Channels32_16_4_feat32"
    / "model_4.pth"
)

# ----------------------------------------------------------------------
# Image / BEV geometry
# ----------------------------------------------------------------------

# Dataset-side raw ground-image size (KITTI rectified) — used by the rotation
# pre-head and as `ori_grdH/ori_grdW` for the camera intrinsic rescaling.
# Source: dataLoader/KITTI_dataset.py:34-35.
RAW_GRD_H = 256
RAW_GRD_W = 1024

# Ground image size that the DINOv2 + DPT branch consumes after F.interpolate.
# Source: models_kitti_nips.py:448 ``F.interpolate(grd_img_left, size=(64, 256))``.
GRD_H = 64
GRD_W = 256

# Satellite image size that the DINOv2 + DPT branch consumes.
# Source: models_kitti_nips.py:454 ``F.interpolate(sat_map, size=(128, 128))``.
SAT_H = 128
SAT_W = 128

# BEV extent (meters across the BEV plane, half-width/half-height fed to the
# orthographic splat). Source: models_kitti_nips.py:571-572.
BEV_WIDTH_M = 101.0
BEV_HEIGHT_M = 101.0
BEV_RES = (128, 128)  # rendered BEV pixel resolution

# Gaussians per ground-image pixel. Source: encoder_feat_nips.py:87.
GAUSSIANS_PER_PIXEL = 3

# ----------------------------------------------------------------------
# Feature pyramid / model knobs
# ----------------------------------------------------------------------

# Only one level is used by the Stage-1 main experiment.
# Source: train command lines in .vscode/launch.json.
LEVEL = 1
LEVEL_STR = "1"          # passed through to legacy args.level
CHANNELS_STR = "32_16_4"  # passed through to legacy args.channels

# Match the legacy implementation's argparse defaults so checkpoint key names
# (e.g. self.FeatureForT vs self.GrdFeatureForT) stay identical.
SHARE_FEATURE_NET = 1     # args.share = 1
CONF_GRD = 1              # args.ConfGrd = 1
CONF_SAT = 0              # args.ConfSat = 0
N_ITERS = 1               # args.N_iters = 1
PROJ = "geo"              # args.proj = "geo"
OPTIMIZER_KIND = "TransV1"  # args.Optimizer = "TransV1"

# ----------------------------------------------------------------------
# Pose perturbation / search range
# ----------------------------------------------------------------------

SHIFT_RANGE_LAT = 20.0     # meters (paper effective search 56x56 m²)
SHIFT_RANGE_LON = 20.0
ROTATION_RANGE = 10.0      # degrees

# ----------------------------------------------------------------------
# Training hyperparameters (Stage 1)
# ----------------------------------------------------------------------

# Optimizer settings used by train_KITTI_weak_nips.py:772.
# NOTE: paper says weight_decay 1e-3; existing code uses 5e-3. We follow code.
LR = 6.25e-5
WEIGHT_DECAY = 5e-3
ADAM_EPS = 1e-8
BATCH_SIZE = 8
EPOCHS = 10

# OneCycleLR settings (train_KITTI_weak_nips.py:775-782).
ONECYCLE_STEPS_PER_EPOCH_BASE = 2456  # train set has 2456 batches at BS=8
ONECYCLE_PCT_START = 0.05
ONECYCLE_ANNEAL = "cos"

# ----------------------------------------------------------------------
# Loss (paper Eq. 1: L_all = L_Weakly + λ₁ · L_GPS)
# ----------------------------------------------------------------------

# λ₁ in the paper. 0 → use only the weakly-supervised triplet on correlation
# peaks; 1 → also include the GPS-error term (assumes noisy location labels
# during training, with the radius set by GPS_ERROR meters).
GPS_ERROR_COE_LAMBDA1 = (0.0, 1.0)
GPS_ERROR_RADIUS_M = 5.0   # args.GPS_error

# ----------------------------------------------------------------------
# Evaluation thresholds (Table 1 columns)
# ----------------------------------------------------------------------

# Distance thresholds in meters for "within X meters" accuracy columns.
DIST_THRESHOLDS_M = (1, 3, 5)
# Angle thresholds in degrees.
ANGLE_THRESHOLDS_DEG = (1, 3, 5)


def stage1_save_dir(
    *,
    stage: int = 1,
    shift_lat: float = SHIFT_RANGE_LAT,
    shift_lon: float = SHIFT_RANGE_LON,
    rotation: float = ROTATION_RANGE,
    n_iters: int = N_ITERS,
    optimizer: str = OPTIMIZER_KIND,
    proj: str = PROJ,
    level_str: str = LEVEL_STR,
    channels_str: str = CHANNELS_STR,
    gps_error: float = GPS_ERROR_RADIUS_M,
    gps_error_coe: float = 0.0,
    conf_sat: int = CONF_SAT,
    share: int = SHARE_FEATURE_NET,
    name: str = "",
) -> Path:
    """Reproduce the save_path layout from train_KITTI_weak_nips.py:942-965.

    Keeping the same directory naming means existing checkpoints can be
    resumed by the refactored trainer without renaming any files.
    """

    parts = (
        f"Stage{stage}/"
        f"lat{shift_lat}m_lon{shift_lon}m_rot{rotation}"
        f"_Nit{n_iters}_{optimizer}_{proj}"
        f"_Level{level_str}_Channels{channels_str}"
    )
    path = CKPT_ROOT / parts
    if conf_sat and stage > 0:
        path = path.parent / (path.name + "_ConfSat")
    if gps_error_coe > 0 and stage > 0:
        path = path.parent / (path.name + f"_GPSerror{gps_error}_Coe{gps_error_coe}")
    if share and stage > 0:
        path = path.parent / (path.name + "_Share")
    if name:
        path = path.parent / (path.name + f"_{name}")
    return path
