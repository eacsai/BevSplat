# `kitti_main/` — clean entry point for the KITTI main experiment

This directory is the recommended way to run the BevSplat KITTI experiments from Table 1 of the paper (arXiv 2502.09080). It is a focused re-skinning of the algorithm that already lives in the repo: the heavy modules (DINOv2 backbone, DPT head, `GaussianFeatEncoder`, CUDA orthographic splatter, KITTI dataloader) are **imported verbatim** from their original files. Only the orchestration around them — argparse, training loop, evaluation, paths — is rewritten for readability.

## What you need

1. **KITTI dataset** mounted at `/data/dataset/KITTI` with these subdirectories:
   - `satmap/` — satellite tiles
   - `depth_data/` — KITTI raw drives with `image_02/grd_no_sky/`, `image_02/data/`, `image_02/grd_depth/*_grd_depth.pt`, `oxts/data/`, `calib_cam_to_cam.txt`
   - File lists `dataLoader/train_files.txt`, `dataLoader/test1_files.txt`, `dataLoader/test2_files.txt` (already in the repo).
2. **CUDA rasterizer extension** built and installed:
   ```bash
   cd /home/qiwei/program/BevSplat/feature_gaussian
   pip install -e .
   ```
   This produces `feat_gaussian._C`, which `gaussian/latent_splat_feat.py` imports via the orthographic splat helper.
3. A Python environment with `torch`, `torchvision`, `einops`, `jaxtyping`, `scipy`, `Pillow`, `matplotlib`, `opencv-python`.
4. The Stage-0 rotation-pre-head checkpoint at `/data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage0/lat20.0m_lon20.0m_rot10.0_Nit1_TransV1_geo_Level1_Channels32_16_4_feat32/model_4.pth`. It's already on disk on the development server; this file is what `--stage1_init` defaults to in `train.py`.

## Architecture in 200 words

The model resolves a ground vehicle's planar pose relative to a satellite tile. The ground image is encoded by DINOv2 + a DPT head into a 32-channel feature map + per-pixel confidence (`models/dino_fit.py` + `models/dpt_single.py`). Each ground-image pixel is then lifted into 3D using pre-computed metric depth (`*_grd_depth.pt`), spawning `Np = 3` Gaussian primitives whose positions, opacities, rotations and scales are predicted by `GaussianFeatEncoder` (which carries its own DINO-pretrained ResNet backbone). The feature/confidence channels of each Gaussian come from the DPT output. These Gaussians are rendered orthographically into a 128×128 BEV plane covering ~101 m × 101 m via the custom CUDA splatter in `feature_gaussian._C` (wrapped by `vis_gaussian_feat.render_projections`). The BEV feature map is cross-correlated with the satellite tile's DPT features (also 32-d, 128×128) by `batch_wise_cross_corr`. The training loss is the InfoNCE-style softplus on correlation peaks (`L_Weakly`) plus, optionally, a peak-consistency term inside a 5 m radius around the noisy GT (`L_GPS`); the combination is `L = L_Weakly + λ₁ · L_GPS` (paper Eq. 1).

```
grd_img (256x1024) ─► resize (64,256) ─► DINOv2 ─► DPT ─► grd_feat (32), grd_conf
                                                ▲
              grd_depth (pre-computed) ─────────┘
                                                ▼
                                       GaussianFeatEncoder
                                                │
                  Stage-0 rotation pre-head (no_grad) ──► heading θ
                                                ▼
                                   render_projections (CUDA orth splat)
                                                │
                                       BEV feat (32, 128, 128)
                                                │
                                                ▼
                                       cross_corr  ─►  argmax / softplus loss
                                                ▲
sat_map (512x512) ─► resize (128,128) ─► DINOv2 ─► DPT ─► sat_feat (32, 128, 128), sat_conf
```

## Reproduce Table 1

The four rows that the paper labels "Ours" map one-to-one to two checkpoints (Same and Cross share the same model — they differ only in which test split is used). Both retraining and eval-only commands are below.

### Retrain from scratch (Stage-0 init → Stage-1 → eval)

The trainer auto-loads the Stage-0 init from `--stage1_init` when `--rotation_range > 0`.

```bash
cd /home/qiwei/program/BevSplat

# Same-Area + Cross-Area, λ₁ = 0     (paper: Same 5.82/2.85, Cross 7.05/3.22)
python -m kitti_main.train \
    --GPS_error_coe 0 \
    --rotation_range 10 \
    --epochs 10 \
    --name reproduce_lambda0

# Same-Area + Cross-Area, λ₁ = 1     (paper: Same 2.87/2.06, Cross 6.20/2.51)
python -m kitti_main.train \
    --GPS_error_coe 1 \
    --rotation_range 10 \
    --epochs 10 \
    --name reproduce_lambda1
```

Each command trains one model, saves `model_0.pth … model_9.pth` under `/data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage1/lat20.0m_lon20.0m_rot10.0_Nit1_TransV1_geo_Level1_Channels32_16_4{_GPSerror5_Coe1.0}_Share_reproduce_lambda{0,1}/`, and after every epoch evaluates on both `test1` (Same-Area) and `test2` (Cross-Area). Allow ~10 hours per run on a single 4090.

### Evaluate existing checkpoints (no retraining)

These checkpoints already exist on the development server:

| Checkpoint dir | λ₁ | rot used during train | Notes |
|---|---|---|---|
| `Stage1/.../Channels32_16_4_Share_feat32_depth/model_9.pth` | 0 | 10° | matches paper λ=0 row |
| `Stage1/.../Channels32_16_4_GPSerror5_Coe1.0_Share_feat32_depth_GPS/model_9.pth` | 1 | 10° | matches paper λ=1 row |

```bash
# λ=0
python -m kitti_main.train --test 1 \
    --ckpt /data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage1/lat20.0m_lon20.0m_rot10.0_Nit1_TransV1_geo_Level1_Channels32_16_4_Share_feat32_depth/model_9.pth \
    --GPS_error_coe 0 --rotation_range 0 --name verify_lambda0

# λ=1
python -m kitti_main.train --test 1 \
    --ckpt /data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage1/lat20.0m_lon20.0m_rot10.0_Nit1_TransV1_geo_Level1_Channels32_16_4_GPSerror5_Coe1.0_Share_feat32_depth_GPS/model_9.pth \
    --GPS_error_coe 1 --rotation_range 0 --name verify_lambda1
```

`--rotation_range 0` matches the test command lines that the legacy `.vscode/launch.json` uses — at eval time, the ground/satellite pairs come without rotation perturbation. The trained checkpoint still contains the rotation pre-head weights from training, but they're frozen and only consulted when the test perturbation is non-zero. Each eval run prints Table-1-shaped metrics to stdout and appends them to `<save_path>/test{1,2}_results.txt`.

### Smoke test (a few batches end-to-end, no training)

```bash
python -m kitti_main.train --epochs 1 --batch_size 2 --max_batches 5 \
    --rotation_range 0 --GPS_error_coe 1 --name smoke
```

## What was dropped from the legacy entry point

The legacy `train_KITTI_weak_nips.py` exposes a long argparse surface. For the Table-1 main experiment, the following CLI flags only have **one meaningful value** and have been removed here. Their fixed values are baked into `kitti_main/config.py`:

| Removed flag | Fixed value | Notes |
|---|---|---|
| `--level` | `"1"` | feature pyramid level used by the DPT branch |
| `--channels` | `"32_16_4"` | DPT output channel widths |
| `--share` | `1` | share feature backbone between sat/grd branches |
| `--ConfGrd` | `1` | use ground-side confidence weighting in correlation |
| `--ConfSat` | `0` | satellite-side confidence weighting (untested code path) |
| `--N_iters` | `1` | rotation pre-head iterations |
| `--proj` | `"geo"` | sat-grd projection method |
| `--Optimizer` | `"TransV1"` | rotation pre-head architecture choice |
| `--task` | `"3DoF"` | task label, only used in save_path |
| `--visualize` | `0` | matplotlib overlay for sample correlation maps |
| `--multi_gpu` | `0` | DataParallel wrapping (no checkpoint exists trained that way) |

Other legacy detritus deleted:

- The 80-line `show_cam_on_image` matplotlib helper (it only fired with `--visualize 1`).
- Five Stage-0-only test functions (`test1_orien`, `test2_orien`, `test1`, `test2`, plus internal duplicates) → one `kitti_main.eval.evaluate(...)`.
- Dead `Model.forward_project` method.
- `args.stage == 0` forward branch (Stage 0 is a separate, frozen pre-head; retrain via `python train_KITTI_weak_nips.py --stage 0` if you ever need to).
- Hardcoded `os.environ['CUDA_VISIBLE_DEVICES'] = '1'` at module load → replaced by `--cuda <id>` flag.
- The "load test checkpoint from this magic path" block at the bottom of the legacy script → replaced by explicit `--test 1 --ckpt <path>`.

## What was kept exactly as the legacy code had it

To preserve numerical behavior with existing checkpoints:

- `weight_decay = 5e-3` (the paper says `1e-3`; the code says `5e-3`; we kept the code value as instructed).
- `OneCycleLR` config with `pct_start=0.05`, cosine annealing, `steps_per_epoch = int(2456 / scale)`.
- Per-level meter-per-pixel scaling via `data_utils.get_meter_per_pixel() * 2**(3-level)`.
- The `if args.ConfSat > 0` branches inside `batch_wise_cross_corr` and `corr_for_translation`, even though the default never hits them — removing them would silently change tensor reshape semantics if anyone enables ConfSat.
- The submodule names inside `BevSplatKITTI.__init__` (`SatFeatureNet`, `FeatureForT`, `TransRefine`, `coe_R`, `coe_T`) match the legacy `Model`, so checkpoints load via `strict=False` with zero key remapping.

## Where the algorithm actually lives

`kitti_main/` does not re-implement the algorithm. Three files are the source of truth:

- `gaussian/encoder_feat_nips.py` — per-pixel Gaussian primitive encoder (means, covariances, opacities, features).
- `vis_gaussian_feat.py` — `render_projections(...)`: the orthographic BEV splat that calls into `feature_gaussian._C`.
- `models/models_kitti_nips.py` — legacy `Model` class; still consulted by the seq / weather experiments. Its Stage-1 forward (lines 495–655) is what `kitti_main/model.py:BevSplatKITTI.forward` mirrors.

Touch those three files if you want to change the algorithm itself; touch `kitti_main/` only to change orchestration, paths, defaults, or reporting.
