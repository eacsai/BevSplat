# `vigor_main/` — clean entry point for the VIGOR 2DoF experiment

Same shape as `kitti_main/` but for the panoramic / VIGOR side of the paper. The algorithm modules (`gaussian/encoder_pano.py`, `vis_gaussian_pano.render_projections`, `models/dino_fit.py`, `models/dpt_single.py`, `dataLoader/Vigor_dataset_gs.py`) are imported verbatim; only the orchestration around them is rewritten.

## What you need

1. **VIGOR dataset** mounted at `/data/dataset/VIGOR` with these per-city subdirectories (NewYork, Chicago, SanFrancisco, Seattle):
   - `satellite/` — 512×512 satellite tiles
   - `panorama/` — original panoramic images
   - `pano_mask_sky/` — sky-masked panoramas (what the model consumes)
   - `UniK3D_<split>_metric/` — pre-computed metric depth as `<id>_depth.npy`
   - `splits__corrected/` — `same_area_balanced_{train,test}__corrected.txt` and `pano_label_balanced__corrected.txt`
2. **CUDA rasterizer extension**:
   ```bash
   cd /home/qiwei/program/BevSplat/pano_feature_gaussian
   pip install -e .
   ```
   Produces `pano_gaussian_feat._C`, called by `gaussian/pano_splat.render_cuda_orthographic`. If `third_party/glm/glm/` is empty (the git submodule is missing in this repo), copy headers from any other 3DGS-style project, e.g.:
   ```bash
   cp -r ~/CVPR26/FiT3D/submodules/diff-feature-gaussian-rasterization/third_party/glm/glm pano_feature_gaussian/third_party/glm/
   ```
3. Python env with `torch`, `torchvision`, `einops`, `jaxtyping`, `scipy`, `Pillow`, `matplotlib`, `e3nn`, `timm`, `open3d` (the last one is imported at the top of `gaussian/encoder_pano.py`).

## Architecture in 200 words

The VIGOR pipeline mirrors KITTI's but adapts to panoramic ground imagery. Sat (512×512) and panoramic grd (any size) are fed to DINOv2 + DPT producing 32-d feature + confidence maps, with two DPT heads (`dpt_sat` / `dpt_grd`) when `share=0` (the default). The panoramic grd is then resized to `80×160` (height × width, 2:1 equirectangular aspect) and passed to `GaussianEncoder` (`gaussian/encoder_pano.py`), which uses a DINO-pretrained ResNet backbone to predict each pixel's `Np=3` Gaussian primitives. Pixel coordinates are converted into 3D rays via the equirectangular projection (`equirectangular_to_xyz`) and scaled by the metric depth. The DPT feature/confidence are attached to each Gaussian. `vis_gaussian_pano.render_projections` then splats the Gaussians orthographically through the `pano_gaussian_feat._C` CUDA kernel into a 128×128 BEV covering `70m × 70m`. The BEV is centre-cropped to `int(128 × 0.4) = 51` and cross-correlated against the satellite DPT features. The loss is the same InfoNCE-style softplus on correlation peaks + an optional GPS-error term — `L = L_Weakly + λ₁ · L_GPS` (paper Eq. 1).

```
panorama (3, H_pano, W_pano) ─► DINOv2 ─► DPT ─► grd_feat (32), grd_conf
                          \
                           ─► resize (80, 160) ───────────────────────┐
                                                                       ▼
        depth_imgs (1, H_pano, W_pano) ─► resize (80, 160) ──► GaussianEncoder (panorama)
                                                                       │
                                                                       ▼
                                                       render_projections (CUDA orth splat)
                                                                       │
                                                                BEV (32, 128, 128) → centre-crop 51×51
                                                                       │
                                                                       ▼
                                                                    cross_corr
                                                                       ▲
sat (3, 512, 512) ─► DINOv2 ─► DPT ─► sat_feat (32, 128, 128), sat_conf
```

## Reproduce the VIGOR rows

### Retrain (Same-Area)

```bash
cd /home/qiwei/program/BevSplat

# Same-Area, λ₁ = 0
python -m vigor_main.train --area same --GPS_error_coe 0 --lr 1.25e-4 --epochs 15 --name reproduce_lambda0

# Same-Area, λ₁ = 1
python -m vigor_main.train --area same --GPS_error_coe 1 --lr 1.25e-4 --epochs 15 --name reproduce_lambda1
```

### Retrain (Cross-Area)

```bash
# Cross-Area, λ₁ = 0
python -m vigor_main.train --area cross --GPS_error_coe 0 --lr 6.5e-5 --epochs 15 --name reproduce_lambda0

# Cross-Area, λ₁ = 1
python -m vigor_main.train --area cross --GPS_error_coe 1 --lr 1e-4 --epochs 15 --name reproduce_lambda1
```

The legacy `train.sh` uses `lr=6.5e-5` for Cross-Area λ=0 and `lr=1e-4` for Cross-Area λ=1; keep those defaults if you want the existing save-paths to line up.

### Evaluate an existing checkpoint (no retraining)

Many trained checkpoints exist under `/data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/`. The ones matching the paper's main results are the `..._0.3_3.0_70_*_depth/` runs. Example:

```bash
# Same-Area, λ=1
python -m vigor_main.train --test 1 --area same \
    --ckpt /data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/same_rot0.0_geo_0.000125_Level1_Channels32_16_4_ConfGrd_GPSerror5_Coe1.0_Weakly_vigor_0.3_3.0_70_1.25e-4_depth/model_14.pth \
    --GPS_error_coe 1 --lr 1.25e-4 --name verify_lambda1

# Cross-Area, λ=1
python -m vigor_main.train --test 1 --area cross \
    --ckpt /data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/cross_rot0.0_geo_0.0001_Level1_Channels32_16_4_ConfGrd_GPSerror5_Coe1.0_Weakly_vigor_0.3_3.0_70_1e-4_depth/model_14.pth \
    --GPS_error_coe 1 --lr 1e-4 --name verify_lambda1
```

(The same-area λ=0 dir is `same_rot0.0_geo_0.000125_..._ConfGrd_Weakly_vigor_0.3_3.0_70_1.25e-4_depth/`; the GPSerror tag in the path implies λ=1.)

### Smoke test

```bash
python -m vigor_main.train --epochs 1 --batch_size 2 --max_batches 5 \
    --area same --GPS_error_coe 1 --name smoke
```

## What was dropped from the legacy entry point

The legacy `train_vigor_2DoF.py` has the same kind of long argparse surface that `train_KITTI_weak_nips.py` had. For the main VIGOR experiment these flags only have **one meaningful value** and are removed from `vigor_main/train.py` — they're fixed in `vigor_main/config.py`:

| Removed flag | Fixed value | Notes |
|---|---|---|
| `--level` | `"1"` | feature pyramid level |
| `--channels` | `"32_16_4"` | DPT channel widths |
| `--share` | `0` | use separate dpt_grd / dpt_sat |
| `--ConfGrd` | `1` | ground confidence weighting |
| `--ConfSat` | `0` | satellite confidence weighting (untested code path) |
| `--N_iters` | `1` | rotation pre-head iterations |
| `--proj` | `"geo"` | sat↔grd projection method |
| `--Optimizer` | `"TransV1"` | unused but consumed by save_path |
| `--use_uncertainty` | `0` | sat-uncertainty re-weighting in corr_for_translation |
| `--Supervision` | `"Weakly"` | the Stage-1 path; the "Gaussian" branch is for the render-loss pretrain |
| `--task` | `"2DoF"` | label only |
| `--visualize` | `0` | matplotlib overlay |
| `--multi_gpu` | `0` | DataParallel wrapping |
| `--grd_res` | `80` | grd panorama height (hardcoded as 80 inside `forward2DoF` of the legacy class, overriding the argparse default of 40) |
| `--sat`, `--grd`, `--sat_grd`, `--debug` | dropped | only used by the Gaussian-render branch we don't expose |

Other legacy cleanup:

- Two near-duplicate inference functions (`test`, `val`) plus the `val`-internal matplotlib block → one `vigor_main.eval.evaluate(...)`.
- Hardcoded `os.environ['CUDA_VISIBLE_DEVICES'] = '3'` at module load → `--cuda <id>` flag.
- The "load from this magic path" block in the `if args.test:` branch → explicit `--test 1 --ckpt <path>`.
- The `args.Supervision == "Gaussian"` (render-loss-only) training branch — orthogonal to Table-2 reproduction and never required in the main pipeline.

## What was kept exactly as the legacy code had it

To preserve numerical behavior with existing checkpoints:

- `weight_decay = 5e-3` (the paper says `1e-3`; the code says `5e-3`; we kept the code value).
- `OneCycleLR(max_lr=lr, pct_start=0.01, anneal_strategy='cos', steps_per_epoch=int(5260/scale))`.
- The hardcoded `grd_res = 80` inside `forward2DoF` (the argparse default of `40` is overridden by the model code).
- The `heading = ones * 90 / rotation_range` trick when `rotation_range > 0` (a numerical-stability scaling rather than a coordinate rotation).
- The `if args.ConfSat > 0` and `args.use_uncertainty` code paths inside `batch_wise_cross_corr` and `corr_for_translation`, kept intact even though the default config never enters them.
- Submodule names (`dpt_sat`, `dpt_grd`, `gaussian_encoder`, `FeatureForT`, `SatFeatureNet`, `GrdFeatureNet`, `camera_k`) match `models/models_vigor.py:ModelVIGOR` exactly, so existing `.pth` files load via `strict=False` with no key remapping.

## Where the algorithm actually lives

`vigor_main/` does not re-implement the algorithm. Touch these files if you need to change the algorithm itself:

- `gaussian/encoder_pano.py` — per-pixel Gaussian encoder for panoramic input.
- `vis_gaussian_pano.py` — `render_projections(...)`: the orthographic BEV splat calling into `pano_gaussian_feat._C`.
- `gaussian/pano_splat.py` — thin wrapper around the CUDA rasterizer.
- `backbone/backbone_pano.py` — the panoramic-specific backbone used inside `GaussianEncoder`.
- `models/models_vigor.py` — the legacy `ModelVIGOR` class; still consulted by other VIGOR experiments not covered by `vigor_main/`.
