# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

BevSplat — weakly-supervised cross-view (ground ↔ satellite) localization that lifts pixel features into **3D Gaussian primitives**, splats them into a **BEV plane** via custom CUDA rasterizers, then correlates BEV-rendered ground features against satellite features to recover the camera pose. Supports KITTI (perspective grd) and VIGOR (panoramic grd).

## Build / install

The two CUDA extensions are **not on PyPI** — they must be built locally before any training script can be imported. Both setups require glm headers in `third_party/glm/glm/` — the original repo never committed them and the git submodule is empty, so populate it from a sibling 3DGS-style project before building (`cp -r ~/CVPR26/FiT3D/submodules/diff-feature-gaussian-rasterization/third_party/glm/glm feature_gaussian/third_party/glm/` works on the dev server; on a fresh machine, vendor glm 0.9.9.8 from any local checkout or install).

```bash
cd feature_gaussian        && pip install -e .   # perspective rasterizer → feat_gaussian._C   (KITTI)
cd ../pano_feature_gaussian && pip install -e .   # panoramic rasterizer  → pano_gaussian_feat._C (VIGOR)
```

Two pip packages are not installed by default and the import of `gaussian.encoder_feat_nips` / `models.dino_fit` will fail without them:

```bash
pip install e3nn timm
```

There is no `requirements.txt`, no environment.yml, and no top-level `setup.py`. The README's "coming soon" install scripts have not landed. Use the existing Python env with PyTorch + CUDA; only the two extensions above are project-local builds.

Quick verification after install:
```bash
python -c "import feat_gaussian, pano_gaussian_feat; print('ok')"
```

## Main-experiment packages — use `kitti_main/` and `vigor_main/`

For the paper's main results, prefer the refactored packages over the legacy `train_*.py` scripts. Each is a thin wrapper around the algorithm modules with all single-value flags fixed at their working defaults and the test/train loops cleaned up.

### KITTI (Table 1)
`kitti_main/` reuses `gaussian/encoder_feat_nips.py`, `vis_gaussian_feat.render_projections`, `models/dino_fit.py`, `models/dpt_single.py`, `dataLoader/KITTI_dataset.py`.

```bash
# Eval an existing trained checkpoint (no retraining)
python -m kitti_main.train --test 1 \
    --ckpt /data/qiwei/nips25/CVLnet2/ModelsKitti/3DoF/Stage1/.../model_9.pth \
    --GPS_error_coe {0|1} --rotation_range 0 --name verify

# Retrain a Table-1 row from Stage-0 init
python -m kitti_main.train --GPS_error_coe {0|1} --rotation_range 10 --epochs 10 --name reproduce
```

### VIGOR (Table 2)
`vigor_main/` reuses `gaussian/encoder_pano.py`, `vis_gaussian_pano.render_projections`, `models/dino_fit.py`, `models/dpt_single.py`, `dataLoader/Vigor_dataset_gs.py`.

```bash
# Eval an existing trained checkpoint
python -m vigor_main.train --test 1 --area {same|cross} \
    --ckpt /data/qiwei/nips25/CVLnet2/ModelsVIGOR/2DoF/.../model_14.pth \
    --GPS_error_coe {0|1} --lr {1.25e-4|1e-4|6.5e-5} --name verify

# Retrain a Table-2 row
python -m vigor_main.train --area {same|cross} --GPS_error_coe {0|1} --lr 1.25e-4 --epochs 15 --name reproduce
```

See `kitti_main/README.md` and `vigor_main/README.md` for full recipe tables, architecture diagrams, and the list of flags dropped vs. preserved.

## Legacy entry points (other experiments)

There is no `pytest` suite. "Run" means "train" or "test" via one of four entry points. The same script handles both — `--test 1` switches to eval and the script loads a hardcoded checkpoint path (see *Hardcoded paths* below).

| Entry point | Dataset | Model file | Notes |
|---|---|---|---|
| `train_KITTI_weak_nips.py` | KITTI | `models/models_kitti_nips.py` | single-frame; canonical KITTI training. **For Table-1 reproduction prefer `kitti_main/train.py`** |
| `train_KITTI_weak_seq.py` | KITTI | `models/models_kitti_seq.py` | multi-frame; takes `--sequence N` |
| `train_KITTI_weak_weather.py` | KITTI | `models/models_kitti_nips.py` | weather-robustness eval |
| `train_vigor_2DoF.py` | VIGOR | `models/models_vigor.py` | panoramic; uses `pano_gaussian_feat` |

**The canonical source of runnable command lines is `.vscode/launch.json`**, not the README. It has working argument sets for every train/test configuration (`Train KITTI Stage 1 Share 1`, `Test VIGOR 2DoF Cross GPS`, etc.). When asked to run something, prefer copying args from there.

**`train.sh` is stale** — it invokes `train_KITTI_weak_nips_orienternet.py`, `train_KITTI_weak_nips_vfa.py`, and `train_KITTI_depth.py`, none of which exist in the tree (though `models/models_kitti_orienternet.py` and `models/models_kitti_vfa.py` do, as orphaned variants). Don't trust `train.sh` blindly; the KITTI-NIPS, KITTI-seq, KITTI-weather, and VIGOR blocks at the bottom are the working ones.

## Argument conventions (shared across all train scripts)

These flags appear in every entry point and mean the same thing — touching them in one script usually means the same change applies elsewhere:

- `--stage` — pipeline stage. `0` = self-supervised pretraining of the feature/Gaussian encoder; `1` = end-to-end pose loss on top of stage-0 weights; `4` = a downstream / fine-tune configuration used by the seq/weather scripts. Stage-1 training expects a stage-0 checkpoint to exist at the matching `restore_path`.
- `--level` — feature-pyramid level used for correlation (e.g. `"1"` or `"0_2"`).
- `--channels` — pyramid channel widths, e.g. `"32_16_4"`.
- `--share` — share the feature backbone between sat and grd branches (1) or use separate backbones (0). KITTI defaults to share=1, VIGOR to share=0.
- `--ConfGrd` / `--ConfSat` — multiply correlation by a learned confidence map on the ground / sat side.
- `--GPS_error_coe` — when > 0, uses `Weakly_supervised_loss_w_GPS_error` instead of the plain `weak_supervise_loss`; `--GPS_error` is the assumed GPS noise radius in meters.
- `--rotation_range`, `--shift_range_lat`, `--shift_range_lon` — perturbation range applied to GT pose to generate training queries.
- `--name` — free-form tag appended to the checkpoint path; used to distinguish runs that share other args.

The path on disk where checkpoints land is computed by each script's `getSavePath(args)` and **encodes most of the args above into the directory name** — changing any of them produces a new save dir, so `--test 1` only works if the args reproduce the dir of an existing trained model.

## Hardcoded paths (NOT in args)

These are baked into the source and must exist (or be edited) on whatever machine runs training:

- **KITTI dataset root**: `/data/dataset/KITTI` — `dataLoader/KITTI_dataset.py:21` (`root_dir`).
- **VIGOR dataset root**: `/data/dataset/VIGOR` — `dataLoader/Vigor_dataset_gs.py:15` (`root`). Expects subdirs `pano_mask_sky/`, `panorama/`, `UniK3D_<split>_metric/` per city, plus `splits__corrected/` labels.
- **Checkpoint root**: `/data/qiwei/nips25/CVLnet2/ModelsKitti/...` and `.../ModelsVIGOR/...`. The `if args.test:` blocks at the bottom of each train script also contain hardcoded `.pth` paths for evaluation — those need updating when reproducing on a different machine.

If the user is on a fresh machine, expect to edit these three locations before anything runs.

## Architecture: pixel features → BEV correlation

The pipeline is the same shape for KITTI and VIGOR; the differences are (a) which CUDA rasterizer, (b) whether the grd input is perspective or panoramic, and (c) which `models/models_*.py` orchestrates it.

```
grd image ──▶ backbone/                         ──▶ per-pixel features + depth
              (BackboneDino | BackboneResnet |       (DINOv2 weights via models/dino*.py;
               BackboneDinoNips | BackbonePano)       depth via models/dpt_single.py)
                       │
                       ▼
              gaussian/encoder_feat*.py         ──▶ N 3D Gaussians per pixel,
              (GaussianFeatEncoder)                  each carrying a feature vector + confidence
                       │  uses build_gaussians.get_world_rays / build_covariance
                       ▼
              feat_gaussian._C  (KITTI)         ──▶ BEV feature map
              pano_gaussian_feat._C (VIGOR)         (differentiable rasterization w/ feature channels)
                       │
                       ▼
              sat backbone ──▶ sat feature map
                       │
                       ▼
              models/models_*.py  (Model / ModelVIGOR)
              cross-correlation: `batch_wise_cross_corr`, `corr_for_translation`
                       │
                       ▼
              weakly-supervised loss:
              `weak_supervise_loss`  or  `Weakly_supervised_loss_w_GPS_error`
              (also `GT_triplet_loss` and `corr_for_accurate_translation_supervision` for variants)
```

Key things to know when navigating:

- **Two parallel CUDA rasterizer copies**: `feature_gaussian/cuda_rasterizer/` and `pano_feature_gaussian/cuda_rasterizer/`. They're forks of the Inria 3DGS rasterizer extended with a feature-channel output. Source files mirror each other (`forward.cu`, `backward.cu`, `rasterizer_impl.cu`); changes typically need to be made in both if they apply to shared logic. The third-party `glm/` headers under each `third_party/` are vendored — leave alone.

- **Multiple encoder variants in `gaussian/`** — `encoder_feat.py`, `encoder_feat_nips.py`, `encoder_feat_seq.py`, `encoder_pano.py`. Each train script imports a specific one; don't unify them without reading the imports in the matching `models/models_kitti_*.py`.

- **`backbone/__init__.py` is a config-driven registry** (`BACKBONES = {"resnet": ..., "dino": ...}`, `get_backbone(cfg, d_in)`), but the dataclasses `BackboneResnetCfg` / `BackboneDinoCfg` are *not* what the training scripts actually use — the train scripts instantiate `BackboneDino` / `BackboneDinoNips` directly inside their model file. The registry is dormant; don't assume editing it changes training behavior.

- **`data_utils.py`, `jacobian.py`, `ply_export.py`, `transformer.py`, `VGG.py`** at the repo root are imported by name throughout the tree (`import data_utils as utils`, `from jacobian import grid_sample`, etc.). They are part of the public surface even though they sit at top level rather than under a package.

- **`visualize.py`, `vis_gaussian_feat.py`, `vis_gaussian_pano.py`, `vis_gaussian_seq.py`** are not just developer tools — `vis_gaussian_feat.render_projections` and helpers from `visualize.py` are imported by `models/models_kitti_nips.py` inside the forward pass for debug renders. Moving or renaming them breaks training imports.

## Notes for future edits

- Don't promise the `pip install -r requirements.txt` flow from the README — it doesn't exist. If the user asks to add it, infer from `import` statements; common: `torch`, `torchvision`, `einops`, `jaxtyping`, `opencv-python`, `scipy`, `matplotlib`, `Pillow`.
- When changing a path-related arg (`--stage`, `--level`, `--share`, ...), also update the `path = '...'` inside the matching script's `if args.test:` block, or eval will load the wrong checkpoint. The latest commit (`feat: fix weights path`) is exactly this kind of fix.
- `models/dino_fit.py` and `models/dino_Fit.py` coexist (case difference); both are referenced by different model files. macOS / Windows checkouts will collide — keep development on the Linux server.
