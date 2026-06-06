"""Clean entry point for BevSplat's KITTI main experiment.

This package reproduces the "Ours" rows of Table 1 in the BevSplat paper
(arXiv 2502.09080). It is a thin, focused wrapper around the algorithm
modules that live elsewhere in the repo:

    gaussian/encoder_feat_nips.py   -- per-pixel Gaussian primitive encoder
    vis_gaussian_feat.py            -- CUDA orthographic BEV splat
    models/dino_fit.py              -- DINOv2 backbone (with FiT weights)
    models/dpt_single.py            -- DPT head producing 32-d feature + confidence
    models/VGGW.py                  -- VGGUnet (rotation pre-head)
    models/swin_transformer.py      -- TransOptimizerG2SP_V1 (rotation pre-head)
    dataLoader/KITTI_dataset.py     -- KITTI sat/grd pair loader

Reproduction recipes live in ``kitti_main/README.md``.
"""
