"""Stage-1 training (and eval-only) entry point for the KITTI main experiment.

Usage examples are in ``kitti_main/README.md``. A typical training run:

    python -m kitti_main.train --GPS_error_coe 1 --name reproduce_lambda1

A typical eval-only run against an existing checkpoint:

    python -m kitti_main.train --test 1 --ckpt /path/to/model_9.pth \\
        --GPS_error_coe 1 --rotation_range 0
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from . import config
from .data import load_test1, load_test2, load_train
from .eval import evaluate
from .losses import Weakly_supervised_loss_w_GPS_error, batch_wise_cross_corr
from .model import BevSplatKITTI


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BevSplat KITTI Stage-1 trainer")

    # Pose-search ranges and rotation perturbation (kept tunable for completeness).
    p.add_argument("--rotation_range", type=float, default=config.ROTATION_RANGE)
    p.add_argument("--shift_range_lat", type=float, default=config.SHIFT_RANGE_LAT)
    p.add_argument("--shift_range_lon", type=float, default=config.SHIFT_RANGE_LON)

    # Loss term weights (paper Eq. 1: L_all = L_Weakly + λ₁·L_GPS).
    p.add_argument("--GPS_error_coe", type=float, default=0.0,
                   help="λ₁ in the paper. 0=Weakly-only, 1=with GPS-error term.")
    p.add_argument("--GPS_error", type=float, default=config.GPS_ERROR_RADIUS_M,
                   help="GPS-noise radius (meters) used to define the L_GPS peak window.")

    # Training schedule.
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--resume", type=int, default=0,
                   help="Epoch index to resume from (loads model_{resume-1}.pth from save_path).")

    # Init / checkpointing.
    p.add_argument("--stage1_init", type=str, default=str(config.STAGE0_INIT_CKPT),
                   help="Stage-0 .pth used to initialize the rotation pre-head (only loaded "
                        "when --rotation_range > 0 and not resuming).")
    p.add_argument("--name", type=str, default="reproduce",
                   help="Tag appended to the auto-generated save directory.")

    # Eval-only path.
    p.add_argument("--test", type=int, default=0,
                   help="If 1, load --ckpt and run evaluate() on test1 and test2 only.")
    p.add_argument("--ckpt", type=str, default="",
                   help="Checkpoint .pth to load (required when --test 1).")

    # Smoke-test escape hatch.
    p.add_argument("--max_batches", type=int, default=0,
                   help="If > 0, stop training after this many batches per epoch (smoke test).")

    # Hardware.
    p.add_argument("--cuda", type=int, default=0,
                   help="CUDA device index (replaces the legacy hardcoded CUDA_VISIBLE_DEVICES).")

    args = p.parse_args(argv)

    # Inject the legacy single-value flags so the algorithm modules
    # (losses.py, model.py) receive the exact Namespace they expect.
    args.stage = 1
    args.level = config.LEVEL_STR
    args.channels = config.CHANNELS_STR
    args.share = config.SHARE_FEATURE_NET
    args.ConfGrd = config.CONF_GRD
    args.ConfSat = config.CONF_SAT
    args.N_iters = config.N_ITERS
    args.proj = config.PROJ
    args.Optimizer = config.OPTIMIZER_KIND
    args.task = "3DoF"
    args.visualize = 0
    args.multi_gpu = 0
    return args


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(net, args, save_path: Path, device):
    save_path.mkdir(parents=True, exist_ok=True)

    params = list(net.feat_gaussian_encoder.parameters()) + list(net.dpt.parameters())
    optimizer = optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
        eps=config.ADAM_EPS,
    )

    scale = float(args.batch_size / config.BATCH_SIZE)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=int(config.ONECYCLE_STEPS_PER_EPOCH_BASE / scale),
        epochs=args.epochs,
        anneal_strategy=config.ONECYCLE_ANNEAL,
        pct_start=config.ONECYCLE_PCT_START,
        cycle_momentum=False,
    )

    t0 = time.time()
    for epoch in range(args.resume, args.epochs):
        net.train()
        train_loader = load_train(
            args.batch_size,
            shift_range_lat=args.shift_range_lat,
            shift_range_lon=args.shift_range_lon,
            rotation_range=args.rotation_range,
        )
        n_batches = len(train_loader)
        print(f"[epoch {epoch}] batch_size={args.batch_size}, num_batches={n_batches}")

        for loop_idx, batch in enumerate(train_loader):
            if args.max_batches and loop_idx >= args.max_batches:
                print(f"[smoke] stopping after {loop_idx} batches")
                break
            optimizer.zero_grad()
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
                sat_align_cam, sat_map, grd_left_imgs, grd_depth, grd_left_imgs_ori,
                left_camera_k, gt_heading, gt_shift_u, gt_shift_v,
                loop=loop_idx, save_dir=str(save_path),
            )

            corr_maps = batch_wise_cross_corr(
                sat_feat_d, sat_conf_d, g2s_feat_d, g2s_conf_d, args, masks=mask_d
            )
            l_weakly, l_gps = Weakly_supervised_loss_w_GPS_error(
                corr_maps, gt_shift_u, gt_shift_v, gt_heading,
                args, net.meters_per_pixel, args.GPS_error,
            )
            loss = l_weakly + l_gps * args.GPS_error_coe
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loop_idx % 10 == 9:
                r_err = (
                    torch.abs(thetas[:, -1, -1].reshape(-1) - gt_heading.reshape(-1)).mean()
                    * args.rotation_range
                )
                cur_lr = scheduler.get_last_lr()[0]
                dt = time.time() - t0
                print(
                    f"[epoch {epoch} loop {loop_idx}] "
                    f"R_err={r_err.item():.3f}  "
                    f"L_Weakly={l_weakly.item():.4f}  L_GPS={l_gps.item():.4f}  "
                    f"lr={cur_lr:.2e}  dt={dt:.1f}s"
                )
                t0 = time.time()

        ckpt_path = save_path / f"model_{epoch}.pth"
        torch.save(net.state_dict(), ckpt_path)
        print(f"[epoch {epoch}] saved {ckpt_path}")

        if not args.max_batches:  # skip eval in smoke runs
            evaluate(net, load_test1(args.batch_size, rotation_range=args.rotation_range),
                     args, save_path, epoch, split_name="test1", device=device)
            evaluate(net, load_test2(args.batch_size, rotation_range=args.rotation_range),
                     args, save_path, epoch, split_name="test2", device=device)


# ---------------------------------------------------------------------------
# Eval-only path
# ---------------------------------------------------------------------------

def _eval_only(net, args, save_path: Path, device):
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"[eval-only] test1 (Same-Area) and test2 (Cross-Area)")
    evaluate(net, load_test1(args.batch_size, rotation_range=args.rotation_range),
             args, save_path, epoch=-1, split_name="test1", device=device)
    evaluate(net, load_test2(args.batch_size, rotation_range=args.rotation_range),
             args, save_path, epoch=-1, split_name="test2", device=device)


# ---------------------------------------------------------------------------
# Weight loading (shape-tolerant)
# ---------------------------------------------------------------------------

def _load_filtered(net, ckpt_path: str, label: str):
    if not ckpt_path or not Path(ckpt_path).is_file():
        print(f"[{label}] WARN: ckpt path not found: {ckpt_path}")
        return
    pre = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    cur = net.state_dict()
    kept, skipped = {}, 0
    for k, v in pre.items():
        if k in cur and v.shape == cur[k].shape:
            kept[k] = v
        elif k in cur:
            print(f"[{label}] skip {k}: pretrained {tuple(v.shape)} vs current {tuple(cur[k].shape)}")
            skipped += 1
    cur.update(kept)
    net.load_state_dict(cur)
    print(f"[{label}] loaded {len(kept)}/{len(pre)} params from {ckpt_path} ({skipped} shape-skip)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    np.random.seed(2022)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = config.stage1_save_dir(
        shift_lat=args.shift_range_lat,
        shift_lon=args.shift_range_lon,
        rotation=args.rotation_range,
        gps_error_coe=args.GPS_error_coe,
        gps_error=args.GPS_error,
        share=args.share,
        conf_sat=args.ConfSat,
        name=args.name,
    )
    print(f"save_path: {save_path}")

    net = BevSplatKITTI(args, device=device).to(device)

    if args.test:
        if not args.ckpt:
            raise SystemExit("--test 1 requires --ckpt <path-to-model.pth>")
        _load_filtered(net, args.ckpt, label="eval")
        _eval_only(net, args, save_path, device)
        return

    if args.resume:
        resume_path = save_path / f"model_{args.resume - 1}.pth"
        _load_filtered(net, str(resume_path), label="resume")
    elif args.rotation_range > 0:
        _load_filtered(net, args.stage1_init, label="stage0-init")

    _train(net, args, save_path, device)


if __name__ == "__main__":
    main()
