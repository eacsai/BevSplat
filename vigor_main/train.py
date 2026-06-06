"""Stage-1 2DoF training (and eval-only) entry point for VIGOR.

Usage examples are in ``vigor_main/README.md``. Typical training:

    python -m vigor_main.train --area same --GPS_error_coe 1 --lr 1.25e-4 --name reproduce

Typical eval-only:

    python -m vigor_main.train --test 1 --area same --ckpt /path/to/model_X.pth \\
        --GPS_error_coe 1 --lr 1.25e-4 --name verify
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

# Avoid "Too many open files" — VIGOR's test sets are 50k+ samples and the
# default file_descriptor sharing strategy leaks fds across dataloader workers.
torch.multiprocessing.set_sharing_strategy("file_system")

from . import config
from .data import load_test, load_train_and_val
from .eval import evaluate
from .losses import Weakly_supervised_loss_w_GPS_error, batch_wise_cross_corr
from .model import BevSplatVIGOR


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BevSplat VIGOR 2DoF trainer")

    p.add_argument("--area", type=str, choices=["same", "cross"], default="same",
                   help="VIGOR test protocol — same-area or cross-area.")

    p.add_argument("--rotation_range", type=float, default=config.ROTATION_RANGE,
                   help="0 for the standard 2DoF protocol; non-zero adds rotation perturbation.")

    p.add_argument("--GPS_error_coe", type=float, default=0.0,
                   help="λ₁ in the paper. 0=Weakly-only, 1=add the GPS-error term.")
    p.add_argument("--GPS_error", type=float, default=config.GPS_ERROR_RADIUS_M)

    p.add_argument("--lr", type=float, default=config.LR,
                   help="Default 1.25e-4 (Same-Area). Use 1e-4 or 6.5e-5 for Cross-Area variants.")
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--amount", type=float, default=config.DATA_AMOUNT,
                   help="Fraction of training data to use (legacy --amount).")

    p.add_argument("--resume", type=int, default=0,
                   help="Epoch index to resume from (loads model_{resume-1}.pth from save_path).")

    p.add_argument("--name", type=str, default="reproduce",
                   help="Tag appended to the auto-generated save directory.")

    p.add_argument("--test", type=int, default=0,
                   help="If 1, load --ckpt and evaluate on the test split only.")
    p.add_argument("--ckpt", type=str, default="",
                   help="Checkpoint .pth to load (required when --test 1).")

    p.add_argument("--max_batches", type=int, default=0,
                   help="If > 0, stop training after this many batches per epoch (smoke test).")

    p.add_argument("--cuda", type=int, default=0,
                   help="CUDA device index (replaces the legacy hardcoded CUDA_VISIBLE_DEVICES=3).")

    args = p.parse_args(argv)

    # Inject the legacy single-value flags so the algorithm modules
    # receive the exact Namespace they expect.
    args.stage = 3                                  # legacy fixed value used by dataloader
    args.level = config.LEVEL_STR
    args.channels = config.CHANNELS_STR
    args.share = config.SHARE_FEATURE_NET
    args.ConfGrd = config.CONF_GRD
    args.ConfSat = config.CONF_SAT
    args.N_iters = config.N_ITERS
    args.proj = config.PROJ
    args.Optimizer = config.OPTIMIZER_KIND
    args.task = "2DoF"
    args.use_uncertainty = config.USE_UNCERTAINTY
    args.Supervision = config.SUPERVISION
    args.grd_res = config.GRD_RES
    args.visualize = 0
    args.multi_gpu = 0
    args.shift_range_lat = 0.0
    args.shift_range_lon = 0.0
    return args


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def _train(net, args, save_path: Path, device):
    save_path.mkdir(parents=True, exist_ok=True)

    if args.share:
        params = net.dpt_sat.parameters()
    else:
        params = list(net.dpt_grd.parameters()) + list(net.dpt_sat.parameters())
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
        train_loader, val_loader = load_train_and_val(
            args.batch_size,
            area=args.area,
            rotation_range=args.rotation_range,
            amount=args.amount,
        )
        n_batches = len(train_loader)
        print(f"[epoch {epoch}] batch_size={args.batch_size}, num_batches={n_batches}")

        for loop_idx, batch in enumerate(train_loader):
            if args.max_batches and loop_idx >= args.max_batches:
                print(f"[smoke] stopping after {loop_idx} batches")
                break
            optimizer.zero_grad()
            (
                grd, sat, depth_imgs, grd_ori,
                gt_shift_u, gt_shift_v, gt_rot, meter_per_pixel,
            ) = (item.to(device) for item in batch)

            sat_feat_d, sat_conf_d, g2s_feat_d, g2s_conf_d, sat_uncer_d = net(
                sat, grd, depth_imgs, grd_ori, meter_per_pixel,
                gt_rot, gt_shift_u, gt_shift_v,
                loop=loop_idx, save_dir=str(save_path),
            )
            corr_maps = batch_wise_cross_corr(
                sat_feat_d, sat_conf_d, g2s_feat_d, g2s_conf_d, args, sat_uncer_d
            )
            l_weakly, l_gps = Weakly_supervised_loss_w_GPS_error(
                corr_maps, gt_shift_u, gt_shift_v, args, meter_per_pixel, args.GPS_error,
            )
            loss = l_weakly + l_gps * args.GPS_error_coe
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loop_idx % 10 == 9:
                cur_lr = scheduler.get_last_lr()[0]
                dt = time.time() - t0
                print(
                    f"[epoch {epoch} loop {loop_idx}] "
                    f"L_Weakly={l_weakly.item():.4f}  L_GPS={l_gps.item():.4f}  "
                    f"lr={cur_lr:.2e}  dt={dt:.1f}s"
                )
                t0 = time.time()

        ckpt_path = save_path / f"model_{epoch}.pth"
        torch.save(net.state_dict(), ckpt_path)
        print(f"[epoch {epoch}] saved {ckpt_path}")

        if not args.max_batches:
            evaluate(net, val_loader, args, save_path, epoch,
                     split_name=f"{args.area}_val", device=device)


def _eval_only(net, args, save_path: Path, device):
    save_path.mkdir(parents=True, exist_ok=True)
    test_loader = load_test(
        args.batch_size,
        area=args.area,
        rotation_range=args.rotation_range,
        amount=args.amount,
    )
    print(f"[eval-only] {args.area}-area")
    evaluate(net, test_loader, args, save_path, epoch=-1,
             split_name=args.area, device=device)


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

    save_path = config.save_dir(
        area=args.area,
        rotation_range=args.rotation_range,
        lr=args.lr,
        gps_error_coe=args.GPS_error_coe,
        gps_error=args.GPS_error,
        share=args.share,
        conf_grd=args.ConfGrd,
        conf_sat=args.ConfSat,
        name=args.name,
    )
    print(f"save_path: {save_path}")

    net = BevSplatVIGOR(args, device=device).to(device)

    if args.test:
        if not args.ckpt:
            raise SystemExit("--test 1 requires --ckpt <path-to-model.pth>")
        _load_filtered(net, args.ckpt, label="eval")
        _eval_only(net, args, save_path, device)
        return

    if args.resume:
        resume_path = save_path / f"model_{args.resume - 1}.pth"
        _load_filtered(net, str(resume_path), label="resume")

    _train(net, args, save_path, device)


if __name__ == "__main__":
    main()
