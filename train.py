# train.py
# Same trainer as v4, but with --target_smooth_window support.
import os
import math
import argparse
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_module import load_and_prepare, make_dataloaders, PreprocessArtifacts
from model import LSTMRegressor


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_setup():
    dist.init_process_group(backend="nccl", init_method="env://")


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank_world() -> Tuple[int, int, int]:
    if is_distributed():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank
    return 0, 1, 0


def inverse_y(y_scaled: torch.Tensor, art: PreprocessArtifacts) -> torch.Tensor:
    mean = torch.tensor(art.y_scaler_mean, device=y_scaled.device, dtype=y_scaled.dtype)
    scale = torch.tensor(art.y_scaler_scale, device=y_scaled.device, dtype=y_scaled.dtype)
    return y_scaled * scale + mean


def inverse_x_feature(x_scaled: torch.Tensor, feat_idx: int, art: PreprocessArtifacts) -> torch.Tensor:
    mean = torch.tensor(float(art.x_scaler_mean[feat_idx]), device=x_scaled.device, dtype=x_scaled.dtype)
    scale = torch.tensor(float(art.x_scaler_scale[feat_idx]), device=x_scaled.device, dtype=x_scaled.dtype)
    return x_scaled * scale + mean


@torch.no_grad()
def eval_model(model: nn.Module, loader, art: PreprocessArtifacts, device: torch.device, distributed: bool) -> Dict[str, float]:
    model.eval()
    agg = {
        "count": 0.0,
        "abs_err_0": 0.0, "sq_err_0": 0.0, "sum_y_0": 0.0, "sum_y2_0": 0.0,
        "abs_err_1": 0.0, "sq_err_1": 0.0, "sum_y_1": 0.0, "sum_y2_1": 0.0,
        "mse_loss_scaled": 0.0,
    }
    mse_sum = nn.MSELoss(reduction="sum")

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yhat = model(X)
        agg["mse_loss_scaled"] += float(mse_sum(yhat, y).item())

        y_orig = inverse_y(y, art)
        yhat_orig = inverse_y(yhat, art)
        err = yhat_orig - y_orig
        abs_err = torch.abs(err)
        sq_err = err * err
        agg["count"] += float(y_orig.shape[0])

        agg["abs_err_0"] += float(abs_err[:, 0].sum().item())
        agg["sq_err_0"]  += float(sq_err[:, 0].sum().item())
        agg["sum_y_0"]   += float(y_orig[:, 0].sum().item())
        agg["sum_y2_0"]  += float((y_orig[:, 0] ** 2).sum().item())

        agg["abs_err_1"] += float(abs_err[:, 1].sum().item())
        agg["sq_err_1"]  += float(sq_err[:, 1].sum().item())
        agg["sum_y_1"]   += float(y_orig[:, 1].sum().item())
        agg["sum_y2_1"]  += float((y_orig[:, 1] ** 2).sum().item())

    if distributed:
        t = torch.tensor(list(agg.values()), device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for k, v in zip(list(agg.keys()), t.tolist()):
            agg[k] = float(v)

    count = max(1.0, agg["count"])
    scaled_mse = agg["mse_loss_scaled"] / count

    def r2(sum_y, sum_y2, sse):
        mean_y = sum_y / count
        sst = sum_y2 - 2 * mean_y * sum_y + count * (mean_y ** 2)
        sst = max(sst, 1e-12)
        return 1.0 - (sse / sst)

    return {
        "scaled_mse": scaled_mse,
        "y0_mae": agg["abs_err_0"] / count,
        "y0_rmse": math.sqrt(agg["sq_err_0"] / count),
        "y0_r2": r2(agg["sum_y_0"], agg["sum_y2_0"], agg["sq_err_0"]),
        "y1_mae": agg["abs_err_1"] / count,
        "y1_rmse": math.sqrt(agg["sq_err_1"] / count),
        "y1_r2": r2(agg["sum_y_1"], agg["sum_y2_1"], agg["sq_err_1"]),
    }


@torch.no_grad()
def eval_persistence_baseline(loader, art: PreprocessArtifacts, device: torch.device, distributed: bool) -> Dict[str, float]:
    # Baseline uses RAW targets from last timestep in X window.
    agg = {
        "count": 0.0,
        "abs_err_0": 0.0, "sq_err_0": 0.0, "sum_y_0": 0.0, "sum_y2_0": 0.0,
        "abs_err_1": 0.0, "sq_err_1": 0.0, "sum_y_1": 0.0, "sum_y2_1": 0.0,
    }

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_orig = inverse_y(y, art)

        x_last_speed = X[:, -1, art.idx_speed_in_X]
        x_last_occ   = X[:, -1, art.idx_occ_in_X]
        yhat0 = inverse_x_feature(x_last_speed, art.idx_speed_in_X, art)
        yhat1 = inverse_x_feature(x_last_occ,   art.idx_occ_in_X,   art)
        yhat = torch.stack([yhat0, yhat1], dim=1)

        err = yhat - y_orig
        abs_err = torch.abs(err)
        sq_err = err * err
        agg["count"] += float(y_orig.shape[0])

        agg["abs_err_0"] += float(abs_err[:, 0].sum().item())
        agg["sq_err_0"]  += float(sq_err[:, 0].sum().item())
        agg["sum_y_0"]   += float(y_orig[:, 0].sum().item())
        agg["sum_y2_0"]  += float((y_orig[:, 0] ** 2).sum().item())

        agg["abs_err_1"] += float(abs_err[:, 1].sum().item())
        agg["sq_err_1"]  += float(sq_err[:, 1].sum().item())
        agg["sum_y_1"]   += float(y_orig[:, 1].sum().item())
        agg["sum_y2_1"]  += float((y_orig[:, 1] ** 2).sum().item())

    if distributed:
        t = torch.tensor(list(agg.values()), device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for k, v in zip(list(agg.keys()), t.tolist()):
            agg[k] = float(v)

    count = max(1.0, agg["count"])

    def r2(sum_y, sum_y2, sse):
        mean_y = sum_y / count
        sst = sum_y2 - 2 * mean_y * sum_y + count * (mean_y ** 2)
        sst = max(sst, 1e-12)
        return 1.0 - (sse / sst)

    return {
        "y0_mae": agg["abs_err_0"] / count,
        "y0_rmse": math.sqrt(agg["sq_err_0"] / count),
        "y0_r2": r2(agg["sum_y_0"], agg["sum_y2_0"], agg["sq_err_0"]),
        "y1_mae": agg["abs_err_1"] / count,
        "y1_rmse": math.sqrt(agg["sq_err_1"] / count),
        "y1_r2": r2(agg["sum_y_1"], agg["sum_y2_1"], agg["sq_err_1"]),
    }


def train_one_epoch(model, loader, optimizer, device, distributed, grad_clip: float = 0.0):
    model.train()
    criterion = nn.MSELoss()
    total = 0.0
    n_batches = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        yhat = model(X)
        loss = criterion(yhat, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total += float(loss.item())
        n_batches += 1

    avg = total / max(1, n_batches)
    if distributed:
        t = torch.tensor([avg], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        avg = float(t.item() / dist.get_world_size())
    return avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--lookback", type=int, default=72)
    p.add_argument("--freq_minutes", type=int, default=5)
    p.add_argument("--target_smooth_window", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--fc_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--save_path", type=str, default="best_lstm.pt")
    args = p.parse_args()

    set_seed(args.seed)

    distributed = is_distributed()
    rank, world_size, local_rank = get_rank_world()

    try:
        if distributed:
            ddp_setup()
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_ds, val_ds, test_ds, art = load_and_prepare(
            args.csv_path,
            lookback=args.lookback,
            freq_minutes=args.freq_minutes,
            target_smooth_window=args.target_smooth_window,
        )

        train_loader, val_loader, test_loader, train_sampler, _, _ = make_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

        model = LSTMRegressor(
            n_features=len(art.feature_names),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            fc_size=args.fc_size,
            out_size=2,
        ).to(device)

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)

        best_val = float("inf")
        bad = 0

        if rank == 0:
            print(f"distributed={distributed} world_size={world_size} device={device}")
            print(f"lookback={args.lookback} freq_minutes={args.freq_minutes} batch={args.batch_size} smooth_window={args.target_smooth_window} | wd={args.weight_decay} grad_clip={args.grad_clip}")
            print(f"targets={art.target_names}")
            print(f"n_features={len(art.feature_names)} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

        for epoch in range(1, args.epochs + 1):
            if distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            tr = train_one_epoch(model, train_loader, opt, device, distributed, grad_clip=args.grad_clip)
            val = eval_model(model, val_loader, art, device, distributed)
            val_loss = val["scaled_mse"]
            sched.step(val_loss)

            if rank == 0:
                print(f"\nEpoch {epoch:03d}")
                print(f"  train_scaled_mse: {tr:.6f}")
                print(f"  val_scaled_mse:   {val_loss:.6f}")
                print(f"  y0: MAE={val['y0_mae']:.3f} RMSE={val['y0_rmse']:.3f} R2={val['y0_r2']:.4f}")
                print(f"  y1: MAE={val['y1_mae']:.3f} RMSE={val['y1_rmse']:.3f} R2={val['y1_r2']:.4f}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad = 0
                if rank == 0:
                    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save({"model_state_dict": state, "artifacts": asdict(art), "args": vars(args)}, args.save_path)
                    print(f"  ✅ saved best model -> {args.save_path}")
            else:
                bad += 1
                if bad >= args.patience:
                    if rank == 0:
                        print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                    break

        test = eval_model(model, test_loader, art, device, distributed)
        base = eval_persistence_baseline(test_loader, art, device, distributed)

        if rank == 0:
            print("\n=== TEST (LSTM) ===")
            print(f"y0: MAE={test['y0_mae']:.3f} RMSE={test['y0_rmse']:.3f} R2={test['y0_r2']:.4f}")
            print(f"y1: MAE={test['y1_mae']:.3f} RMSE={test['y1_rmse']:.3f} R2={test['y1_r2']:.4f}")

            print("\n=== TEST (Baseline Persistência: raw y(t)=y(t-1)) ===")
            print(f"y0: MAE={base['y0_mae']:.3f} RMSE={base['y0_rmse']:.3f} R2={base['y0_r2']:.4f}")
            print(f"y1: MAE={base['y1_mae']:.3f} RMSE={base['y1_rmse']:.3f} R2={base['y1_r2']:.4f}")

    finally:
        if distributed:
            ddp_cleanup()


if __name__ == "__main__":
    main()
