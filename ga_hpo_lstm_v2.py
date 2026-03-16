#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA (Genetic Algorithm) Hyperparameter Optimization for the PyTorch LSTMRegressor.

This version was inspired by your GA.ipynb:
- Tournament selection
- 1-point crossover
- Per-gene mutation
- Elitism
- Optional parallel evaluation (CPU or multi-GPU)

Works with the project files:
  - data_module.py  (load_and_prepare, make_dataloaders, PreprocessArtifacts)
  - model.py        (LSTMRegressor)

Typical usage (single GPU, sequential evaluation):
  python ga_hpo_lstm_v2.py --csv_path smart_mobility_dataset.csv --device cuda:0 --generations 10 --pop_size 20

Multi-GPU parallel evaluation (one process per GPU):
  python ga_hpo_lstm_v2.py --csv_path smart_mobility_dataset.csv --devices 0,1,2,3,4 --n_jobs 5 --generations 10 --pop_size 25

CPU parallel evaluation (disables CUDA):
  python ga_hpo_lstm_v2.py --csv_path smart_mobility_dataset.csv --cpu_only --n_jobs 8

Output:
- hpo_runs/<timestamp>/results.csv
- hpo_runs/<timestamp>/best_config.json

Notes:
- HPO uses a small number of epochs with early stopping for speed.
- After finding the best config, run a full training with train.py (and 5 GPUs if you want).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from data_module import load_and_prepare, make_dataloaders, PreprocessArtifacts
from model import LSTMRegressor


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Metrics helpers
# -----------------------
def inverse_y(y_scaled: torch.Tensor, art: PreprocessArtifacts) -> torch.Tensor:
    mean = torch.tensor(art.y_scaler_mean, device=y_scaled.device, dtype=y_scaled.dtype)
    scale = torch.tensor(art.y_scaler_scale, device=y_scaled.device, dtype=y_scaled.dtype)
    return y_scaled * scale + mean


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    var = float(np.var(y_true))
    if var <= 1e-12:
        return float("nan")
    mse = float(np.mean((y_true - y_pred) ** 2))
    return 1.0 - (mse / var)


@torch.no_grad()
def eval_val_metrics(model: nn.Module, loader, art: PreprocessArtifacts, device: torch.device) -> Dict[str, float]:
    model.eval()
    mse_sum = nn.MSELoss(reduction="sum")

    total = 0.0
    n = 0
    y_true_all, y_pred_all = [], []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        yhat = model(X)
        total += float(mse_sum(yhat, y).item())
        n += int(y.shape[0])

        y_true_all.append(inverse_y(y, art).detach().cpu().numpy())
        y_pred_all.append(inverse_y(yhat, art).detach().cpu().numpy())

    val_scaled_mse = total / max(1, n)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    r2_0 = r2_score_np(y_true[:, 0], y_pred[:, 0])
    r2_1 = r2_score_np(y_true[:, 1], y_pred[:, 1])
    avg_r2 = float(np.nanmean([r2_0, r2_1]))

    return {
        "val_scaled_mse": float(val_scaled_mse),
        "y0_r2": float(r2_0),
        "y1_r2": float(r2_1),
        "avg_r2": float(avg_r2),
    }


# -----------------------
# Trial training (fast)
# -----------------------
def train_quick(
    csv_path: str,
    cfg: Dict[str, Any],
    device: torch.device,
    max_epochs: int,
    patience: int,
    num_workers: int,
    seed: int,
) -> Dict[str, float]:
    set_seed(seed)

    train_ds, val_ds, _, art = load_and_prepare(
        csv_path,
        lookback=int(cfg["lookback"]),
        freq_minutes=int(cfg["freq_minutes"]),
        target_smooth_window=int(cfg["target_smooth_window"]),
    )

    train_loader, val_loader, _, _, _, _ = make_dataloaders(
        train_ds, val_ds, val_ds,
        batch_size=int(cfg["batch_size"]),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        distributed=False,
    )

    model = LSTMRegressor(
        n_features=len(art.feature_names),
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        fc_size=int(cfg["fc_size"]),
        out_size=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.MSELoss()

    best = float("inf")
    bad = 0

    t0 = time.time()

    for _epoch in range(1, max_epochs + 1):
        model.train()
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            yhat = model(X)
            loss = criterion(yhat, y)
            loss.backward()

            gc = float(cfg["grad_clip"])
            if gc > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gc)

            optimizer.step()

        m = eval_val_metrics(model, val_loader, art, device)
        val_mse = m["val_scaled_mse"]

        if val_mse < best - 1e-6:
            best = val_mse
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    m["train_time_s"] = float(time.time() - t0)
    return m


# -----------------------
# GA search space
# -----------------------
GENE_SPACE = {
    # data / target
    "lookback": [36, 72, 144],
    "target_smooth_window": [3, 6, 12],
    "freq_minutes": [5],

    # training
    "batch_size": [64, 128, 256],
    "lr": ("log_uniform", 1e-4, 3e-3),
    "weight_decay": ("log_uniform_or_zero", 1e-6, 1e-3),
    "grad_clip": ("uniform", 0.0, 2.0),

    # model
    "hidden_size": [32, 64, 128, 256],
    "num_layers": [1, 2],
    "dropout": ("uniform", 0.0, 0.5),
    "fc_size": [32, 64, 128],
}
GENE_KEYS = list(GENE_SPACE.keys())


def sample_gene(key: str) -> Any:
    spec = GENE_SPACE[key]
    if isinstance(spec, list):
        return random.choice(spec)

    kind = spec[0]
    if kind == "uniform":
        lo, hi = float(spec[1]), float(spec[2])
        return random.uniform(lo, hi)
    if kind == "log_uniform":
        lo, hi = float(spec[1]), float(spec[2])
        return 10 ** random.uniform(math.log10(lo), math.log10(hi))
    if kind == "log_uniform_or_zero":
        lo, hi = float(spec[1]), float(spec[2])
        if random.random() < 0.30:
            return 0.0
        return 10 ** random.uniform(math.log10(lo), math.log10(hi))

    raise ValueError(f"Unknown gene spec: {spec}")


def init_individual() -> Dict[str, Any]:
    return {k: sample_gene(k) for k in GENE_KEYS}


def mutate(ind: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
    out = dict(ind)
    for k in GENE_KEYS:
        if random.random() < mutation_rate:
            out[k] = sample_gene(k)
    return out


def crossover_1point(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    child: Dict[str, Any] = {}
    n_params = len(GENE_KEYS)
    cp = random.randint(1, n_params - 1)
    for i, k in enumerate(GENE_KEYS):
        child[k] = a[k] if i < cp else b[k]
    return child


def tournament_select(pop: List[Dict[str, Any]], fitness: List[float], k: int) -> Dict[str, Any]:
    idxs = random.sample(range(len(pop)), k)
    best_i = min(idxs, key=lambda i: fitness[i])  # minimize
    return pop[best_i]


def config_to_key(cfg: Dict[str, Any]) -> str:
    norm = {}
    for k in GENE_KEYS:
        v = cfg[k]
        if isinstance(v, float):
            norm[k] = round(v, 10)
        else:
            norm[k] = int(v)
    return json.dumps(norm, sort_keys=True)


def _worker_eval(args: Tuple[str, Dict[str, Any], Optional[int], bool, int, int, int, int]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    csv_path, cfg, gpu_id, cpu_only, max_epochs, patience, num_workers, seed = args

    if cpu_only:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            if gpu_id is not None:
                torch.cuda.set_device(int(gpu_id))
                device = torch.device(f"cuda:{int(gpu_id)}")
            else:
                device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    metrics = train_quick(
        csv_path=csv_path,
        cfg=cfg,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        num_workers=num_workers,
        seed=seed,
    )
    return cfg, metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True)
    p.add_argument("--seed", type=int, default=42)

    # Evaluation mode
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                   help="Used when n_jobs=1. Ignored when using --devices with n_jobs>1.")
    p.add_argument("--cpu_only", action="store_true", help="Force CPU evaluation (useful for parallel CPU runs).")
    p.add_argument("--devices", default="", help="Comma-separated GPU ids for parallel runs, e.g., 0,1,2,3,4")
    p.add_argument("--n_jobs", type=int, default=1, help="Number of parallel workers (1 = sequential).")

    # GA params
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--pop_size", type=int, default=20)
    p.add_argument("--elite", type=int, default=2)
    p.add_argument("--tournament_k", type=int, default=3)
    p.add_argument("--mutation_rate", type=float, default=0.05)

    # Trial training params
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers per trial. Use 0 for stability when running many processes.")

    # Fitness objective
    p.add_argument("--fitness", choices=["val_mse", "neg_avg_r2"], default="val_mse",
                   help="val_mse: minimize scaled MSE; neg_avg_r2: minimize (-avg_r2).")
    args = p.parse_args()

    set_seed(args.seed)

    gpu_list: List[int] = []
    if args.devices.strip():
        gpu_list = [int(x.strip()) for x in args.devices.split(",") if x.strip()]

    run_dir = Path("hpo_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"
    best_json = run_dir / "best_config.json"

    fieldnames = ["gen", "idx", "fitness", "val_scaled_mse", "y0_r2", "y1_r2", "avg_r2", "train_time_s"] + GENE_KEYS
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    cache: Dict[str, Dict[str, float]] = {}

    def fitness_from_metrics(m: Dict[str, float]) -> float:
        return float(m["val_scaled_mse"]) if args.fitness == "val_mse" else float(-m["avg_r2"])

    def evaluate_population(pop: List[Dict[str, Any]], gen: int) -> Tuple[List[float], List[Dict[str, float]]]:
        # Parallel evaluation: simpler (no cache)
        if args.n_jobs > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")

            tasks = []
            for i, cfg in enumerate(pop):
                gpu_id = None
                if (not args.cpu_only) and gpu_list:
                    gpu_id = gpu_list[i % len(gpu_list)]
                tasks.append((args.csv_path, cfg, gpu_id, args.cpu_only, args.max_epochs, args.patience, args.num_workers, args.seed + gen * 1000 + i))

            with ctx.Pool(processes=args.n_jobs) as pool:
                out = pool.map(_worker_eval, tasks)

            fitness_list, metrics_list = [], []
            for idx, (cfg, m) in enumerate(out):
                fit = fitness_from_metrics(m)
                fitness_list.append(fit)
                metrics_list.append(m)
                row = {"gen": gen, "idx": idx, "fitness": fit, **m, **cfg}
                with results_csv.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            return fitness_list, metrics_list

        # Sequential evaluation with cache
        device = torch.device("cpu" if args.cpu_only else args.device)

        fitness_list, metrics_list = [], []
        for idx, cfg in enumerate(pop):
            key = config_to_key(cfg)
            if key in cache:
                m = cache[key]
            else:
                m = train_quick(
                    csv_path=args.csv_path,
                    cfg=cfg,
                    device=device,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    num_workers=args.num_workers,
                    seed=args.seed + gen * 1000 + idx,
                )
                cache[key] = m

            fit = fitness_from_metrics(m)
            fitness_list.append(fit)
            metrics_list.append(m)

            row = {"gen": gen, "idx": idx, "fitness": fit, **m, **cfg}
            with results_csv.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        return fitness_list, metrics_list

    print(f"Run dir: {run_dir}")
    print(f"Fitness objective: {args.fitness} | pop={args.pop_size} | gen={args.generations} | elite={args.elite}")
    if args.n_jobs > 1:
        mode = "CPU parallel" if args.cpu_only else (f"GPU parallel devices={gpu_list}" if gpu_list else "GPU parallel cuda:0")
        print(f"Evaluation mode: {mode} | n_jobs={args.n_jobs}")
    else:
        print(f"Evaluation mode: {'CPU' if args.cpu_only else args.device}")

    pop = [init_individual() for _ in range(args.pop_size)]

    best_global_cfg: Optional[Dict[str, Any]] = None
    best_global_fit = float("inf")
    best_global_metrics: Optional[Dict[str, float]] = None

    for gen in range(0, args.generations + 1):
        fitness, metrics = evaluate_population(pop, gen=gen)

        order = sorted(range(len(pop)), key=lambda i: fitness[i])
        pop = [pop[i] for i in order]
        fitness = [fitness[i] for i in order]
        metrics = [metrics[i] for i in order]

        best_cfg = pop[0]
        best_fit = fitness[0]
        best_m = metrics[0]

        if best_fit < best_global_fit:
            best_global_fit = best_fit
            best_global_cfg = best_cfg
            best_global_metrics = best_m

        print(f"\n=== GEN {gen} BEST ===")
        print(f"fitness={best_fit:.6f} | val_mse={best_m['val_scaled_mse']:.6f} | avg_r2={best_m['avg_r2']:.4f} | time={best_m['train_time_s']:.1f}s")
        print("cfg:", {k: best_cfg[k] for k in GENE_KEYS})

        if gen == args.generations:
            break

        elites = pop[: max(0, min(args.elite, len(pop)))]
        next_gen: List[Dict[str, Any]] = list(elites)

        while len(next_gen) < args.pop_size:
            p1 = tournament_select(pop, fitness, k=args.tournament_k)
            p2 = tournament_select(pop, fitness, k=args.tournament_k)
            child = crossover_1point(p1, p2)
            child = mutate(child, mutation_rate=args.mutation_rate)
            next_gen.append(child)

        pop = next_gen

    assert best_global_cfg is not None and best_global_metrics is not None

    out = {
        "fitness": float(best_global_fit),
        "metrics": best_global_metrics,
        "config": {k: best_global_cfg[k] for k in GENE_KEYS},
        "gene_space": {k: GENE_SPACE[k] for k in GENE_KEYS},
        "suggested_train_command": (
            "torchrun --nproc_per_node=5 train.py "
            f"--csv_path {args.csv_path} "
            f"--lookback {int(best_global_cfg['lookback'])} "
            f"--batch_size {int(best_global_cfg['batch_size'])} "
            f"--lr {float(best_global_cfg['lr'])} "
            f"--hidden_size {int(best_global_cfg['hidden_size'])} "
            f"--num_layers {int(best_global_cfg['num_layers'])} "
            f"--dropout {float(best_global_cfg['dropout'])} "
            f"--fc_size {int(best_global_cfg['fc_size'])} "
            f"--target_smooth_window {int(best_global_cfg['target_smooth_window'])}"
        ),
    }
    best_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n==============================")
    print("FINAL BEST (global)")
    print("==============================")
    print(json.dumps(out["config"], indent=2, ensure_ascii=False))
    print("metrics:", out["metrics"])
    print("\nSuggested full training command:")
    print(out["suggested_train_command"])
    print(f"\nSaved: {best_json}")
    print(f"All trials: {results_csv}")


if __name__ == "__main__":
    main()
