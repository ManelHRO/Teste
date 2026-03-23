from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.allclose(np.var(y_true), 0.0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with y0_true/y0_pred/y1_true/y1_pred columns.")
    p.add_argument("--out", default="pred_vs_real.pdf", help="Output figure path (.pdf or .png).")
    p.add_argument("--title0", default="Traffic_Speed_kmh", help="Label for target 0 (speed).")
    p.add_argument("--title1", default="Road_Occupancy_%", help="Label for target 1 (occupancy).")
    p.add_argument("--xlabel", default="Timesteps (Conjunto de Teste)", help="X-axis label.")
    p.add_argument("--ylabel0", default="km/h", help="Y-axis label for speed subplot.")
    p.add_argument("--ylabel1", default="% Ocupação", help="Y-axis label for occupancy subplot.")
    p.add_argument("--n_points", type=int, default=0, help="If >0, plot only the first N points.")
    p.add_argument("--start", type=int, default=0, help="Start index for plotting.")
    p.add_argument("--use_timestamp", action="store_true", help="Use 'Timestamp' column as x-axis if present.")
    p.add_argument("--fig_w", type=float, default=7.0, help="Figure width in inches.")
    p.add_argument("--fig_h", type=float, default=5.2, help="Figure height in inches.")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs (png).")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    needed = ["y0_true", "y0_pred", "y1_true", "y1_pred"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    start = max(0, int(args.start))
    end = None
    if args.n_points and args.n_points > 0:
        end = start + int(args.n_points)

    dfp = df.iloc[start:end].reset_index(drop=True)

    y0_true = dfp["y0_true"].to_numpy(dtype=float)
    y0_pred = dfp["y0_pred"].to_numpy(dtype=float)
    y1_true = dfp["y1_true"].to_numpy(dtype=float)
    y1_pred = dfp["y1_pred"].to_numpy(dtype=float)

    r2_0 = _safe_r2(y0_true, y0_pred)
    r2_1 = _safe_r2(y1_true, y1_pred)

    if args.use_timestamp and "Timestamp" in dfp.columns:
        x = pd.to_datetime(dfp["Timestamp"]).to_numpy()
        x_label = "Timestamp"
    else:
        x = np.arange(len(dfp))
        x_label = args.xlabel

    fig, axs = plt.subplots(2, 1, figsize=(args.fig_w, args.fig_h), sharex=True)

    axs[0].plot(x, y0_true, label="Real (Velocidade)")
    axs[0].plot(x, y0_pred, linestyle="--", label="Predito (Velocidade)")
    axs[0].set_ylabel(args.ylabel0)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")
    axs[0].set_title(f"Target: {args.title0} | R²: {r2_0:.4f}" if np.isfinite(r2_0) else f"Target: {args.title0} | R²: N/A")

    axs[1].plot(x, y1_true, label="Real (Ocupação)")
    axs[1].plot(x, y1_pred, linestyle="--", label="Predito (Ocupação)")
    axs[1].set_ylabel(args.ylabel1)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")
    axs[1].set_title(f"Target: {args.title1} | R²: {r2_1:.4f}" if np.isfinite(r2_1) else f"Target: {args.title1} | R²: N/A")

    axs[1].set_xlabel(x_label)

    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, bbox_inches="tight")

    print(f"Saved figure -> {out_path.resolve()}")


if __name__ == "__main__":
    main()