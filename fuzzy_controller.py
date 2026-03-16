# fuzzy_controller.py
# USER fuzzy wrapped as a SEQUENTIAL ΔGreen controller using G_prev.
#
# ΔG(t) = clip(G_fuzzy(t) - G_prev(t-1), [-dg_max, dg_max])
# G_prev(t) = clip(G_prev(t-1) + ΔG(t), [G_min, G_max])
#
# Run:
#   python fuzzy_controller.py --csv_path smart_mobility_dataset.csv --ckpt best_lstm.pt --lookback 72 --target_smooth_window 12
#
#   --save_csv saida_seq.csv

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from data_module import load_and_prepare, PreprocessArtifacts
from model import LSTMRegressor


def trap(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    idx = (a < x) & (x < b)
    if b != a:
        y[idx] = (x[idx] - a) / (b - a)

    y[(b <= x) & (x <= c)] = 1.0

    idx = (c < x) & (x < d)
    if d != c:
        y[idx] = (d - x[idx]) / (d - c)

    return np.clip(y, 0.0, 1.0)


def centroid(x, mu):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    area = np.trapz(mu, x)
    if area == 0:
        return float(np.mean(x))
    return float(np.trapz(mu * x, x) / area)


@dataclass
class UserFuzzyConfig:
    speed_min: float = 0.0
    speed_max: float = 80.0
    occ_min: float = 0.0
    occ_max: float = 100.0
    green_min: float = 10.0
    green_max: float = 120.0

    speed_points: int = 801
    occ_points: int = 1001
    green_points: int = 1111


class UserFuzzyGreenController:
    def __init__(self, cfg: Optional[UserFuzzyConfig] = None):
        self.cfg = cfg or UserFuzzyConfig()
        self.green_u = np.linspace(self.cfg.green_min, self.cfg.green_max, self.cfg.green_points)

        # Output MFs (your original)
        self.green_short = trap(self.green_u, 10, 10, 25, 40)
        self.green_med   = trap(self.green_u, 30, 45, 55, 70)
        self.green_long  = trap(self.green_u, 60, 80, 95, 110)
        self.green_maxmf = trap(self.green_u, 95, 110, 120, 120)

    def green_time(self, speed_kmh: float, occ_pct: float) -> float:
        speed_kmh = float(np.clip(speed_kmh, self.cfg.speed_min, self.cfg.speed_max))
        occ_pct = float(np.clip(occ_pct, self.cfg.occ_min, self.cfg.occ_max))

        sL = float(trap([speed_kmh], 0, 0, 40, 55)[0])
        sM = float(trap([speed_kmh], 40, 52, 65, 75)[0])
        sH = float(trap([speed_kmh], 65, 72, 80, 80)[0])

        oL = float(trap([occ_pct], 0, 0, 25, 45)[0])
        oM = float(trap([occ_pct], 30, 45, 55, 70)[0])
        oH = float(trap([occ_pct], 60, 80, 100, 100)[0])

        r1 = min(oH, sL)  # MAX
        r2 = min(oH, sM)  # LONG
        r3 = min(oM, sL)  # LONG
        r4 = min(oM, sM)  # MED
        r5 = min(oM, sH)  # SHORT
        r6 = min(oL, sL)  # MED
        r7 = min(oL, sM)  # SHORT
        r8 = min(oL, sH)  # SHORT
        r9 = min(oH, sH)  # LONG

        out1 = np.minimum(self.green_maxmf, r1)
        out2 = np.minimum(self.green_long,  r2)
        out3 = np.minimum(self.green_long,  r3)
        out4 = np.minimum(self.green_med,   r4)
        out5 = np.minimum(self.green_short, r5)
        out6 = np.minimum(self.green_med,   r6)
        out7 = np.minimum(self.green_short, r7)
        out8 = np.minimum(self.green_short, r8)
        out9 = np.minimum(self.green_long,  r9)

        out = out1
        for o in (out2, out3, out4, out5, out6, out7, out8, out9):
            out = np.maximum(out, o)

        return float(centroid(self.green_u, out))


def _load_checkpoint(ckpt_path: str, device: torch.device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _artifacts_from_ckpt(art_dict: Dict) -> PreprocessArtifacts:
    return PreprocessArtifacts(
        feature_names=list(art_dict["feature_names"]),
        target_names=list(art_dict["target_names"]),
        x_scaler_mean=np.asarray(art_dict["x_scaler_mean"], dtype=np.float64),
        x_scaler_scale=np.asarray(art_dict["x_scaler_scale"], dtype=np.float64),
        y_scaler_mean=np.asarray(art_dict["y_scaler_mean"], dtype=np.float64),
        y_scaler_scale=np.asarray(art_dict["y_scaler_scale"], dtype=np.float64),
        idx_speed_in_X=int(art_dict["idx_speed_in_X"]),
        idx_occ_in_X=int(art_dict["idx_occ_in_X"]),
    )


def inverse_y(y_scaled: torch.Tensor, art: PreprocessArtifacts) -> torch.Tensor:
    mean = torch.tensor(art.y_scaler_mean, device=y_scaled.device, dtype=y_scaled.dtype)
    scale = torch.tensor(art.y_scaler_scale, device=y_scaled.device, dtype=y_scaled.dtype)
    return y_scaled * scale + mean


def simulate(ctrl: UserFuzzyGreenController, speed_occ: np.ndarray, g_init: float, g_min: float, g_max: float, dg_max: float):
    N = speed_occ.shape[0]
    g_series = np.zeros(N, dtype=np.float64)
    dg_series = np.zeros(N, dtype=np.float64)
    g_fuzzy = np.zeros(N, dtype=np.float64)

    g_prev = float(np.clip(g_init, g_min, g_max))
    for t in range(N):
        sp, oc = float(speed_occ[t, 0]), float(speed_occ[t, 1])
        g_des = float(np.clip(ctrl.green_time(sp, oc), g_min, g_max))
        g_fuzzy[t] = g_des

        dg = float(np.clip(g_des - g_prev, -dg_max, dg_max))
        g_prev = float(np.clip(g_prev + dg, g_min, g_max))

        dg_series[t] = dg
        g_series[t] = g_prev

    return g_series, dg_series, g_fuzzy


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best_lstm.pt")
    p.add_argument("--lookback", type=int, default=72)
    p.add_argument("--freq_minutes", type=int, default=5)
    p.add_argument("--target_smooth_window", type=int, default=12)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--g_init", type=float, default=45.0)
    p.add_argument("--g_min", type=float, default=10.0)
    p.add_argument("--g_max", type=float, default=120.0)
    p.add_argument("--dg_max", type=float, default=20.0)

    p.add_argument("--save_csv", type=str, default="")
    args = p.parse_args()

    device = torch.device(args.device)

    _, _, test_ds, _ = load_and_prepare(
        args.csv_path,
        lookback=args.lookback,
        freq_minutes=args.freq_minutes,
        target_smooth_window=args.target_smooth_window,
    )

    ckpt = _load_checkpoint(args.ckpt, device)
    art = _artifacts_from_ckpt(ckpt["artifacts"])

    model = LSTMRegressor(
        n_features=len(art.feature_names),
        hidden_size=ckpt["args"].get("hidden_size", 64),
        num_layers=ckpt["args"].get("num_layers", 1),
        dropout=ckpt["args"].get("dropout", 0.2),
        fc_size=ckpt["args"].get("fc_size", 64),
        out_size=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X = test_ds.X.to(device)
    y_true = inverse_y(test_ds.y.to(device), art).cpu().numpy()
    y_pred = inverse_y(model(X), art).cpu().numpy()

    ctrl = UserFuzzyGreenController(UserFuzzyConfig(green_min=args.g_min, green_max=args.g_max))

    g_pred, dg_pred, g_fuzzy_pred = simulate(ctrl, y_pred, args.g_init, args.g_min, args.g_max, abs(args.dg_max))
    g_oracle, dg_oracle, g_fuzzy_oracle = simulate(ctrl, y_true, args.g_init, args.g_min, args.g_max, abs(args.dg_max))

    mae_dg = float(np.mean(np.abs(dg_pred - dg_oracle)))
    rmse_dg = float(np.sqrt(np.mean((dg_pred - dg_oracle) ** 2)))
    mae_g = float(np.mean(np.abs(g_pred - g_oracle)))
    rmse_g = float(np.sqrt(np.mean((g_pred - g_oracle) ** 2)))

    sat_up = float(np.mean(np.isclose(dg_pred, abs(args.dg_max))))
    sat_dn = float(np.mean(np.isclose(dg_pred, -abs(args.dg_max))))

    print("=== USER Fuzzy SEQUENTIAL ΔGreen controller (test) ===")
    print(f"Targets fed to fuzzy: predicted {art.target_names}")
    print(f"Init green: {args.g_init:.1f} sec, clipped to [{args.g_min:.1f}, {args.g_max:.1f}]")
    print(f"ΔG clip: ±{abs(args.dg_max):.1f} sec  |  sat_up={sat_up*100:.1f}% sat_dn={sat_dn*100:.1f}%")
    print(f"ΔGreen MAE vs oracle:  {mae_dg:.3f} sec")
    print(f"ΔGreen RMSE vs oracle: {rmse_dg:.3f} sec")
    print(f"Green MAE vs oracle:   {mae_g:.3f} sec")
    print(f"Green RMSE vs oracle:  {rmse_g:.3f} sec")
    print("First 10 ΔG:", dg_pred[:10].round(2).tolist())
    print("First 10 G:", g_pred[:10].round(2).tolist())

    if args.save_csv:
        import pandas as pd
        out = pd.DataFrame({
            "y0_true": y_true[:, 0],
            "y1_true": y_true[:, 1],
            "y0_pred": y_pred[:, 0],
            "y1_pred": y_pred[:, 1],
            "g_fuzzy_oracle": g_fuzzy_oracle,
            "g_fuzzy_pred": g_fuzzy_pred,
            "dg_oracle": dg_oracle,
            "dg_pred": dg_pred,
            "g_oracle": g_oracle,
            "g_pred": g_pred,
            "dg_abs_err": np.abs(dg_pred - dg_oracle),
            "g_abs_err": np.abs(g_pred - g_oracle),
        })
        out.to_csv(args.save_csv, index=False)
        print("Saved:", args.save_csv)


if __name__ == "__main__":
    main()
