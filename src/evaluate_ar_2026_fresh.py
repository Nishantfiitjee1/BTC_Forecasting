"""Fresh 2026 evaluation of the locked AR(3) model vs persistence; no refit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/processed/btc_usd_hourly_extended_model.csv")
OUT = Path("results/backtests/ar_2026_fresh")
START = pd.Timestamp("2026-01-01T00:00:00Z")
COEF = np.array([-0.006587255086993749, -0.015670938780031313, 0.00362343996296227])
INTERCEPT = 1.420610050224954e-05

def main():
    d = pd.read_csv(INPUT, parse_dates=["timestamp_utc", "target_timestamp_utc"])
    d = d.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep=False)
    d = d[(d.target_available_1h == 1) & (d.target_status == "valid")].copy().reset_index(drop=True)
    d["pred_ar_return"] = np.nan
    for i in range(3, len(d)):
        hist = d.iloc[i-3:i]
        expected = pd.date_range(end=d.at[i, "timestamp_utc"]-pd.Timedelta(hours=1), periods=3, freq="h")
        if list(hist.timestamp_utc) != list(expected):
            continue
        # Previous returns ordered lag 1, lag 2, lag 3
        x = hist.target_return_1h.to_numpy(dtype=float)[::-1]
        d.at[i, "pred_ar_return"] = INTERCEPT + float(COEF @ x)
    s = d[(d.timestamp_utc >= START) & d.pred_ar_return.notna()].copy()
    if s.empty:
        raise RuntimeError("No eligible 2026 predictions found.")
    s["pred_ar_close"] = s.close * np.exp(s.pred_ar_return)
    s["pred_persistence_close"] = s.close
    actual_r = s.target_return_1h.to_numpy(float)
    actual_c = s.target_close_1h.to_numpy(float)
    def met(pr, pc):
        return {"n_predictions": len(s),
                "rmse_return": float(np.sqrt(np.mean((pr-actual_r)**2))),
                "mae_return": float(np.mean(np.abs(pr-actual_r))),
                "mae_close": float(np.mean(np.abs(pc-actual_c))),
                "rmse_close": float(np.sqrt(np.mean((pc-actual_c)**2)))}
    ar = met(s.pred_ar_return.to_numpy(float), s.pred_ar_close.to_numpy(float))
    base = met(np.zeros(len(s)), s.pred_persistence_close.to_numpy(float))
    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / "fresh_2026_predictions.csv"
    s.to_csv(pred_path, index=False)
    summary = {"evaluation":"fresh_2026_out_of_sample", "model":"locked_static_AR(3)_with_intercept",
               "coefficients_lag1_to_lag3":COEF.tolist(), "intercept":INTERCEPT,
               "refit_performed":False, "model_selection_performed":False,
               "ar_metrics":ar, "persistence_metrics":base,
               "ar_minus_persistence_rmse_return":ar["rmse_return"]-base["rmse_return"],
               "ar_minus_persistence_mae_close":ar["mae_close"]-base["mae_close"],
               "predictions_csv":str(pred_path)}
    summary_path = OUT / "fresh_2026_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(f"2026 predictions: {len(s):,}")
    print(f"AR return RMSE:          {ar['rmse_return']:.8f}")
    print(f"Persistence return RMSE: {base['rmse_return']:.8f}")
    print(f"AR close MAE:            {ar['mae_close']:.4f}")
    print(f"Persistence close MAE:   {base['mae_close']:.4f}")
    print(f"Summary: {summary_path}")
    print(f"Predictions: {pred_path}")

if __name__ == "__main__":
    main()
