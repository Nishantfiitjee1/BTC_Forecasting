"""Leakage-safe fixed-notional P&L translation of locked AR test forecasts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/backtests/ar_static/selected_test_predictions.csv"
DEFAULT_OUT = ROOT / "results/backtests/trading_pnl"
REQUIRED = {"timestamp_utc","forecast_origin_close","predicted_return_1h","predicted_close_1h","actual_return_1h","actual_close_1h"}

class BacktestError(ValueError): pass

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file(): raise BacktestError(f"Missing input: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing: raise BacktestError(f"Missing columns: {sorted(missing)}")
    if df.empty: raise BacktestError("No prediction rows.")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="raise")
    if df["timestamp_utc"].isna().any() or df["timestamp_utc"].duplicated().any():
        raise BacktestError("Invalid or duplicate timestamps.")
    if not df["timestamp_utc"].is_monotonic_increasing: raise BacktestError("Timestamps not chronological.")
    cols = ["forecast_origin_close","predicted_return_1h","predicted_close_1h","actual_return_1h","actual_close_1h"]
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    x = df[cols].to_numpy(float)
    if not np.isfinite(x).all(): raise BacktestError("NaN/Inf in prediction data.")
    if (df["forecast_origin_close"] <= 0).any() or (df["actual_close_1h"] <= 0).any():
        raise BacktestError("Prices must be positive.")
    pclose = df["forecast_origin_close"].to_numpy() * np.exp(df["predicted_return_1h"].to_numpy())
    if not np.allclose(pclose, df["predicted_close_1h"].to_numpy(), rtol=1e-8, atol=1e-8):
        raise BacktestError("Predicted close inconsistent with predicted return.")
    aret = np.log(df["actual_close_1h"].to_numpy()/df["forecast_origin_close"].to_numpy())
    if not np.allclose(aret, df["actual_return_1h"].to_numpy(), rtol=1e-8, atol=1e-10):
        raise BacktestError("Actual return inconsistent with close prices.")
    return df

def max_drawdown(equity, capital):
    path = np.r_[capital, equity]
    peaks = np.maximum.accumulate(path)
    return float(np.min(np.divide(path-peaks, peaks, out=np.zeros_like(path), where=peaks != 0)))

def evaluate(df, capital, fee_bps):
    if not np.isfinite(capital) or capital <= 0: raise BacktestError("Capital must be positive.")
    if not np.isfinite(fee_bps) or fee_bps < 0: raise BacktestError("Fee must be non-negative.")
    pos = np.sign(df["predicted_return_1h"].to_numpy()).astype(int)
    simple_ret = np.expm1(df["actual_return_1h"].to_numpy())
    active = pos != 0
    gross_ret = pos * simple_ret
    fee_ret = active * (2 * fee_bps / 10000)
    gross = capital * gross_ret
    fees = capital * fee_ret
    net = gross - fees
    equity = capital + np.cumsum(net)
    first, last = float(df["forecast_origin_close"].iloc[0]), float(df["actual_close_1h"].iloc[-1])
    bh_gross = capital * (last/first - 1)
    bh_fee = capital * (2 * fee_bps / 10000)
    summary = {
        "strategy":"sign_of_locked_AR_forecast_one_hour_hold",
        "position_rule":"long if forecast return > 0, short if < 0, flat if == 0",
        "holding_period":"one forecasted hour, then close",
        "sizing":"fixed notional equal to initial capital; no compounding",
        "initial_capital":float(capital),"fee_bps_per_side":float(fee_bps),
        "round_trip_fee_bps":float(2*fee_bps),"n_forecasts":int(len(df)),
        "n_trades":int(active.sum()),"long_trades":int((pos>0).sum()),
        "short_trades":int((pos<0).sum()),"flat_forecasts":int((pos==0).sum()),
        "gross_pnl":float(gross.sum()),"fees_paid":float(fees.sum()),
        "net_pnl":float(net.sum()),"gross_roi_pct":float(gross.sum()/capital*100),
        "net_roi_pct":float(net.sum()/capital*100),
        "max_drawdown_pct":float(max_drawdown(equity, capital)*100),
        "positive_trade_rate_pct":float((gross_ret[active]>0).mean()*100) if active.any() else None,
        "buy_hold_gross_pnl":float(bh_gross),"buy_hold_fees_paid":float(bh_fee),
        "buy_hold_net_pnl":float(bh_gross-bh_fee),
        "buy_hold_net_roi_pct":float((bh_gross-bh_fee)/capital*100),
        "test_used_for_model_selection":False,"model_refit_performed":False,
        "fee_note":"Fee rates are sensitivity assumptions, not verified exchange fees."
    }
    detail = pd.DataFrame({
        "timestamp_utc":df["timestamp_utc"],"forecast_origin_close":df["forecast_origin_close"],
        "predicted_return_1h":df["predicted_return_1h"],"actual_return_1h":df["actual_return_1h"],
        "position":pos,"gross_return_on_notional":gross_ret,"fees_paid":fees,
        "gross_pnl":gross,"net_pnl":net,"equity":equity
    })
    return summary, detail

def run(input_path, outdir, capital, fees):
    df=load_predictions(input_path)
    if not fees or len(set(fees)) != len(fees): raise BacktestError("Fee scenarios must be nonempty and unique.")
    outdir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for fee in fees:
        s,d=evaluate(df,capital,fee); rows.append(s)
        label=f"{fee:g}bps_per_side".replace(".","p")
        d.to_csv(outdir/f"trades_{label}.csv",index=False,float_format="%.17g")
    result={
        "method":"fixed_notional_one_hour_directional_backtest",
        "input_file":str(input_path),
        "test_period_start_utc":df.timestamp_utc.iloc[0].isoformat(),
        "test_period_last_origin_utc":df.timestamp_utc.iloc[-1].isoformat(),
        "initial_capital":float(capital),"fee_scenarios":rows,
        "notes":[
            "Uses already-generated locked AR test predictions; no model fitting or tuning.",
            "Each signal is held for its one-hour target and closed; no positions bridge data gaps.",
            "Fixed notional equals initial capital; P&L is not compounded.",
            "0/5/10 bps fee scenarios are assumptions; verify actual exchange fees and slippage before any live-trading claim.",
            "This is a simple backtest translation, not evidence of deployable profitability."
        ]
    }
    out=outdir/"trading_pnl_summary.json"
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Rows: {len(df)} | Period: {result['test_period_start_utc']} -> {result['test_period_last_origin_utc']}")
    for s in rows:
        print(f"Fee {s['fee_bps_per_side']:g} bps/side | trades {s['n_trades']} | gross P&L {s['gross_pnl']:.2f} | fees {s['fees_paid']:.2f} | net P&L {s['net_pnl']:.2f} | net ROI {s['net_roi_pct']:.3f}% | buy&hold net ROI {s['buy_hold_net_roi_pct']:.3f}%")
    print(f"Saved: {out}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--initial-capital",type=float,default=10000.0)
    p.add_argument("--fees-bps",type=float,nargs="+",default=[0.0,5.0,10.0])
    a=p.parse_args()
    try: run(a.input,a.output_dir,a.initial_capital,a.fees_bps)
    except (BacktestError,OSError,ValueError,json.JSONDecodeError) as e:
        print(f"ERROR: {e}"); return 1
    return 0
if __name__=="__main__": raise SystemExit(main())
