
"""Project release audit for BTC_Forecasting. Run from any working directory."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "src/download_btc_hourly.py",
    "src/prepare_hourly_model_data.py",
    "src/fit_ar_baseline.py",
    "src/backtest_ar.py",
    "src/garch_volatility.py",
    "src/backtest_garch.py",
    "src/forecast_one_hour.py",
    "src/backtest_trading_pnl.py",
    "tests",
    "results/forecasts/latest_forecast.json",
    "results/backtests/trading_pnl/trading_pnl_summary.json",
]

FORECAST_REQUIRED = {
    "forecast_timestamp_utc",
    "target_timestamp_utc",
    "input_latest_timestamp_utc",
    "forecast_origin_close",
    "predicted_return_1h",
    "predicted_close_1h",
    "predicted_variance_1h",
    "predicted_volatility_1h",
    "production_refit_scope",
    "point_model",
    "volatility_model",
    "interval_calibration",
    "protocol",
}

EXPERIMENT_NAME_FRAGMENTS = (
    "student_t_garch_experiment",
    "garch_order_experiment",
    "gjr_garch_experiment",
)


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    return bool(ok)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def main() -> int:
    print(f"BTC Forecasting release audit\nProject root: {ROOT}\n")
    passed = True

    # Check required files and directories.
    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        passed &= check(
            f"Required path: {relative_path}",
            path.exists(),
        )

    # Compile source and test files without executing them.
    py_files = sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "tests").rglob("*.py")
    )

    syntax_errors = []

    for path in py_files:
        try:
            compile(
                path.read_text(encoding="utf-8-sig"),
                str(path),
                "exec",
            )
        except Exception as exc:
            syntax_errors.append(
                f"{path.relative_to(ROOT)}: {exc}"
            )

    passed &= check(
        "Python syntax",
        not syntax_errors,
        "; ".join(syntax_errors[:5])
        if syntax_errors
        else f"{len(py_files)} files compiled",
    )

    # Validate the saved forecast artifact.
    forecast_path = ROOT / "results/forecasts/latest_forecast.json"

    if forecast_path.is_file():
        try:
            forecast = read_json(forecast_path)

            missing = sorted(
                FORECAST_REQUIRED - set(forecast)
            )

            numeric_fields = (
                "forecast_origin_close",
                "predicted_return_1h",
                "predicted_close_1h",
                "predicted_variance_1h",
                "predicted_volatility_1h",
            )

            bad_numbers = [
                key
                for key in numeric_fields
                if not finite_number(forecast.get(key))
            ]

            try:
                origin = datetime.fromisoformat(
                    str(
                        forecast["forecast_timestamp_utc"]
                    ).replace("Z", "+00:00")
                )
                target = datetime.fromisoformat(
                    str(
                        forecast["target_timestamp_utc"]
                    ).replace("Z", "+00:00")
                )

                time_ok = (
                    origin.tzinfo is not None
                    and target.tzinfo is not None
                    and target > origin
                )

            except (KeyError, TypeError, ValueError):
                time_ok = False

            valid = (
                not missing
                and not bad_numbers
                and time_ok
            )

            issues = []

            if missing:
                issues.append(f"missing keys={missing}")

            if bad_numbers:
                issues.append(
                    f"invalid numeric fields={bad_numbers}"
                )

            if not time_ok:
                issues.append(
                    "timestamps must be timezone-aware "
                    "and target must follow origin"
                )

            passed &= check(
                "Forecast JSON schema and values",
                valid,
                "; ".join(issues)
                if issues
                else "fields present, finite numbers, valid time order",
            )

        except Exception as exc:
            passed &= check(
                "Forecast JSON readable",
                False,
                str(exc),
            )
    else:
        passed &= check(
            "Forecast JSON readable",
            False,
            "file is missing",
        )

    # Validate the P&L summary and its limitations.
    pnl_path = (
        ROOT
        / "results/backtests/trading_pnl/trading_pnl_summary.json"
    )

    if pnl_path.is_file():
        try:
            pnl = read_json(pnl_path)

            scenarios = pnl.get("fee_scenarios")

            required_scenario_keys = {
                "fee_bps_per_side",
                "n_trades",
                "gross_pnl",
                "fees_paid",
                "net_pnl",
                "net_roi_pct",
                "buy_hold_net_roi_pct",
            }

            valid = (
                isinstance(scenarios, list)
                and len(scenarios) > 0
            )

            if valid:
                valid = all(
                    isinstance(scenario, dict)
                    and required_scenario_keys.issubset(
                        scenario
                    )
                    and all(
                        finite_number(scenario[key])
                        for key in required_scenario_keys
                    )
                    for scenario in scenarios
                )

            raw_notes = pnl.get("notes", [])

            if isinstance(raw_notes, str):
                notes = raw_notes.lower()
            elif isinstance(raw_notes, list):
                notes = " ".join(
                    str(note).lower()
                    for note in raw_notes
                )
            else:
                notes = ""

            method_ok = (
                pnl.get("method")
                == "fixed_notional_one_hour_directional_backtest"
            )

            profitability_caveat = (
                "not evidence of deployable profitability"
                in notes
            )

            no_compounding_caveat = (
                "no compounding" in notes
                or "not compounded" in notes
            )

            caution = (
                method_ok
                and profitability_caveat
                and no_compounding_caveat
            )

            passed &= check(
                "P&L summary schema",
                valid,
                f"{len(scenarios) if isinstance(scenarios, list) else 0} fee scenario(s)",
            )

            passed &= check(
                "P&L caveats documented",
                caution,
                "fixed-notional, non-compounded diagnostic",
            )

        except Exception as exc:
            passed &= check(
                "P&L summary readable",
                False,
                str(exc),
            )
    else:
        passed &= check(
            "P&L summary readable",
            False,
            "file is missing",
        )

    # Check that rejected experiment artifacts are absent.
    found = []

    for base in (ROOT / "src", ROOT / "results"):
        if base.exists():
            for path in base.rglob("*"):
                if (
                    path.is_file()
                    and any(
                        fragment in path.name.lower()
                        for fragment in EXPERIMENT_NAME_FRAGMENTS
                    )
                ):
                    found.append(
                        str(path.relative_to(ROOT))
                    )

    passed &= check(
        "Rejected experiment artifacts absent",
        not found,
        ", ".join(found) if found else "none found",
    )

    # Run only this project's test suite.
    tests_dir = ROOT / "tests"

    if tests_dir.is_dir():
        print("\nRunning project tests only:")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(tests_dir),
            ],
            cwd=ROOT,
            text=True,
        )

        passed &= check(
            "Project test suite",
            result.returncode == 0,
            f"pytest exit code {result.returncode}",
        )
    else:
        passed &= check(
            "Project test suite",
            False,
            "tests directory is missing",
        )

    # Flag .dist for manual inspection; never delete automatically.
    odd_dist = ROOT / ".dist"

    passed &= check(
        "Unexpected .dist directory",
        not odd_dist.exists(),
        (
            "not present"
            if not odd_dist.exists()
            else "inspect contents; not deleted automatically"
        ),
    )

    print(
        "\n"
        + (
            "RELEASE AUDIT PASSED"
            if passed
            else "RELEASE AUDIT NEEDS FIXES"
        )
    )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
