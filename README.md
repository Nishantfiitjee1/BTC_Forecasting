# BTC/USD One-Hour Forecasting

Leakage-controlled Bitcoin forecasting pipeline for the Glimpse Bitcoin Forecasting Hackathon 2026.

## Objective

Forecast the next hourly BTC/USD close from information available at the forecast origin.

The pipeline produces:

- one-hour-ahead point return forecast
- one-hour-ahead BTC/USD price forecast
- one-hour conditional volatility forecast
- calibrated prediction intervals

The current research snapshot uses Coinbase BTC-USD hourly OHLCV data from 2021-01-01 through 2026-01-01.

## Final model architecture

```text
Coinbase BTC-USD hourly OHLCV
            |
            v
     gap-aware processing
            |
      +-----+------+
      |            |
      v            v
    AR(3)       GARCH(1,1)
      |            |
      |       volatility
      |            |
      +-----+------+
            |
            v
     interval calibration
            |
            v
       1-hour forecast
```

### Mean model

The locked mean model is **AR(3) with an intercept**.

It predicts the next hourly log return using the current and previous two hourly returns. Model selection is performed on the validation period only; the 2025 test period is excluded from selection.

### Volatility model

The locked volatility model is **Normal GARCH(1,1)** with a constant conditional mean.

GARCH models conditional return variance. The model is evaluated using QLIKE and squared-return MSE against a rolling-24-hour variance baseline.

### Interval calibration

Prediction intervals are produced from the locked AR and GARCH forecasts.

Validation-only calibration experiments found:

- 50% interval: empirical residual quantiles
- 80% interval: empirical residual quantiles
- 90% interval: empirical residual quantiles
- 95% interval: Gaussian quantile

These choices are a calibration layer and do not change the locked AR/GARCH models.

## Data

### Source

- Exchange/data source: Coinbase Exchange
- Product: BTC-USD
- Frequency: 1 hour
- Timezone: UTC
- Historical development snapshot: 2021-01-01 to 2026-01-01

The raw OHLCV snapshot is retained unchanged after acquisition.

The frozen raw dataset contains 8 missing hourly timestamps. They are reported rather than filled or interpolated:

- 2023-03-04 18:00, 19:00, 20:00 UTC
- 2025-10-25 16:00 through 20:00 UTC

The raw audit found no duplicate timestamps, NaNs, non-positive prices, negative volumes, or OHLC consistency violations.

### Gap handling

The pipeline does not bridge missing hours.

A one-hour target is valid only when the next timestamp is exactly one hour after the forecast origin. Return calculations are also gap-aware.

No price interpolation or silent forward filling is performed.

## Chronological evaluation

The historical snapshot is divided into:

| Period | Dates | Purpose |
|---|---|---|
| Development | 2021-01-01 to 2024-01-01 | Model fitting |
| Validation | 2024-01-01 to 2025-01-01 | Model/calibration selection |
| Test | 2025-01-01 to 2026-01-01 | One-time held-out evaluation |

The test period is not used for AR model selection.

The locked AR artifact records test-isolation metadata and training-slice provenance.

## Baselines

### Persistence price baseline

The persistence baseline predicts:

```text
next return = 0
next close = current close
```

On the 2025 held-out test period:

- Persistence return RMSE: 0.00477590
- AR(3) return RMSE: 0.00477810

Therefore the AR model should **not** be described as a strong return-prediction edge over persistence.

AR(3) produced a small improvement in close-price MAE on the same held-out period, but the return forecast comparison is the more direct measure for this target.

### Rolling volatility baseline

The volatility baseline uses a trailing 24-hour realized-return variance.

On the 2025 held-out comparison, GARCH(1,1) produced lower QLIKE than the rolling-24-hour baseline. The advantage remained after removing the 2025-10-25 data-gap/recovery window.

## Model selection and rejected experiments

The project deliberately keeps the final model simple.

Several alternatives were evaluated on validation data and rejected rather than retained merely because they were more complex.

Rejected experiments included:

- Student-t GARCH
- alternative symmetric GARCH orders
- GJR-GARCH

The rejected experiments did not improve the locked validation objective sufficiently to justify replacing the selected Normal GARCH(1,1).

## Statistical comparison

Model comparisons use paired forecasts on common timestamps.

For dependent hourly observations, comparisons include:

- Diebold-Mariano-style HAC test statistics
- moving-block bootstrap confidence intervals

The statistical tests are supplementary evidence; model selection remains chronological and validation-based.

## Forecast engine

Run:

```bash
python src/forecast_one_hour.py
```

The engine:

1. loads the processed hourly dataset
2. verifies the latest row is the forecast origin
3. prevents a known future target from being used at the forecast origin
4. fits the production mean model without using the future target
5. uses the latest observed return for the volatility forecast
6. produces the one-hour point forecast and intervals
7. writes:

```text
results/forecasts/latest_forecast.json
```

The forecast JSON contains the forecast timestamp, target timestamp, origin close, predicted return, predicted close, predicted variance/volatility, model identifiers, calibration method, and protocol metadata.

### Important snapshot limitation

The current reproducibility dataset ends at **2026-01-01**.

Therefore the forecast currently produced from this frozen snapshot is a historical/reproducibility forecast, **not the final September 2026 competition forecast**.

For the final competition run, the historical data must be refreshed through the permitted forecast origin and the same leakage-controlled pipeline rerun.

## Backtesting commands

### AR selection

```bash
python src/backtest_ar.py --stage select
```

### AR held-out test

```bash
python src/backtest_ar.py --stage test
```

### GARCH selection/test

Use the corresponding `backtest_garch.py` stages documented by the script.

### Full tests

Run only the project's test directory:

```bash
python -m pytest -q tests
```

## Project structure

```text
BTC_Forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── backtests/
│   └── forecasts/
├── src/
├── tests/
├── README.md
└── requirements.txt
```

## Reproducibility

The project records dataset/model provenance and uses deterministic numerical procedures where applicable.

Important frozen dataset hashes:

```text
Raw BTC/USD CSV:
f79ed20eae17367b62c2c3bcd1fccd4d3471cdbb456523d959ad583b00f1c237

Processed model CSV:
59267d0808f12520218fe8e11b081e362a99e85a57cd10fc3a1cace83c354afa
```

## Methodological principles

This project follows several non-negotiable rules:

- no future target values in forecast features
- no random train/test split for time-series evaluation
- no interpolation across missing market hours
- no silent deletion of suspicious observations
- no test-period model selection
- no test-period hyperparameter tuning
- no hidden future information in interval calibration
- preserve the raw dataset separately from processed modeling data

## Status

The research/modeling pipeline is complete through the current frozen historical snapshot.

The remaining work is release engineering:

- final dependency/environment verification
- final cleanup of rejected artifacts
- final release audit
- refresh data for the actual competition forecast period
- final reproducibility run
