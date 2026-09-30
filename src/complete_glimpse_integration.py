"""Create Glimpse-compatible BTC price-range probabilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "results" / "forecasts" / "latest_forecast.json"


def build_ranges(
    origin_close: float,
    predicted_return: float,
    predicted_volatility: float,
    width: float = 200.0,
) -> list[dict]:
    """Map the existing one-hour predictive distribution into price ranges."""

    expected_price = origin_close * math.exp(predicted_return)

    # Five central $200 ranges, plus two open-ended tails.
    center = math.floor(expected_price / width) * width
    boundaries = [
        center - 2 * width,
        center - width,
        center,
        center + width,
        center + 2 * width,
        center + 3 * width,
    ]

    normal = NormalDist()

    def cdf(price: float) -> float:
        if price <= 0:
            return 0.0

        z = (
            math.log(price / origin_close) - predicted_return
        ) / predicted_volatility

        return normal.cdf(z)

    ranges = []

    # Lower tail
    ranges.append({
        "range": f"<${boundaries[0]:,.0f}",
        "low": None,
        "high": boundaries[0],
        "probability": cdf(boundaries[0]),
    })

    # Central ranges
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        ranges.append({
            "range": f"${low:,.0f}-${high:,.0f}",
            "low": low,
            "high": high,
            "probability": cdf(high) - cdf(low),
        })

    # Upper tail
    ranges.append({
        "range": f">=${boundaries[-1]:,.0f}",
        "low": boundaries[-1],
        "high": None,
        "probability": 1.0 - cdf(boundaries[-1]),
    })

    total = sum(x["probability"] for x in ranges)

    if total <= 0:
        raise RuntimeError("Invalid probability distribution.")

    for item in ranges:
        item["probability"] = float(item["probability"] / total)
        item["probability_pct"] = float(
            item["probability"] * 100.0
        )

    return ranges


def main() -> None:
    if not FORECAST.exists():
        raise SystemExit(f"Missing forecast: {FORECAST}")

    data = json.loads(
        FORECAST.read_text(encoding="utf-8")
    )

    origin_close = float(data["forecast_origin_close"])
    predicted_return = float(data["predicted_return_1h"])
    predicted_volatility = float(
        data["predicted_volatility_1h"]
    )

    ranges = build_ranges(
        origin_close=origin_close,
        predicted_return=predicted_return,
        predicted_volatility=predicted_volatility,
    )

    data["glimpse_output"] = {
        "format": "expected_price_plus_probability_by_price_range",
        "current_price": origin_close,
        "expected_price_1h": float(
            data["predicted_close_1h"]
        ),
        "target_timestamp_utc": data[
            "target_timestamp_utc"
        ],
        "range_width_usd": 200.0,
        "distribution": (
            "Normal log-return predictive distribution "
            "using AR(3) mean and GARCH(1,1) volatility"
        ),
        "probability_sum": float(
            sum(x["probability"] for x in ranges)
        ),
        "ranges": ranges,
    }

    FORECAST.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("GLIMPSE RANGE PROBABILITIES")
    print("=" * 50)
    print(
        f"Current BTC:   ${origin_close:,.2f}"
    )
    print(
        f"Expected BTC:  ${data['predicted_close_1h']:,.2f}"
    )
    print(
        f"Target:        {data['target_timestamp_utc']}"
    )
    print()

    for item in ranges:
        print(
            f"{item['range']:>22} : "
            f"{item['probability_pct']:7.3f}%"
        )

    total = sum(
        x["probability"] for x in ranges
    )

    print()
    print(f"Probability sum: {total:.12f}")
    print()
    print(f"Updated: {FORECAST}")


if __name__ == "__main__":
    main()