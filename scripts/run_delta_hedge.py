"""Sell a one-month at-the-money-forward GBPUSD call every trading day and delta-hedge it daily.

The option is priced with the trailing one-month (21 trading days) realised volatility, a stand-in
for implied volatility, which is not freely available. Spot is FRED's daily GBPUSD noon buying
rate in New York; interest rates are OECD monthly 3-month rates, each day using the previous
month's average.

Writes to results/: delta_hedge_summary.csv, delta_hedge_by_decade.csv, delta_hedge_worst.csv,
delta_hedge_paths.csv and delta_hedge.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fxlab import DeltaHedgeSimulator, FxDataStore, by_code, daily_rates_from_monthly, to_continuous
from fxlab.currencies import USD_DAY_COUNT_BASIS
from fxlab.report import plot_hedge

NOTIONAL_GBP = 1e6


def ols(y: pd.Series, x: pd.Series) -> tuple[float, float, float]:
    """Intercept, slope and R-squared of y on x."""
    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return float(intercept), float(slope), r2


def distribution(outcomes: pd.DataFrame, label: str) -> dict:
    pnl = outcomes["pnl"] * NOTIONAL_GBP
    gt = outcomes["gamma_theta_pnl"] * NOTIONAL_GBP
    intercept, slope, r2 = ols(pnl, gt)
    _, _, r2_var = ols(pnl, outcomes["variance_gap"])
    return {
        "sample": label,
        "options": len(outcomes),
        "first_start": f"{outcomes['start'].iloc[0]:%Y-%m-%d}",
        "last_expiry": f"{outcomes['expiry'].iloc[-1]:%Y-%m-%d}",
        "mean_premium_usd": float((outcomes["premium"] * NOTIONAL_GBP).mean()),
        "mean_pnl_usd": float(pnl.mean()),
        "median_pnl_usd": float(pnl.median()),
        "std_pnl_usd": float(pnl.std()),
        "p05_pnl_usd": float(pnl.quantile(0.05)),
        "p95_pnl_usd": float(pnl.quantile(0.95)),
        "min_pnl_usd": float(pnl.min()),
        "max_pnl_usd": float(pnl.max()),
        "share_positive": float((pnl > 0).mean()),
        "skew_pnl": float(pnl.skew()),
        "mean_pricing_vol": float(outcomes["pricing_vol"].mean()),
        "mean_realised_vol": float(outcomes["realised_vol"].mean()),
        "share_realised_above_pricing": float((outcomes["realised_vol"] > outcomes["pricing_vol"]).mean()),
        "corr_pnl_variance_gap": float(np.corrcoef(pnl, outcomes["variance_gap"])[0, 1]),
        "r2_pnl_on_variance_gap": r2_var,
        "slope_pnl_on_gamma_theta": slope,
        "intercept_pnl_on_gamma_theta_usd": intercept,
        "r2_pnl_on_gamma_theta": r2,
    }


def worst_episodes(outcomes: pd.DataFrame, k: int = 10, gap_days: int = 31) -> pd.DataFrame:
    """The k worst options, keeping only one per episode (starts at least `gap_days` apart)."""
    chosen = []
    for _, row in outcomes.sort_values("pnl").iterrows():
        if all(abs((row["start"] - c["start"]).days) >= gap_days for c in chosen):
            chosen.append(row)
        if len(chosen) == k:
            break
    table = pd.DataFrame(chosen)
    return pd.DataFrame({
        "start": table["start"].dt.strftime("%Y-%m-%d"),
        "expiry": table["expiry"].dt.strftime("%Y-%m-%d"),
        "spot_start": table["spot0"],
        "spot_expiry": table["spot_expiry"],
        "pricing_vol_pct": 100 * table["pricing_vol"],
        "realised_vol_pct": 100 * table["realised_vol"],
        "premium_usd": table["premium"] * NOTIONAL_GBP,
        "pnl_usd": table["pnl"] * NOTIONAL_GBP,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/fred")
    parser.add_argument("--out", default="results")
    parser.add_argument("--start", default="1975-01-01", help="first trade date")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    gbp = by_code("GBP")
    store = FxDataStore(args.data)
    spot = store.series(gbp.spot_series)  # USD per GBP
    rates = store.monthly_rates()
    month = 1 / 12
    r_usd = daily_rates_from_monthly(rates["USD"], spot.index).pipe(to_continuous, USD_DAY_COUNT_BASIS, month)
    r_gbp = daily_rates_from_monthly(rates["GBP"], spot.index).pipe(to_continuous, gbp.day_count_basis, month)
    r_usd = pd.Series(r_usd, index=spot.index)
    r_gbp = pd.Series(r_gbp, index=spot.index)

    simulator = DeltaHedgeSimulator(spot, r_dom=r_usd, r_for=r_gbp, tenor_months=1, vol_window=21)
    outcomes = simulator.run(start=args.start)
    print(f"{len(outcomes):,} options, first start {outcomes['start'].iloc[0]:%d %b %Y}, "
          f"last expiry {outcomes['expiry'].iloc[-1]:%d %b %Y}")

    # Every trading day starts an option, so neighbouring options overlap. The first trading day of
    # each month gives a sample of options that do not overlap, as a check.
    first_of_month = outcomes.groupby(outcomes["start"].dt.to_period("M")).head(1)
    summary = pd.DataFrame([distribution(outcomes, "every trading day"),
                            distribution(first_of_month, "first trading day of each month")])
    summary.to_csv(out / "delta_hedge_summary.csv", index=False, float_format="%.4f")
    print(summary.T.to_string())

    decade = (outcomes["start"].dt.year // 10 * 10).astype(str) + "s"
    by_decade = outcomes.assign(decade=decade, pnl_usd=outcomes["pnl"] * NOTIONAL_GBP).groupby("decade").agg(
        options=("pnl_usd", "size"),
        mean_pnl_usd=("pnl_usd", "mean"),
        p05_pnl_usd=("pnl_usd", lambda s: s.quantile(0.05)),
        mean_pricing_vol=("pricing_vol", "mean"),
        mean_realised_vol=("realised_vol", "mean"),
    )
    by_decade.to_csv(out / "delta_hedge_by_decade.csv", float_format="%.4f")
    print(by_decade.round(4).to_string())

    worst = worst_episodes(outcomes)
    worst.to_csv(out / "delta_hedge_worst.csv", index=False, float_format="%.4f")
    print(worst.round(4).to_string(index=False))

    paths = outcomes.assign(
        start=outcomes["start"].dt.strftime("%Y-%m-%d"),
        expiry=outcomes["expiry"].dt.strftime("%Y-%m-%d"),
        premium_usd=outcomes["premium"] * NOTIONAL_GBP,
        pnl_usd=outcomes["pnl"] * NOTIONAL_GBP,
        gamma_theta_usd=outcomes["gamma_theta_pnl"] * NOTIONAL_GBP,
    )[["start", "expiry", "steps", "spot0", "strike", "spot_expiry", "pricing_vol", "realised_vol",
       "premium_usd", "pnl_usd", "gamma_theta_usd"]]
    paths.to_csv(out / "delta_hedge_paths.csv", index=False, float_format="%.5f")

    plot_hedge(outcomes, out / "delta_hedge.png", notional=NOTIONAL_GBP)


if __name__ == "__main__":
    main()
