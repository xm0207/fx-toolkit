"""Backtest carry, momentum and the dollar factor on nine G10 currencies against the US dollar.

Writes to results/: strategy_summary.csv, cost_sensitivity.csv, worst_months.csv,
carry_positions.csv, carry_weights.csv (weights held in each month), strategy_returns.csv,
forward_points.csv, rate_gaps_filled.csv and strategies.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fxlab import Carry, DollarFactor, FxDataStore, Momentum, MonthlyFxPanel, StrategyBacktest
from fxlab.report import plot_strategies


def drawdown_window(result) -> tuple[str, str]:
    """Month of the peak before the largest drawdown, and month of its trough."""
    wealth = result.wealth()
    drawdown = result.drawdown()
    trough = drawdown.idxmin()
    before = wealth.loc[:trough]
    peak = before.idxmax() if before.max() >= 1.0 else "start"
    return str(peak), str(trough)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/fred")
    parser.add_argument("--out", default="results")
    parser.add_argument("--cost-bp", type=float, default=5.0, help="cost per dollar of notional traded, in bp")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    store = FxDataStore(args.data)
    store.filled_rate_months().to_csv(out / "rate_gaps_filled.csv", index=False)
    panel = MonthlyFxPanel.build(store.spot(), store.monthly_rates()).balanced()
    months = panel.complete_months()
    print(f"Balanced panel: forwards agreed at month-ends {months[0]} to {months[-1]} ({len(months)} months)")

    strategies = [Carry(n=3), Momentum(n=3, lookback=3), DollarFactor()]
    results = StrategyBacktest(panel, cost_bp=args.cost_bp).run_all(strategies)

    nasdaq = store.monthly_equity("NASDAQCOM").pct_change()
    rows = []
    for result in results:
        row = result.stats()
        row["corr_nasdaq"] = result.correlation(nasdaq)
        row["drawdown_peak"], row["drawdown_trough"] = drawdown_window(result)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "strategy_summary.csv", index=False, float_format="%.4f")
    print(summary.round(3).to_string(index=False))

    sensitivity = []
    start = results[0].frame.index[0]
    for cost in (0.0, 2.0, 5.0, 10.0, 20.0):
        for strategy in strategies:
            r = StrategyBacktest(panel, cost_bp=cost).run(strategy, start=start).stats()
            sensitivity.append({"strategy": r["strategy"], "cost_bp": cost, "ann_return": r["ann_return"],
                                "sharpe": r["sharpe"], "ann_cost": r["ann_cost"]})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(out / "cost_sensitivity.csv", index=False, float_format="%.4f")
    print(sensitivity.round(4).to_string(index=False))

    worst = pd.concat([r.worst_months(5) for r in results], ignore_index=True)
    worst.to_csv(out / "worst_months.csv", index=False, float_format="%.4f")
    print(worst.round(4).to_string(index=False))

    carry = results[0]
    positions = pd.DataFrame({
        "share_long": (carry.held > 0).mean(),
        "share_short": (carry.held < 0).mean(),
        "avg_carry_pct_pa": 100 * panel.carry.loc[carry.held.index[0] - 1:carry.held.index[-1] - 1].mean(),
    }).rename_axis("currency")
    positions.to_csv(out / "carry_positions.csv", float_format="%.4f")
    print(positions.round(3).to_string())
    held = carry.held.copy()
    held.index = held.index.astype(str)
    held.to_csv(out / "carry_weights.csv", index_label="month", float_format="%.4f")

    returns = pd.DataFrame({r.name: r.net for r in results})
    returns.index = returns.index.astype(str)
    returns.to_csv(out / "strategy_returns.csv", index_label="month", float_format="%.6f")

    points = panel.forward_points_table(months[-1])
    points.to_csv(out / "forward_points.csv", index=False, float_format="%.4f")
    print(points.round(4).to_string(index=False))

    plot_strategies(results, out / "strategies.png")


if __name__ == "__main__":
    main()
