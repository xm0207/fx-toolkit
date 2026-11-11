"""Monthly backtests of currency strategies, with transaction costs, and their performance statistics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forwards import MonthlyFxPanel
from .strategies import Strategy


def max_drawdown(returns: pd.Series) -> float:
    """Largest fall of cumulative wealth from a previous peak, as a negative fraction (-0.25 = -25%).

    Wealth starts at 1 and compounds the returns; the starting value counts as a peak.
    """
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns, dtype=float))])
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min())


def performance_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised mean, volatility and Sharpe ratio, skewness, t-statistic and maximum drawdown.

    The returns are excess returns (a forward needs no cash), so the Sharpe ratio is simply the
    annualised mean over the annualised volatility. Skewness is of the monthly returns.
    """
    r = pd.Series(returns, dtype=float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    return {
        "months": len(r),
        "ann_return": mean * periods_per_year,
        "ann_vol": std * np.sqrt(periods_per_year),
        "sharpe": mean / std * np.sqrt(periods_per_year) if std > 0 else np.nan,
        "t_stat": mean / std * np.sqrt(len(r)) if std > 0 else np.nan,
        "skew": float(r.skew()),
        "max_drawdown": max_drawdown(r),
        "worst_month": r.min(),
        "best_month": r.max(),
    }


@dataclass
class BacktestResult:
    """Monthly returns of one strategy, before and after costs, and the weights held each month.

    `frame` is indexed by the month in which the return is earned, with columns:
    gross (weights x excess returns), turnover (sum of absolute weight changes traded at the start
    of the month), cost and net. `held` holds the weights in force during each month.
    """

    name: str
    cost_bp: float
    frame: pd.DataFrame
    held: pd.DataFrame

    @property
    def net(self) -> pd.Series:
        return self.frame["net"]

    def stats(self) -> dict:
        row = {"strategy": self.name, "cost_bp": self.cost_bp,
               "start": str(self.frame.index[0]), "end": str(self.frame.index[-1])}
        row.update(performance_stats(self.net))
        row["avg_turnover"] = float(self.frame["turnover"].mean())
        row["ann_cost"] = float(self.frame["cost"].mean() * 12)
        return row

    def worst_months(self, k: int = 5) -> pd.DataFrame:
        worst = self.net.nsmallest(k)
        return pd.DataFrame({"strategy": self.name, "month": worst.index.astype(str), "net_return": worst.to_numpy()})

    def wealth(self) -> pd.Series:
        return (1.0 + self.net).cumprod()

    def drawdown(self) -> pd.Series:
        wealth = self.wealth()
        return wealth / np.maximum(wealth.cummax(), 1.0) - 1.0

    def correlation(self, other: pd.Series) -> float:
        """Correlation of the net monthly returns with another monthly series, over common months."""
        both = pd.concat([self.net, other], axis=1, join="inner").dropna()
        return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


class StrategyBacktest:
    """Runs strategies month by month on a `MonthlyFxPanel` and charges transaction costs.

    Timing: the weights decided at the end of month t - 1 are held through month t and earn that
    month's forward excess returns. To set them, the portfolio trades the change from the weights
    held in month t - 1, and pays `cost_bp` basis points per dollar of notional traded. The first
    month pays for building the whole position. Rolling an unchanged position into a new one-month
    forward is not charged separately (see the README's Limitations).

    Public interface: `run(strategy)` and `run_all(strategies)`; the latter starts every strategy
    in the same month so that their statistics cover the same period.
    """

    def __init__(self, panel: MonthlyFxPanel, cost_bp: float = 5.0):
        if cost_bp < 0:
            raise ValueError("cost_bp cannot be negative")
        self.panel = panel
        self.cost_bp = cost_bp

    def run(self, strategy: Strategy, start: pd.Period | None = None) -> BacktestResult:
        decided = strategy.weights(self.panel)
        held = decided.shift(1)  # decided at the end of t - 1, held during t
        active = held.notna().any(axis=1)
        if not active.any():
            raise ValueError(f"{strategy.name} never takes a position")
        first = held.index[active.to_numpy()][0]
        last = held.index[active.to_numpy()][-1]
        if start is not None:
            first = max(first, pd.Period(start, freq="M"))
        held = held.loc[first:last].fillna(0.0)

        returns = self.panel.excess_returns.loc[first:last, held.columns]
        if (returns.isna() & (held != 0)).any().any():
            raise ValueError(f"{strategy.name} holds a currency whose return is missing")
        gross = (held * returns.fillna(0.0)).sum(axis=1)
        previous = held.shift(1).fillna(0.0)
        turnover = (held - previous).abs().sum(axis=1)
        cost = turnover * self.cost_bp / 1e4
        frame = pd.DataFrame({"gross": gross, "turnover": turnover, "cost": cost, "net": gross - cost})
        return BacktestResult(strategy.name, self.cost_bp, frame, held)

    def run_all(self, strategies: Sequence[Strategy]) -> list[BacktestResult]:
        """Run every strategy from the first month in which all of them hold a position."""
        firsts = [self.run(s).frame.index[0] for s in strategies]
        start = max(firsts)
        return [self.run(s, start=start) for s in strategies]
