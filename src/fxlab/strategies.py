"""Currency strategies: carry, momentum and the dollar factor.

A strategy only decides weights. At each month-end it looks at what is known then and returns
one weight per currency: the dollar notional of the one-month forward bought (positive) or sold
(negative) per dollar of capital. Applying the weights to next month's returns, and charging
transaction costs, is the backtest's job (see backtest.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .forwards import MonthlyFxPanel


def long_short_weights(signal: pd.Series, n: int) -> pd.Series:
    """Weights +1/n on the n highest signals and -1/n on the n lowest; 0 for everything else.

    Currencies with a missing signal get weight 0. If fewer than 2n signals are available the
    result is all missing, meaning "no position this month". Ties are broken by column order, so
    the result never depends on chance.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    weights = pd.Series(0.0, index=signal.index)
    valid = signal.dropna()
    if len(valid) < 2 * n:
        return weights * np.nan
    order = valid.sort_values(kind="mergesort").index  # stable sort: ties keep column order
    weights[order[-n:]] = 1.0 / n
    weights[order[:n]] = -1.0 / n
    return weights


class Strategy(ABC):
    """Base class. A strategy turns a monthly panel into a table of weights.

    Public interface: `name` and `weights(panel)`. The weights table has one row per month-end
    (formation month) and one column per currency. A row of missing values means no position.
    """

    name = "base"

    @abstractmethod
    def weights(self, panel: MonthlyFxPanel) -> pd.DataFrame:
        """Weights decided at each month-end, using only information known at that month-end."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SortedLongShort(Strategy):
    """Ranks currencies on a signal each month; long the top n, short the bottom n, equal weights.

    The portfolio is dollar-neutral: it is long n/n = 1 dollar of foreign currency and short 1
    dollar, so the US dollar's own moves largely cancel. A subclass only defines `signal()`.
    """

    def __init__(self, n: int = 3):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n

    @abstractmethod
    def signal(self, panel: MonthlyFxPanel) -> pd.DataFrame:
        """Higher signal = more attractive to hold. Same shape as the panel's spot table."""

    def weights(self, panel: MonthlyFxPanel) -> pd.DataFrame:
        # a currency can only be traded in a month when its forward can be priced
        signal = self.signal(panel).where(panel.forward.notna())
        rows = [long_short_weights(signal.loc[month], self.n) for month in signal.index]
        return pd.DataFrame(rows, index=signal.index, columns=signal.columns)


class Carry(SortedLongShort):
    """Long the highest-yielding currencies, short the lowest, ranked by the forward discount.

    The forward discount (S / F - 1, annualised) is the interest differential implied by the
    forward, so this is the textbook carry trade, built as a cross-sectional sort.
    """

    name = "Carry"

    def signal(self, panel):
        return panel.carry

    def __repr__(self) -> str:
        return f"Carry(n={self.n})"


class Momentum(SortedLongShort):
    """Long recent winners, short recent losers, ranked by the past `lookback` months' excess return.

    The signal at month-end t compounds the excess returns of months t - lookback + 1 to t, all of
    which are known at the end of month t.
    """

    name = "Momentum"

    def __init__(self, n: int = 3, lookback: int = 3):
        super().__init__(n)
        if lookback < 1:
            raise ValueError("lookback must be at least 1")
        self.lookback = lookback
        self.name = f"Momentum ({lookback}m)"

    def signal(self, panel):
        growth = np.log1p(panel.excess_returns)
        return np.expm1(growth.rolling(self.lookback, min_periods=self.lookback).sum())

    def __repr__(self) -> str:
        return f"Momentum(n={self.n}, lookback={self.lookback})"


class DollarFactor(Strategy):
    """Equal-weighted long position in every available foreign currency against the US dollar.

    This is the "dollar factor" of Lustig, Roussanov and Verdelhan (2011): the average excess return
    of all currencies against the dollar. Its weights sum to 1, so it is short one dollar of USD.
    """

    name = "Dollar factor"

    def weights(self, panel):
        available = panel.forward.notna()
        count = available.sum(axis=1).replace(0, np.nan)
        return available.astype(float).div(count, axis=0)
