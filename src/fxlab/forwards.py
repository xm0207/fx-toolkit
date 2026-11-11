"""Covered-interest-parity forwards and the monthly excess returns of currency forwards.

Notation: S is spot and F the forward, both in units of domestic currency per unit of foreign
currency. Inside the package the domestic currency is always the US dollar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .currencies import G10, USD_DAY_COUNT_BASIS, Currency


def cip_forward(spot, r_dom, r_for, days, basis_dom: int = 360, basis_for: int = 360):
    """Forward price from covered interest parity with simple money-market rates.

    Borrowing S units of domestic currency for `days` days, converting to one unit of foreign
    currency and depositing it must cost the same as agreeing today to buy that foreign currency
    forward, so F = S * (1 + r_dom * days / basis_dom) / (1 + r_for * days / basis_for).
    Rates are decimals (0.05 for 5%). Works element-wise on arrays and pandas objects.
    """
    return spot * (1 + r_dom * days / basis_dom) / (1 + r_for * days / basis_for)


def forward_points(spot, forward, pip: float = 1e-4):
    """Forward minus spot, in pips of the quote (1e-4 for most pairs, 1e-2 for yen pairs)."""
    return (forward - spot) / pip


@dataclass(frozen=True)
class MonthlyFxPanel:
    """Month-end spot rates, one-month CIP forwards and forward excess returns for many currencies.

    Why a class: the strategies, the backtest and the forward-points table all need the same
    aligned monthly tables. Building them once, in one place, keeps the timing conventions
    (what is known at which month-end) consistent everywhere.

    Timing. Row t holds what is known at the end of calendar month t: the month-end spot, the
    month-t average of each 3-month rate, and the one-month forward agreed at that month-end for
    delivery at the end of month t + 1. `excess_returns` row t is the return realised over month
    t on the forward agreed at the end of month t - 1.

    All tables share a complete monthly PeriodIndex, so shifting by one row is shifting by one
    calendar month.
    """

    spot: pd.DataFrame  # USD per unit of foreign currency
    forward: pd.DataFrame  # one-month forward agreed at the month-end, same units
    rates: pd.DataFrame  # decimals; one column per currency plus "USD"
    month_end: pd.Series  # date of the last observation in each month
    days: pd.Series  # calendar days from this month-end to the next
    currencies: tuple[Currency, ...] = G10

    @classmethod
    def build(cls, spot_daily: pd.DataFrame, rates: pd.DataFrame, currencies: tuple[Currency, ...] = G10):
        """Build the panel from daily spot (USD per unit) and monthly rates (decimals, PeriodIndex)."""
        codes = [c.code for c in currencies]
        spot_daily = spot_daily[codes].dropna(how="all")
        months = spot_daily.index.to_period("M")
        full = pd.period_range(months.min(), months.max(), freq="M")
        spot = spot_daily.groupby(months).last().reindex(full)  # last valid value in each month
        month_end = pd.Series(spot_daily.index, index=months).groupby(level=0).max().reindex(full)
        next_end = month_end.shift(-1)
        fallback = month_end + pd.DateOffset(months=1)  # for the final month, whose successor is unknown
        days = (next_end.fillna(fallback) - month_end).dt.days.astype(float)

        rates = rates.reindex(full)[["USD", *codes]]
        forward = pd.DataFrame(index=full, columns=codes, dtype=float)
        for c in currencies:
            forward[c.code] = cip_forward(
                spot[c.code], rates["USD"], rates[c.code], days, USD_DAY_COUNT_BASIS, c.day_count_basis
            )
        return cls(spot, forward, rates, month_end, days, tuple(currencies))

    @property
    def codes(self) -> list[str]:
        return [c.code for c in self.currencies]

    @property
    def carry(self) -> pd.DataFrame:
        """Annualised forward discount, (S / F - 1) * 365 / days.

        This is what a long position in the foreign currency forward earns per year if spot does not
        move, i.e. roughly the foreign minus the US interest rate.
        """
        return (self.spot / self.forward - 1).mul(365.0 / self.days, axis=0)

    @property
    def excess_returns(self) -> pd.DataFrame:
        """Return in month t of buying one dollar's worth of foreign currency forward at the end of t - 1.

        The position pays S_t / F_{t-1} - 1 per dollar of notional. No money changes hands when the
        forward is agreed, so this is an excess return over the US dollar interest rate.
        """
        return self.spot / self.forward.shift(1) - 1

    def complete_months(self) -> pd.PeriodIndex:
        """Months in which every currency has a spot rate, a 3-month rate and a forward."""
        ok = self.forward.notna().all(axis=1) & self.rates.notna().all(axis=1)
        return self.forward.index[ok.to_numpy()]

    def balanced(self) -> "MonthlyFxPanel":
        """Keep only the span in which the whole cross-section is available.

        Forwards and rates outside the span are set to missing; spot is kept, because the return on
        the last forward is realised in the month after the span ends. Raises if the span has gaps.
        """
        complete = self.complete_months()
        if len(complete) == 0:
            raise ValueError("No month has data for every currency")
        span = pd.period_range(complete[0], complete[-1], freq="M")
        missing = span.difference(complete)
        if len(missing):
            raise ValueError(f"Cross-section incomplete inside the span, e.g. {missing[0]}")
        outside = ~self.forward.index.isin(span)
        forward, rates = self.forward.copy(), self.rates.copy()
        forward.loc[outside] = np.nan
        rates.loc[outside] = np.nan
        return replace(self, forward=forward, rates=rates)

    def forward_points_table(self, month: pd.Period | str) -> pd.DataFrame:
        """Spot, one-month CIP forward and forward points at one month-end, in market quotes."""
        month = pd.Period(month, freq="M")
        rows = []
        for c in self.currencies:
            s = c.usd_per_unit_to_market(self.spot.at[month, c.code])
            f = c.usd_per_unit_to_market(self.forward.at[month, c.code])
            rows.append(
                {
                    "pair": c.pair,
                    "date": self.month_end[month].date(),
                    "days": int(self.days[month]),
                    "rate_usd_pct": 100 * self.rates.at[month, "USD"],
                    "rate_foreign_pct": 100 * self.rates.at[month, c.code],
                    "spot": s,
                    "forward_1m": f,
                    "forward_points_pips": forward_points(s, f, c.pip),
                    "carry_pct_pa": 100 * self.carry.at[month, c.code],
                }
            )
        return pd.DataFrame(rows)
