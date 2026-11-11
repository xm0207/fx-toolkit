"""Delta-hedging a short FX option on historical spot paths.

The experiment: on a given day, sell a one-month at-the-money-forward call on the foreign
currency, priced with Garman-Kohlhagen at a chosen volatility, and hedge it by holding the
option's spot delta in foreign currency, rebalanced once a day at the close. At expiry, pay the
option's payoff and close the hedge. What is left is the hedger's P&L.

If spot followed a geometric Brownian motion with exactly the pricing volatility, and hedging were
continuous, the P&L would be zero. Over each small step the hedged short option instead earns

    0.5 * gamma * S^2 * (vol^2 * dt - (dS / S)^2),

the theta collected minus the gamma loss on the day's move. Summed over the option's life, the
P&L is therefore roughly proportional to implied variance minus realised variance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .options import GarmanKohlhagen

DAYS_PER_YEAR = 365.0  # calendar-time clock (ACT/365) for both volatility and interest


def daily_rates_from_monthly(monthly: pd.Series, dates: pd.DatetimeIndex, lag_months: int = 1) -> pd.Series:
    """Spread monthly-average rates over daily dates, using the average of an earlier month.

    With lag_months=1 the rate used on any day of March is February's average, which is fully
    known by the end of February, so no future information leaks into the simulation.
    """
    lagged = monthly.copy()
    lagged.index = lagged.index + lag_months
    return pd.Series(lagged.reindex(dates.to_period("M")).to_numpy(), index=dates, name=monthly.name)


def realised_vol(spot: np.ndarray, dates: pd.DatetimeIndex | np.ndarray) -> float:
    """Annualised zero-mean realised volatility of log returns on a calendar-time clock.

    Squared daily log returns are summed and divided by the calendar time they span, so a weekend
    return counts for three days, as the option's time decay does.
    """
    r = np.diff(np.log(np.asarray(spot, dtype=float)))
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / DAYS_PER_YEAR
    return float(np.sqrt(np.sum(r**2) / years))


@dataclass(frozen=True)
class HedgeOutcome:
    """Result of one simulated option, per unit of foreign notional, in domestic currency at expiry."""

    start: pd.Timestamp
    expiry: pd.Timestamp
    steps: int
    spot0: float
    strike: float
    spot_expiry: float
    pricing_vol: float
    realised_vol: float
    premium: float
    pnl: float
    gamma_theta_pnl: float

    @property
    def variance_gap(self) -> float:
        """Pricing (implied) variance minus realised variance over the option's life."""
        return self.pricing_vol**2 - self.realised_vol**2


class DeltaHedgeSimulator:
    """Sells a European call and delta-hedges it daily along a historical spot path.

    Why a class: the spot history, the rate histories and the trade design (tenor, how the pricing
    vol is chosen) are fixed for a whole study, while the start date varies. The object holds the
    fixed parts; `simulate(i)` runs one start date and `run()` runs them all.

    Inputs are aligned daily Series: spot in domestic per foreign (USD per GBP), and continuously
    compounded domestic and foreign rates (decimals). The pricing vol is the trailing realised
    volatility over the last `vol_window` daily returns before the trade, a stand-in for the implied
    volatility that the market would quote. If `pricing_vol` is given, it is used instead.
    """

    def __init__(self, spot: pd.Series, r_dom: pd.Series, r_for: pd.Series, tenor_months: int = 1,
                 vol_window: int = 21, pricing_vol: float | None = None):
        frame = pd.concat({"spot": spot, "r_dom": r_dom, "r_for": r_for}, axis=1).dropna()
        if frame.empty:
            raise ValueError("spot and rates have no dates in common")
        self.dates = pd.DatetimeIndex(frame.index)
        self.spot = frame["spot"].to_numpy(dtype=float)
        self.r_dom = frame["r_dom"].to_numpy(dtype=float)
        self.r_for = frame["r_for"].to_numpy(dtype=float)
        self.tenor_months = tenor_months
        self.vol_window = vol_window
        self.fixed_vol = pricing_vol

    def expiry_index(self, i: int) -> int | None:
        """Index of the expiry: the first date on or after start + tenor; None if beyond the data."""
        target = self.dates[i] + pd.DateOffset(months=self.tenor_months)
        j = int(self.dates.searchsorted(target))
        return j if j < len(self.dates) else None

    def pricing_vol(self, i: int) -> float | None:
        if self.fixed_vol is not None:
            return self.fixed_vol
        if i < self.vol_window:
            return None
        lo = i - self.vol_window
        return realised_vol(self.spot[lo:i + 1], self.dates[[lo, i]])

    def simulate(self, i: int) -> HedgeOutcome | None:
        """Sell one call on the start date with index i and hedge it to expiry."""
        j = self.expiry_index(i)
        vol = self.pricing_vol(i)
        if j is None or vol is None or vol <= 0:
            return None
        dates = self.dates[i:j + 1]
        s = self.spot[i:j + 1]
        rd, rf = self.r_dom[i:j + 1], self.r_for[i:j + 1]
        tau = (dates[-1] - dates).days.to_numpy() / DAYS_PER_YEAR
        dt = np.diff(tau) * -1.0

        # Strike at the start-date forward; each day the option is valued with that day's rates.
        live = GarmanKohlhagen(s[:-1], rd[:-1], rf[:-1], tau[:-1])
        strike = float(live.forward[0])
        deltas = live.delta(strike, vol, call=True)
        gammas = live.gamma(strike, vol)
        premium = float(live.price(strike, vol, call=True)[0])

        # Self-financing hedge: the cash account earns r_dom, foreign currency held earns r_for.
        cash = premium - deltas[0] * s[0]
        units = deltas[0]
        for k in range(1, len(s)):
            cash *= np.exp(rd[k - 1] * dt[k - 1])
            units *= np.exp(rf[k - 1] * dt[k - 1])
            if k < len(s) - 1:
                cash -= (deltas[k] - units) * s[k]
                units = deltas[k]
        payoff = max(s[-1] - strike, 0.0)
        pnl = cash + units * s[-1] - payoff

        moves = s[1:] / s[:-1] - 1.0
        gamma_theta = float(np.sum(0.5 * gammas * s[:-1] ** 2 * (vol**2 * dt - moves**2)))
        return HedgeOutcome(
            start=dates[0], expiry=dates[-1], steps=len(s) - 1, spot0=float(s[0]), strike=strike,
            spot_expiry=float(s[-1]), pricing_vol=vol, realised_vol=realised_vol(s, dates[[0, -1]]),
            premium=premium, pnl=float(pnl), gamma_theta_pnl=gamma_theta,
        )

    def run(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Simulate every start date in [start, end] whose option expires inside the data."""
        lo = 0 if start is None else int(self.dates.searchsorted(pd.Timestamp(start)))
        hi = len(self.dates) if end is None else int(self.dates.searchsorted(pd.Timestamp(end), side="right"))
        rows = []
        for i in range(lo, hi):
            outcome = self.simulate(i)
            if outcome is not None:
                row = asdict(outcome)
                row["variance_gap"] = outcome.variance_gap
                rows.append(row)
        return pd.DataFrame(rows)
