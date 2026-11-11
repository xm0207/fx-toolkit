"""Garman-Kohlhagen pricing of European FX options, Greeks, and delta-strike conversion.

Conventions. For a pair FORDOM (e.g. GBPUSD: foreign GBP, domestic USD):
- spot S and strike K are in domestic currency per unit of foreign currency (USD per GBP);
- a call is the right to buy one unit of foreign currency at K; prices are in domestic currency
  per unit of foreign notional (USD per GBP), which the market calls "domestic pips";
- r_dom and r_for are continuously compounded, tau is in years (ACT/365).

Garman and Kohlhagen (1983) is Black-Scholes with the foreign interest rate as a continuous
dividend yield: holding foreign currency earns r_for.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

DeltaKind = Literal["spot", "forward"]


class GarmanKohlhagen:
    """Prices European options on one currency pair for one expiry.

    Why a class: the market inputs (spot, two interest rates, time to expiry) are shared by every
    option on the same pair and expiry, while strike and volatility change from option to option.
    The object stores the market once; each method takes only (strike, vol, call).

    Public interface:
        forward                          F = S exp((r_dom - r_for) tau)
        price(K, vol, call)              premium in domestic currency per unit of foreign notional
        delta(K, vol, call, kind, premium_adjusted)
        gamma(K, vol), vega(K, vol), theta(K, vol, call)
        strike_from_delta(delta, vol, call, kind, premium_adjusted)
        atm_dns_strike(vol, premium_adjusted)

    All inputs may be NumPy arrays; the formulas broadcast element by element, which the delta-hedge
    simulation uses to price a whole path at once. `strike_from_delta` with premium adjustment is
    the exception: it solves numerically and takes scalars.
    """

    def __init__(self, spot, r_dom, r_for, tau):
        self.spot = np.asarray(spot, dtype=float)
        self.r_dom = np.asarray(r_dom, dtype=float)
        self.r_for = np.asarray(r_for, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        if np.any(self.spot <= 0):
            raise ValueError("spot must be positive")
        if np.any(self.tau <= 0):
            raise ValueError("time to expiry must be positive")

    # ----- market quantities -----------------------------------------------------------------
    @property
    def df_dom(self):
        """Domestic discount factor, exp(-r_dom tau)."""
        return np.exp(-self.r_dom * self.tau)

    @property
    def df_for(self):
        """Foreign discount factor, exp(-r_for tau)."""
        return np.exp(-self.r_for * self.tau)

    @property
    def forward(self):
        """Outright forward from covered interest parity, S exp((r_dom - r_for) tau)."""
        return self.spot * self.df_for / self.df_dom

    def _d1_d2(self, strike, vol):
        strike = np.asarray(strike, dtype=float)
        vol = np.asarray(vol, dtype=float)
        if np.any(strike <= 0) or np.any(vol <= 0):
            raise ValueError("strike and vol must be positive")
        sd = vol * np.sqrt(self.tau)
        d1 = (np.log(self.forward / strike) + 0.5 * sd**2) / sd
        return d1, d1 - sd

    @staticmethod
    def _omega(call: bool) -> float:
        return 1.0 if call else -1.0

    # ----- price and Greeks ------------------------------------------------------------------
    def price(self, strike, vol, call: bool = True):
        """Premium in domestic currency per unit of foreign notional (USD per GBP for GBPUSD)."""
        w = self._omega(call)
        d1, d2 = self._d1_d2(strike, vol)
        return w * self.df_dom * (self.forward * norm.cdf(w * d1) - strike * norm.cdf(w * d2))

    def delta(self, strike, vol, call: bool = True, kind: DeltaKind = "spot", premium_adjusted: bool = False):
        """Delta in one of the four market conventions.

        kind="spot": units of foreign currency to hold now to hedge (dV/dS).
        kind="forward": notional of the forward contract that hedges, dV/dF in forward terms.
        premium_adjusted=True: the delta quoted when the premium is paid in foreign currency
        (e.g. USDJPY, premium in USD). The premium itself then carries spot risk, so it is
        subtracted: spot premium-adjusted delta = spot delta - price / spot.
        """
        w = self._omega(call)
        d1, d2 = self._d1_d2(strike, vol)
        if premium_adjusted:
            raw = w * (np.asarray(strike) / self.forward) * norm.cdf(w * d2)
        else:
            raw = w * norm.cdf(w * d1)
        if kind == "spot":
            return self.df_for * raw
        if kind == "forward":
            return raw * np.ones_like(self.df_for)
        raise ValueError("kind must be 'spot' or 'forward'")

    def gamma(self, strike, vol):
        """d2V/dS2: change in spot delta per unit change in spot. The same for calls and puts."""
        d1, _ = self._d1_d2(strike, vol)
        return self.df_for * norm.pdf(d1) / (self.spot * np.asarray(vol) * np.sqrt(self.tau))

    def vega(self, strike, vol):
        """dV/dvol, per unit of volatility (divide by 100 for one vol point). Same for calls and puts."""
        d1, _ = self._d1_d2(strike, vol)
        return self.spot * self.df_for * norm.pdf(d1) * np.sqrt(self.tau)

    def theta(self, strike, vol, call: bool = True):
        """dV/dt as calendar time passes, per year (divide by 365 for one day), all else fixed."""
        w = self._omega(call)
        d1, d2 = self._d1_d2(strike, vol)
        decay = -self.spot * self.df_for * norm.pdf(d1) * np.asarray(vol) / (2 * np.sqrt(self.tau))
        carry = w * (self.r_for * self.spot * self.df_for * norm.cdf(w * d1)
                     - self.r_dom * np.asarray(strike) * self.df_dom * norm.cdf(w * d2))
        return decay + carry

    # ----- delta <-> strike --------------------------------------------------------------------
    def strike_from_delta(self, delta, vol, call: bool = True, kind: DeltaKind = "spot",
                          premium_adjusted: bool = False):
        """The strike whose delta, at volatility `vol`, equals `delta` (calls: delta > 0, puts: < 0).

        Without premium adjustment the delta formula inverts in closed form. With it, the call
        delta is not monotonic in the strike (very low strikes have a large premium to subtract),
        so the equation has two roots; the market convention, used here, takes the higher strike.
        """
        if kind not in ("spot", "forward"):
            raise ValueError("kind must be 'spot' or 'forward'")
        w = self._omega(call)
        scale = self.df_for if kind == "spot" else 1.0
        x =np.asarray(delta, dtype=float) / scale  # forward-delta equivalent
        if np.any(w * x <= 0) or np.any(w * x >= 1):
            raise ValueError("delta must be strictly between 0 and the discount factor, with the option's sign")
        sd = np.asarray(vol, dtype=float) * np.sqrt(self.tau)
        if not premium_adjusted:
            d1 = w * norm.ppf(w * x)
            return self.forward * np.exp(-d1 * sd + 0.5 * sd**2)
        return self._strike_from_pa_delta(float(x), float(sd), w)

    def _strike_from_pa_delta(self, x: float, sd: float, w: float) -> float:
        forward = float(self.forward)

        def pa_delta(k: float) -> float:  # premium-adjusted forward delta as a function of strike
            d2 = (np.log(forward / k) - 0.5 * sd**2) / sd
            return w * (k / forward) * norm.cdf(w * d2)

        d1_plain = w * norm.ppf(w * x)
        k_plain = forward * np.exp(-d1_plain * sd + 0.5 * sd**2)  # strike without the adjustment
        if w > 0:
            # (K/F) N(d2) peaks where sd * N(d2) = pdf(d2); the conventional root lies above the peak
            d2_peak = brentq(lambda d: sd * norm.cdf(d) - norm.pdf(d), -sd, 10.0)
            k_peak = forward * np.exp(-d2_peak * sd - 0.5 * sd**2)
            if pa_delta(k_peak) < x:
                raise ValueError(f"premium-adjusted call delta cannot reach {x:.4f} at this vol")
            return float(brentq(lambda k: pa_delta(k) - x, k_peak, k_plain))
        low = k_plain * np.exp(-10 * sd)
        return float(brentq(lambda k: pa_delta(k) - x, low, k_plain))

    def atm_dns_strike(self, vol, premium_adjusted: bool = False):
        """Delta-neutral straddle strike: the call and put deltas cancel, the usual ATM for FX.

        Without premium adjustment K = F exp(vol^2 tau / 2); with it K = F exp(-vol^2 tau / 2).
        """
        half_var = 0.5 * np.asarray(vol, dtype=float) ** 2 * self.tau
        return self.forward * np.exp(-half_var if premium_adjusted else half_var)
