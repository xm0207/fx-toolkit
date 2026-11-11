"""A volatility smile for one expiry, built from ATM, risk-reversal and butterfly quotes.

The FX options market quotes volatility by delta, not by strike. For one expiry the standard
quotes are:
- ATM: the volatility of the at-the-money straddle;
- RR25, the 25-delta risk reversal: vol(25-delta call) - vol(25-delta put), the skew;
- BF25, the 25-delta butterfly: how much the 25-delta wings are above ATM, the smile's curvature.

Conventions used here (stated because the market has several):
- ATM is the delta-neutral straddle (DNS) strike;
- deltas are spot deltas without premium adjustment by default, the market convention for
  GBPUSD up to one year, because the premium is paid in USD (the domestic currency);
- the butterfly is the *simplified smile strangle*: vol(25C) = ATM + BF + RR/2 and
  vol(25P) = ATM + BF - RR/2. Brokers quote a "market strangle" butterfly, which needs an extra
  calibration step to convert; that step is not implemented.

The three (strike, vol) pillars are then interpolated, and extrapolated, with SABR
(Hagan, Kumar, Lesniewski and Woodward, 2002), which has exactly three free parameters once beta
is fixed, so it passes through all three pillars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .options import DeltaKind, GarmanKohlhagen


def sabr_vol(strike, forward: float, tau: float, alpha: float, beta: float, rho: float, nu: float):
    """Hagan et al. (2002) lognormal (Black) implied volatility of the SABR model."""
    k = np.asarray(strike, dtype=float)
    one_b = 1.0 - beta
    fk = forward * k
    log_fk = np.log(forward / k)
    fk_pow = fk ** (one_b / 2)
    z = (nu / alpha) * fk_pow * log_fk
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        z_over_x = np.where(np.abs(z) < 1e-10, 1.0, z / x)
    denom = fk_pow * (1 + one_b**2 / 24 * log_fk**2 + one_b**4 / 1920 * log_fk**4)
    correction = 1 + (one_b**2 / 24 * alpha**2 / fk**one_b
                      + rho * beta * nu * alpha / (4 * fk_pow)
                      + (2 - 3 * rho**2) / 24 * nu**2) * tau
    return alpha / denom * z_over_x * correction


@dataclass(frozen=True)
class SmileQuotes:
    """Volatility quotes for one expiry, as decimals (0.08 = 8%).

    These must be labelled with where they come from. In this project they are illustrative
    inputs chosen by hand, not market data: FX option quotes are not freely available.
    """

    atm: float
    rr25: float
    bf25: float
    source: str = "illustrative"

    @property
    def call25(self) -> float:
        """Vol of the 25-delta call under the simplified smile-strangle convention."""
        return self.atm + self.bf25 + 0.5 * self.rr25

    @property
    def put25(self) -> float:
        """Vol of the 25-delta put under the simplified smile-strangle convention."""
        return self.atm + self.bf25 - 0.5 * self.rr25


class SabrSmile:
    """Implied volatility as a function of strike for one pair and one expiry.

    Why a class: a smile is a few fitted numbers (the SABR parameters) plus the market it was fitted
    to. Once built it answers two questions: "what vol at this strike?" and "what strike, and vol,
    at this delta?". Build it with `SabrSmile.calibrate(pricer, quotes)`.

    Public interface: calibrate(...), vol(strike), strike_at_delta(delta, call), pillars (a table
    of the three quoted points and the fitted vols there), and the parameters alpha, beta, rho, nu.
    """

    def __init__(self, pricer: GarmanKohlhagen, alpha: float, beta: float, rho: float, nu: float,
                 delta_kind: DeltaKind = "spot", premium_adjusted: bool = False,
                 pillars: pd.DataFrame | None = None):
        self.pricer = pricer
        self.alpha, self.beta, self.rho, self.nu = alpha, beta, rho, nu
        self.delta_kind = delta_kind
        self.premium_adjusted = premium_adjusted
        self.pillars = pillars

    @classmethod
    def calibrate(cls, pricer: GarmanKohlhagen, quotes: SmileQuotes, beta: float = 1.0,
                  delta_kind: DeltaKind = "spot", premium_adjusted: bool = False) -> "SabrSmile":
        """Turn the three quotes into (strike, vol) pillars and fit SABR through them."""
        conv = {"kind": delta_kind, "premium_adjusted": premium_adjusted}
        k_put = float(pricer.strike_from_delta(-0.25, quotes.put25, call=False, **conv))
        k_atm = float(pricer.atm_dns_strike(quotes.atm, premium_adjusted))
        k_call = float(pricer.strike_from_delta(0.25, quotes.call25, call=True, **conv))
        strikes = np.array([k_put, k_atm, k_call])
        targets = np.array([quotes.put25, quotes.atm, quotes.call25])

        forward, tau = float(pricer.forward), float(pricer.tau)

        def residuals(p):
            alpha, rho, nu = p
            return sabr_vol(strikes, forward, tau, alpha, beta, rho, nu) - targets

        alpha0 = quotes.atm * forward ** (1 - beta)
        rho0 = float(np.clip(np.sign(quotes.rr25) * 0.3, -0.9, 0.9))
        fit = least_squares(residuals, x0=[alpha0, rho0, 1.0],
                            bounds=([1e-8, -0.9999, 1e-8], [np.inf, 0.9999, 20.0]),
                            xtol=1e-15, ftol=1e-15, gtol=1e-15)
        alpha, rho, nu = fit.x
        pillars = pd.DataFrame(
            {
                "pillar": ["25D put", "ATM (DNS)", "25D call"],
                "strike": strikes,
                "quoted_vol": targets,
                "sabr_vol": sabr_vol(strikes, forward, tau, alpha, beta, rho, nu),
            }
        )
        return cls(pricer, float(alpha), beta, float(rho), float(nu), delta_kind, premium_adjusted, pillars)

    def vol(self, strike):
        """SABR implied volatility at the given strike(s)."""
        return sabr_vol(strike, float(self.pricer.forward), float(self.pricer.tau),
                        self.alpha, self.beta, self.rho, self.nu)

    def strike_at_delta(self, delta: float, call: bool = True, tol: float = 1e-12, max_iter: int = 100):
        """Strike and vol of the option with this delta on the smile.

        Delta depends on the vol and the vol depends on the strike, so this iterates
        K -> strike_from_delta(delta, vol(K)) until the strike stops moving.
        """
        k = float(self.pricer.forward)
        for _ in range(max_iter):
            vol = float(self.vol(k))
            k_new = float(self.pricer.strike_from_delta(delta, vol, call, self.delta_kind, self.premium_adjusted))
            if abs(k_new - k) < tol * k:
                return k_new, float(self.vol(k_new))
            k = k_new
        raise RuntimeError(f"strike for delta {delta} did not converge")
