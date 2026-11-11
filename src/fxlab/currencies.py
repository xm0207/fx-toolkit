"""The G10 currencies, their FRED series and their market conventions.

Everything inside the package works in one direction: US dollars per unit of foreign currency.
A rise in that number means the foreign currency has strengthened against the dollar. The
market quotes some pairs the other way round (USDJPY is yen per dollar), so `Currency` knows how
to convert between its FRED quote, the internal direction and the market quote.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Currency:
    """One foreign currency traded against the US dollar.

    Why a class: each currency carries several facts that must travel together (where its data
    lives, which way FRED quotes it, its money-market day count, how the market quotes the pair).
    Keeping them in one small immutable object stops these facts drifting apart across modules.

    Attributes:
        code: ISO code, e.g. "GBP".
        spot_series: FRED H.10 daily series, e.g. "DEXUSUK".
        fred_usd_per_unit: True if FRED quotes US dollars per unit of this currency (DEXUSUK),
            False if it quotes units of this currency per dollar (DEXJPUS).
        rate_series: OECD 3-month interbank rate on FRED, monthly average, in percent.
        day_count_basis: money-market day count, 360 or 365 days a year.
        market_usd_per_unit: True if the market quotes the pair as USD per unit (GBPUSD),
            False if it quotes units per USD (USDJPY).
    """

    code: str
    spot_series: str
    fred_usd_per_unit: bool
    rate_series: str
    day_count_basis: int
    market_usd_per_unit: bool

    @property
    def pair(self) -> str:
        """Market name of the pair, base currency first: "GBPUSD", "USDJPY"."""
        return f"{self.code}USD" if self.market_usd_per_unit else f"USD{self.code}"

    @property
    def pip(self) -> float:
        """Size of one pip in the market quote: 0.01 for yen pairs, 0.0001 otherwise."""
        return 0.01 if self.code == "JPY" else 0.0001

    def fred_to_usd_per_unit(self, values):
        """Convert FRED quotes to US dollars per unit of this currency."""
        return values if self.fred_usd_per_unit else 1.0 / values

    def usd_per_unit_to_market(self, values):
        """Convert US dollars per unit to the market quote of the pair."""
        return values if self.market_usd_per_unit else 1.0 / values


USD_RATE_SERIES = "IR3TIB01USM156N"
USD_DAY_COUNT_BASIS = 360

# Money-market day counts: ACT/365 for sterling and the Commonwealth dollars, ACT/360 otherwise.
G10: tuple[Currency, ...] = (
    Currency("EUR", "DEXUSEU", True, "IR3TIB01EZM156N", 360, True),
    Currency("JPY", "DEXJPUS", False, "IR3TIB01JPM156N", 360, False),
    Currency("GBP", "DEXUSUK", True, "IR3TIB01GBM156N", 365, True),
    Currency("CHF", "DEXSZUS", False, "IR3TIB01CHM156N", 360, False),
    Currency("CAD", "DEXCAUS", False, "IR3TIB01CAM156N", 365, False),
    Currency("AUD", "DEXUSAL", True, "IR3TIB01AUM156N", 365, True),
    Currency("NZD", "DEXUSNZ", True, "IR3TIB01NZM156N", 365, True),
    Currency("NOK", "DEXNOUS", False, "IR3TIB01NOM156N", 360, False),
    Currency("SEK", "DEXSDUS", False, "IR3TIB01SEM156N", 360, False),
)


def by_code(code: str, currencies: tuple[Currency, ...] = G10) -> Currency:
    """Look up a currency by its ISO code."""
    for currency in currencies:
        if currency.code == code:
            return currency
    raise KeyError(f"Unknown currency {code!r}")


def to_continuous(rate, basis: int, tau: float):
    """Convert a simple money-market rate (decimal, ACT/basis) to a continuously compounded rate.

    The Garman-Kohlhagen formula wants continuously compounded rates on an ACT/365 clock. A deposit
    of `tau` years (ACT/365) at simple rate `rate` grows by 1 + rate * tau * 365 / basis.
    """
    growth = 1.0 + np.asarray(rate, dtype=float) * tau * 365.0 / basis
    out = np.log(growth) / tau
    return float(out) if np.ndim(out) == 0 else out


def usd_per_unit_frame(fred_quotes: pd.DataFrame, currencies: tuple[Currency, ...] = G10) -> pd.DataFrame:
    """Turn a table of FRED quotes (one column per spot series) into USD per unit, columns by code."""
    out = {c.code: c.fred_to_usd_per_unit(fred_quotes[c.spot_series]) for c in currencies}
    return pd.DataFrame(out, index=fred_quotes.index)
