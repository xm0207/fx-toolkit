"""Small synthetic fixtures, so that every test runs offline."""

import numpy as np
import pandas as pd
import pytest

from fxlab import Currency, MonthlyFxPanel

# Four made-up currencies, quoted as USD per unit, all on ACT/360, so the arithmetic is easy to follow.
TOY = tuple(Currency(code, f"SPOT_{code}", True, f"RATE_{code}", 360, True) for code in ("AAA", "BBB", "CCC", "DDD"))


def make_panel(spot_by_month: dict[str, list[float]], rates_by_month: dict[str, list[float]], start: str = "2020-01"):
    """Build a MonthlyFxPanel with one spot observation on the last business day of each month.

    rates_by_month must include "USD"; all values are decimals.
    """
    n = len(next(iter(spot_by_month.values())))
    months = pd.period_range(start, periods=n, freq="M")
    dates = pd.DatetimeIndex([pd.offsets.BMonthEnd().rollforward(m.to_timestamp()) for m in months])
    spot = pd.DataFrame(spot_by_month, index=dates)
    rates = pd.DataFrame(rates_by_month, index=months)
    currencies = tuple(c for c in TOY if c.code in spot_by_month)
    return MonthlyFxPanel.build(spot, rates, currencies)


@pytest.fixture
def toy_panel():
    """Four currencies, six months, constant rates 1%, 2%, 3%, 4% and USD 2.5%."""
    rng = np.random.default_rng(1)
    spot = {c.code: list(1.0 + 0.02 * rng.standard_normal(6).cumsum()) for c in TOY}
    rates = {"USD": [0.025] * 6, "AAA": [0.01] * 6, "BBB": [0.02] * 6, "CCC": [0.03] * 6, "DDD": [0.04] * 6}
    return make_panel(spot, rates)
