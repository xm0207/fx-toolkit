import numpy as np
import pandas as pd
import pytest

from fxlab import by_code, cip_forward, forward_points
from fxlab.currencies import USD_DAY_COUNT_BASIS
from fxlab.forwards import MonthlyFxPanel

from .conftest import TOY, make_panel


def test_cip_forward_hand_calculation():
    # GBPUSD 1.30, USD 5% (ACT/360), GBP 4% (ACT/365), 90 days
    expected = 1.30 * (1 + 0.05 * 90 / 360) / (1 + 0.04 * 90 / 365)
    assert cip_forward(1.30, 0.05, 0.04, 90, 360, 365) == pytest.approx(expected, rel=1e-15)


def test_cip_leaves_no_arbitrage():
    # Borrow S dollars, buy one unit of foreign currency, deposit it, sell the proceeds forward:
    # the forward sale exactly repays the dollar loan.
    spot, r_dom, r_for, days = 1.10, 0.03, 0.01, 182
    forward = cip_forward(spot, r_dom, r_for, days, 360, 360)
    foreign_at_maturity = 1 + r_for * days / 360
    dollars_owed = spot * (1 + r_dom * days / 360)
    assert foreign_at_maturity * forward == pytest.approx(dollars_owed, rel=1e-15)


def test_forward_points_in_pips():
    assert forward_points(1.2500, 1.2512) == pytest.approx(12.0)
    assert forward_points(150.00, 149.70, pip=0.01) == pytest.approx(-30.0)


def test_currency_conventions():
    jpy, gbp = by_code("JPY"), by_code("GBP")
    assert (jpy.pair, jpy.pip, gbp.pair, gbp.pip) == ("USDJPY", 0.01, "GBPUSD", 0.0001)
    assert jpy.fred_to_usd_per_unit(150.0) == pytest.approx(1 / 150)
    assert jpy.usd_per_unit_to_market(1 / 150) == pytest.approx(150.0)


def test_panel_forward_and_excess_return_timing():
    spot = {"AAA": [1.00, 1.02, 0.99]}
    rates = {"USD": [0.05] * 3, "AAA": [0.01] * 3}
    panel = make_panel(spot, rates)
    days = panel.days.iloc[0]
    assert days == (panel.month_end.iloc[1] - panel.month_end.iloc[0]).days
    f0 = cip_forward(1.00, 0.05, 0.01, days, USD_DAY_COUNT_BASIS, 360)
    assert panel.forward["AAA"].iloc[0] == pytest.approx(f0)
    # the forward agreed at the end of month 0 pays off in month 1
    rx = panel.excess_returns["AAA"]
    assert np.isnan(rx.iloc[0])
    assert rx.iloc[1] == pytest.approx(1.02 / f0 - 1)
    # a low-yielding foreign currency trades at a forward premium, so its carry is negative
    assert panel.carry["AAA"].iloc[0] < 0


def test_carry_is_about_the_interest_differential():
    spot = {"AAA": [1.0] * 4}
    rates = {"USD": [0.02] * 4, "AAA": [0.06] * 4}
    panel = make_panel(spot, rates)
    assert panel.carry["AAA"].iloc[0] == pytest.approx(0.04, abs=0.002)


def test_month_end_uses_the_last_observation_of_the_month():
    dates = pd.to_datetime(["2021-01-05", "2021-01-29", "2021-02-10", "2021-02-26"])
    spot = pd.DataFrame({"AAA": [1.0, 1.1, 1.2, 1.3]}, index=dates)
    rates = pd.DataFrame({"USD": [0.01, 0.01], "AAA": [0.01, 0.01]}, index=pd.period_range("2021-01", periods=2, freq="M"))
    panel = MonthlyFxPanel.build(spot, rates, TOY[:1])
    assert list(panel.spot["AAA"]) == [1.1, 1.3]
    assert panel.days.iloc[0] == 28  # 29 Jan to 26 Feb


def test_balanced_keeps_only_the_complete_span():
    spot = {"AAA": [1.0] * 5, "BBB": [1.0] * 5}
    rates = {"USD": [0.01] * 5, "AAA": [0.01] * 5, "BBB": [np.nan, 0.02, 0.02, 0.02, np.nan]}
    balanced = make_panel(spot, rates).balanced()
    assert [str(m) for m in balanced.complete_months()] == ["2020-02", "2020-03", "2020-04"]
    # returns run from the month after the first complete month to the month after the last
    rx = balanced.excess_returns.dropna()
    assert [str(m) for m in rx.index] == ["2020-03", "2020-04", "2020-05"]


def test_balanced_rejects_holes_inside_the_span():
    spot = {"AAA": [1.0] * 4}
    rates = {"USD": [0.01, np.nan, 0.01, 0.01], "AAA": [0.01] * 4}
    with pytest.raises(ValueError):
        make_panel(spot, rates).balanced()


def test_forward_points_table_uses_market_quotes():
    usd_jpy = by_code("JPY")
    dates = pd.to_datetime(["2021-01-29", "2021-02-26"])
    spot = pd.DataFrame({"JPY": [1 / 100.0, 1 / 101.0]}, index=dates)
    rates = pd.DataFrame({"USD": [0.02, 0.02], "JPY": [0.0, 0.0]}, index=pd.period_range("2021-01", periods=2, freq="M"))
    panel = MonthlyFxPanel.build(spot, rates, (usd_jpy,))
    row = panel.forward_points_table("2021-01").iloc[0]
    assert row["pair"] == "USDJPY"
    assert row["spot"] == pytest.approx(100.0)
    # the dollar earns more than the yen, so USDJPY trades at a forward discount: negative points
    expected_forward = 100.0 * (1 + 0.0) / (1 + 0.02 * 28 / 360)
    assert row["forward_1m"] == pytest.approx(expected_forward)
    assert row["forward_points_pips"] == pytest.approx((expected_forward - 100.0) / 0.01)
