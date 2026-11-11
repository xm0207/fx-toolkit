import numpy as np
import pandas as pd
import pytest

from fxlab import Carry, DollarFactor, Momentum, long_short_weights

from .conftest import make_panel


def test_long_short_weights_top_and_bottom():
    signal = pd.Series({"A": 0.5, "B": -1.0, "C": 2.0, "D": 0.0, "E": 1.0, "F": -0.5, "G": 3.0})
    w = long_short_weights(signal, n=3)
    assert w[["G", "C", "E"]].tolist() == pytest.approx([1 / 3] * 3)
    assert w[["B", "F", "D"]].tolist() == pytest.approx([-1 / 3] * 3)
    assert w["A"] == 0.0
    assert w.sum() == pytest.approx(0.0)  # dollar-neutral
    assert w.abs().sum() == pytest.approx(2.0)  # one dollar long, one dollar short


def test_missing_signals_are_not_traded():
    signal = pd.Series({"A": 1.0, "B": np.nan, "C": 3.0, "D": 2.0})
    w = long_short_weights(signal, n=1)
    assert w.to_dict() == {"A": -1.0, "B": 0.0, "C": 1.0, "D": 0.0}


def test_too_few_signals_means_no_position():
    w = long_short_weights(pd.Series({"A": 1.0, "B": np.nan, "C": 2.0}), n=2)
    assert w.isna().all()


def test_ties_are_broken_by_column_order():
    w = long_short_weights(pd.Series({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}), n=1)
    assert w.to_dict() == {"A": -1.0, "B": 0.0, "C": 0.0, "D": 1.0}


def test_carry_goes_long_high_yielders_and_short_low_yielders(toy_panel):
    w = Carry(n=1).weights(toy_panel)
    first = w.iloc[0]
    assert first["DDD"] == 1.0 and first["AAA"] == -1.0  # rates 4% and 1%
    assert first[["BBB", "CCC"]].tolist() == [0.0, 0.0]


def test_carry_has_no_position_where_a_forward_is_missing():
    spot = {"AAA": [1.0] * 3, "BBB": [1.0] * 3}
    rates = {"USD": [0.01] * 3, "AAA": [0.02, np.nan, 0.02], "BBB": [0.03] * 3}
    w = Carry(n=1).weights(make_panel(spot, rates))
    assert w.iloc[1].isna().all()
    assert w.iloc[0].tolist() == [-1.0, 1.0]


def test_momentum_signal_compounds_past_returns_only(toy_panel):
    strategy = Momentum(n=1, lookback=2)
    signal = strategy.signal(toy_panel)
    rx = toy_panel.excess_returns
    month = signal.index[3]
    expected = (1 + rx.loc[month - 1]) * (1 + rx.loc[month]) - 1
    assert np.allclose(signal.loc[month], expected)
    assert signal.iloc[:2].isna().all().all()  # month 0 has no return yet, month 1 has only one


def test_momentum_weights_do_not_see_the_future(toy_panel):
    before = Momentum(n=1, lookback=2).weights(toy_panel)
    shocked_spot = toy_panel.spot.copy()
    shocked_spot.iloc[-1] *= np.array([2.0, 0.5, 1.5, 0.7])  # change only the last month's spot
    shocked = toy_panel.__class__(shocked_spot, toy_panel.forward, toy_panel.rates, toy_panel.month_end,
                                  toy_panel.days, toy_panel.currencies)
    after = Momentum(n=1, lookback=2).weights(shocked)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


def test_dollar_factor_is_equally_weighted(toy_panel):
    w = DollarFactor().weights(toy_panel)
    assert np.allclose(w.to_numpy(), 0.25)
