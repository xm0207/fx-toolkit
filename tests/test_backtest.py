import numpy as np
import pandas as pd
import pytest

from fxlab import StrategyBacktest, max_drawdown, performance_stats
from fxlab.strategies import Strategy

MONTHS = pd.period_range("2021-01", periods=4, freq="M")


class ToyPanel:
    """Just the one attribute the backtest reads: monthly excess returns."""

    def __init__(self, excess_returns: pd.DataFrame):
        self.excess_returns = excess_returns


class FixedWeights(Strategy):
    name = "fixed"

    def __init__(self, table: pd.DataFrame):
        self.table = table

    def weights(self, panel):
        return self.table


@pytest.fixture
def toy():
    returns = pd.DataFrame({"A": [np.nan, 0.02, -0.01, 0.03], "B": [np.nan, -0.01, 0.04, 0.00]}, index=MONTHS)
    # decided at the end of Jan, Feb and Mar; nothing decided at the end of Apr
    weights = pd.DataFrame({"A": [1.0, 1.0, -1.0, np.nan], "B": [-1.0, 0.0, 1.0, np.nan]}, index=MONTHS)
    return ToyPanel(returns), FixedWeights(weights)


def test_backtest_arithmetic_by_hand(toy):
    panel, strategy = toy
    result = StrategyBacktest(panel, cost_bp=10).run(strategy)
    f = result.frame
    assert [str(m) for m in f.index] == ["2021-02", "2021-03", "2021-04"]
    # Feb holds (A +1, B -1): 0.02 + 0.01; Mar holds (A +1, B 0): -0.01; Apr holds (A -1, B +1): -0.03
    assert f["gross"].tolist() == pytest.approx([0.03, -0.01, -0.03])
    # trades: build |1| + |-1| = 2, then |0| + |1| = 1, then |-2| + |1| = 3
    assert f["turnover"].tolist() == pytest.approx([2.0, 1.0, 3.0])
    assert f["cost"].tolist() == pytest.approx([0.002, 0.001, 0.003])
    assert f["net"].tolist() == pytest.approx([0.028, -0.011, -0.033])


def test_zero_cost_net_equals_gross(toy):
    panel, strategy = toy
    f = StrategyBacktest(panel, cost_bp=0).run(strategy).frame
    assert np.allclose(f["net"], f["gross"])


def test_start_month_rebuilds_the_position(toy):
    panel, strategy = toy
    f = StrategyBacktest(panel, cost_bp=10).run(strategy, start="2021-03").frame
    assert f["turnover"].iloc[0] == pytest.approx(1.0)  # buys A from nothing


def test_missing_return_on_a_held_currency_is_an_error(toy):
    panel, strategy = toy
    panel.excess_returns.loc[MONTHS[2], "A"] = np.nan
    with pytest.raises(ValueError):
        StrategyBacktest(panel).run(strategy)


def test_max_drawdown_by_hand():
    # wealth 1 -> 1.1 -> 0.88 -> 0.924: worst fall is from 1.1 to 0.88, i.e. -20%
    assert max_drawdown(pd.Series([0.1, -0.2, 0.05])) == pytest.approx(-0.2)
    assert max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)  # the start counts as a peak
    assert max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_performance_stats_annualise_monthly_returns():
    r = pd.Series([0.01, -0.02, 0.03, 0.00, 0.02, -0.01])
    stats = performance_stats(r)
    assert stats["ann_return"] == pytest.approx(r.mean() * 12)
    assert stats["ann_vol"] == pytest.approx(r.std(ddof=1) * np.sqrt(12))
    assert stats["sharpe"] == pytest.approx(stats["ann_return"] / stats["ann_vol"])
    assert stats["worst_month"] == -0.02
    assert stats["skew"] == pytest.approx(r.skew())


def test_run_all_uses_a_common_start(toy):
    panel, strategy = toy
    late_table = strategy.table.copy()
    late_table.iloc[0] = np.nan  # this strategy decides nothing at the end of January
    late = FixedWeights(late_table)
    results = StrategyBacktest(panel).run_all([strategy, late])
    assert {str(r.frame.index[0]) for r in results} == {"2021-03"}


def test_correlation_aligns_months(toy):
    panel, strategy = toy
    result = StrategyBacktest(panel, cost_bp=0).run(strategy)
    other = pd.Series([0.03, -0.01, -0.03, 99.0], index=pd.period_range("2021-02", periods=4, freq="M"))
    assert result.correlation(other) == pytest.approx(1.0)
