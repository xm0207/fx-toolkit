import numpy as np
import pandas as pd
import pytest

from fxlab import DeltaHedgeSimulator, GarmanKohlhagen, daily_rates_from_monthly, realised_vol

RD, RF = 0.04, 0.02


def constant(dates, value):
    return pd.Series(value, index=dates)


def test_no_volatility_path_keeps_the_premium():
    # Spot moves exactly along the forward, so the hedge breaks even and the at-the-money-forward
    # call expires worthless: the seller ends with the premium plus interest on it.
    dates = pd.bdate_range("2022-01-03", periods=60)
    years = (dates - dates[0]).days.to_numpy() / 365
    spot = pd.Series(1.25 * np.exp((RD - RF) * years), index=dates)
    sim = DeltaHedgeSimulator(spot, constant(dates, RD), constant(dates, RF), pricing_vol=0.10)
    out = sim.simulate(0)
    tau = (out.expiry - out.start).days / 365
    assert out.realised_vol < 0.005  # only the small, steady forward drift
    assert out.spot_expiry == pytest.approx(out.strike)
    assert out.pnl == pytest.approx(out.premium * np.exp(RD * tau), rel=1e-10)


def test_premium_is_the_garman_kohlhagen_price_at_the_forward():
    dates = pd.bdate_range("2022-01-03", periods=40)
    spot = pd.Series(1.30, index=dates)
    out = DeltaHedgeSimulator(spot, constant(dates, RD), constant(dates, RF), pricing_vol=0.08).simulate(0)
    tau = (out.expiry - out.start).days / 365
    gk = GarmanKohlhagen(1.30, RD, RF, tau)
    assert out.strike == pytest.approx(float(gk.forward))
    assert out.premium == pytest.approx(float(gk.price(gk.forward, 0.08)))


def test_pnl_tracks_implied_minus_realised_variance():
    # Geometric Brownian motion with 10% volatility on a calendar-time clock; options priced at 6%,
    # 10% and 14%. Underpricing loses on average, overpricing gains, and the gamma-theta estimate
    # explains most of the path-by-path P&L.
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2000-01-03", periods=1500)
    dt = np.diff(dates).astype("timedelta64[D]").astype(float) / 365
    shocks = rng.standard_normal(dt.size) * 0.10 * np.sqrt(dt) - 0.5 * 0.10**2 * dt + (RD - RF) * dt
    spot = pd.Series(1.2 * np.exp(np.concatenate([[0.0], np.cumsum(shocks)])), index=dates)
    means = {}
    for vol in (0.06, 0.10, 0.14):
        sim = DeltaHedgeSimulator(spot, constant(dates, RD), constant(dates, RF), pricing_vol=vol)
        out = sim.run()
        means[vol] = out["pnl"].mean()
        if vol == 0.10:
            assert abs(means[vol]) < 0.05 * out["premium"].mean()
            assert np.corrcoef(out["pnl"], out["gamma_theta_pnl"])[0, 1] > 0.9
    assert means[0.06] < 0 < means[0.14]


def test_trailing_pricing_vol_uses_only_past_returns():
    dates = pd.bdate_range("2022-01-03", periods=80)
    values = 1.3 * np.exp(0.004 * (-1.0) ** np.arange(80))
    spot = pd.Series(values, index=dates)
    sim = DeltaHedgeSimulator(spot, constant(dates, RD), constant(dates, RF), vol_window=21)
    assert sim.pricing_vol(20) is None  # not enough history yet
    expected = realised_vol(values[9:31], dates[[9, 30]])
    assert sim.pricing_vol(30) == pytest.approx(expected)


def test_realised_vol_on_a_calendar_clock():
    dates = pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2022-01-05"])  # two one-day returns
    spot = np.array([1.0, np.exp(0.01), 1.0])
    assert realised_vol(spot, dates) == pytest.approx(np.sqrt(2 * 0.01**2 / (2 / 365)))


def test_expiry_is_first_date_on_or_after_one_month():
    dates = pd.DatetimeIndex(["2022-01-14", "2022-02-11", "2022-02-15", "2022-02-16"])
    sim = DeltaHedgeSimulator(pd.Series(1.0, index=dates), constant(dates, RD), constant(dates, RF), pricing_vol=0.1)
    assert sim.dates[sim.expiry_index(0)] == pd.Timestamp("2022-02-15")
    assert sim.expiry_index(2) is None


def test_daily_rates_use_the_previous_months_average():
    monthly = pd.Series([0.01, 0.02, 0.03], index=pd.period_range("2022-01", periods=3, freq="M"))
    dates = pd.DatetimeIndex(["2022-01-31", "2022-02-01", "2022-03-15", "2022-04-01"])
    daily = daily_rates_from_monthly(monthly, dates)
    assert np.isnan(daily.iloc[0])
    assert daily.iloc[1:].tolist() == [0.01, 0.02, 0.03]
