import numpy as np
import pytest

from fxlab import GarmanKohlhagen, SabrSmile, SmileQuotes, sabr_vol

QUOTES = SmileQuotes(atm=0.08, rr25=-0.008, bf25=0.0025)


@pytest.fixture
def pricer():
    return GarmanKohlhagen(1.34, 0.04, 0.037, 30 / 365)


def test_simplified_strangle_vols():
    assert QUOTES.call25 - QUOTES.put25 == pytest.approx(QUOTES.rr25)
    assert 0.5 * (QUOTES.call25 + QUOTES.put25) - QUOTES.atm == pytest.approx(QUOTES.bf25)


@pytest.mark.parametrize("premium_adjusted", [False, True])
def test_smile_reproduces_the_input_quotes(pricer, premium_adjusted):
    smile = SabrSmile.calibrate(pricer, QUOTES, premium_adjusted=premium_adjusted)
    p = smile.pillars
    assert np.allclose(smile.vol(p["strike"].to_numpy()), [QUOTES.put25, QUOTES.atm, QUOTES.call25], atol=1e-8)


def test_pillar_strikes_have_the_quoted_deltas(pricer):
    smile = SabrSmile.calibrate(pricer, QUOTES)
    k_put, k_atm, k_call = smile.pillars["strike"]
    assert pricer.delta(k_put, QUOTES.put25, call=False) == pytest.approx(-0.25, abs=1e-10)
    assert pricer.delta(k_call, QUOTES.call25, call=True) == pytest.approx(0.25, abs=1e-10)
    straddle = pricer.delta(k_atm, QUOTES.atm, True) + pricer.delta(k_atm, QUOTES.atm, False)
    assert straddle == pytest.approx(0.0, abs=1e-12)


def test_strike_at_delta_is_consistent_with_the_pillars(pricer):
    smile = SabrSmile.calibrate(pricer, QUOTES)
    strike, vol = smile.strike_at_delta(0.25, call=True)
    assert strike == pytest.approx(smile.pillars["strike"].iloc[2], rel=1e-7)
    assert vol == pytest.approx(QUOTES.call25, abs=1e-8)


def test_negative_risk_reversal_gives_downside_skew(pricer):
    smile = SabrSmile.calibrate(pricer, QUOTES)
    forward = float(pricer.forward)
    assert smile.rho < 0
    assert smile.vol(0.95 * forward) > smile.vol(1.05 * forward)


def test_sabr_without_vol_of_vol_is_flat():
    strikes = np.array([0.8, 1.0, 1.2])
    vols = sabr_vol(strikes, forward=1.0, tau=0.5, alpha=0.1, beta=1.0, rho=0.0, nu=1e-12)
    assert np.allclose(vols, 0.1, atol=1e-10)
