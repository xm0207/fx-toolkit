import numpy as np
import pytest

from fxlab import GarmanKohlhagen

S, RD, RF, T = 1.30, 0.045, 0.035, 0.25
VOL = 0.10


@pytest.fixture
def gk():
    return GarmanKohlhagen(S, RD, RF, T)


def test_forward_is_covered_interest_parity(gk):
    assert gk.forward == pytest.approx(S * np.exp((RD - RF) * T))


@pytest.mark.parametrize("strike", [1.10, 1.25, 1.30, 1.35, 1.50])
def test_put_call_parity(gk, strike):
    # C - P = exp(-r_dom T) (F - K): a long call and a short put is a forward
    call, put = gk.price(strike, VOL, call=True), gk.price(strike, VOL, call=False)
    assert call - put == pytest.approx(gk.df_dom * (gk.forward - strike), abs=1e-14)


def test_prices_are_vectorised(gk):
    strikes = np.array([1.2, 1.3, 1.4])
    one_by_one = [float(gk.price(k, VOL)) for k in strikes]
    assert np.allclose(gk.price(strikes, VOL), one_by_one)


def _price(spot=S, rd=RD, rf=RF, tau=T, strike=1.32, vol=VOL, call=True):
    return float(GarmanKohlhagen(spot, rd, rf, tau).price(strike, vol, call))


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("strike", [1.20, 1.32, 1.45])
def test_greeks_match_finite_differences(call, strike):
    gk = GarmanKohlhagen(S, RD, RF, T)
    h, h2 = 1e-5, 1e-4  # step for first derivatives; a larger one for the second derivative
    delta_fd = (_price(spot=S + h, strike=strike, call=call) - _price(spot=S - h, strike=strike, call=call)) / (2 * h)
    gamma_fd = (_price(spot=S + h2, strike=strike, call=call) - 2 * _price(strike=strike, call=call)
                + _price(spot=S - h2, strike=strike, call=call)) / h2**2
    vega_fd = (_price(vol=VOL + h, strike=strike, call=call) - _price(vol=VOL - h, strike=strike, call=call)) / (2 * h)
    theta_fd = -(_price(tau=T + h, strike=strike, call=call) - _price(tau=T - h, strike=strike, call=call)) / (2 * h)
    assert gk.delta(strike, VOL, call) == pytest.approx(delta_fd, rel=1e-6)
    assert gk.gamma(strike, VOL) == pytest.approx(gamma_fd, rel=1e-4)
    assert gk.vega(strike, VOL) == pytest.approx(vega_fd, rel=1e-6)
    assert gk.theta(strike, VOL, call) == pytest.approx(theta_fd, rel=1e-6)


@pytest.mark.parametrize("call", [True, False])
def test_forward_delta_is_derivative_of_undiscounted_value_wrt_forward(call):
    # F = S exp((rd - rf) T), so moving S by h moves F by h exp((rd - rf) T)
    strike, h = 1.31, 1e-4
    gk = GarmanKohlhagen(S, RD, RF, T)
    growth = np.exp((RD - RF) * T)
    up = _price(spot=S + h, strike=strike, call=call) / gk.df_dom
    down = _price(spot=S - h, strike=strike, call=call) / gk.df_dom
    assert gk.delta(strike, VOL, call, kind="forward") == pytest.approx((up - down) / (2 * h * growth), rel=1e-6)


@pytest.mark.parametrize("call", [True, False])
def test_premium_adjusted_delta_subtracts_the_premium(gk, call):
    strike = 1.33
    expected = gk.delta(strike, VOL, call) - gk.price(strike, VOL, call) / S
    assert gk.delta(strike, VOL, call, premium_adjusted=True) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("kind", ["spot", "forward"])
@pytest.mark.parametrize("premium_adjusted", [False, True])
@pytest.mark.parametrize("delta, call", [(0.10, True), (0.25, True), (0.40, True), (-0.10, False), (-0.25, False), (-0.45, False)])
def test_delta_strike_round_trip(gk, kind, premium_adjusted, delta, call):
    strike = gk.strike_from_delta(delta, VOL, call, kind, premium_adjusted)
    assert gk.delta(strike, VOL, call, kind, premium_adjusted) == pytest.approx(delta, abs=1e-10)


def test_premium_adjusted_call_takes_the_upper_strike(gk):
    # The premium-adjusted call delta rises then falls with the strike; the convention takes the
    # root on the falling side, where a higher strike means a lower delta.
    strike = float(gk.strike_from_delta(0.25, 0.30, True, "spot", premium_adjusted=True))
    h = 1e-5
    slope = (gk.delta(strike + h, 0.30, True, premium_adjusted=True) - gk.delta(strike - h, 0.30, True, premium_adjusted=True))
    assert slope < 0
    assert strike < gk.strike_from_delta(0.25, 0.30, True, "spot")


@pytest.mark.parametrize("premium_adjusted", [False, True])
def test_atm_dns_straddle_is_delta_neutral(gk, premium_adjusted):
    k = gk.atm_dns_strike(VOL, premium_adjusted)
    straddle = gk.delta(k, VOL, True, premium_adjusted=premium_adjusted) + gk.delta(k, VOL, False, premium_adjusted=premium_adjusted)
    assert straddle == pytest.approx(0.0, abs=1e-12)


def test_bad_inputs_are_rejected(gk):
    with pytest.raises(ValueError):
        GarmanKohlhagen(S, RD, RF, 0.0)
    with pytest.raises(ValueError):
        gk.price(1.3, -0.1)
    with pytest.raises(ValueError):
        gk.strike_from_delta(-0.25, VOL, call=True)  # a call cannot have a negative delta
    with pytest.raises(ValueError):
        gk.delta(1.3, VOL, kind="vega")
