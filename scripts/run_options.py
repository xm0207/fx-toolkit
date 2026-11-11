"""Price a one-month GBPUSD option grid off a SABR smile built from ILLUSTRATIVE vol quotes.

Spot comes from FRED and the interest rates are the latest OECD 3-month rates on FRED. The
volatility quotes (ATM, 25-delta risk reversal and butterfly) are NOT market data: FX option
quotes are not freely available, so they are set by hand below and labelled as illustrative in
every output.

Writes to results/: option_inputs.csv, smile_pillars.csv, option_grid.csv, delta_conventions.csv
and smile.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fxlab import FxDataStore, GarmanKohlhagen, SabrSmile, SmileQuotes, by_code, to_continuous
from fxlab.currencies import USD_DAY_COUNT_BASIS
from fxlab.report import plot_smile

ILLUSTRATIVE_QUOTES = SmileQuotes(atm=0.080, rr25=-0.008, bf25=0.0025, source="illustrative, set by hand")
TENOR_DAYS = 30
NOTIONAL_GBP = 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/fred")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)

    gbp = by_code("GBP")
    store = FxDataStore(args.data)
    spot_series = store.series(gbp.spot_series)
    rates = store.monthly_rates()
    usd_rate, gbp_rate = rates["USD"].dropna(), rates["GBP"].dropna()
    tau = TENOR_DAYS / 365.0
    r_usd = to_continuous(usd_rate.iloc[-1], USD_DAY_COUNT_BASIS, tau)
    r_gbp = to_continuous(gbp_rate.iloc[-1], gbp.day_count_basis, tau)
    pricer = GarmanKohlhagen(float(spot_series.iloc[-1]), r_usd, r_gbp, tau)

    inputs = pd.DataFrame([
        {"input": "GBPUSD spot", "value": float(spot_series.iloc[-1]), "source": f"FRED DEXUSUK, {spot_series.index[-1]:%Y-%m-%d}"},
        {"input": "USD rate (cont. comp.)", "value": r_usd, "source": f"FRED IR3TIB01USM156N, {usd_rate.index[-1]} average"},
        {"input": "GBP rate (cont. comp.)", "value": r_gbp, "source": f"FRED IR3TIB01GBM156N, {gbp_rate.index[-1]} average (latest available)"},
        {"input": "time to expiry (years)", "value": tau, "source": f"{TENOR_DAYS} calendar days / 365"},
        {"input": "forward", "value": float(pricer.forward), "source": "covered interest parity"},
        {"input": "ATM vol", "value": ILLUSTRATIVE_QUOTES.atm, "source": ILLUSTRATIVE_QUOTES.source},
        {"input": "25D risk reversal", "value": ILLUSTRATIVE_QUOTES.rr25, "source": ILLUSTRATIVE_QUOTES.source},
        {"input": "25D butterfly", "value": ILLUSTRATIVE_QUOTES.bf25, "source": ILLUSTRATIVE_QUOTES.source},
    ])
    inputs.to_csv(out / "option_inputs.csv", index=False, float_format="%.6f")
    print(inputs.to_string(index=False))

    smile = SabrSmile.calibrate(pricer, ILLUSTRATIVE_QUOTES, beta=1.0, delta_kind="spot", premium_adjusted=False)
    pillars = smile.pillars.assign(alpha=smile.alpha, beta=smile.beta, rho=smile.rho, nu=smile.nu,
                                   quotes="illustrative, not market data")
    pillars.to_csv(out / "smile_pillars.csv", index=False, float_format="%.8f")
    print(pillars.to_string(index=False))

    rows = []
    for label, delta, call in [("10D put", -0.10, False), ("25D put", -0.25, False), ("ATM (DNS)", None, True),
                               ("25D call", 0.25, True), ("10D call", 0.10, True)]:
        if delta is None:
            strike = float(smile.pillars.loc[smile.pillars["pillar"] == "ATM (DNS)", "strike"].iloc[0])
            vol = float(smile.vol(strike))
        else:
            strike, vol = smile.strike_at_delta(delta, call)
        price = float(pricer.price(strike, vol, call))
        rows.append({
            "option": label,
            "strike": strike,
            "vol_pct": 100 * vol,
            "premium_usd_pips": price / 1e-4,
            "premium_pct_gbp": 100 * price / float(pricer.spot),
            "premium_usd": price * NOTIONAL_GBP,
            "spot_delta": float(pricer.delta(strike, vol, call, "spot")),
            "forward_delta": float(pricer.delta(strike, vol, call, "forward")),
            "spot_delta_pa": float(pricer.delta(strike, vol, call, "spot", premium_adjusted=True)),
            "gamma_gbp_per_1pct": float(pricer.gamma(strike, vol)) * float(pricer.spot) * 0.01 * NOTIONAL_GBP,
            "vega_usd_per_volpt": float(pricer.vega(strike, vol)) * 0.01 * NOTIONAL_GBP,
            "theta_usd_per_day": float(pricer.theta(strike, vol, call)) / 365 * NOTIONAL_GBP,
        })
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "option_grid.csv", index=False, float_format="%.6f")
    print(grid.round(4).to_string(index=False))

    # The same 25-delta call and put at the ATM vol, under each of the four delta conventions.
    conventions = []
    for kind in ("spot", "forward"):
        for pa in (False, True):
            conventions.append({
                "convention": f"{kind} delta{', premium-adjusted' if pa else ''}",
                "strike_25d_call": float(pricer.strike_from_delta(0.25, ILLUSTRATIVE_QUOTES.atm, True, kind, pa)),
                "strike_25d_put": float(pricer.strike_from_delta(-0.25, ILLUSTRATIVE_QUOTES.atm, False, kind, pa)),
                "atm_dns_strike": float(pricer.atm_dns_strike(ILLUSTRATIVE_QUOTES.atm, pa)),
            })
    conventions = pd.DataFrame(conventions)
    conventions.to_csv(out / "delta_conventions.csv", index=False, float_format="%.6f")
    print(conventions.round(5).to_string(index=False))

    plot_smile(smile, out / "smile.png")


if __name__ == "__main__":
    main()
