"""Charts for the README. Static PNGs drawn with matplotlib."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestResult  # noqa: E402
from .smile import SabrSmile  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order: blue, orange, aqua
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _style(ax) -> None:
    ax.grid(color=GRID, lw=0.6, alpha=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_strategies(results: Sequence[BacktestResult], path: str | Path) -> None:
    """Cumulative wealth (log scale) and drawdowns of each strategy, after costs."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for color, result in zip(SERIES, results):
        wealth = result.wealth()
        x = wealth.index.to_timestamp(how="end")
        top.plot(x, wealth.to_numpy(), color=color, lw=1.6, label=result.name)
        top.annotate(result.name, (x[-1], wealth.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                     color=INK, fontsize=8, va="center")
        bottom.plot(x, 100 * result.drawdown().to_numpy(), color=color, lw=1.2)
    top.set_yscale("log")
    top.yaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]))
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    top.yaxis.set_minor_formatter(NullFormatter())
    top.set_ylabel("Wealth, start = 1 (log scale)", color=MUTED, fontsize=9)
    top.set_title(f"Currency strategies after {results[0].cost_bp:g} bp costs, "
                  f"{results[0].frame.index[0].strftime('%b %Y')} to {results[0].frame.index[-1].strftime('%b %Y')}",
                  loc="left", fontsize=10, color=INK)
    top.legend(loc="upper left", fontsize=8, frameon=False)
    bottom.set_ylabel("Drawdown, %", color=MUTED, fontsize=9)
    for ax in (top, bottom):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_smile(smile: SabrSmile, path: str | Path, pair: str = "GBPUSD", tenor: str = "1M") -> None:
    """The SABR smile through the three quoted pillars, against strike."""
    pillars = smile.pillars
    lo, hi = pillars["strike"].min(), pillars["strike"].max()
    width = hi - lo
    strikes = np.linspace(lo - 0.8 * width, hi + 0.8 * width, 300)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(strikes, 100 * smile.vol(strikes), color=SERIES[0], lw=2, label="SABR smile (beta = 1)")
    ax.scatter(pillars["strike"], 100 * pillars["quoted_vol"], s=40, color=INK, zorder=3,
               label="Quoted pillars (illustrative, not market data)")
    for _, row in pillars.iterrows():
        ax.annotate(row["pillar"], (row["strike"], 100 * row["quoted_vol"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axvline(float(smile.pricer.forward), color=MUTED, lw=0.8, ls="--")
    ax.annotate(" forward", (float(smile.pricer.forward), ax.get_ylim()[0]), fontsize=8, color=MUTED, va="bottom")
    ax.set_xlabel(f"Strike ({pair})", color=MUTED, fontsize=9)
    ax.set_ylabel("Implied volatility, %", color=MUTED, fontsize=9)
    ax.set_title(f"{pair} {tenor} smile from illustrative ATM / 25D RR / 25D BF quotes", loc="left",
                 fontsize=10, color=INK)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hedge(outcomes: pd.DataFrame, path: str | Path, notional: float = 1e6) -> None:
    """Left: hedged P&L against pricing minus realised variance. Right: P&L against its gamma-theta estimate."""
    pnl = outcomes["pnl"] * notional / 1e3
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    left.scatter(100**2 * outcomes["variance_gap"], pnl, s=3, alpha=0.25, color=SERIES[0], linewidths=0)
    left.axhline(0, color=MUTED, lw=0.8)
    left.axvline(0, color=MUTED, lw=0.8)
    left.set_xlabel("Pricing vol² − realised vol² (vol points²)", color=MUTED, fontsize=9)
    left.set_ylabel("Hedged P&L at expiry, $000 per £1m", color=MUTED, fontsize=9)
    left.set_title("P&L follows implied minus realised variance", loc="left", fontsize=10, color=INK)

    approx = outcomes["gamma_theta_pnl"] * notional / 1e3
    right.scatter(approx, pnl, s=3, alpha=0.25, color=SERIES[0], linewidths=0)
    lim = [min(approx.min(), pnl.min()), max(approx.max(), pnl.max())]
    right.plot(lim, lim, color=SERIES[1], lw=1.2, label="45° line")
    right.set_xlabel("Sum of ½ Γ S² (σ² dt − (dS/S)²), $000 per £1m", color=MUTED, fontsize=9)
    right.set_ylabel("Hedged P&L at expiry, $000 per £1m", color=MUTED, fontsize=9)
    right.set_title("Gamma-theta estimate vs simulated P&L", loc="left", fontsize=10, color=INK)
    right.legend(loc="upper left", fontsize=8, frameon=False)
    for ax in (left, right):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
