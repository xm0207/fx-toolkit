"""FX forwards, currency strategy backtests, FX option pricing and delta-hedging, on public FRED data."""

from .backtest import BacktestResult, StrategyBacktest, max_drawdown, performance_stats
from .currencies import G10, USD_RATE_SERIES, Currency, by_code, to_continuous
from .data import FredClient, FxDataStore
from .forwards import MonthlyFxPanel, cip_forward, forward_points
from .hedging import DeltaHedgeSimulator, HedgeOutcome, daily_rates_from_monthly, realised_vol
from .options import GarmanKohlhagen
from .smile import SabrSmile, SmileQuotes, sabr_vol
from .strategies import Carry, DollarFactor, Momentum, SortedLongShort, Strategy, long_short_weights

__all__ = [
    "BacktestResult",
    "Carry",
    "Currency",
    "DeltaHedgeSimulator",
    "DollarFactor",
    "FredClient",
    "FxDataStore",
    "G10",
    "GarmanKohlhagen",
    "HedgeOutcome",
    "Momentum",
    "MonthlyFxPanel",
    "SabrSmile",
    "SmileQuotes",
    "SortedLongShort",
    "Strategy",
    "StrategyBacktest",
    "USD_RATE_SERIES",
    "by_code",
    "cip_forward",
    "daily_rates_from_monthly",
    "forward_points",
    "long_short_weights",
    "max_drawdown",
    "performance_stats",
    "realised_vol",
    "sabr_vol",
    "to_continuous",
]
