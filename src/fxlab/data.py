"""Download FRED series and read them back as tidy tables.

Raw data lives in plain CSV files (data/fred/<SERIES>.csv), one file per series, so that it is
easy to inspect and the tests can build a fake data folder in a few lines.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from .currencies import G10, USD_RATE_SERIES, Currency, usd_per_unit_frame

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


class FredClient:
    """Fetches one series from FRED's public CSV endpoint (no API key needed).

    Why a class: the HTTP session and timeout are set once and reused for every series, and a test
    can pass in a fake session so that no network is needed.
    """

    def __init__(self, session: requests.Session | None = None, timeout: float = 60.0):
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch(self, series_id: str) -> pd.Series:
        """Return the series as floats indexed by date. FRED's missing values (".") are dropped."""
        response = self.session.get(FRED_CSV_URL.format(series_id=series_id), timeout=self.timeout)
        response.raise_for_status()
        header = response.text.split("\n", 1)[0].strip()
        if header != f"observation_date,{series_id}":
            raise ValueError(f"Unexpected response from FRED for {series_id}: {header[:80]!r}")
        frame = pd.read_csv(io.StringIO(response.text), na_values=["."])
        series = pd.Series(frame.iloc[:, 1].to_numpy(dtype=float), index=pd.to_datetime(frame.iloc[:, 0]))
        series.index.name = "date"
        series.name = series_id
        return series.dropna()


class FxDataStore:
    """Reads the downloaded CSV files and returns the tables the analysis needs.

    Public interface:
        save(series)          write one series to the folder
        series(series_id)     one series as a pandas Series
        coverage(ids)         first date, last date and count of each series
        spot(currencies)      daily spot rates, US dollars per unit of foreign currency
        monthly_rates(...)    monthly 3-month rates in decimals, indexed by calendar month
        monthly_equity(...)   month-end levels of an equity index
    """

    def __init__(self, folder: str | Path = "data/fred"):
        self.folder = Path(folder)

    def path(self, series_id: str) -> Path:
        return self.folder / f"{series_id}.csv"

    def save(self, series: pd.Series) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path(str(series.name))
        series.rename("value").to_csv(path, index_label="date")
        return path

    def series(self, series_id: str) -> pd.Series:
        path = self.path(series_id)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run scripts/download_data.py first")
        frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return frame["value"].astype(float).rename(series_id)

    def coverage(self, series_ids: list[str]) -> pd.DataFrame:
        rows = []
        for series_id in series_ids:
            s = self.series(series_id)
            rows.append({"series_id": series_id, "first": s.index[0].date(), "last": s.index[-1].date(),
                         "observations": len(s)})
        return pd.DataFrame(rows)

    def spot(self, currencies: tuple[Currency, ...] = G10) -> pd.DataFrame:
        """Daily spot rates in US dollars per unit of each currency; NaN where FRED has no value."""
        quotes = pd.concat([self.series(c.spot_series) for c in currencies], axis=1, sort=True)
        return usd_per_unit_frame(quotes, currencies)

    def monthly_rates(self, currencies: tuple[Currency, ...] = G10, usd_series: str = USD_RATE_SERIES,
                      max_gap: int = 1) -> pd.DataFrame:
        """Monthly-average 3-month interbank rates as decimals, one column per currency plus "USD".

        FRED dates each OECD observation on the first day of the month it averages; the index here
        is the calendar month itself (a pandas Period), which avoids confusing it with a daily date.

        A gap of at most `max_gap` months inside a series is filled with the last earlier value
        (the USD series has no value for April 2020). Nothing is filled before a series' first
        observation or after its last one, so ragged starts and ends stay visible.
        """
        columns = {"USD": self.series(usd_series)}
        columns.update({c.code: self.series(c.rate_series) for c in currencies})
        table = pd.concat(columns, axis=1, sort=True) / 100.0
        table.index = table.index.to_period("M")
        table = table.reindex(pd.period_range(table.index.min(), table.index.max(), freq="M"))
        inside = table.bfill().notna() & table.ffill().notna()  # between first and last observation
        if max_gap < 1:
            return table
        return table.ffill(limit=max_gap).where(inside)

    def filled_rate_months(self, currencies: tuple[Currency, ...] = G10, max_gap: int = 1) -> pd.DataFrame:
        """The months that `monthly_rates` fills, for reporting."""
        raw = self.monthly_rates(currencies, max_gap=0)
        filled = self.monthly_rates(currencies, max_gap=max_gap)
        mask = raw.isna() & filled.notna()
        rows = [(col, str(month)) for col in mask for month in mask.index[mask[col].to_numpy()]]
        return pd.DataFrame(rows, columns=["series", "month"])

    def monthly_equity(self, series_id: str = "NASDAQCOM") -> pd.Series:
        """Last close of each calendar month."""
        s = self.series(series_id)
        return s.groupby(s.index.to_period("M")).last()
