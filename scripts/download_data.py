"""Download every FRED series the project uses into data/fred/ and record their coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

from fxlab import G10, USD_RATE_SERIES, FredClient, FxDataStore


def series_list() -> dict[str, str]:
    series = {}
    for c in G10:
        unit = f"USD per {c.code}" if c.fred_usd_per_unit else f"{c.code} per USD"
        series[c.spot_series] = f"{c.code} spot, {unit}, daily (H.10)"
    series[USD_RATE_SERIES] = "USD 3-month interbank rate, % p.a., monthly average (OECD)"
    for c in G10:
        series[c.rate_series] = f"{c.code} 3-month interbank rate, % p.a., monthly average (OECD)"
    series["NASDAQCOM"] = "NASDAQ Composite index, daily close"
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="data/fred", help="where to write one CSV per series")
    parser.add_argument("--out", default="results", help="where to write data_coverage.csv")
    args = parser.parse_args()

    client, store = FredClient(), FxDataStore(args.folder)
    descriptions = series_list()
    for series_id, description in descriptions.items():
        series = client.fetch(series_id)
        store.save(series)
        print(f"{series_id:<17} {len(series):>6} rows  {series.index[0]:%Y-%m-%d} to {series.index[-1]:%Y-%m-%d}  {description}")

    coverage = store.coverage(list(descriptions))
    coverage["description"] = coverage["series_id"].map(descriptions)
    Path(args.out).mkdir(exist_ok=True)
    coverage.to_csv(Path(args.out) / "data_coverage.csv", index=False)


if __name__ == "__main__":
    main()
