import numpy as np
import pandas as pd
import pytest

from fxlab import FredClient, FxDataStore, by_code, to_continuous

FRED_TEXT = "observation_date,DEXJPUS\n2024-01-02,141.5\n2024-01-03,.\n2024-01-04,143.0\n"


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, text):
        self.text = text
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return FakeResponse(self.text)


def test_fred_client_parses_csv_and_drops_missing_values():
    session = FakeSession(FRED_TEXT)
    series = FredClient(session=session).fetch("DEXJPUS")
    assert session.urls == ["https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS"]
    assert series.name == "DEXJPUS"
    assert series.tolist() == [141.5, 143.0]


def test_fred_client_rejects_unexpected_responses():
    with pytest.raises(ValueError):
        FredClient(session=FakeSession("<html>blocked</html>")).fetch("DEXJPUS")


def write(store, series_id, dates, values):
    store.save(pd.Series(values, index=pd.to_datetime(dates), name=series_id))


def test_store_round_trip_and_spot_direction(tmp_path):
    store = FxDataStore(tmp_path)
    jpy, gbp = by_code("JPY"), by_code("GBP")
    write(store, jpy.spot_series, ["2024-01-02", "2024-01-03"], [150.0, 160.0])
    write(store, gbp.spot_series, ["2024-01-02", "2024-01-03"], [1.25, 1.30])
    spot = store.spot((jpy, gbp))
    assert spot["JPY"].tolist() == pytest.approx([1 / 150, 1 / 160])  # yen per dollar -> dollars per yen
    assert spot["GBP"].tolist() == pytest.approx([1.25, 1.30])


def test_monthly_rates_fill_short_gaps_but_not_ragged_ends(tmp_path):
    store = FxDataStore(tmp_path)
    gbp = by_code("GBP")
    write(store, "USDRATE", ["2024-01-01", "2024-03-01", "2024-04-01"], [5.0, 4.0, 3.0])  # February missing
    write(store, gbp.rate_series, ["2024-01-01", "2024-02-01"], [4.5, 4.4])  # ends in February
    rates = store.monthly_rates((gbp,), usd_series="USDRATE")
    assert [str(m) for m in rates.index] == ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert rates["USD"].tolist() == pytest.approx([0.05, 0.05, 0.04, 0.03])
    assert rates["GBP"].iloc[:2].tolist() == pytest.approx([0.045, 0.044])
    assert rates["GBP"].iloc[2:].isna().all()


def test_filled_rate_months_reports_the_fill(tmp_path):
    store = FxDataStore(tmp_path)
    gbp = by_code("GBP")
    write(store, "IR3TIB01USM156N", ["2024-01-01", "2024-03-01"], [5.0, 4.0])
    write(store, gbp.rate_series, ["2024-01-01", "2024-02-01", "2024-03-01"], [4.5, 4.4, 4.3])
    report = store.filled_rate_months((gbp,))
    assert report.to_dict("records") == [{"series": "USD", "month": "2024-02"}]


def test_missing_file_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        FxDataStore(tmp_path).series("DEXUSUK")


def test_to_continuous_matches_the_deposit_growth():
    # 1 month at 5% ACT/360 grows by 1 + 0.05 * (1/12) * 365/360
    r = to_continuous(0.05, 360, 1 / 12)
    assert np.exp(r / 12) == pytest.approx(1 + 0.05 / 12 * 365 / 360)
