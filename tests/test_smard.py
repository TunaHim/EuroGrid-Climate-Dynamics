from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from eurogrid.data.smard import (
    CAPACITY_FILTERS,
    GENERATION_FILTERS,
    SmardClient,
    SmardError,
    capacity_factor,
    week_timestamps,
)

WEEK_MS = 7 * 24 * 3600 * 1000
# Mondays 00:00 CET expressed in UTC ms, as SMARD publishes them.
W0 = int(datetime(2022, 12, 25, 23, tzinfo=UTC).timestamp() * 1000)
WEEKS = [W0 + i * WEEK_MS for i in range(6)]


class FakeSmard:
    """Serves a SMARD-shaped JSON API from memory and counts hits."""

    def __init__(self, hourly_value=100.0):
        self.hits: list[str] = []
        self.hourly_value = hourly_value

    def __call__(self, url: str):
        self.hits.append(url)
        parts = url.rsplit("/", 3)  # base, filter, region, file
        fid, region, fname = int(parts[1]), parts[2], parts[3]
        if fid == GENERATION_FILTERS["wind_offshore"] and region in {"Amprion", "TransnetBW"}:
            return None  # SMARD 404s here
        if fname.startswith("index_"):
            return {"timestamps": WEEKS}
        resolution, ts = fname[:-5].split("_")[-2:]
        step = 15 if resolution == "quarterhour" else 60
        n = 7 * 24 * 60 // step
        t0 = int(ts)
        series = [[t0 + i * step * 60_000, self.hourly_value * step / 60] for i in range(n)]
        series[5][1] = None  # one gap, as in the real feed
        if fid in CAPACITY_FILTERS.values():
            series = [[t, 1000.0] for t, _ in series]
        return {"series": series, "meta_data": {"version": 1}}


@pytest.fixture
def client(tmp_path):
    fake = FakeSmard()
    return SmardClient(cache_dir=tmp_path, fetch_json=fake, pause_s=0), fake


def test_week_timestamps_overlap():
    start, end = datetime(2023, 1, 10), datetime(2023, 1, 20)
    # Files start Sun 23:00 UTC: 01-08 and 01-15 overlap [01-10, 01-20); 01-01's ends 01-08 23:00.
    assert week_timestamps(WEEKS, start, end) == [WEEKS[2], WEEKS[3]]
    # UTC midnight on Monday falls one hour into the local week, so the previous file is not needed.
    assert week_timestamps(WEEKS, datetime(2023, 1, 2), datetime(2023, 1, 3)) == [WEEKS[1]]


def test_generation_is_mw_and_trimmed(client):
    c, fake = client
    start, end = datetime(2023, 1, 2), datetime(2023, 1, 9)
    gen = c.load_generation(start, end, zones=["50Hertz"], variables=["wind_onshore"])
    assert gen.columns == ["time", "zone", "variable", "value_mw"]
    assert gen.height == 7 * 24
    assert gen["time"].min() >= datetime(2023, 1, 2, tzinfo=UTC)
    assert gen["time"].max() < datetime(2023, 1, 9, tzinfo=UTC)
    assert gen["value_mw"].drop_nulls().unique().to_list() == [100.0]
    assert gen["value_mw"].null_count() == 1


def test_quarterhour_scaled_to_mw(client):
    c, _ = client
    gen = c.load_generation(
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
        zones=["TenneT"],
        variables=["solar"],
        resolution="quarterhour",
    )
    assert gen.height == 96
    assert gen["value_mw"].drop_nulls().unique().to_list() == [100.0]  # 25 MWh / 15 min == 100 MW


def test_offshore_zero_for_inland_zones(client):
    c, _ = client
    gen = c.load_generation(datetime(2023, 1, 2), datetime(2023, 1, 3), variables=["wind_offshore"])
    by_zone = gen.group_by("zone").agg(pl.col("value_mw").sum()).sort("zone")
    assert dict(zip(by_zone["zone"], by_zone["value_mw"], strict=True)) == {
        "50Hertz": 2300.0,  # 24 h x 100 MW minus the one null hour
        "Amprion": 0.0,
        "TenneT": 2300.0,
        "TransnetBW": 0.0,
    }
    assert gen.filter(pl.col("zone") == "Amprion").height == 24


def test_missing_non_offshore_series_is_an_error(tmp_path):
    c = SmardClient(cache_dir=tmp_path, fetch_json=lambda url: None, pause_s=0)
    with pytest.raises(SmardError):
        c.load_generation(datetime(2023, 1, 2), datetime(2023, 1, 3), zones=["50Hertz"], variables=["solar"])


def test_weekly_files_are_cached(client):
    c, fake = client
    args = (datetime(2023, 1, 2), datetime(2023, 1, 16), ["Amprion"], ["solar"])
    c.load_generation(*args)
    n_first = len(fake.hits)
    # UTC window spans local weeks starting 01-02, 01-09 and 01-16 (the last covers 23:00-00:00 UTC).
    assert sum("index_" not in u for u in fake.hits) == 3
    c.load_generation(*args)
    assert all("index_" in u for u in fake.hits[n_first:])  # only the index is re-read


def test_capacity_factor_join(client):
    c, _ = client
    s, e = datetime(2023, 1, 2), datetime(2023, 1, 3)
    gen = c.load_generation(s, e, zones=["50Hertz"], variables=["wind_onshore"])
    cap = c.load_installed_capacity(s, e, zones=["50Hertz"], variables=["wind_onshore"])
    cf = capacity_factor(gen, cap)
    assert cf["capacity_mw"].unique().to_list() == [1000.0]
    assert cf["capacity_factor"].drop_nulls().unique().to_list() == [0.1]


def test_capacity_factor_null_where_no_capacity(client):
    c, _ = client
    s, e = datetime(2023, 1, 2), datetime(2023, 1, 3)
    gen = c.load_generation(s, e, zones=["Amprion"], variables=["wind_offshore"])
    cap = c.load_installed_capacity(s, e, zones=["Amprion"], variables=["wind_offshore"])
    cf = capacity_factor(gen, cap)
    assert cf["capacity_factor"].null_count() == cf.height
    assert not cf["capacity_factor"].is_nan().any()


def test_bad_region_rejected(client):
    c, _ = client
    with pytest.raises(ValueError):
        c.fetch_series(4067, "Bavaria", datetime(2023, 1, 1), datetime(2023, 1, 1) + timedelta(days=1))
