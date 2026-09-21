"""SMARD (Bundesnetzagentur) client: German generation and installed capacity by TSO zone.

Endpoint layout (undocumented but stable, used by smard.de itself)::

    {base}/{filter}/{region}/index_{resolution}.json
        -> {"timestamps": [ms, ...]}          one entry per weekly file (Mon 00:00 local)
    {base}/{filter}/{region}/{filter}_{region}_{resolution}_{ts}.json
        -> {"series": [[ms, value|null], ...], "meta_data": {...}}

Values are energy per interval (MWh); at hourly resolution that equals mean MW.
Generation filter ids are documented in the bundesAPI OpenAPI spec. The installed
capacity ids (186/187/188) are *not* documented; they were identified by matching
Germany-wide yearly values against known capacity (2015: 37.7 GW onshore,
1.0 GW offshore, 37.3 GW PV; 2024: 59.8 / 8.5 / 76.6 GW).

Offshore wind is only published for 50Hertz and TenneT: the endpoint returns
404 for Amprion and TransnetBW, which this module maps to an explicit zero series.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl

log = logging.getLogger(__name__)

BASE_URL: Final = "https://www.smard.de/app/chart_data"

ZONES: Final[tuple[str, ...]] = ("50Hertz", "TenneT", "Amprion", "TransnetBW")
OFFSHORE_ZONES: Final[frozenset[str]] = frozenset({"50Hertz", "TenneT"})
REGIONS: Final[frozenset[str]] = frozenset({"DE", *ZONES})

GENERATION_FILTERS: Final[dict[str, int]] = {
    "wind_onshore": 4067,
    "wind_offshore": 1225,
    "solar": 4068,
}
CAPACITY_FILTERS: Final[dict[str, int]] = {
    "wind_onshore": 186,
    "wind_offshore": 187,
    "solar": 188,
}
RESOLUTIONS: Final[dict[str, int]] = {"quarterhour": 15, "hour": 60}

JsonFetcher = Callable[[str], object]
"""Callable returning parsed JSON for a URL, or ``None`` on HTTP 404."""


class SmardError(RuntimeError):
    pass


def http_fetch_json(url: str, retries: int = 5, timeout: float = 30.0) -> object | None:
    """Default fetcher: urllib with exponential backoff; 404 -> None."""
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code < 500 and e.code != 429:
                raise SmardError(f"{url}: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise SmardError(f"{url}: {e}") from e
        time.sleep(delay)
        delay *= 2
    raise SmardError(f"{url}: gave up after {retries} attempts")


def _ms_to_utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _to_utc(t: datetime) -> datetime:
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def week_timestamps(index_ts: Iterable[int], start: datetime, end: datetime) -> list[int]:
    """Weekly file timestamps whose 7-day window overlaps ``[start, end)``."""
    start, end = _to_utc(start), _to_utc(end)
    ts = sorted(index_ts)
    out = []
    for i, ms in enumerate(ts):
        t0 = _ms_to_utc(ms)
        # A file runs until the next one starts; the last file gets a DST-tolerant week.
        t1 = _ms_to_utc(ts[i + 1]) if i + 1 < len(ts) else t0 + timedelta(days=7, hours=1)
        if t0 < end and t1 > start:
            out.append(ms)
    return out


class SmardClient:
    """Fetch SMARD series into tidy Polars frames, caching each weekly file as parquet."""

    def __init__(
        self,
        cache_dir: str | Path = "data/smard",
        fetch_json: JsonFetcher = http_fetch_json,
        base_url: str = BASE_URL,
        pause_s: float = 0.2,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.fetch_json = fetch_json
        self.base_url = base_url
        self.pause_s = pause_s

    # -- low level --------------------------------------------------------

    def _index(self, filter_id: int, region: str, resolution: str) -> list[int] | None:
        data = self.fetch_json(f"{self.base_url}/{filter_id}/{region}/index_{resolution}.json")
        if data is None:
            return None
        return list(data["timestamps"])

    def _week(self, filter_id: int, region: str, resolution: str, ts: int) -> pl.DataFrame:
        path = self.cache_dir / f"{filter_id}_{region}_{resolution}_{ts}.parquet"
        if path.exists():
            return pl.read_parquet(path)
        url = f"{self.base_url}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts}.json"
        data = self.fetch_json(url)
        if data is None:
            raise SmardError(f"missing weekly file {url}")
        series = data["series"]
        df = pl.DataFrame(
            {
                "time": [_ms_to_utc(int(row[0])) for row in series],
                "value": [None if row[1] is None else float(row[1]) for row in series],
            },
            schema={"time": pl.Datetime("us", "UTC"), "value": pl.Float64},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        if self.pause_s:
            time.sleep(self.pause_s)
        return df

    def fetch_series(
        self,
        filter_id: int,
        region: str,
        start: datetime,
        end: datetime,
        resolution: str = "hour",
    ) -> pl.DataFrame | None:
        """Raw series (``time`` UTC, ``value`` MWh per interval) for ``[start, end)``.

        Returns ``None`` when SMARD publishes nothing for this filter/region (HTTP 404 or empty index).
        """
        if region not in REGIONS:
            raise ValueError(f"unknown region {region!r}")
        if resolution not in RESOLUTIONS:
            raise ValueError(f"unknown resolution {resolution!r}")
        index = self._index(filter_id, region, resolution)
        if not index:  # 404, or an index with no files (offshore capacity for inland zones)
            return None
        weeks = week_timestamps(index, start, end)
        if not weeks:
            raise SmardError(f"no weekly files for filter {filter_id} {region} in [{start}, {end})")
        frames = [self._week(filter_id, region, resolution, ts) for ts in weeks]
        df = pl.concat(frames).unique(subset="time").sort("time")
        return df.filter((pl.col("time") >= _to_utc(start)) & (pl.col("time") < _to_utc(end)))

    # -- tidy loaders -----------------------------------------------------

    def _tidy(
        self,
        filters: dict[str, int],
        variables: Iterable[str],
        zones: Iterable[str],
        start: datetime,
        end: datetime,
        resolution: str,
        to_mw: bool,
    ) -> pl.DataFrame:
        per_hour = 60 / RESOLUTIONS[resolution]
        frames: list[pl.DataFrame] = []
        template: pl.DataFrame | None = None
        pending_zero: list[tuple[str, str]] = []
        for var in variables:
            fid = filters[var]
            for zone in zones:
                if var == "wind_offshore" and zone not in OFFSHORE_ZONES:
                    pending_zero.append((var, zone))  # structurally zero: no coastline
                    continue
                raw = self.fetch_series(fid, zone, start, end, resolution)
                if raw is None:
                    raise SmardError(f"SMARD has no data for {var} in {zone}")
                value = pl.col("value") * per_hour if to_mw else pl.col("value")
                df = raw.select(
                    pl.col("time"),
                    pl.lit(zone).alias("zone"),
                    pl.lit(var).alias("variable"),
                    value.alias("value_mw"),
                )
                frames.append(df)
                template = template if template is not None else df.select("time")
        if template is None:  # only structurally-zero series requested: synthesise the time axis
            template = pl.DataFrame(
                {
                    "time": pl.datetime_range(
                        _to_utc(start), _to_utc(end), f"{RESOLUTIONS[resolution]}m", closed="left", eager=True
                    )
                }
            )
        for var, zone in pending_zero:
            frames.append(
                template.with_columns(
                    pl.lit(zone).alias("zone"),
                    pl.lit(var).alias("variable"),
                    pl.lit(0.0).alias("value_mw"),
                )
            )
        return pl.concat(frames).sort(["variable", "zone", "time"])

    def load_generation(
        self,
        start: datetime,
        end: datetime,
        zones: Iterable[str] = ZONES,
        variables: Iterable[str] = tuple(GENERATION_FILTERS),
        resolution: str = "hour",
    ) -> pl.DataFrame:
        """Long frame ``time, zone, variable, value_mw`` of mean feed-in power (MW)."""
        return self._tidy(GENERATION_FILTERS, variables, zones, start, end, resolution, to_mw=True)

    def load_installed_capacity(
        self,
        start: datetime,
        end: datetime,
        zones: Iterable[str] = ZONES,
        variables: Iterable[str] = tuple(CAPACITY_FILTERS),
        resolution: str = "hour",
    ) -> pl.DataFrame:
        """Long frame ``time, zone, variable, value_mw`` of installed capacity (MW, step-wise)."""
        return self._tidy(CAPACITY_FILTERS, variables, zones, start, end, resolution, to_mw=False)


def capacity_factor(generation: pl.DataFrame, capacity: pl.DataFrame) -> pl.DataFrame:
    """Join generation and capacity on (time, zone, variable) and add ``capacity_factor``.

    Null (not NaN) where installed capacity is zero, e.g. offshore in inland zones.
    """
    cap = capacity.rename({"value_mw": "capacity_mw"})
    cf = pl.when(pl.col("capacity_mw") > 0).then(pl.col("value_mw") / pl.col("capacity_mw"))
    return generation.join(cap, on=["time", "zone", "variable"], how="left").with_columns(
        cf.alias("capacity_factor")
    )


def _main() -> None:  # pragma: no cover - thin CLI
    import argparse

    p = argparse.ArgumentParser(description="Populate the SMARD parquet cache.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="exclusive")
    p.add_argument("--cache-dir", default="data/smard")
    p.add_argument("--resolution", default="hour", choices=sorted(RESOLUTIONS))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    c = SmardClient(a.cache_dir)
    s, e = datetime.fromisoformat(a.start), datetime.fromisoformat(a.end)
    gen = c.load_generation(s, e, resolution=a.resolution)
    cap = c.load_installed_capacity(s, e, resolution=a.resolution)
    cf = capacity_factor(gen, cap)
    print(cf.group_by(["variable", "zone"]).agg(pl.col("capacity_factor").mean()).sort(["variable", "zone"]))


if __name__ == "__main__":
    _main()
