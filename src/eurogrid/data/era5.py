"""ERA5 loader: ARCO-ERA5 (Google Cloud, anonymous) -> canonical dataset, cached locally.

Store facts verified 2026-09-21 against
``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``:

* One store carries everything we need: ``100m_u_component_of_wind``,
  ``100m_v_component_of_wind``, ``surface_pressure`` (2-D) and ``geopotential``
  on 37 pressure levels (3-D). Hourly, 0.25 deg, ``latitude`` 90 -> -90,
  ``longitude`` 0 -> 359.75 (needs :func:`~eurogrid.contract.roll_longitude`).
* The time axis is padded to 2050; ``attrs["valid_time_stop"]`` marks the
  real end (2026-06-30 at verification). Requests past it are rejected here.
* Chunks are one timestep x the whole globe: ``(1, 721, 1440)`` for 2-D
  fields, ``(1, 37, 721, 1440)`` for 3-D. Spatial subsetting does not
  reduce transfer, and *every hour of Z500 costs a full 37-level chunk* -
  so Z500 is pulled at a coarser stride (6 h default; TM1990 is a daily index).

Output variables follow :mod:`eurogrid.contract`: ``u100``, ``v100``, ``ws100``,
``sp`` (hourly) and ``z500`` (strided), all on ``(time, lat, lon)`` cropped to
the domain. Each call is persisted once to a local Zarr under ``cache_dir`` and
re-opened from there afterwards.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr

from eurogrid.contract import G0, roll_longitude, subset_domain, validate_canonical

log = logging.getLogger(__name__)

ARCO_ERA5_URL: Final = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

#: ARCO name -> canonical name for the 2-D (single-level) fields.
SINGLE_LEVEL: Final[dict[str, str]] = {
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "surface_pressure": "sp",
}
GEOPOTENTIAL: Final = "geopotential"
Z500_LEVEL_HPA: Final = 500


def open_arco_era5(url: str = ARCO_ERA5_URL) -> xr.Dataset:
    """Open the ARCO-ERA5 store lazily with anonymous GCS access."""
    return xr.open_zarr(url, storage_options={"token": "anon"}, consolidated=True, chunks={})


def _check_time_window(ds: xr.Dataset, start: datetime, end: datetime) -> None:
    stop = ds.attrs.get("valid_time_stop")
    if stop is not None and np.datetime64(end) > np.datetime64(stop) + np.timedelta64(1, "D"):
        raise ValueError(f"requested end {end} is after the store's valid_time_stop {stop}")
    if start >= end:
        raise ValueError("start must be before end")


def normalise(raw: xr.Dataset, z500_stride_hours: int) -> xr.Dataset:
    """Map a raw ARCO-ERA5 subset onto the canonical contract (lazy).

    ``raw`` must contain the :data:`SINGLE_LEVEL` fields and ``geopotential``
    with a ``level`` dimension, on ``(time, latitude, longitude)``.
    """
    ds = raw.rename({"latitude": "lat", "longitude": "lon"})
    out = xr.Dataset(attrs={"source": "era5"})
    for src, dst in SINGLE_LEVEL.items():
        out[dst] = ds[src].transpose("time", "lat", "lon")
    out["u100"].attrs["units"] = out["v100"].attrs["units"] = "m s-1"
    out["sp"].attrs["units"] = "Pa"
    out["ws100"] = np.hypot(out["u100"], out["v100"])
    out["ws100"].attrs["units"] = "m s-1"

    z = ds[GEOPOTENTIAL].sel(level=Z500_LEVEL_HPA, drop=True)
    z = z.isel(time=slice(None, None, z500_stride_hours)).transpose("time", "lat", "lon")
    z500 = (z / G0).rename(time="time_z500")
    z500.attrs = {"units": "m", "long_name": "500 hPa geopotential height", "stride_hours": z500_stride_hours}
    out["z500"] = z500
    for v in out.data_vars:
        out[v].attrs.setdefault("long_name", v)
    return roll_longitude(out)


def _validate(ds: xr.Dataset) -> xr.Dataset:
    # z500 lives on its own strided time axis; validate it separately against the same contract.
    hourly = ds.drop_vars("z500").drop_dims("time_z500", errors="ignore")
    validate_canonical(hourly)
    validate_canonical(ds[["z500"]].rename(time_z500="time"))
    return ds


def load_era5(
    start: datetime,
    end: datetime,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cache_dir: str | Path = "data/era5",
    z500_stride_hours: int = 6,
    store: xr.Dataset | None = None,
) -> xr.Dataset:
    """Canonical ERA5 dataset for ``[start, end)`` over a lat/lon box, cached as local Zarr.

    ``store`` lets tests inject an in-memory stand-in for the ARCO dataset.
    The cache key covers window, box and stride, so a different request never
    reads a stale file.
    """
    cache_dir = Path(cache_dir)
    key = (
        f"era5_{start:%Y%m%dT%H}_{end:%Y%m%dT%H}"
        f"_lat{lat_min:g}_{lat_max:g}_lon{lon_min:g}_{lon_max:g}_z{z500_stride_hours}h.zarr"
    )
    path = cache_dir / key
    if path.exists():
        log.info("era5 cache hit %s", path)
        return _validate(xr.open_zarr(path, consolidated=True))

    raw = store if store is not None else open_arco_era5()
    _check_time_window(raw, start, end)
    store_name = "injected" if store is not None else ARCO_ERA5_URL
    raw = raw[[*SINGLE_LEVEL, GEOPOTENTIAL]].sel(time=slice(start, end), level=[Z500_LEVEL_HPA])
    raw = raw.sel(time=raw.time < np.datetime64(end))  # half-open window
    if raw.sizes["time"] == 0:
        raise ValueError(f"no ERA5 timesteps in [{start}, {end})")
    ds = subset_domain(normalise(raw, z500_stride_hours), lat_min, lat_max, lon_min, lon_max)
    ds.attrs["store"] = store_name
    _validate(ds)

    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("era5 fetching %d hourly steps -> %s", ds.sizes["time"], path)
    ds = ds.chunk({"time": 24, "time_z500": 24 // z500_stride_hours})
    for var in ds.variables:
        ds[var].encoding = {}  # otherwise the store's (1, 721, 1440) chunking is carried over
    ds.to_zarr(path, mode="w", consolidated=True)
    return _validate(xr.open_zarr(path, consolidated=True))


def load_era5_z500(
    start: datetime,
    end: datetime,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    cache_dir: str | Path = "data/era5",
    z500_stride_hours: int = 6,
    store: xr.Dataset | None = None,
) -> xr.DataArray:
    """Load only Z500 for blocking studies and persist it as a local Zarr.

    This avoids downloading the hourly wind and surface-pressure fields when
    a blocking benchmark needs only the 500 hPa geopotential-height field.
    """
    cache_dir = Path(cache_dir)
    key = (
        f"era5_z500_{start:%Y%m%dT%H}_{end:%Y%m%dT%H}"
        f"_lat{lat_min:g}_{lat_max:g}_lon{lon_min:g}_{lon_max:g}_z{z500_stride_hours}h.zarr"
    )
    path = cache_dir / key
    if path.exists():
        return xr.open_zarr(path, consolidated=True)["z500"]

    raw = store if store is not None else open_arco_era5()
    _check_time_window(raw, start, end)
    raw = raw[["geopotential"]].sel(
        time=slice(start, end),
        level=[Z500_LEVEL_HPA],
    )
    raw = raw.sel(time=raw.time < np.datetime64(end))
    if raw.sizes["time"] == 0:
        raise ValueError(f"no ERA5 timesteps in [{start}, {end})")

    z500 = raw["geopotential"].sel(level=Z500_LEVEL_HPA, drop=True)
    z500 = z500.isel(time=slice(None, None, z500_stride_hours))
    z500 = (z500.rename({"latitude": "lat", "longitude": "lon"}) / G0).rename(time="time_z500")
    z500 = z500.transpose("time_z500", "lat", "lon")
    z500.attrs = {
        "units": "m",
        "long_name": "500 hPa geopotential height",
        "stride_hours": z500_stride_hours,
    }
    z500 = subset_domain(
        roll_longitude(z500),
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )
    _validate_z500(z500)

    cache_dir.mkdir(parents=True, exist_ok=True)
    z500 = z500.chunk({"time_z500": 24})
    dataset = z500.to_dataset(name="z500")
    dataset.attrs["source"] = "era5"
    for var in dataset.variables:
        dataset[var].encoding = {}
    dataset.to_zarr(path, mode="w", consolidated=True)
    return xr.open_zarr(path, consolidated=True)["z500"]


def _validate_z500(z500: xr.DataArray) -> xr.DataArray:
    """Validate a standalone Z500 array against the canonical contract."""
    dataset = z500.rename(time_z500="time").to_dataset(name="z500")
    dataset.attrs["source"] = "era5"
    validate_canonical(dataset)
    return z500


def _main() -> None:  # pragma: no cover - thin CLI
    import argparse

    import yaml

    p = argparse.ArgumentParser(description="Populate the ERA5 local Zarr cache for a window.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="exclusive")
    p.add_argument("--settings", default="config/settings.yaml")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open(a.settings) as f:
        cfg = yaml.safe_load(f)
    ds = load_era5(
        datetime.fromisoformat(a.start),
        datetime.fromisoformat(a.end),
        **cfg["domain"],
        cache_dir=cfg["era5"]["cache_dir"],
        z500_stride_hours=cfg["era5"]["z500_stride_hours"],
    )
    print(ds)


if __name__ == "__main__":
    _main()
