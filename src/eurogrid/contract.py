"""Canonical gridded-dataset contract shared by every loader and diagnostic.

Loaders (ERA5, EERIE) normalise their source into this shape; diagnostics
accept only this shape and call :func:`validate_canonical` on entry. The
contract is the seam that makes "run the same diagnostic on a different
source" true.

Contract
--------
* Dimensions, in order: ``(time, lat, lon)`` (plus any extra leading dims a
  loader documents, e.g. ``level``).
* ``lat``: strictly decreasing, in degrees north, within [-90, 90].
* ``lon``: strictly increasing, in degrees east, within [-180, 180).
* ``time``: ``datetime64[ns]`` on the proleptic Gregorian calendar.
* Variables use the names and units in :data:`VARIABLES`.
* ``ds.attrs["source"]`` names the loader that produced the dataset.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import xarray as xr

DIMS: Final[tuple[str, str, str]] = ("time", "lat", "lon")

#: Canonical variable name -> required ``units`` attribute.
VARIABLES: Final[dict[str, str]] = {
    "u100": "m s-1",  # eastward wind at (or proxy for) 100 m
    "v100": "m s-1",  # northward wind at (or proxy for) 100 m
    "ws100": "m s-1",  # wind speed at (or proxy for) 100 m
    "z500": "m",  # 500 hPa geopotential *height* (geopotential / g)
    "z_level": "m",  # geopotential height of the proxy pressure level
    "sp": "Pa",  # surface pressure
    "below_ground": "1",  # 1 where the proxy level is underground, else 0
}

#: Accepted values of ``ds.attrs["source"]``.
SOURCES: Final[frozenset[str]] = frozenset({"era5", "eerie", "synthetic"})

#: Standard gravity used to convert geopotential to geopotential height.
G0: Final[float] = 9.80665


class ContractError(ValueError):
    """Raised when a dataset violates the canonical contract."""


def _strictly(values: np.ndarray, increasing: bool) -> bool:
    if values.size < 2:
        return True
    d = np.diff(values.astype("float64"))
    return bool(np.all(d > 0) if increasing else np.all(d < 0))


def validate_canonical(ds: xr.Dataset) -> xr.Dataset:
    """Assert ``ds`` satisfies the contract; return it unchanged.

    Raises :class:`ContractError` with a message naming the first violation.
    Coordinates are inspected eagerly (they are small); data variables are
    not loaded.
    """
    for dim in DIMS:
        if dim not in ds.dims:
            raise ContractError(f"missing dimension {dim!r}; have {tuple(ds.dims)}")
        if dim not in ds.coords:
            raise ContractError(f"dimension {dim!r} has no coordinate")

    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ContractError("lat and lon must be 1-D coordinates")
    if not _strictly(lat, increasing=False):
        raise ContractError("lat must be strictly decreasing (90 -> -90)")
    if lat.min() < -90 or lat.max() > 90:
        raise ContractError("lat outside [-90, 90]")
    if not _strictly(lon, increasing=True):
        raise ContractError(
            "lon must be strictly increasing; EERIE gr025 rows run 0..180 then "
            "-180..0 and need roll_longitude() first"
        )
    if lon.min() < -180 or lon.max() >= 180:
        raise ContractError("lon outside [-180, 180)")

    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        raise ContractError(f"time must be datetime64, got {ds['time'].dtype}")

    for name, var in ds.data_vars.items():
        if name not in VARIABLES:
            raise ContractError(f"unknown variable {name!r}; allowed: {sorted(VARIABLES)}")
        expected = VARIABLES[name]
        units = var.attrs.get("units")
        if units != expected:
            raise ContractError(f"{name}: units={units!r}, expected {expected!r}")
        trailing = tuple(var.dims)[-3:]
        if trailing != DIMS:
            raise ContractError(f"{name}: trailing dims {trailing} != {DIMS}")

    source = ds.attrs.get("source")
    if source not in SOURCES:
        raise ContractError(f"attrs['source']={source!r} not in {sorted(SOURCES)}")
    return ds


def wrap_longitude(lon: np.ndarray) -> np.ndarray:
    """Map longitudes to [-180, 180)."""
    return ((np.asarray(lon, dtype="float64") + 180.0) % 360.0) - 180.0


def roll_longitude(obj: xr.Dataset | xr.DataArray, lon: str = "lon") -> xr.Dataset | xr.DataArray:
    """Return ``obj`` with ``lon`` wrapped to [-180, 180) and strictly increasing.

    Works on any ordering (0..360, or EERIE's 0..180 then -180..0). Lazy on
    dask-backed data: only an index permutation is applied.
    """
    wrapped = wrap_longitude(obj[lon].values)
    order = np.argsort(wrapped, kind="stable")
    if not _strictly(wrapped[order], increasing=True):
        raise ContractError("duplicate longitudes after wrapping")
    out = obj.isel({lon: order})
    return out.assign_coords({lon: wrapped[order]})


def reshape_flat_grid(
    da: xr.DataArray,
    lat_flat: np.ndarray,
    lon_flat: np.ndarray,
    flat_dim: str = "value",
) -> xr.DataArray:
    """Unflatten a ``(..., value)`` array into ``(..., lat, lon)``.

    EERIE ``gr025`` stores publish regular-grid fields as a single flattened
    dimension with 1-D ``lat``/``lon`` companions. The flattening is verified
    to be C-order with latitude as the slow axis; the function raises if the
    companions do not describe such a layout. The resulting ``lon`` is *not*
    necessarily monotonic — call :func:`roll_longitude` afterwards.
    """
    lat_flat = np.asarray(lat_flat)
    lon_flat = np.asarray(lon_flat)
    n = da.sizes[flat_dim]
    if lat_flat.shape != (n,) or lon_flat.shape != (n,):
        raise ContractError("lat/lon companions must be 1-D with the flat dim's length")

    lat_axis = np.unique(lat_flat)[::-1]  # descending
    nlat = lat_axis.size
    if n % nlat:
        raise ContractError("flat length not divisible by number of latitudes")
    nlon = n // nlat

    lat2d = lat_flat.reshape(nlat, nlon)
    lon2d = lon_flat.reshape(nlat, nlon)
    if not np.all(lat2d == lat2d[:, :1]):
        raise ContractError("latitude is not constant along rows: not C-order (lat, lon)")
    if not np.all(lon2d == lon2d[:1, :]):
        raise ContractError("longitude rows differ: not a regular grid")
    if not _strictly(lat2d[:, 0], increasing=False):
        raise ContractError("row latitudes not strictly decreasing")

    other = [d for d in da.dims if d != flat_dim]
    data = da.transpose(*other, flat_dim).data
    reshaped = data.reshape(*[da.sizes[d] for d in other], nlat, nlon)
    coords = {d: da.coords[d] for d in other if d in da.coords}
    coords["lat"] = ("lat", lat2d[:, 0])
    coords["lon"] = ("lon", lon2d[0, :])
    return xr.DataArray(reshaped, dims=(*other, "lat", "lon"), coords=coords, attrs=da.attrs, name=da.name)


def subset_domain(
    obj: xr.Dataset | xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset | xr.DataArray:
    """Select a lat/lon box on a contract-conforming object (lat descending)."""
    return obj.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
