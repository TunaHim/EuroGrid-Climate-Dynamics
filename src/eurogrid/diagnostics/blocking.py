"""Tibaldi–Molteni blocking diagnostics for canonical 500 hPa fields.

The diagnostic follows the original three-band idea: at each longitude, a
blocking candidate has a positive south gradient between roughly 40°N and
60°N and a strongly negative north gradient between roughly 60°N and 80°N.
The three bands are shifted together by a small latitude offset to avoid
making the result depend on one exact grid row.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import xarray as xr

from eurogrid.contract import ContractError

Z500_DIMS: Final[tuple[str, str, str]] = ("time_z500", "lat", "lon")


def _require_z500(z500: xr.DataArray) -> None:
    if tuple(z500.dims) != Z500_DIMS:
        raise ContractError(f"z500 dimensions must be {Z500_DIMS}, got {z500.dims}")
    if z500.attrs.get("units") != "m":
        raise ContractError(f"z500 units must be 'm', got {z500.attrs.get('units')!r}")
    lat = np.asarray(z500["lat"].values, dtype=float)
    lon = np.asarray(z500["lon"].values, dtype=float)
    if lat.size < 3 or lon.size < 2:
        raise ContractError("z500 needs at least three latitudes and two longitudes")
    if not np.all(np.diff(lat) < 0):
        raise ContractError("z500 latitude must be strictly decreasing")
    if not np.all(np.diff(lon) > 0):
        raise ContractError("z500 longitude must be strictly increasing")


def _at_latitude(z500: xr.DataArray, latitude: float) -> xr.DataArray:
    values = np.asarray(z500["lat"].values, dtype=float)
    index = int(np.argmin(np.abs(values - latitude)))
    spacing = float(np.min(np.abs(np.diff(values))))
    if not np.isclose(values[index], latitude, atol=spacing + 1e-6):
        raise ContractError(f"no grid latitude is close to requested {latitude:g}°N")
    return z500.isel(lat=index)


def meridional_gradients(
    z500: xr.DataArray,
    *,
    phi0_deg: float = 60.0,
    arm_deg: float = 20.0,
    scan_offsets_deg: tuple[float, ...] = (-4.0, 0.0, 4.0),
) -> xr.Dataset:
    """Compute south and north height gradients for shifted TM1990 bands."""
    _require_z500(z500)
    records: list[xr.Dataset] = []
    for offset in scan_offsets_deg:
        center = phi0_deg + offset
        south = _at_latitude(z500, center - arm_deg)
        middle = _at_latitude(z500, center)
        north = _at_latitude(z500, center + arm_deg)
        records.append(
            xr.Dataset(
                {
                    "ghgs": (middle - south) / arm_deg,
                    "ghgn": (north - middle) / arm_deg,
                }
            ).expand_dims({"scan_latitude": [center]})
        )
    gradients = xr.concat(records, dim="scan_latitude")
    gradients["ghgs"].attrs = {
        "units": "m degree-1",
        "long_name": "south meridional geopotential-height gradient",
    }
    gradients["ghgn"].attrs = {
        "units": "m degree-1",
        "long_name": "north meridional geopotential-height gradient",
    }
    gradients.attrs.update({"phi0_deg": phi0_deg, "arm_deg": arm_deg})
    return gradients


def _retain_longitude_sectors(
    candidate: np.ndarray,
    lon: np.ndarray,
    min_sector_width_deg: float,
) -> np.ndarray:
    """Retain contiguous candidate runs whose coordinate span is wide enough."""
    qualified = np.zeros_like(candidate, dtype=bool)
    for time_index, row in enumerate(candidate):
        starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
        ends = np.flatnonzero(row & ~np.r_[row[1:], False])
        for start, end in zip(starts, ends, strict=True):
            if lon[end] - lon[start] >= min_sector_width_deg:
                qualified[time_index, start : end + 1] = True
    return qualified


def detect_tm1990(
    z500: xr.DataArray,
    *,
    phi0_deg: float = 60.0,
    arm_deg: float = 20.0,
    scan_offsets_deg: tuple[float, ...] = (-4.0, 0.0, 4.0),
    ghgn_threshold_m_per_deg: float = -10.0,
    min_sector_width_deg: float = 20.0,
) -> xr.Dataset:
    """Return TM1990 gradients, candidate longitudes, and sector masks."""
    gradients = meridional_gradients(
        z500,
        phi0_deg=phi0_deg,
        arm_deg=arm_deg,
        scan_offsets_deg=scan_offsets_deg,
    )
    candidate_by_latitude = (gradients["ghgs"] > 0) & (gradients["ghgn"] < ghgn_threshold_m_per_deg)
    candidate = candidate_by_latitude.any("scan_latitude")
    blocking = xr.DataArray(
        _retain_longitude_sectors(
            candidate.values,
            np.asarray(z500["lon"].values, dtype=float),
            min_sector_width_deg,
        ),
        dims=("time_z500", "lon"),
        coords={"time_z500": z500["time_z500"], "lon": z500["lon"]},
        name="blocking",
        attrs={
            "units": "1",
            "long_name": "TM1990 sector-qualified blocking longitude mask",
        },
    )
    result = gradients.assign(
        candidate=candidate,
        candidate_by_latitude=candidate_by_latitude,
        blocking=blocking,
    )
    result["candidate"].attrs = {
        "units": "1",
        "long_name": "TM1990 gradient-qualified candidate longitude mask",
    }
    result["candidate_by_latitude"].attrs = {
        "units": "1",
        "long_name": "candidate mask by scanned central latitude",
    }
    result.attrs.update(
        {
            "ghgn_threshold_m_per_deg": ghgn_threshold_m_per_deg,
            "min_sector_width_deg": min_sector_width_deg,
        }
    )
    return result


def persistence_summary(
    blocking: xr.DataArray,
    *,
    timestep_hours: float,
    min_persistence_days: float = 5.0,
) -> xr.Dataset:
    """Summarise longitude coverage and domain-wide persistence."""
    if tuple(blocking.dims) != ("time_z500", "lon"):
        raise ContractError("blocking must have dimensions ('time_z500', 'lon')")
    min_steps = int(np.ceil(min_persistence_days * 24.0 / timestep_hours))
    present = blocking.any("lon")
    values = present.values.astype(bool)
    run_lengths = np.zeros(values.size, dtype=int)
    start = 0
    while start < values.size:
        if not values[start]:
            start += 1
            continue
        end = start
        while end < values.size and values[end]:
            end += 1
        run_lengths[start:end] = end - start
        start = end
    persistent = xr.DataArray(
        values & (run_lengths >= min_steps),
        dims=("time_z500",),
        coords={"time_z500": blocking["time_z500"]},
        name="persistent_blocking",
        attrs={"units": "1", "long_name": "domain-wide blocking persistence mask"},
    )
    return xr.Dataset(
        {
            "blocking_longitude_count": blocking.sum("lon"),
            "blocking_longitude_fraction": blocking.mean("lon"),
            "persistent_blocking": persistent,
        },
        attrs={
            "minimum_persistence_days": min_persistence_days,
            "minimum_persistence_steps": min_steps,
            "timestep_hours": timestep_hours,
        },
    )
