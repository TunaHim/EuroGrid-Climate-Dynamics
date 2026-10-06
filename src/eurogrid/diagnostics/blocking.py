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


def _longitude_dilation(values: np.ndarray, half_width: int) -> np.ndarray:
    """OR each column with its ``half_width`` neighbours along axis 1 (no wrap)."""
    cum = np.cumsum(values.astype(np.int64), axis=1)
    nlon = values.shape[1]
    padded = np.concatenate([np.zeros((values.shape[0], 1), dtype=np.int64), cum], axis=1)
    lo = np.clip(np.arange(nlon) - half_width, 0, nlon)
    hi = np.clip(np.arange(nlon) + half_width + 1, 0, nlon)
    return (padded[:, hi] - padded[:, lo]) > 0


def _persistence_run_lengths(active: np.ndarray) -> np.ndarray:
    """Per-cell length of the consecutive-True run along axis 0 (vectorised)."""
    out = np.zeros(active.shape, dtype=np.int64)
    for col in range(active.shape[1]):
        series = active[:, col]
        padded = np.r_[False, series, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for start, end in zip(starts, ends, strict=True):
            out[start:end, col] = end - start
    return out


def persistence_summary(
    blocking: xr.DataArray,
    *,
    timestep_hours: float,
    min_persistence_days: float = 5.0,
    tolerance_deg: float = 10.0,
) -> xr.Dataset:
    """Longitude coverage plus persistence evaluated *per longitude*.

    A longitude is called persistent while it sits inside an unbroken run
    (>= ``min_persistence_days``) during which some longitude within
    +/- ``tolerance_deg`` was blocked every step. The tolerance lets a
    slow-drifting block count as one event instead of several. Because the
    persistence flag is only applied where the sector mask is already True,
    candidate >= sector >= persistent cell fractions share one denominator
    and the funnel is monotonic.
    """
    if tuple(blocking.dims) != ("time_z500", "lon"):
        raise ContractError("blocking must have dimensions ('time_z500', 'lon')")
    lon = np.asarray(blocking["lon"].values, dtype=float)
    dlon = float(np.min(np.abs(np.diff(lon))))
    half_width = int(np.round(tolerance_deg / dlon))
    min_steps = int(np.ceil(min_persistence_days * 24.0 / timestep_hours))

    blocked = np.asarray(blocking.values, dtype=bool)
    neighbourhood = _longitude_dilation(blocked, half_width)
    run_lengths = _persistence_run_lengths(neighbourhood)
    persistent_values = blocked & (run_lengths >= min_steps)
    persistent = xr.DataArray(
        persistent_values,
        dims=("time_z500", "lon"),
        coords={"time_z500": blocking["time_z500"], "lon": blocking["lon"]},
        name="persistent_blocking",
        attrs={"units": "1", "long_name": "per-longitude blocking persistence mask"},
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
            "longitude_tolerance_deg": tolerance_deg,
        },
    )


def daily_mean_z500(z500: xr.DataArray) -> xr.DataArray:
    """Daily-mean Z500: TM1990 is a daily index, so 6-hourly input is averaged.

    Daily means remove the within-day cycle and make one timestep equal one
    day, so the 5-day persistence test is on actual days.
    """
    if "time_z500" not in z500.dims:
        raise ContractError("z500 must carry a 'time_z500' dimension")
    z500 = z500.reset_index("time_z500").assign_coords(
        time_z500=z500["time_z500"].values.astype("datetime64[ns]")
    )
    daily = z500.resample(time_z500="1D").mean()
    daily.attrs = dict(z500.attrs)
    daily.attrs["temporal_aggregation"] = "daily mean"
    return daily


def blocked_day_frequency(blocking: xr.DataArray) -> xr.DataArray:
    """Blocked-day frequency in % at each longitude (mean over time)."""
    if "lon" not in blocking.dims:
        raise ContractError("blocking must carry a 'lon' dimension")
    freq = blocking.mean("time_z500") * 100.0
    return freq.rename("blocked_day_frequency").assign_attrs(units="%", long_name="blocked-day frequency")
