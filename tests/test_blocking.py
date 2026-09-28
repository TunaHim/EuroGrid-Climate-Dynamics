import numpy as np
import xarray as xr

from eurogrid.diagnostics.blocking import (
    detect_tm1990,
    meridional_gradients,
    persistence_summary,
)


def _height_field(blocking: bool = True) -> xr.DataArray:
    lat = np.arange(85.0, 29.0, -0.25)
    lon = np.arange(-30.0, 30.25, 0.25)
    time = np.arange("2000-01-01T00", "2000-01-06T00", dtype="datetime64[6h]")
    values = np.empty((time.size, lat.size, lon.size), dtype=float)
    for index, latitude in enumerate(lat):
        values[:, index, :] = 5300.0 + 5.0 * latitude
    if blocking:
        ridge = (lat <= 64.0) & (lat >= 56.0)
        sector = (lon >= -10.0) & (lon <= 10.0)
        for index in np.flatnonzero(ridge):
            values[:, index, sector] += 600.0
    return xr.DataArray(
        values,
        dims=("time_z500", "lat", "lon"),
        coords={"time_z500": time, "lat": lat, "lon": lon},
        attrs={"units": "m"},
        name="z500",
    )


def test_meridional_gradients_have_expected_dimensions():
    gradients = meridional_gradients(_height_field())
    assert gradients.sizes["scan_latitude"] == 3
    assert gradients.sizes["lon"] == 241
    assert gradients["ghgs"].attrs["units"] == "m degree-1"


def test_tm1990_detects_sector_qualified_pattern():
    result = detect_tm1990(_height_field(), min_sector_width_deg=5.0)
    assert bool(result["candidate"].isel(time_z500=0, lon=120))
    assert bool(result["blocking"].isel(time_z500=0, lon=120))


def test_persistence_requires_minimum_consecutive_steps():
    result = detect_tm1990(_height_field(), min_sector_width_deg=5.0)
    summary = persistence_summary(result["blocking"], timestep_hours=6, min_persistence_days=5)
    assert bool(summary["persistent_blocking"].all())
    assert summary.attrs["minimum_persistence_steps"] == 20


def test_nonblocking_pattern_does_not_trigger():
    result = detect_tm1990(_height_field(blocking=False), min_sector_width_deg=5.0)
    assert not bool(result["blocking"].any())
