import numpy as np
import xarray as xr

from eurogrid.diagnostics.blocking import (
    blocked_day_frequency,
    daily_mean_z500,
    detect_tm1990,
    meridional_gradients,
    persistence_summary,
)


def _height_field(blocking: bool = True) -> xr.DataArray:
    lat = np.arange(85.0, 29.0, -0.25)
    lon = np.arange(-30.0, 30.25, 0.25)
    time = np.arange(np.datetime64("2000-01-01T00"), np.datetime64("2000-01-06T00"), np.timedelta64(6, "h"))
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
    assert bool(summary["persistent_blocking"].isel(lon=120).all())
    # every flagged cell was already sector-blocked -> funnel is monotonic
    assert bool((summary["persistent_blocking"] <= result["blocking"]).all())
    assert summary.attrs["minimum_persistence_steps"] == 20


def test_persistence_tolerance_allows_longitude_drift():
    # blocked ridge drifts 6 deg east mid-run: a strict per-longitude test
    # splits it, the +/-10 deg tolerance keeps one event
    z500 = _height_field()
    lon = z500["lon"].values
    values = z500.values.copy()
    values[:, :, :] = 0.0
    for index, latitude in enumerate(z500["lat"].values):
        values[:, index, :] = 5300.0 + 5.0 * latitude
    ridge = (z500["lat"].values <= 64.0) & (z500["lat"].values >= 56.0)
    first = (lon >= -10.0) & (lon <= 10.0)
    second = (lon >= -4.0) & (lon <= 16.0)
    for index in np.flatnonzero(ridge):
        values[:10, index, first] += 600.0
        values[10:, index, second] += 600.0
    drifting = xr.DataArray(values, dims=z500.dims, coords=z500.coords, attrs=z500.attrs, name="z500")
    result = detect_tm1990(drifting, min_sector_width_deg=5.0)
    summary = persistence_summary(
        result["blocking"], timestep_hours=6, min_persistence_days=4, tolerance_deg=10.0
    )
    second_only = second & ~first
    assert bool(summary["persistent_blocking"].isel(time_z500=15).sel(lon=second_only).all())
    summary_zero = persistence_summary(
        result["blocking"], timestep_hours=6, min_persistence_days=4, tolerance_deg=0.0
    )
    tail = summary_zero["persistent_blocking"].isel(time_z500=15).sel(lon=second_only)
    assert not bool(tail.all())  # strict test splits the drifting run


def test_daily_mean_and_blocked_day_frequency():
    z500 = _height_field()
    daily = daily_mean_z500(z500)
    assert daily.sizes["time_z500"] == 5
    assert np.all(np.diff(daily["time_z500"].values.astype("datetime64[D]")) == 1)
    result = detect_tm1990(daily, min_sector_width_deg=5.0)
    freq = blocked_day_frequency(result["blocking"])
    assert freq.sel(lon=0.0).item() == 100.0
    assert freq.sel(lon=-25.0).item() == 0.0
    assert freq.attrs["units"] == "%"


def test_nonblocking_pattern_does_not_trigger():
    result = detect_tm1990(_height_field(blocking=False), min_sector_width_deg=5.0)
    assert not bool(result["blocking"].any())
