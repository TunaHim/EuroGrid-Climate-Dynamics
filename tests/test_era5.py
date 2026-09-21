from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from eurogrid.contract import G0, validate_canonical
from eurogrid.data.era5 import load_era5, normalise


def fake_arco(hours: int = 48) -> xr.Dataset:
    """A tiny stand-in with ARCO-ERA5's layout: lat 90->-90, lon 0->359.75, padded time axis."""
    lat = np.arange(90, -90.1, -10.0)
    lon = np.arange(0, 360, 10.0)
    time = np.datetime64("2023-01-01") + np.arange(hours) * np.timedelta64(1, "h")
    level = np.array([300, 500, 1000])
    shape = (hours, lat.size, lon.size)
    # u = longitude wrapped to [-180,180); lets the test see that rolling kept values with coords.
    u = np.broadcast_to(((lon + 180) % 360 - 180)[None, None, :], shape).astype("float32")
    v = np.full(shape, 3.0, dtype="float32")
    sp = np.full(shape, 101325.0, dtype="float32")
    z = np.broadcast_to((level * 100.0)[None, :, None, None], (hours, 3, lat.size, lon.size)).astype(
        "float32"
    )
    ds = xr.Dataset(
        {
            "100m_u_component_of_wind": (("time", "latitude", "longitude"), u),
            "100m_v_component_of_wind": (("time", "latitude", "longitude"), v),
            "surface_pressure": (("time", "latitude", "longitude"), sp),
            "geopotential": (("time", "level", "latitude", "longitude"), z),
        },
        coords={"time": time, "latitude": lat, "longitude": lon, "level": level},
        attrs={"valid_time_stop": "2023-01-05"},
    )
    return ds.chunk({"time": 1})


BOX = dict(lat_min=30, lat_max=75, lon_min=-40, lon_max=40)


def test_normalise_meets_contract_and_rolls_longitude():
    ds = normalise(fake_arco(), z500_stride_hours=6)
    validate_canonical(ds.drop_vars("z500"))
    assert ds.lon.values[0] == -180 and np.all(np.diff(ds.lon.values) > 0)
    # value travelled with its coordinate through the roll
    np.testing.assert_allclose(ds["u100"].isel(time=0, lat=0).values, ds.lon.values)
    np.testing.assert_allclose(ds["ws100"].isel(time=0, lat=0, lon=0), np.hypot(-180, 3))
    assert ds.sizes["time_z500"] == 48 // 6
    np.testing.assert_allclose(ds["z500"].isel(time_z500=0, lat=0, lon=0), 500 * 100 / G0)


def test_load_era5_crops_and_caches(tmp_path):
    store = fake_arco()
    s, e = datetime(2023, 1, 1), datetime(2023, 1, 2)
    ds = load_era5(s, e, **BOX, cache_dir=tmp_path, store=store)
    assert ds.attrs["source"] == "era5"
    assert ds.sizes["time"] == 24 and ds.sizes["time_z500"] == 4  # half-open, strided
    assert ds.lat.values.max() <= 75 and ds.lat.values.min() >= 30
    assert ds.lon.values.min() >= -40 and ds.lon.values.max() <= 40
    assert set(ds.data_vars) == {"u100", "v100", "ws100", "sp", "z500"}
    zarrs = list(tmp_path.glob("*.zarr"))
    assert len(zarrs) == 1

    # Second call must not touch the store at all.
    poisoned = store.isel(time=slice(0, 0))
    again = load_era5(s, e, **BOX, cache_dir=tmp_path, store=poisoned)
    xr.testing.assert_allclose(again["ws100"].load(), ds["ws100"].load())


def test_load_era5_rejects_out_of_range(tmp_path):
    store = fake_arco()
    with pytest.raises(ValueError, match="valid_time_stop"):
        load_era5(datetime(2023, 1, 1), datetime(2023, 3, 1), **BOX, cache_dir=tmp_path, store=store)
    with pytest.raises(ValueError, match="no ERA5 timesteps"):
        load_era5(datetime(2023, 1, 3), datetime(2023, 1, 4), **BOX, cache_dir=tmp_path, store=store)
