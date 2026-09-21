import numpy as np
import pytest
import xarray as xr

from eurogrid.contract import (
    ContractError,
    reshape_flat_grid,
    roll_longitude,
    subset_domain,
    validate_canonical,
    wrap_longitude,
)


def _canonical(nt=4, nlat=5, nlon=8):
    lat = np.linspace(70, 30, nlat)
    lon = np.linspace(-30, 40, nlon, endpoint=False)
    time = np.array(["2023-01-01T00"], dtype="datetime64[ns]") + np.arange(nt) * np.timedelta64(6, "h")
    ds = xr.Dataset(
        {
            "u100": (("time", "lat", "lon"), np.zeros((nt, nlat, nlon)), {"units": "m s-1"}),
            "z500": (("time", "lat", "lon"), np.full((nt, nlat, nlon), 5500.0), {"units": "m"}),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"source": "synthetic"},
    )
    return ds


def test_canonical_passes():
    ds = _canonical()
    assert validate_canonical(ds) is ds


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda ds: ds.isel(lat=slice(None, None, -1)), "strictly decreasing"),
        (lambda ds: ds.assign_coords(lon=ds.lon.values + 200), r"lon outside|roll_longitude"),
        (lambda ds: ds.rename(u100="wind"), "unknown variable"),
        (lambda ds: ds.assign(u100=ds.u100.assign_attrs(units="knots")), "units"),
        (lambda ds: ds.transpose("lat", "time", "lon"), "trailing dims"),
        (lambda ds: ds.assign_attrs(source="model"), "source"),
        (lambda ds: ds.assign_coords(time=np.arange(ds.sizes["time"])), "datetime64"),
    ],
)
def test_canonical_rejects(mutate, match):
    with pytest.raises(ContractError, match=match):
        validate_canonical(mutate(_canonical()))


def test_wrap_longitude():
    np.testing.assert_allclose(wrap_longitude([0, 179.75, 180, 359.75, -180]), [0, 179.75, -180, -0.25, -180])


def _eerie_like_lonlat(nlat=5, nlon=8):
    """EERIE gr025 layout: C-order (lat, lon), lat 90->-90, lon 0..180 then -180..0."""
    dlon = 360 / nlon
    lon_row = np.concatenate([np.arange(0, 180, dlon), np.arange(-180, 0, dlon)])
    lat_col = np.linspace(90, -90, nlat)
    lat_flat = np.repeat(lat_col, nlon)
    lon_flat = np.tile(lon_row, nlat)
    return lat_col, lon_row, lat_flat, lon_flat


def test_reshape_flat_grid_recovers_2d_field():
    nt, nlat, nlon = 2, 5, 8
    lat_col, lon_row, lat_flat, lon_flat = _eerie_like_lonlat(nlat, nlon)
    # Field encodes its own position so we can check the unflattening.
    field2d = lat_col[:, None] * 1000 + lon_row[None, :]
    flat = np.broadcast_to(field2d.reshape(-1), (nt, nlat * nlon))
    da = xr.DataArray(flat, dims=("time", "value"), coords={"time": np.arange(nt)}, name="u")

    out = reshape_flat_grid(da, lat_flat, lon_flat)
    assert out.dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(out.lat, lat_col)
    np.testing.assert_array_equal(out.lon, lon_row)
    np.testing.assert_array_equal(out.isel(time=0), field2d)


def test_reshape_rejects_non_c_order():
    nlat, nlon = 5, 8
    lat_col, lon_row, lat_flat, lon_flat = _eerie_like_lonlat(nlat, nlon)
    da = xr.DataArray(np.zeros(nlat * nlon), dims=("value",))
    # Fortran-order companions: lon slow, lat fast.
    with pytest.raises(ContractError, match="C-order"):
        reshape_flat_grid(da, np.tile(lat_col, nlon), np.repeat(lon_row, nlat))


def test_roll_longitude_makes_eerie_order_monotonic_and_keeps_values():
    nlat, nlon = 5, 8
    lat_col, lon_row, lat_flat, lon_flat = _eerie_like_lonlat(nlat, nlon)
    field2d = lat_col[:, None] * 1000 + lon_row[None, :]
    da = xr.DataArray(field2d, dims=("lat", "lon"), coords={"lat": lat_col, "lon": lon_row})

    rolled = roll_longitude(da)
    assert np.all(np.diff(rolled.lon) > 0)
    assert rolled.lon[0] == -180
    # Every value still sits at the lon it was labelled with.
    np.testing.assert_array_equal(
        rolled.values % 1000, np.broadcast_to(rolled.lon.values % 1000, rolled.shape)
    )


def test_sel_on_unrolled_eerie_lon_is_wrong_but_rolled_is_right():
    """Regression for the silent-garbage failure mode the contract exists to prevent."""
    nlat, nlon = 3, 16
    lat_col, lon_row, lat_flat, lon_flat = _eerie_like_lonlat(nlat, nlon)
    da = xr.DataArray(
        np.broadcast_to(lon_row, (nlat, nlon)).copy(),
        dims=("lat", "lon"),
        coords={"lat": lat_col, "lon": lon_row},
    )
    # Both labels exist, so no KeyError: the slice silently spans the wrap and is empty.
    naive = da.sel(lon=slice(-22.5, 22.5))
    assert naive.sizes["lon"] == 0
    correct = subset_domain(roll_longitude(da), -90, 90, -22.5, 22.5)
    np.testing.assert_array_equal(correct.lon, [-22.5, 0, 22.5])


def test_roll_is_lazy_on_dask():
    dask = pytest.importorskip("dask.array")
    nlat, nlon = 5, 8
    lat_col, lon_row, *_ = _eerie_like_lonlat(nlat, nlon)
    da = xr.DataArray(
        dask.zeros((nlat, nlon), chunks=(nlat, nlon)),
        dims=("lat", "lon"),
        coords={"lat": lat_col, "lon": lon_row},
    )
    rolled = roll_longitude(da)
    assert isinstance(rolled.data, dask.Array)
