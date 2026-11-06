from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import xarray as xr

from eurogrid.models.transfer import (
    FLEET_BOXES,
    PowerCurve,
    fleet_wind_speed,
    modelled_capacity_factor,
    validation_metrics,
)


def era5_ds(ws_field: np.ndarray, hours: int = 24) -> xr.Dataset:
    """Canonical-era5-shaped dataset with a given ws100 field (time, lat, lon)."""
    lat = np.arange(75, 29.9, -0.25)
    lon = np.arange(-40, 40, 0.25)
    time = np.datetime64("2023-01-01") + np.arange(hours) * np.timedelta64(1, "h")
    comp = np.sqrt(ws_field**2 / 2).astype("float32")
    return xr.Dataset(
        {
            "u100": (("time", "lat", "lon"), comp, {"units": "m s-1"}),
            "v100": (("time", "lat", "lon"), comp, {"units": "m s-1"}),
            "ws100": (("time", "lat", "lon"), ws_field.astype("float32"), {"units": "m s-1"}),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"source": "era5"},
    )


def tidy_frame(zone: str, variable: str, values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [datetime(2023, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(len(values))],
            "zone": zone,
            "variable": variable,
            "value_mw": values,
        }
    )


def model_frame(variable: str, cf: float | list[float], n: int = 24) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [datetime(2023, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)],
            "variable": variable,
            "cf_model": cf,
        }
    )


def test_power_curve_shape():
    c = PowerCurve()
    assert c(0.0) == c(2.9) == 0.0
    assert c(26.0) == 0.0
    assert c(12.0) == pytest.approx(1.0)
    assert 0 < c(7.0) < 1
    assert c(np.array([6.0]))[0] == pytest.approx((216 - 27) / (1728 - 27))


def test_fleet_wind_speed_area_weighted():
    ws = np.full((24, 181, 320), 10.0)
    out = fleet_wind_speed(era5_ds(ws), FLEET_BOXES["wind_onshore"])
    assert out.sizes["time"] == 24
    np.testing.assert_allclose(out.values, 10.0)
    with pytest.raises(ValueError):
        fleet_wind_speed(era5_ds(ws), (60, 70, 100, 120))


def test_modelled_cf_uses_variable_curves():
    ws = np.full((24, 181, 320), 14.0)
    on = modelled_capacity_factor(era5_ds(ws), "wind_onshore")
    off = modelled_capacity_factor(era5_ds(ws), "wind_offshore")
    assert on["cf_model"].unique().to_list() == [1.0]  # 14 > rated 12
    assert off["cf_model"].unique().to_list() == [1.0]  # 14 > rated 13
    assert on["time"].dtype == pl.Datetime(time_zone="UTC")
    with pytest.raises(ValueError):
        modelled_capacity_factor(era5_ds(ws), "solar")


def test_validation_metrics_perfect_match():
    n = 24
    pattern = [0.5, 0.6, 0.4, 0.7] * 6
    mod = model_frame("wind_onshore", pattern, n)
    gen = tidy_frame("50Hertz", "wind_onshore", [v * 1000.0 for v in pattern])
    cap = tidy_frame("50Hertz", "wind_onshore", [1000.0] * n)
    m = validation_metrics(mod, gen, cap, "50Hertz")
    assert m["n"] == n and m["corr"] == pytest.approx(1.0)
    assert m["rmse_mw"] == pytest.approx(0.0) and m["bias_mw"] == pytest.approx(0.0)
    assert m["cf_obs"] == pytest.approx(0.55) and m["cf_model"] == pytest.approx(0.55)


def test_validation_metrics_bias_direction():
    mod = model_frame("wind_onshore", 0.8)
    gen = tidy_frame("TenneT", "wind_onshore", [400.0] * 24)
    cap = tidy_frame("TenneT", "wind_onshore", [1000.0] * 24)
    m = validation_metrics(mod, gen, cap, "TenneT")
    assert m["bias_mw"] == pytest.approx(400.0)  # model overpredicts
    assert m["cf_model"] - m["cf_obs"] == pytest.approx(0.4)
