from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import xarray as xr

from eurogrid.models.transfer import (
    FLEET_BOXES,
    PowerCurve,
    fleet_box,
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


def model_frame(zone: str, variable: str, cf: float | list[float], n: int = 24) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [datetime(2023, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)],
            "zone": zone,
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


def test_fleet_box_zone_aware():
    assert fleet_box("wind_offshore", "50Hertz") == (54.0, 55.2, 12.0, 15.0)  # Baltic
    assert fleet_box("wind_offshore", "TenneT") == (53.5, 56.0, 3.0, 8.5)  # North Sea
    assert fleet_box("wind_onshore", "Amprion") == fleet_box("wind_onshore", "50Hertz")
    with pytest.raises(ValueError):
        fleet_box("wind_offshore", "Amprion")  # southern zones have no offshore fleet
    with pytest.raises(ValueError):
        fleet_box("solar", "50Hertz")


def test_fleet_wind_speed_area_weighted():
    ws = np.full((24, 181, 320), 10.0)
    out = fleet_wind_speed(era5_ds(ws), FLEET_BOXES[("wind_onshore", "50Hertz")])
    assert out.sizes["time"] == 24
    np.testing.assert_allclose(out.values, 10.0)
    with pytest.raises(ValueError):
        fleet_wind_speed(era5_ds(ws), (60, 70, 100, 120))


def test_modelled_cf_uses_variable_curves():
    ws = np.full((24, 181, 320), 14.0)
    on = modelled_capacity_factor(era5_ds(ws), "wind_onshore", "50Hertz")
    off = modelled_capacity_factor(era5_ds(ws), "wind_offshore", "50Hertz")
    assert on["cf_model"].unique().to_list() == pytest.approx([1.0])  # 14 > rated 12
    assert off["cf_model"].unique().to_list() == pytest.approx([1.0])  # 14 > rated 13
    assert on["time"].dtype == pl.Datetime(time_zone="UTC")
    assert on["zone"].unique().to_list() == ["50Hertz"]
    with pytest.raises(ValueError):
        modelled_capacity_factor(era5_ds(ws), "solar", "50Hertz")
    with pytest.raises(ValueError):
        modelled_capacity_factor(era5_ds(ws), "wind_offshore", "Amprion")
    with pytest.raises(ValueError):
        modelled_capacity_factor(era5_ds(ws), "wind_onshore", "50Hertz", aggregation="x")


def test_field_aggregation_beats_box_mean_across_cut_in():
    # Cells straddling the 3 m/s cut-in: P(mean u) = 0 but mean P(u) > 0
    ws = np.full((24, 181, 320), 4.0)
    ws[:, :, :160] = 2.0  # half the cells below cut-in, mean u = 3.0 at cut-in
    box = FLEET_BOXES[("wind_onshore", "50Hertz")]
    field = modelled_capacity_factor(era5_ds(ws), "wind_onshore", "50Hertz", aggregation="field")
    box_mean = modelled_capacity_factor(era5_ds(ws), "wind_onshore", "50Hertz", aggregation="box_mean")
    assert (field["cf_model"] >= box_mean["cf_model"]).all()
    assert field["cf_model"].mean() > box_mean["cf_model"].mean()
    assert set(box) == set(fleet_box("wind_onshore", "50Hertz"))


def test_loss_factor_scales_linearly():
    ws = np.full((24, 181, 320), 10.0)
    full = modelled_capacity_factor(era5_ds(ws), "wind_onshore", "50Hertz")
    half = modelled_capacity_factor(era5_ds(ws), "wind_onshore", "50Hertz", loss_factor=0.5)
    np.testing.assert_allclose(half["cf_model"].to_numpy(), full["cf_model"].to_numpy() * 0.5)
    lit = modelled_capacity_factor(era5_ds(ws), "wind_onshore", "50Hertz", loss_factor=0.88)
    np.testing.assert_allclose(lit["cf_model"].to_numpy(), full["cf_model"].to_numpy() * 0.88)
    with pytest.raises(ValueError):
        modelled_capacity_factor(era5_ds(ws), "wind_onshore", "50Hertz", loss_factor=1.1)


def test_validation_metrics_perfect_match():
    n = 24
    pattern = [0.5, 0.6, 0.4, 0.7] * 6
    mod = model_frame("50Hertz", "wind_onshore", pattern, n)
    gen = tidy_frame("50Hertz", "wind_onshore", [v * 1000.0 for v in pattern])
    cap = tidy_frame("50Hertz", "wind_onshore", [1000.0] * n)
    m = validation_metrics(mod, gen, cap, "50Hertz")
    assert m["n"] == n and m["corr"] == pytest.approx(1.0)
    assert m["rmse_mw"] == pytest.approx(0.0) and m["bias_mw"] == pytest.approx(0.0)
    assert m["cf_obs"] == pytest.approx(0.55) and m["cf_model"] == pytest.approx(0.55)
    assert m["bias_cf"] == pytest.approx(0.0)
    assert m["rmse_cf"] == pytest.approx(0.0)
    assert m["rmse_cf_rel"] == pytest.approx(0.0)
    assert m["low_wind_hit_rate"] != m["low_wind_hit_rate"]  # no sub-0.10 hours -> nan


def test_validation_metrics_low_wind_hit_rate():
    pattern = [0.05, 0.08, 0.5] * 8
    mod = model_frame("50Hertz", "wind_onshore", pattern, 24)
    gen = tidy_frame("50Hertz", "wind_onshore", [v * 1000.0 for v in pattern])
    cap = tidy_frame("50Hertz", "wind_onshore", [1000.0] * 24)
    m = validation_metrics(mod, gen, cap, "50Hertz")
    assert m["low_wind_hit_rate"] == pytest.approx(1.0)
    assert m["low_wind_far"] == pytest.approx(0.0)
    assert m["low_wind_csi"] == pytest.approx(1.0)


def test_validation_metrics_low_wind_false_alarms():
    # model predicts calm on 3 of 4 obs-calm hours and calls calm on 6 hours
    # total: hit rate 3/4, FAR 1 - 3/6, CSI 3/7
    obs_pattern = [0.05, 0.08, 0.09, 0.07] + [0.5] * 20
    mod_pattern = [0.05, 0.08, 0.09, 0.5] + [0.05, 0.06, 0.07] + [0.5] * 17
    mod = model_frame("50Hertz", "wind_onshore", mod_pattern, 24)
    gen = tidy_frame("50Hertz", "wind_onshore", [v * 1000.0 for v in obs_pattern])
    cap = tidy_frame("50Hertz", "wind_onshore", [1000.0] * 24)
    m = validation_metrics(mod, gen, cap, "50Hertz")
    assert m["low_wind_hit_rate"] == pytest.approx(0.75)
    assert m["low_wind_far"] == pytest.approx(0.5)
    assert m["low_wind_csi"] == pytest.approx(3 / 7)


def test_validation_metrics_bias_direction():
    mod = model_frame("TenneT", "wind_onshore", 0.8)
    gen = tidy_frame("TenneT", "wind_onshore", [400.0] * 24)
    cap = tidy_frame("TenneT", "wind_onshore", [1000.0] * 24)
    m = validation_metrics(mod, gen, cap, "TenneT")
    assert m["bias_mw"] == pytest.approx(400.0)  # model overpredicts
    assert m["cf_model"] - m["cf_obs"] == pytest.approx(0.4)
    assert m["bias_cf"] == pytest.approx(0.4)
    assert m["rmse_cf"] == pytest.approx(0.4)
    assert m["rmse_cf_rel"] == pytest.approx(1.0)
