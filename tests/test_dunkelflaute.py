from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import xarray as xr

from eurogrid.diagnostics.dunkelflaute import (
    capacity_weights,
    conditional_drought_probability,
    daily_mean,
    detect_events,
    event_days,
    event_verification,
    smard_renewable_cf,
    solar_capacity_factor,
    threshold_duration_matrix,
)


def hourly_frame(cfs):
    times = pl.datetime_range(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=len(cfs) - 1),
        "1h",
        eager=True,
    )
    return pl.DataFrame({"time": times, "cf": cfs})


def test_capacity_weights_splits_offshore_by_basin():
    cap = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1, tzinfo=UTC)] * 6,
            "zone": ["50Hertz", "50Hertz", "TenneT", "TenneT", "Amprion", "TransnetBW"],
            "variable": [
                "wind_onshore",
                "wind_offshore",
                "wind_onshore",
                "wind_offshore",
                "solar",
                "solar",
            ],
            "value_mw": [10000.0, 1000.0, 20000.0, 8000.0, 30000.0, 4000.0],
        }
    )
    w = capacity_weights(cap)
    assert w["wind_onshore"] == 30000.0
    assert w["wind_offshore_baltic"] == 1000.0
    assert w["wind_offshore_north_sea"] == 8000.0
    assert w["solar"] == 34000.0


def test_solar_capacity_factor_clips_and_scales():
    ssrd = xr.DataArray(
        np.array([0.0, 500.0, 1000.0, 1200.0]),
        dims=("x",),
        attrs={"units": "W m-2"},
    )
    cf = solar_capacity_factor(ssrd)
    np.testing.assert_allclose(cf.values, [0.0, 0.5, 1.0, 1.0])


def test_smard_renewable_cf_capacity_weighted():
    gen = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            "zone": ["50Hertz", "TenneT"],
            "variable": ["wind_onshore", "solar"],
            "value_mw": [3000.0, 1000.0],
        }
    )
    cap = gen.with_columns(pl.lit(10000.0).alias("value_mw"))
    out = smard_renewable_cf(gen, cap)
    assert out["cf"].item() == pytest.approx((3000.0 + 1000.0) / 20000.0)


def test_detect_events_min_duration():
    # 3 days low (72 h), 1 day high, 5 days low (120 h)
    cfs = [0.05] * 72 + [0.5] * 24 + [0.08] * 120
    frame = hourly_frame(cfs)
    events = detect_events(frame, threshold=0.10, min_duration_hours=48.0)
    assert events.height == 2
    assert events["duration_hours"].to_list() == [72.0, 120.0]
    events72 = detect_events(frame, threshold=0.10, min_duration_hours=96.0)
    assert events72.height == 1
    assert detect_events(frame, threshold=0.10, min_duration_hours=24.0).height == 2


def test_threshold_duration_matrix_shape():
    frame = hourly_frame([0.05] * 72 + [0.5] * 24)
    m = threshold_duration_matrix(frame, [0.10, 0.20], [24, 48])
    assert m.height == 4
    assert m.filter(pl.col("threshold_cf") == 0.10, pl.col("min_duration_hours") == 24)["events"].item() == 1


def test_event_verification_overlap():
    obs = pl.DataFrame(
        {
            "start": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)],
            "end": [datetime(2024, 1, 3, tzinfo=UTC), datetime(2024, 2, 3, tzinfo=UTC)],
        }
    )
    mod = pl.DataFrame(
        {
            "start": [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC)],
            "end": [datetime(2024, 1, 4, tzinfo=UTC), datetime(2024, 3, 2, tzinfo=UTC)],
        }
    )
    out = event_verification(mod, obs)
    assert out == {
        "observed_events": 2,
        "modelled_events": 2,
        "hits": 1,
        "misses": 1,
        "false_alarms": 1,
    }


def test_conditional_drought_probability():
    days = pl.datetime_range(
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC), "1d", eager=True
    )
    frame = pl.DataFrame({"time": days, "cf": [0.05, 0.5, 0.05, 0.5, 0.05]})
    blocked = {days[0].date(), days[2].date(), days[3].date()}
    out = conditional_drought_probability(frame, 0.10, blocked)
    assert out["p_drought"] == pytest.approx(3 / 5)
    assert out["p_drought_given_blocked"] == pytest.approx(2 / 3)


def test_daily_mean_resamples():
    frame = hourly_frame([0.1] * 24 + [0.9] * 24)
    daily = daily_mean(frame)
    assert daily.height == 2
    assert daily["cf"].to_list() == pytest.approx([0.1, 0.9])


def daily_frame(dates, cfs):
    return pl.DataFrame({"time": [datetime(*d, tzinfo=UTC) for d in dates], "cf": cfs})


def test_detect_events_does_not_bridge_gap_between_winters():
    # Low at the end of one winter and the start of the next: two events, not one.
    dates = [(2021, 2, 27), (2021, 2, 28), (2021, 12, 1), (2021, 12, 2)]
    ev = detect_events(daily_frame(dates, [0.05, 0.05, 0.05, 0.05]), 0.10, 48)
    assert ev.height == 2
    assert ev["duration_hours"].to_list() == [48.0, 48.0]


def test_detect_events_gap_inside_run_splits_it():
    dates = [(2024, 1, 1), (2024, 1, 2), (2024, 1, 4), (2024, 1, 5)]
    assert detect_events(daily_frame(dates, [0.05] * 4), 0.10, 72).height == 0


def test_event_days_covers_every_day_of_each_event():
    dates = [(2024, 1, d) for d in range(1, 8)]
    ev = detect_events(daily_frame(dates, [0.2, 0.05, 0.05, 0.05, 0.2, 0.05, 0.05]), 0.10, 48)
    days = {d.day for d in event_days(ev)}
    assert days == {2, 3, 4, 6, 7}


def test_smard_renewable_cf_ignores_capacity_of_missing_generation():
    t = datetime(2024, 1, 1, tzinfo=UTC)
    gen = pl.DataFrame(
        {
            "time": [t, t],
            "zone": ["TenneT", "Amprion"],
            "variable": ["wind_onshore", "wind_onshore"],
            "value_mw": [5000.0, None],
        }
    )
    cap = pl.DataFrame(
        {
            "time": [t, t],
            "zone": ["TenneT", "Amprion"],
            "variable": ["wind_onshore", "wind_onshore"],
            "value_mw": [10000.0, 10000.0],
        }
    )
    assert smard_renewable_cf(gen, cap)["cf"].item() == pytest.approx(0.5)
