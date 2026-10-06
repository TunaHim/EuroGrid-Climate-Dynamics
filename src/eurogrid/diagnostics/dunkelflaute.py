"""Dunkelflaute (renewable drought) diagnostics.

A Dunkelflaute is a sustained period of low wind-plus-solar output. Because
solar is zero every night, hourly thresholds mostly count the diurnal
cycle; the literature therefore works on daily means or multi-day running
means (e.g. Kittel & Schill 2024, Li et al. 2021). This module keeps the
definition transparent: a national capacity-weighted renewable capacity
factor on daily means, an event = consecutive days below a threshold.

The national index weights technologies by SMARD installed capacity -
onshore wind dominates the German fleet (~60 GW) vs ~9 GW offshore - so an
offshore-only proxy would describe a small slice of the system.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
import xarray as xr

from eurogrid.models.transfer import fleet_box, modelled_capacity_factor

#: SMARD offshore zone -> sea basin, for splitting offshore capacity.
OFFSHORE_BASIN: Final[dict[str, str]] = {"50Hertz": "baltic", "TenneT": "north_sea"}

#: Germany box used for the solar field (same as the onshore wind box).
GERMANY_BOX: Final[tuple[float, float, float, float]] = fleet_box("wind_onshore", "50Hertz")

STC_REFERENCE_WM2: Final[float] = 1000.0


def solar_capacity_factor(ssrd: xr.DataArray, reference_wm2: float = STC_REFERENCE_WM2) -> xr.DataArray:
    """Transparent v1 solar CF: G / G_STC clipped to [0, 1].

    G_STC = 1000 W m-2 is the standard-test-condition irradiance. This is a
    resource proxy, not a PV-system model (no tilt, temperature, inverter).
    """
    cf = (ssrd / reference_wm2).clip(0.0, 1.0)
    cf.attrs.update(units="1", long_name="solar capacity factor (ssrd/STC proxy)")
    return cf


def _cos_mean_box(da: xr.DataArray, box: tuple[float, float, float, float]) -> xr.DataArray:
    lat_min, lat_max, lon_min, lon_max = box
    sub = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    if sub.sizes["lat"] == 0 or sub.sizes["lon"] == 0:
        raise ValueError(f"box {box} selects no grid cells")
    return sub.weighted(np.cos(np.deg2rad(sub["lat"]))).mean(dim=["lat", "lon"])


def capacity_weights(capacity: pl.DataFrame) -> dict[str, float]:
    """National installed capacity (MW) per technology, offshore split by basin.

    ``capacity`` is the SMARD installed-capacity frame (time, zone, variable,
    value_mw). The latest timestamp per (zone, variable) is used.
    """
    latest = capacity.sort("time").group_by(["zone", "variable"]).last()
    weights: dict[str, float] = {}
    for row in latest.iter_rows(named=True):
        var, zone, mw = row["variable"], row["zone"], row["value_mw"]
        if mw <= 0:
            continue
        if var == "wind_offshore":
            key = f"wind_offshore_{OFFSHORE_BASIN[zone]}"
        else:
            key = var
        weights[key] = weights.get(key, 0.0) + float(mw)
    return weights


def era5_renewable_cf(
    ds: xr.Dataset,
    weights: dict[str, float],
    *,
    aggregation: str = "field",
    loss_factor: float = 1.0,
) -> pl.DataFrame:
    """Hourly national renewable CF from ERA5, capacity-weighted.

    Onshore wind uses the shared Germany box, offshore uses the two basin
    boxes (North Sea / Baltic) weighted by zonal installed capacity, solar
    uses the ssrd/STC proxy over the Germany box. ``weights`` is the output
    of :func:`capacity_weights`.
    """
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("capacity weights sum to zero")
    parts: dict[str, pl.DataFrame] = {}
    parts["wind_onshore"] = modelled_capacity_factor(
        ds, "wind_onshore", "50Hertz", aggregation=aggregation, loss_factor=loss_factor
    )
    parts["wind_offshore_north_sea"] = modelled_capacity_factor(
        ds, "wind_offshore", "TenneT", aggregation=aggregation, loss_factor=loss_factor
    )
    parts["wind_offshore_baltic"] = modelled_capacity_factor(
        ds, "wind_offshore", "50Hertz", aggregation=aggregation, loss_factor=loss_factor
    )
    solar_cf = _cos_mean_box(solar_capacity_factor(ds["ssrd"]), GERMANY_BOX)
    parts["solar"] = pl.DataFrame(
        {"time": solar_cf["time"].values, "cf_model": solar_cf.values.astype(float)}
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC"))

    out = pl.DataFrame({"time": parts["wind_onshore"]["time"]})
    for key, frame in parts.items():
        w = weights.get(key, 0.0) / total
        part = frame.select("time", pl.col("cf_model").alias(key)).with_columns(
            pl.col("time").dt.cast_time_unit("us")
        )
        out = out.join(part, on="time").with_columns((pl.col(key) * w).alias(key + "__w"))
    weighted = sum(pl.col(key + "__w") for key in parts)
    return out.select(pl.col("time"), weighted.alias("cf"))


def smard_renewable_cf(generation: pl.DataFrame, capacity: pl.DataFrame) -> pl.DataFrame:
    """Hourly national renewable CF from SMARD, capacity-weighted over all zones."""
    cap = capacity.rename({"value_mw": "capacity_mw"})
    joined = generation.join(cap, on=["time", "zone", "variable"], how="left")
    agg = joined.group_by("time").agg(
        pl.col("value_mw").sum().alias("gen_mw"),
        pl.col("capacity_mw").sum().alias("cap_mw"),
    )
    return agg.select(pl.col("time"), (pl.col("gen_mw") / pl.col("cap_mw")).alias("cf")).sort("time")


def daily_mean(frame: pl.DataFrame, value: str = "cf") -> pl.DataFrame:
    """Resample an hourly (time, value) frame to daily means (UTC days)."""
    return (
        frame.sort("time")
        .filter(pl.col(value).is_finite())
        .group_by_dynamic("time", every="1d")
        .agg(pl.col(value).mean().alias(value))
    )


def detect_events(frame: pl.DataFrame, threshold: float, min_duration_hours: float) -> pl.DataFrame:
    """Consecutive runs with ``cf < threshold`` lasting >= ``min_duration_hours``.

    ``frame`` must be sorted (time, cf). Returns one row per event:
    start, end (exclusive), duration_hours, min_cf, mean_cf.
    """
    rows = list(frame.select("time", "cf").iter_rows())
    events: list[dict] = []
    run: list[tuple] = []
    step_hours = 24.0
    if len(rows) > 1:
        step_hours = (rows[1][0] - rows[0][0]).total_seconds() / 3600.0
    for time_, cf in rows:
        if cf is not None and cf < threshold:
            run.append((time_, cf))
        else:
            if run:
                dur = len(run) * step_hours
                if dur >= min_duration_hours:
                    events.append(
                        {
                            "start": run[0][0],
                            "end": run[-1][0],
                            "duration_hours": dur,
                            "min_cf": min(v for _, v in run),
                            "mean_cf": float(np.mean([v for _, v in run])),
                        }
                    )
                run = []
    if run:
        dur = len(run) * step_hours
        if dur >= min_duration_hours:
            events.append(
                {
                    "start": run[0][0],
                    "end": run[-1][0],
                    "duration_hours": dur,
                    "min_cf": min(v for _, v in run),
                    "mean_cf": float(np.mean([v for _, v in run])),
                }
            )
    return pl.DataFrame(
        events,
        schema={
            "start": pl.Datetime("us", "UTC"),
            "end": pl.Datetime("us", "UTC"),
            "duration_hours": pl.Float64,
            "min_cf": pl.Float64,
            "mean_cf": pl.Float64,
        },
    )


def threshold_duration_matrix(
    frame: pl.DataFrame, thresholds: list[float], min_durations: list[float]
) -> pl.DataFrame:
    """Event counts for every (threshold, min duration) combination."""
    rows = [
        {
            "threshold_cf": t,
            "min_duration_hours": d,
            "events": detect_events(frame, t, d).height,
        }
        for t in thresholds
        for d in min_durations
    ]
    return pl.DataFrame(rows)


def _overlap_days(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and b_start <= a_end


def event_verification(model_events: pl.DataFrame, observed_events: pl.DataFrame) -> dict[str, float]:
    """Hits / misses / false alarms between modelled and observed event lists.

    An observed event counts as hit when any modelled event overlaps it;
    otherwise it is a miss. A modelled event overlapping no observed event
    is a false alarm.
    """
    obs = observed_events.select("start", "end").iter_rows()
    mod = model_events.select("start", "end").iter_rows()
    obs_list, mod_list = list(obs), list(mod)
    hits = sum(any(_overlap_days(o[0], o[1], m[0], m[1]) for m in mod_list) for o in obs_list)
    misses = len(obs_list) - hits
    false_alarms = sum(not any(_overlap_days(o[0], o[1], m[0], m[1]) for o in obs_list) for m in mod_list)
    return {
        "observed_events": len(obs_list),
        "modelled_events": len(mod_list),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
    }


def conditional_drought_probability(
    daily_cf: pl.DataFrame, threshold: float, blocked_days: set
) -> dict[str, float]:
    """P(CF < threshold | blocked day) vs P(CF < threshold) overall.

    ``blocked_days`` is a set of dates (or datetimes) flagged by the
    persistence-filtered blocking mask. Small autocorrelated sample: report
    the numbers, not a significance claim.
    """
    dates = daily_cf["time"].dt.date().to_list()
    cf = daily_cf["cf"].to_list()
    blocked = [c < threshold for d, c in zip(dates, cf, strict=True) if d in blocked_days]
    all_days = [c < threshold for c in cf]
    p_all = float(np.mean(all_days)) if all_days else float("nan")
    p_blocked = float(np.mean(blocked)) if blocked else float("nan")
    return {
        "n_days": len(all_days),
        "n_blocked_days": len(blocked),
        "p_drought": p_all,
        "p_drought_given_blocked": p_blocked,
    }
