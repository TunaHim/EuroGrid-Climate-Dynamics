"""Wind-to-generation transfer: ERA5 ws100 -> fleet mean wind speed -> power curve -> MW.

Honest scope (see blueprint Phase 03): this is a *deliberately simple* v1 whose
bias against SMARD is a reported result, not something tuned away. Known missing
pieces, in order of expected impact:

* MaStR turbine locations/capacities -> capacity-weighted spatial aggregation
  (here: area-weighted mean over static "fleet proxy" boxes).
* Wake, availability and curtailment losses are bundled into an explicit
  ``loss_factor`` (default 1.0 = no losses); it is reported, never tuned
  to SMARD.
* A generic IEC-class curve standing in for a fleet mix of turbine models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import polars as pl
import xarray as xr

from eurogrid.contract import G0 as G0  # re-export: single definition lives in contract
from eurogrid.contract import validate_canonical


@dataclass(frozen=True)
class PowerCurve:
    """Generic IEC-class power curve: 0 below cut-in, cubic up to rated, 0 above cut-out."""

    cut_in: float = 3.0  # m/s
    rated: float = 12.0  # m/s
    cut_out: float = 25.0  # m/s

    def __call__(self, ws: np.ndarray) -> np.ndarray:
        """Fraction of rated capacity (0..1) for wind speed(s) ``ws`` in m/s."""
        ws = np.asarray(ws, dtype="float64")
        f = np.clip((ws**3 - self.cut_in**3) / (self.rated**3 - self.cut_in**3), 0.0, 1.0)
        f = np.where((ws < self.cut_in) | (ws >= self.cut_out), 0.0, f)
        return f


#: Fleet-proxy boxes keyed ``(variable, zone)`` -> (lat_min, lat_max, lon_min,
#: lon_max). Static stand-ins until MaStR capacity-weighted aggregation exists.
#: Onshore uses one Germany box for every zone for now; offshore is split by
#: basin because the 50Hertz fleet (Baltic 1/2, Wikinger, Arkona, Baltic Eagle)
#: sits in the Baltic, not the North Sea Bight.
FLEET_BOXES: Final[dict[tuple[str, str], tuple[float, float, float, float]]] = {
    ("wind_onshore", "50Hertz"): (47.5, 55.5, 5.0, 15.0),
    ("wind_onshore", "TenneT"): (47.5, 55.5, 5.0, 15.0),
    ("wind_onshore", "Amprion"): (47.5, 55.5, 5.0, 15.0),
    ("wind_onshore", "TransnetBW"): (47.5, 55.5, 5.0, 15.0),
    ("wind_offshore", "50Hertz"): (54.0, 55.2, 12.0, 15.0),  # Baltic
    ("wind_offshore", "TenneT"): (53.5, 56.0, 3.0, 8.5),  # North Sea Bight
}


def fleet_box(variable: str, zone: str) -> tuple[float, float, float, float]:
    """Return the fleet-proxy box for ``(variable, zone)``; raise on unknown pairs."""
    try:
        return FLEET_BOXES[(variable, zone)]
    except KeyError:
        raise ValueError(f"no fleet box for ({variable!r}, {zone!r}); have {sorted(FLEET_BOXES)}") from None


def _fleet_field(ds: xr.Dataset, box: tuple[float, float, float, float]) -> xr.DataArray:
    """``ds.ws100`` cropped to ``box`` (validated, non-empty)."""
    validate_canonical(ds.drop_vars([v for v in ("z500",) if v in ds.data_vars]))
    lat_min, lat_max, lon_min, lon_max = box
    sub = ds["ws100"].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    if sub.sizes["lat"] == 0 or sub.sizes["lon"] == 0:
        raise ValueError(f"fleet box {box} selects no grid cells")
    return sub


def fleet_wind_speed(ds: xr.Dataset, box: tuple[float, float, float, float]) -> xr.DataArray:
    """Area-weighted mean of ``ds.ws100`` over ``box`` -> 1-D (time) series in m/s."""
    sub = _fleet_field(ds, box)
    w = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(w).mean(dim=["lat", "lon"]).rename("ws_fleet")


def modelled_capacity_factor(
    ds: xr.Dataset,
    variable: str,
    zone: str,
    *,
    aggregation: Literal["field", "box_mean"] = "field",
    loss_factor: float = 1.0,
) -> pl.DataFrame:
    """Modelled capacity factor (0..1) time series for ``variable`` in ``zone``.

    ``aggregation="field"`` applies the power curve to every grid cell and
    then takes the cos(lat)-weighted mean (``mean(P(u))``). ``"box_mean"``
    keeps the previous behaviour: curve on the box-mean wind (``P(mean(u))``),
    which under-predicts in light/moderate wind (Jensen, convex part of curve).

    ``loss_factor`` is the fraction of ideal output retained after wake,
    availability and curtailment losses (1.0 = none). It is a documented,
    literature-level knob, never calibrated to SMARD.
    """
    if not 0.0 <= loss_factor <= 1.0:
        raise ValueError(f"loss_factor must be in [0, 1], got {loss_factor}")
    box = fleet_box(variable, zone)
    curve = PowerCurve(rated=13.0, cut_out=30.0) if variable == "wind_offshore" else PowerCurve()
    sub = _fleet_field(ds, box)
    if aggregation == "field":
        cf_cell = xr.apply_ufunc(curve, sub.load())
        w = np.cos(np.deg2rad(sub["lat"]))
        cf = cf_cell.weighted(w).mean(dim=["lat", "lon"])
    elif aggregation == "box_mean":
        ws = sub.weighted(np.cos(np.deg2rad(sub["lat"]))).mean(dim=["lat", "lon"]).load()
        cf = xr.apply_ufunc(curve, ws)
    else:
        raise ValueError(f"aggregation must be 'field' or 'box_mean', got {aggregation!r}")
    return pl.DataFrame(
        {
            "time": cf["time"].values.astype("datetime64[us]"),
            "zone": zone,
            "variable": variable,
            "cf_model": np.clip(np.asarray(cf.values, dtype="float64"), 0.0, 1.0) * loss_factor,
        }
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC"))


def validation_metrics(
    modelled: pl.DataFrame,
    observed_gen: pl.DataFrame,
    capacity: pl.DataFrame,
    zone: str,
) -> dict[str, float]:
    """Compare modelled CF against SMARD-observed generation for one zone/variable.

    ``modelled``: output of :func:`modelled_capacity_factor`.
    ``observed_gen``/``capacity``: tidy frames from :class:`SmardClient` filtered
    to the same variable. Returns a metrics dict; ``cf_obs`` is observed mean CF,
    ``cf_model`` modelled mean CF, rmse/bias in MW (time-varying capacity).
    """
    var = modelled["variable"][0]
    obs = (
        observed_gen.filter(pl.col("zone") == zone, pl.col("variable") == var)
        .join(
            capacity.filter(pl.col("zone") == zone, pl.col("variable") == var).rename(
                {"value_mw": "capacity_mw"}
            ),
            on=["time", "zone"],
            how="inner",
        )
        .join(
            modelled.filter(pl.col("zone") == zone).drop("variable"),
            on=["time", "zone"],
            how="inner",
        )
        .with_columns(
            pl.when(pl.col("capacity_mw") > 0)
            .then(pl.col("value_mw") / pl.col("capacity_mw"))
            .alias("cf_obs"),
            (pl.col("cf_model") - pl.col("value_mw") / pl.col("capacity_mw"))
            .mul(pl.col("capacity_mw"))
            .alias("err_mw"),
        )
        .with_columns((pl.col("cf_model") - pl.col("cf_obs")).alias("err_cf"))
    )
    if obs.height == 0:
        raise ValueError(f"no overlapping timestamps for {var} in {zone}")
    low_obs = obs.filter(pl.col("cf_obs") < 0.10)
    low_hit = (
        float("nan")
        if low_obs.height == 0
        else float(low_obs.filter(pl.col("cf_model") < 0.10).height / low_obs.height)
    )
    cf_obs_mean = float(obs["cf_obs"].mean())
    rmse_cf = float(obs["err_cf"].pow(2).mean() ** 0.5)
    return {
        "zone": zone,
        "variable": var,
        "n": obs.height,
        "corr": obs.select(pl.corr("cf_obs", "cf_model")).item(),
        "rmse_mw": float(obs["err_mw"].pow(2).mean() ** 0.5),
        "bias_mw": float(obs["err_mw"].mean()),
        "cf_obs": cf_obs_mean,
        "cf_model": float(obs["cf_model"].mean()),
        "bias_cf": float(obs["err_cf"].mean()),
        "rmse_cf": rmse_cf,
        "rmse_cf_rel": rmse_cf / cf_obs_mean if cf_obs_mean > 0 else float("nan"),
        "low_wind_hit_rate": low_hit,
    }
