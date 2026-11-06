"""Wind-to-generation transfer: ERA5 ws100 -> fleet mean wind speed -> power curve -> MW.

Honest scope (see blueprint Phase 03): this is a *deliberately simple* v1 whose
bias against SMARD is a reported result, not something tuned away. Known missing
pieces, in order of expected impact:

* MaStR turbine locations/capacities -> capacity-weighted spatial aggregation
  (here: area-weighted mean over static "fleet proxy" boxes).
* Power curve applied to the box-mean speed, not to the field
  (`mean(P(u))` != `P(mean(u))`; Jensen bias).
* Air density, wakes, availability, curtailment losses (~10-15%).
* A generic IEC-class curve standing in for a fleet mix of turbine models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
import xarray as xr

from eurogrid.contract import validate_canonical

G0: Final = 9.80665


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


#: Fleet-proxy boxes (lat_min, lat_max, lon_min, lon_max). Static stand-ins until
#: MaStR capacity-weighted aggregation exists.
FLEET_BOXES: Final[dict[str, tuple[float, float, float, float]]] = {
    # Onshore: Germany land box (fleet concentrated in the north/east; southern
    # zones get the same proxy, which inflates their modelled CF - noted in bias).
    "wind_onshore": (47.5, 55.5, 5.0, 15.0),
    # Offshore: German North Sea Bight (Baltic omitted: small share, 50Hertz only).
    "wind_offshore": (53.5, 56.0, 3.0, 8.5),
}


def fleet_wind_speed(ds: xr.Dataset, box: tuple[float, float, float, float]) -> xr.DataArray:
    """Area-weighted mean of ``ds.ws100`` over ``box`` -> 1-D (time) series in m/s."""
    validate_canonical(ds.drop_vars([v for v in ("z500",) if v in ds.data_vars]))
    lat_min, lat_max, lon_min, lon_max = box
    sub = ds["ws100"].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    if sub.sizes["lat"] == 0 or sub.sizes["lon"] == 0:
        raise ValueError(f"fleet box {box} selects no grid cells")
    w = np.cos(np.deg2rad(sub["lat"]))
    return sub.weighted(w).mean(dim=["lat", "lon"]).rename("ws_fleet")


def modelled_capacity_factor(ds: xr.Dataset, variable: str) -> pl.DataFrame:
    """Modelled capacity factor (0..1) time series for ``variable``'s fleet box."""
    if variable not in FLEET_BOXES:
        raise ValueError(f"no fleet box for {variable!r}; have {sorted(FLEET_BOXES)}")
    curve = PowerCurve(rated=13.0, cut_out=30.0) if variable == "wind_offshore" else PowerCurve()
    ws = fleet_wind_speed(ds, FLEET_BOXES[variable]).load()
    return pl.DataFrame(
        {
            "time": ws["time"].values.astype("datetime64[us]"),
            "variable": variable,
            "cf_model": curve(ws.values),
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
        .join(modelled.drop("variable"), on="time", how="inner")
        .with_columns(
            pl.when(pl.col("capacity_mw") > 0)
            .then(pl.col("value_mw") / pl.col("capacity_mw"))
            .alias("cf_obs"),
            (pl.col("cf_model") - pl.col("value_mw") / pl.col("capacity_mw"))
            .mul(pl.col("capacity_mw"))
            .alias("err_mw"),
        )
    )
    if obs.height == 0:
        raise ValueError(f"no overlapping timestamps for {var} in {zone}")
    return {
        "zone": zone,
        "variable": var,
        "n": obs.height,
        "corr": obs.select(pl.corr("cf_obs", "cf_model")).item(),
        "rmse_mw": float(obs["err_mw"].pow(2).mean() ** 0.5),
        "bias_mw": float(obs["err_mw"].mean()),
        "cf_obs": float(obs["cf_obs"].mean()),
        "cf_model": float(obs["cf_model"].mean()),
    }
