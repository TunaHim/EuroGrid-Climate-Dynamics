# EuroGrid-Climate-Dynamics

Does a kilometre-scale coupled climate model (EERIE IFS-FESOM2) reproduce
observed Euro-Atlantic blocking and European wind-drought statistics, and
what would that imply for German renewable generation at mid-century?

## The claim this repository makes

We build and validate a blocking / Dunkelflaute / ramp-rate pipeline on
ERA5 reanalysis and SMARD (German TSO) generation observations, then run the
*same* diagnostics on the EERIE `ifs-fesom2-sr` historical simulation to
evaluate its Euro-Atlantic blocking against reanalysis. A projection under
SSP2-4.5 is reported only alongside the historical bias that qualifies it.

We do **not** claim that ocean coupling causes or controls blocking
persistence: that requires a coupled-vs-uncoupled experiment pair, which is
out of scope for v1.

## Things that are true about the data (verified 2026-09-21)

- ERA5 provides hourly, true 100 m winds (`100u`, `100v`) on a 0.25° grid.
  Z500 lives in a *different* (pressure-level) ARCO store.
- EERIE `gr025` atmosphere output is **6-hourly** and has **no 100 m wind**.
  We use the 1000 hPa level (925 hPa as an upper bracket) as a hub-height
  proxy, report its actual height from `z/g`, and use it only for
  blocked-state and drought composites, flagging samples where the level is
  underground. A blanket `sp >= 1000 hPa` mask is deliberately *not* used:
  it deletes cyclone cases and biases wind statistics low.
- Consequently, 1 h / 3 h ramp rates come from ERA5 and SMARD only. The
  aliasing incurred by 6-hourly sampling is quantified as a result.
- EERIE `gr025` arrays are flattened (`value` = 1440 x 721, C-order, lat
  90 -> -90, lon 0 -> 179.75 then -180 -> -0.25). Stored chunks are
  `[3 time, 3 level, global]`, so spatial subsetting saves no bandwidth; we
  crop once and cache locally.
- `gr025` is a 0.25° regridded product. Nothing here is a km-scale analysis
  unless a native-resolution case study is added (stretch goal).

## Layout

```
config/settings.yaml      domain, periods, dataset ids, verified store facts, diagnostic parameters
src/eurogrid/contract.py  canonical (time, lat, lon) dataset contract every loader must satisfy
src/eurogrid/data/        loaders: smard, era5, eerie          (phases 01, 02, 06)
src/eurogrid/diagnostics/ blocking, dunkelflaute, ramp_rates   (phase 04)
src/eurogrid/models/      wind -> generation transfer          (phase 03)
dashboard/                thin Streamlit app                   (phase 05)
tests/                    offline; CI runs with sockets disabled
```

## Build order and gates

| Phase | Deliverable | Gate |
|---|---|---|
| 00 | scaffold, dataset contract, CI | green CI, contract assertions tested |
| 01 | SMARD client + installed capacity | Jan 2023 and Nov–Dec 2024 cached, offline fixtures |
| 02 | ERA5 loader (100 m winds, Z500) | one wind map that looks like Europe |
| 03 | transfer function + SMARD validation | bias/correlation figure by zone |
| 04 | TM1990 blocking, Dunkelflaute sweep, ramps + aliasing | blocking frequency vs longitude matches published climatology |
| 05 | thin dashboard | builds headless |
| 06 | EERIE historical blocking evaluation | model-vs-reanalysis figure, below-ground fraction reported |
| 07 | SSP2-4.5 projection (gated on 06) | historical bias attached to every future statistic |

## Development

```
conda env create -f environment.yml && conda activate eurogrid
pytest          # network disabled
ruff check . && ruff format --check .
```
