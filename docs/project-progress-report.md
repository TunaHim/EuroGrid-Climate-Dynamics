# EuroGrid-Climate-Dynamics: Progress Report

**Purpose:** document what has been built, what the current notebooks show, what
is scientifically established so far, and what remains to be done.

## 1. Project goal

The project evaluates whether the EERIE `IFS-FESOM2` climate model reproduces
historical Euro-Atlantic atmospheric blocking and renewable-energy weather
patterns.

The evidence chain is intended to be:

1. Use ERA5 reanalysis as the historical meteorological benchmark.
2. Use SMARD observations as the German renewable-generation benchmark.
3. Validate wind-to-generation calculations against observed generation.
4. Test blocking, wind-drought, and ramp-rate diagnostics on the historical
   benchmark.
5. Apply the same diagnostics to EERIE historical simulations.
6. Only then interpret future projections, together with the model's historical
   bias.

This project does **not** claim that ocean coupling causes blocking persistence.
It evaluates model behaviour against observations and reanalysis.

## 2. What has been completed

### Phase 00 — project foundation

Completed:

- Repository structure and configuration.
- A common dataset contract for `(time, latitude, longitude)` data.
- Offline tests and continuous integration.
- Configuration for domains, periods, datasets, and diagnostic parameters.
- Ruff formatting/linting setup.

The tests are designed to run without downloading data or requiring network
access.

### Phase 01 — SMARD observations

Completed:

- SMARD client for German renewable-generation data.
- Installed-capacity data by German transmission-system operator (TSO) zone.
- Wind onshore, wind offshore, and solar categories.
- Local parquet caching and offline fixtures.
- Initial data coverage for January 2023 and November–December 2024.

Important limitation: SMARD provides zonal aggregates, not plant-level
coordinates. The exploratory capacity maps therefore use approximate zone
locations, not the locations of individual power plants.

### Phase 02 — ERA5 reanalysis loader

Completed:

- ERA5 loading from the public ARCO-ERA5 stores.
- Hourly 100 m eastward and northward wind components.
- Derived 100 m wind speed:

  \[
  \mathrm{ws100}=\sqrt{u100^2+v100^2}.
  \]

- Surface pressure.
- 500 hPa geopotential height (`Z500`) from the pressure-level store.
- Spatial cropping to the project domain.
- Local Zarr caching.
- Coordinate normalization to the project contract.

The ERA5 domain used for the blocking work reaches approximately 30–85°N and
40°W–40°E. The northern edge is deliberately extended to 85°N because the
standard blocking calculation samples around 80°N.

### Phase 03 — wind-to-generation transfer

Completed:

- A first wind-to-generation transfer model.
- Fleet proxy boxes for onshore and offshore wind.
- Cosine-latitude-weighted spatial averaging.
- Comparison with SMARD generation observations.
- Validation notebook with timing and correlation diagnostics.

The initial model captures the timing of onshore generation reasonably well.
Offshore performance is weaker because a generic wind curve and simple spatial
box cannot represent the actual fleet mix, turbine availability, curtailment,
or grid constraints.

This is a useful baseline, not a final production model.

### Phase 04 — ERA5 blocking diagnostics

Completed:

- TM1990/Tibaldi–Molteni-style blocking detector.
- Local meridional height-gradient test.
- Requirement for a connected longitude sector at least 20° wide.
- Persistence calculation using a five-day threshold.
- Five complete December–February pilot winters:
  - 2020–21
  - 2021–22
  - 2022–23
  - 2023–24
  - 2024–25
- Z500-only local caches for those five winters.
- Spatial teaching figures showing:
  - the Z500 field;
  - the 40°N, 60°N, and 80°N sampling bands;
  - candidate longitudes;
  - sector-qualified longitudes;
  - longitude–time evolution;
  - persistent detections.

The detector is intentionally presented as a sequence:

1. Does the pressure-height pattern look like blocking?
2. Does it cover a connected longitude sector rather than one isolated grid
   cell?
3. Does it last at least five days, long enough to matter for persistent
   renewable-weather conditions?

## 3. Current notebooks

### `01_eda.ipynb` — initial data review

This notebook checks what was downloaded and whether it looks physically and
technically plausible.

It includes:

- variable inventory;
- sources, units, dimensions, and spatial coverage;
- ERA5 wind-speed maps and distributions;
- pressure and Z500 checks;
- temporal coverage and missing-value checks;
- SMARD generation and installed-capacity checks;
- approximate zonal capacity maps;
- explanations of geopotential height, temporal resolution, and weighting;
- explicit documentation that GHI was **not** downloaded in the current phase.

### `03_transfer_validation.ipynb` — wind-to-generation validation

This notebook compares the first modelled renewable-generation proxy with
SMARD observations. It is used to understand what the simple transfer function
captures and where it is biased.

### `02_era5_blocking_diagnostics.ipynb` — teaching notebook

This notebook introduces the blocking calculation step by step:

- what geopotential height means;
- why Z500 is useful for describing large-scale circulation;
- how the 40°N–60°N–80°N pattern is evaluated;
- a worked numerical example;
- a visual explanation of the sampling bands;
- the first ERA5 blocking result;
- what the result can and cannot establish.

### `03_era5_blocking_benchmark.ipynb` — five-winter benchmark

This notebook applies the same detector to five complete winters. It contains:

- cache and coverage checks;
- readable winter-by-winter summary tables;
- intermediate candidate → sector → persistence figures;
- a Z500 spatial map with detector masks;
- a longitude–time heatmap;
- pooled longitude-frequency plots;
- winter-to-winter comparison;
- a separate offshore-box wind-resource check.

The offshore wind section currently uses the wind caches that exist locally:

- January 2023;
- November–December 2024.

The five blocking winters have complete **Z500** coverage, but the matching
100 m wind fields have not yet been downloaded for every winter. Missing wind
months are shown as missing; they are not treated as zero wind.

## 4. What the current results mean

The five-winter blocking analysis is a pilot benchmark, not a final
climatology. It demonstrates that:

- the ERA5 Z500 data are available for the selected winters;
- the detector runs consistently over all five windows;
- isolated local pressure patterns are filtered before being called spatially
  extended blocking;
- persistence is a separate time-based test;
- the detected longitude extent varies substantially between events and
  winters.

The results do **not** yet establish:

- a robust multi-decadal blocking climatology;
- a causal relationship between blocking and renewable generation;
- that EERIE reproduces blocking;
- future changes in blocking or Dunkelflaute events;
- a complete relationship between offshore wind speed and German offshore
  generation.

Those claims require the next stages of the project.

## 5. Data currently available

### ERA5

Available in local caches:

- hourly 100 m wind components and derived wind speed for selected windows;
- surface pressure for selected windows;
- six-hourly Z500 for the five blocking benchmark winters;
- local Zarr cache files under `data/era5/`.

### SMARD

Available:

- renewable generation observations;
- installed capacity by TSO zone;
- wind onshore, wind offshore, and solar categories;
- parquet caches for the currently downloaded windows.

### Not yet available

- GHI / surface solar radiation for the current download set;
- complete 100 m wind coverage for every blocking benchmark winter;
- EERIE historical data;
- EERIE future data;
- plant-level MaStR locations;
- a validated Dunkelflaute event catalogue;
- ramp-rate and sampling-aliasing results.

## 6. Quality checks completed

The current branch has been checked with:

- notebook execution with zero cell errors;
- Ruff linting;
- Ruff format checking;
- the offline pytest suite;
- `git diff --check`;
- pull-request CI.

The latest local test suite contains 36 passing tests. The benchmark notebook
has been executed successfully after the latest repair.

## 7. Recommended next steps

### Next step 1 — complete the historical ERA5 renewable-weather benchmark

Download the missing 100 m wind fields for the remaining December–February
benchmark winters. Then produce the requested three-point winter plot:

- December mean offshore-box wind speed;
- January mean offshore-box wind speed;
- February mean offshore-box wind speed.

After that, implement the Dunkelflaute threshold sweep using ERA5 wind,
solar-related variables when available, and SMARD generation.

### Next step 2 — add EERIE historical evaluation

Define and test the EERIE loader against the same canonical data contract.
Compare EERIE historical blocking and renewable-weather statistics with ERA5
before using any future projection.

The EERIE comparison must report model bias first. Future changes should only be
interpreted together with that historical bias.

## 8. How to view the project online

The repository can be browsed like a normal cloud folder on GitHub:

- Repository: <https://github.com/TunaHim/EuroGrid-Climate-Dynamics>
- Current blocking branch: <https://github.com/TunaHim/EuroGrid-Climate-Dynamics/tree/devin/1790244127-blocking-diagnostics>
- All notebooks: <https://github.com/TunaHim/EuroGrid-Climate-Dynamics/tree/devin/1790244127-blocking-diagnostics/notebooks>
- Configuration: <https://github.com/TunaHim/EuroGrid-Climate-Dynamics/tree/devin/1790244127-blocking-diagnostics/config>
- Source code: <https://github.com/TunaHim/EuroGrid-Climate-Dynamics/tree/devin/1790244127-blocking-diagnostics/src>
- Tests: <https://github.com/TunaHim/EuroGrid-Climate-Dynamics/tree/devin/1790244127-blocking-diagnostics/tests>

For a rendered, read-only notebook view, use nbviewer:

- <https://nbviewer.org/github/TunaHim/EuroGrid-Climate-Dynamics/blob/devin/1790244127-blocking-diagnostics/notebooks/01_eda.ipynb>
- <https://nbviewer.org/github/TunaHim/EuroGrid-Climate-Dynamics/blob/devin/1790244127-blocking-diagnostics/notebooks/02_era5_blocking_diagnostics.ipynb>
- <https://nbviewer.org/github/TunaHim/EuroGrid-Climate-Dynamics/blob/devin/1790244127-blocking-diagnostics/notebooks/03_era5_blocking_benchmark.ipynb>

GitHub is best for browsing folders and source files. Nbviewer is best for
reading executed notebook outputs. The temporary JupyterLab preview is best
when cells need to be rerun interactively, but it stops when the Devin VM
sleeps.

## 9. Current project status in one sentence

The project has a tested ERA5/SMARD foundation, a validated first
wind-to-generation baseline, and an explained five-winter ERA5 blocking pilot;
the next scientific work is to complete the wind benchmark and then evaluate
historical EERIE behaviour.
