# Radar Selection & NaN Analysis

## Overview

This document summarizes our investigation into improving radar selection for the
NEXRAD turbulence prediction pipeline. We explored altitude-aware beam geometry
scoring, tournament-style radar selection, and NaN fraction filtering. This work
was conducted using February 2024 PIREP data (49,320 reports).

---

## Problem Statement

The original pipeline selected the **5 geographically closest** NEXRAD radar
stations for each PIREP. This approach has two issues:

1. **Beam geometry mismatch at altitude**: NEXRAD radars scan at fixed elevation
   angles (0.5° to 19.5°). At higher altitudes, the closest radar's beams may
   pass entirely below the PIREP altitude, contributing zero data to the grid.
   A radar further away might actually have beams passing through the correct
   altitude band.

2. **Grid sparsity**: The 3D reflectivity grid (10x16x16 = 2,560 cells) covering
   0.25° x 0.25° x 10,000ft is small enough that most radar beams don't
   intersect it, resulting in high NaN fractions even when weather is present.

---

## What We Built

### 1. Beam Geometry Scoring (`beam_geometry.py`)

A utility module that computes how well a radar station covers a PIREP's
altitude:

- **`beam_height(distance, elevation_angle)`**: Calculates beam center height at
  a given distance using the 4/3 effective earth radius refraction model
- **`count_covering_angles(distance, altitude)`**: Counts how many of the 15
  standard NEXRAD elevation angles have beams passing through the altitude band
  (±5,000ft around the PIREP)
- **`score_radar_for_pirep(distance, altitude)`**: Returns a 0.0-1.0 score
  combining coverage (70% weight) and distance (30% weight)
- **`get_num_candidates(altitude_ft)`**: Returns how many nearby stations to
  consider: 5 for <15,000ft, 10 for <30,000ft, 20 for ≥30,000ft

### 2. Altitude-Aware Radar Selection (`get_radars_for_pirep.py`)

Updated `find_candidate_sites()` to:
1. Query the KDTree for more candidates than needed (5/10/20 based on altitude)
2. Score each candidate using `score_radar_for_pirep()`
3. Return the **top 5** by score

This means at high altitudes, a radar 200km away with good beam coverage can
outrank a closer radar with none. At low altitudes, this behaves identically
to the original closest-5 approach since nearby radars always score highest.

### 3. NaN Fraction Filter (`radar_data_to_model_input.py`)

Added a post-gridding filter: if the final grid has >90% NaN cells, the
NetCDF file is not output. This prevents the model from training on samples
with essentially no reflectivity signal.

### 4. S3 Bucket Migration

The NEXRAD Level 2 archive migrated from `noaa-nexrad-level2` (deprecated
September 2025) to `unidata-nexrad-level2`. Updated all code and added
backward-compatible bucket name replacement for pre-existing CSVs.

---

## What We Tested

### Tournament-Style Radar Selection (Tested and Rejected)

We implemented and tested a two-phase tournament approach:

**Phase 1**: Grid the 5 closest radars together. If NaN fraction ≤ 90%, use
them (fast path).

**Phase 2**: If coverage was poor, score each of the 5 individually, drop
any with ≥99% NaN, then download and score additional candidates (up to 20
total) to find replacements with better coverage.

**Why we rejected it**:
- At altitudes where the tournament activated (≥FL150), the NaN problem was
  overwhelmingly caused by **lack of weather** (clear air), not poor radar
  selection. No amount of radar swapping helps when there's nothing to reflect.
- The tournament added massive compute cost: downloading and individually
  gridding 10-20 radars per PIREP (vs 5 downloads + 1 grid without it)
- At 100k PIREPs with 250-way HPC parallelism, the tournament would roughly
  triple processing time (from ~3 hours to ~10 hours)
- The marginal improvement on borderline cases was not worth the cost

### NaN Fraction Survey (50 PIREPs, 10 Altitude Bands)

We sampled 5 PIREPs per altitude band and gridded each with the 5 closest
radars to understand the NaN distribution:

```
Band          Count  Mean NaN%   Min NaN%   Max NaN%   ≤90% NaN
--------------------------------------------------------------
FL030-050         5      80.8%      63.3%      99.6%     3/5
FL050-080         5      80.6%      48.7%      99.8%     2/5
FL080-100         5      77.9%      54.6%      99.8%     3/5
FL100-140         5      59.4%      38.2%      92.8%     4/5
FL140-180         5      66.5%      42.2%      91.2%     4/5
FL180-220         5      83.6%      42.7%     100.0%     2/5
FL220-280         5      84.4%      53.9%     100.0%     2/5
FL280-330         5      94.5%      72.7%     100.0%     1/5
FL330-380         5      99.0%      95.2%     100.0%     0/5
FL380-450         5     100.0%      99.9%     100.0%     0/5
--------------------------------------------------------------
TOTAL            50                                     21/50
Overall pass rate (≤90% NaN): 42.0%
```

Full results are saved in `pireps/nan_fraction_survey_results.csv`.

### Key Findings

1. **The sweet spot is FL100-FL180** (10,000-18,000ft): 4/5 PIREPs passed the
   90% filter with mean NaN around 60-66%. This is where reflectivity-based
   turbulence prediction is most viable.

2. **Low altitude (FL030-100) is NOT guaranteed good**: 2-3 out of 5 failed
   per band. The NaN problem at low altitude is caused by clear-air PIREPs
   (turbulence without precipitation), not beam geometry.

3. **FL180-280 is mixed**: Some PIREPs have excellent data (42-54% NaN), others
   are empty. Depends entirely on whether there's precipitation.

4. **FL330+ is effectively unusable** for reflectivity-only models: 0/10 passed.
   At these altitudes, turbulence is predominantly Clear Air Turbulence (CAT)
   caused by wind shear and jet streams — phenomena that produce no radar
   reflectivity returns.

5. **NaN fraction is weather-dependent, not altitude-dependent**: The scatter
   plot shows high variance at every altitude. A FL200 PIREP in a storm can
   have 42% NaN while a FL050 PIREP in clear air has 100% NaN.

---

## Why Reflectivity Alone Can't Detect Clear Air Turbulence

Radar reflectivity measures the return signal from hydrometeors (rain, snow,
ice crystals). In clear air, there is nothing to reflect, so reflectivity is
NaN/missing everywhere. Clear Air Turbulence (CAT) is caused by:

- Wind shear at jet stream boundaries
- Mountain wave propagation
- Temperature inversions and Kelvin-Helmholtz instability

None of these produce reflectivity returns. If the model is fed a grid of
all -32.0 values (NaN replacement) for both "CAT with moderate turbulence"
and "clear sky with no turbulence," it cannot distinguish between them.
The input is identical in both cases — the model learns nothing.

---

## Final Pipeline Design

### What We Kept

- **Beam geometry scoring** in `get_radars_for_pirep.py`: Still pulls extra
  candidates at higher altitudes (10 at FL150-300, 20 at FL300+) and scores
  them by beam coverage, returning the best 5. This ensures the selected
  radars are the ones most likely to have data at the PIREP altitude.

- **90% NaN filter** in `radar_data_to_model_input.py`: Grids that are >90%
  NaN are not output as training samples. This prevents the model from
  training on empty inputs.

- **`nan_fraction` metadata** in `create_grid.py`: The gridding function now
  computes and stores NaN fraction in the xarray Dataset attrs for downstream
  filtering.

### What We Removed

- **Tournament-style selection**: All tournament code was removed from
  `radar_data_to_model_input.py`. The file went from ~280 lines back to ~165.
  The compute cost was not justified by the marginal improvement.

- **Variable radar counts**: The CSV always contains exactly 5 radar files
  per PIREP. `radar_data_to_model_input.py` uses all 5 in a single grid.

- **NUM_RADARS and NAN_FRACTION NetCDF attrs**: Removed from output to keep
  attrs matching the original format expected by the dataloader (LAT, LON,
  ALT, DELTA_T, TURB).

### Processing Cost (Estimated for 100k PIREPs, 250-way HPC parallelism)

- `get_radars_for_pirep.py`: ~5 minutes (unchanged, S3 listing is the bottleneck)
- `radar_data_to_model_input.py`: ~2-3 hours (5 downloads + 1 grid per PIREP)

---

## Possible Future Improvements

These were identified during our analysis but are outside the current scope:

1. **Additional input features for CAT detection**: Wind shear data from
   RAP/HRRR models, temperature gradients, jet stream proximity, Richardson
   number. These are available on AWS and would give the model signal for
   clear-air turbulence.

2. **Composite reflectivity (2D)**: Collapse the altitude axis using column-max
   reflectivity. Produces denser grids but loses vertical structure.

3. **Larger grid for high altitudes**: Increase from 0.25° to 0.5° or 1.0° at
   higher flight levels. More area = more beam intersections, but lower spatial
   resolution per cell.

4. **Satellite imagery inputs**: Infrared/visible satellite data provides
   coverage regardless of precipitation presence.

---

## Files Modified

| File | Changes |
|------|---------|
| `radars/beam_geometry.py` | **New file**. Beam height calculations, coverage scoring, candidate count logic. |
| `radars/get_radars_for_pirep.py` | Replaced naive closest-5 with beam-geometry-scored best-5. Added `haversine` dependency. S3 bucket updated to `unidata-nexrad-level2`. |
| `radars/radar_data_to_model_input.py` | Added 90% NaN filter. Added backward-compatible S3 bucket name fix. Simplified from tournament version back to direct 5-radar gridding. |
| `radars/create_grid.py` | Added `nan_fraction` and `num_radars` to output xarray Dataset attrs. |
| `radars/README.md` | Updated S3 bucket references. |
| `pireps/single_pirep.ipynb` | End-to-end pipeline notebook with altitude survey (Step 10). |
| `pireps/nan_fraction_survey_results.csv` | Raw survey data from 50-PIREP altitude analysis. |
