# DIMER Workshop Specification: Weather & Earth-System Forecasting with Aurora

**Status:** Proposed  
**Notebook specification:** DIMER `NOTEBOOK_SPEC` **2.1**  
**Notebook profile:** `E2E`  
**Pedagogical mode:** `WORKSHOP`  
**Proposed filename:** `DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb`  
**Anchor repository:** `kurtvalcorza/aurora-earth-system-pipeline`  
**Canonical runtime:** NVIDIA Tesla T4 or equivalent  
**Canonical execution:** standalone, credential-free, top-to-bottom `Run all`

---

# 1. Purpose

This workshop demonstrates how an Earth-system foundation model can produce **short-range global weather forecasts** from gridded atmospheric analyses and how its performance should be evaluated against a strong physical baseline.

The workshop centers on:

**Microsoft Aurora 0.25° Small Pretrained**

and teaches the complete workflow:

```text
ERA5 analysis
→ validate Earth-system state
→ persistence forecast
→ frozen Aurora rollout
→ diagnose domain/resolution shift
→ bounded LoRA adaptation
→ validation selection
→ freeze experiment
→ independent 6/12/18/24 h forecast evaluation
→ spatial/error interpretation
→ new-origin forecast
→ adapter export and fresh reload
```

The core question is:

> Can a pretrained Earth-system model provide forecast skill beyond simply carrying the latest atmospheric analysis forward, and what happens when the deployment grid differs substantially from the grid on which it was trained?

---

# 2. Scientific scope

Preferred terminology:

- **weather forecasting**
- **Earth-system forecasting**
- **atmospheric forecasting**
- **short-range forecasting**
- **foundation-model adaptation**

The notebook MUST NOT describe its 6–24-hour forecasts as:

- climate projections;
- climate-change predictions;
- seasonal climate outlooks; or
- long-range climate simulation.

A concise distinction SHOULD appear near the beginning:

> Weather forecasting predicts the evolving atmospheric state from an initial condition over hours to days. Climate projection studies statistical changes in the Earth system over much longer periods under changing forcings. This workshop demonstrates the former.

---

# 3. Notebook profile

The notebook SHALL declare:

**Profile:** `E2E`  
**Mode:** `WORKSHOP`

The canonical path includes actual adaptation and artifact export.

This is required by the supported capability and `NOTEBOOK_SPEC 2.1`:

```text
sample acquisition
→ validation
→ baseline
→ frozen inference
→ adaptation
→ validation selection
→ independent evaluation
→ new-data inference
→ artifact export
→ fresh reload verification
```

LoRA MUST run on the default `Run all` path.

---

# 4. Model identity

**Model:** `microsoft/aurora`  
**Revision:**  
`a96afd7ee6d65e3bd2d476f3be798a25a56f2296`

Checkpoint:

**Aurora 0.25° Small Pretrained**

Source asset:

```text
aurora-0.25-small-pretrained.ckpt
451,339,106 bytes
SHA-256:
f80f78de1524a9faba8c9053e4a8ce6a2114ec01cff7f7b4efe9377200d50621
```

Static fields:

```text
aurora-0.25-static.pickle
12,459,115 bytes
SHA-256:
e382103f6b24bcf1f996cc0af217c71ff2fc66507a5221e1300b5017581bd318
```

Converted serving files:

```text
aurora-0.25-small-pretrained.safetensors
451,230,408 bytes
SHA-256:
fc03b5fc5764e08e1f7a05f06ec4debb87fbb66fa2496ec203142221d0843370
```

```text
aurora-0.25-static.safetensors
12,459,136 bytes
SHA-256:
9bd430b666d9267aca34d5c7924d594c3474296c9e39d85701df389816303bc0
```

License:

**MIT**

---

# 5. Important checkpoint boundary

The workshop MUST state prominently:

> `AuroraSmallPretrained` is the approximately 113M-parameter small Aurora checkpoint that upstream describes as a debugging/testing model. It is not the 1.3B-parameter production-scale Aurora checkpoint.

Approximate parameters:

```text
112,797,584
```

The full Aurora 0.25° checkpoint is approximately:

```text
5.03 GB
```

and remains outside the canonical workshop resource envelope.

The workshop MUST NOT infer production-model skill from the small checkpoint.

---

# 6. Why the small model

The small checkpoint is appropriate for a workshop because it allows:

- model acquisition in a normal hosted runtime;
- global forecasting;
- autoregressive 24-hour rollout;
- LoRA adaptation;
- artifact export;
- fresh reload;
- meaningful evaluation against persistence.

The pedagogical objective is the **forecasting/adaptation methodology**, not reproducing Aurora's published production scores.

---

# 7. Canonical dataset

Reuse the live Aurora pipeline's real WeatherBench 2 sample.

Source:

**ERA5 via WeatherBench 2**

Dataset:

```text
1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr
```

Resolution:

**1.5°**

Grid:

```text
121 latitudes × 240 longitudes
```

Frequency:

**6-hourly**

Total pinned tutorial acquisition:

approximately **196 MB**

from **57 digest-pinned WeatherBench 2 objects**.

---

# 8. Canonical windows

The sample consists of four independent two-day windows.

| Role | Window | Analyses |
|---|---|---:|
| Train | January 2019 | 8 × 6-hourly |
| Train | July 2019 | 8 × 6-hourly |
| Validation | April 2020 | 8 × 6-hourly |
| Test | October 2021 | 8 × 6-hourly |

Canonical identifiers:

```text
train-2019-01
train-2019-07
val-2020-04
test-2021-10
```

These windows intentionally differ by:

- year;
- season; and
- synoptic situation.

The test window MUST remain unavailable for adaptation or epoch selection.

---

# 9. Why temporal windows must not be randomly split

Atmospheric analyses six hours apart are strongly dependent.

Randomly shuffling individual timesteps would create severe temporal leakage.

The notebook MUST explain:

> Forecasting splits belong to periods/windows, not randomly shuffled rows.

Training, validation and testing therefore remain separated by whole time windows.

---

# 10. Resolution mismatch as a central lesson

Aurora was trained at:

**0.25°**

The tutorial uses:

**1.5°**

Therefore the tutorial grid is **six times coarser per horizontal axis**.

A model token spanning 4×4 grid cells corresponds approximately to:

- 1°×1° at native 0.25°;
- 6°×6° at tutorial 1.5°.

The notebook MUST explicitly state:

> Every forecast metric in this workshop is an out-of-distribution resolution experiment. It is not an estimate of Aurora's native-resolution forecasting skill.

This should be treated as a core workshop concept rather than a footnote.

---

# 11. Learning objectives

Participants should be able to:

1. explain the structure of a global atmospheric model state;
2. distinguish surface, pressure-level and static variables;
3. explain why two historical analyses are needed to initialize Aurora;
4. construct an autoregressive 6–24-hour forecast;
5. implement and interpret a persistence baseline;
6. calculate latitude-weighted global RMSE;
7. explain why equal-latitude grids require area weighting;
8. compare model RMSE against persistence through relative skill;
9. identify resolution/domain shift;
10. perform bounded LoRA adaptation;
11. preserve a leakage-safe train/validation/test design;
12. interpret error growth with lead time;
13. export and reload a LoRA adapter; and
14. distinguish an experimental forecast from an operational weather product.

---

# 12. Aurora input state

Each analysis contains three kinds of information.

## 12.1 Surface variables

```text
2t   — 2 m temperature       [K]
10u  — 10 m zonal wind       [m/s]
10v  — 10 m meridional wind  [m/s]
msl  — mean sea-level pressure [Pa]
```

---

# 13. Atmospheric variables

For each pressure level:

```text
t — temperature
u — zonal wind
v — meridional wind
q — specific humidity
z — geopotential
```

Units:

```text
t : K
u : m/s
v : m/s
q : kg/kg
z : m²/s²
```

---

# 14. Pressure levels

Exactly 13 standard pressure levels:

```text
50
100
150
200
250
300
400
500
600
700
850
925
1000 hPa
```

The workshop MUST reject another level set.

---

# 15. Static Earth fields

Aurora also uses:

```text
lsm — land-sea mask
z   — surface geopotential
slt — soil type
```

These do not vary through the forecast window.

---

# 16. History contract

Aurora requires:

**two consecutive analyses**

to initialize the forecast.

Therefore:

```text
history size = 2
time interval = 6 hours
```

Example:

```text
00 UTC analysis
06 UTC analysis
→ forecast 12 UTC
```

The model then rolls forward autoregressively:

```text
00,06 observed
→ 12 forecast
→ 18 forecast
→ 00+1 forecast
→ 06+1 forecast
```

---

# 17. Forecast horizons

Canonical workshop evaluation:

| Step | Lead |
|---:|---:|
| 1 | 6 h |
| 2 | 12 h |
| 3 | 18 h |
| 4 | 24 h |

The workshop SHALL evaluate all four leads.

Although the DIMER pipeline supports up to eight rollout steps, longer horizons SHOULD remain optional.

---

# 18. Data validation

Validation MUST execute before model construction.

Required checks:

- latitude exists;
- longitude exists;
- latitude spans the poles;
- longitude covers the globe;
- grid dimensions within supported bounds;
- longitude dimension divisible by patch size;
- latitude dimension either divisible by four or one row larger;
- exactly 13 pressure levels;
- correct level ordering;
- 3–64 analyses;
- exact 6-hour spacing;
- required surface variables;
- required atmospheric variables;
- required static fields;
- finite numerical values;
- variable shapes agree;
- plausibility ranges respected.

---

# 19. Plausibility checks

Examples from the live contract:

```text
2t  : 150–350 K
10u : -150–150 m/s
10v : -150–150 m/s
msl : 85,000–110,000 Pa

t : 150–350 K
u : -300–300 m/s
v : -300–300 m/s
q : -0.001–0.1 kg/kg
z : -10,000–250,000 m²/s²
```

These ranges detect obvious unit/data errors.

The notebook MUST explain:

> Passing range validation does not mean an atmospheric state is dynamically or meteorologically realistic.

---

# 20. Earth-system data primer

Before model execution, show:

1. latitude/longitude grid;
2. 2-m temperature field;
3. mean sea-level pressure;
4. 500-hPa geopotential (`z500`);
5. 850-hPa temperature or wind;
6. land-sea mask.

Participants should see that this is not a table of independent observations.

It is a coupled global state.

---

# 21. Spatial weighting

A regular latitude-longitude grid does not give equal physical area to every grid cell.

Cells near the poles represent less area than cells near the equator.

Therefore errors MUST be weighted by:

\[
w(\phi)=\cos(\phi)
\]

where \(\phi\) is latitude.

Weights SHOULD be normalized to unit mean before calculating RMSE.

---

# 22. Primary metric — latitude-weighted RMSE

For forecast field \(f\) and reference analysis \(y\):

\[
RMSE =
\sqrt{
\frac{\sum_i w_i(f_i-y_i)^2}
{\sum_i w_i}
}
\]

Atmospheric variables SHOULD be averaged across their 13 pressure levels for the primary variable-level score.

---

# 23. Headline geopotential metric

Also report:

**z500**

geopotential at:

**500 hPa**

as a separate meteorologically recognizable diagnostic.

`z500` SHOULD NOT be counted as a tenth independent variable in the nine-variable mean skill summary because it is derived from the existing `z` atmospheric field.

---

# 24. Baseline — persistence

Persistence carries the latest observed atmospheric analysis forward unchanged.

For all future leads:

\[
\hat y_{t+h}=y_t
\]

This is a strong short-range baseline.

The notebook MUST run persistence **before Aurora**.

Pedagogical question:

> Is the learned model actually more useful than simply assuming the current atmospheric state persists?

---

# 25. Skill relative to persistence

For each variable and lead:

\[
Skill =
\frac{RMSE_{model}}
{RMSE_{persistence}}
\]

Interpretation:

| Skill | Meaning |
|---:|---|
| `< 1` | model beats persistence |
| `= 1` | same RMSE |
| `> 1` | persistence performs better |

Because lower is better, this MUST NOT be called a conventional "higher-is-better skill score" without qualification.

Preferred language:

> **RMSE ratio versus persistence**

The existing `skill` field MAY remain for semantic parity.

---

# 26. Nine-variable summary

Primary variables:

```text
2t
10u
10v
msl
t
u
v
q
z
```

At each lead report:

```text
mean RMSE ratio vs persistence
variables beating persistence / 9
```

This produces an intuitive summary such as:

```text
6 h: 6 / 9 variables beat persistence
```

without hiding the individual fields.

---

# 27. Stage A — persistence

Run persistence on the **validation window** first.

Show:

- RMSE per variable;
- RMSE by lead;
- z500;
- selected spatial maps.

This establishes the difficulty of the forecast before using a neural model.

---

# 28. Stage B — frozen Aurora

Load the pinned small Aurora model with:

```text
LoRA = zero / inactive
```

Run the same 6/12/18/24-hour validation rollout.

Compare directly against persistence.

The workshop MUST state the expected educational outcome:

> At this 1.5° out-of-distribution resolution, the frozen small model may perform worse than persistence. That is an informative result, not a notebook failure.

---

# 29. Frozen-model diagnostic

Create a validation table:

| Variable | 6 h | 12 h | 18 h | 24 h |
|---|---:|---:|---:|---:|
| Persistence RMSE | | | | |
| Frozen Aurora RMSE | | | | |
| RMSE ratio | | | | |

Then plot:

```text
lead time → RMSE ratio
```

with a horizontal line at:

```text
1.0 = persistence
```

---

# 30. Why adaptation is justified

The notebook SHOULD explicitly ask:

> If a pretrained model was trained at 0.25°, should we expect it to behave optimally when every input token suddenly represents roughly 36 times the physical area?

This motivates adaptation as **domain/resolution transfer**, not generic fine-tuning for higher benchmark performance.

---

# 31. Canonical adaptation scope

Default adaptation:

**LoRA only**

Aurora exposes rank-8 LoRA adapters in the attention projections.

Trainable tensors:

```text
80
```

Trainable parameters:

```text
540,672
```

This is approximately:

**0.48% of the small model's parameters**

and SHOULD be highlighted as parameter-efficient adaptation.

---

# 32. Adaptation scope exclusions

The canonical path MUST NOT update:

- full backbone;
- all model parameters;
- static fields;
- input data;
- test window.

Optional:

```text
lora+heads
```

MAY appear as an advanced experiment, disabled by default.

The existing build evidence suggests it changes tradeoffs and is less stable than LoRA alone.

---

# 33. Training samples

Training windows:

```text
January 2019
July 2019
```

Each contains eight analyses.

With:

```text
2-step history
1-step target
```

each window yields six one-step supervised samples.

Total:

```text
12 training forecast samples
```

The notebook MUST make this small adaptation scale explicit.

---

# 34. Training objective

Adaptation trains on:

**one-step, 6-hour forecasts**

not on full 24-hour rollout loss.

Each variable SHOULD use the live pipeline's scaling constants:

```text
2t  : 10
10u : 5
10v : 5
msl : 1000
t   : 10
u   : 15
v   : 15
q   : 0.004
z   : 25000
```

The workshop should explain that the scales prevent variables with large numerical units from dominating the joint loss.

---

# 35. Adaptation hyperparameters

Canonical defaults SHOULD preserve the live qualified recipe:

```text
trainable = "lora"
epochs = 6
learning rate = 1e-3
seed = 0
```

Optimizer:

**AdamW**

Selection:

> lowest validation one-step loss

The validation window MUST be:

```text
April 2020
```

---

# 36. Validation history

Display:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 0 | — | frozen baseline |
| 1 | | |
| ... | | |
| 6 | | |

Epoch 0 is the frozen model and MUST remain eligible as the selected policy.

If no adaptation epoch improves validation loss, the workshop SHOULD preserve the frozen policy rather than assuming adaptation must win.

---

# 37. Forecast skill during adaptation

At least for:

- epoch 0;
- selected epoch;

show validation:

- 6-h RMSE ratio;
- 24-h RMSE ratio;
- variables beating persistence.

This helps learners distinguish:

> optimizing one-step training loss

from:

> improving an autoregressive multi-step forecast.

---

# 38. Freeze-before-test

After validation selection write:

```text
outputs/frozen/frozen_experiment.json
```

including:

```text
dataset digests
training window identities
validation window identity
test window identity
base-model revision
converted-weight digests
grid resolution
variable set
pressure levels
history size
timestep
LoRA rank/scope
trainable tensor names
epochs
learning rate
seed
best epoch
loss scales
forecast leads
metric definition
persistence definition
```

No choice may be changed after final test metrics are viewed.

---

# 39. Independent test

Test window:

**October 2021**

Evaluate:

1. persistence;
2. frozen Aurora;
3. selected adapted Aurora.

For:

```text
6 h
12 h
18 h
24 h
```

using every valid forecast origin available at each lead.

---

# 40. Test comparison table

Primary summary:

| Lead | Persistence | Frozen mean ratio | Adapted mean ratio | Frozen vars beating persistence | Adapted vars beating persistence |
|---:|---:|---:|---:|---:|---:|
| 6 h | baseline | | | /9 | /9 |
| 12 h | baseline | | | /9 | /9 |
| 18 h | baseline | | | /9 | /9 |
| 24 h | baseline | | | /9 | /9 |

The persistence column SHOULD not be expressed as a mean ratio because by definition its self-ratio is 1.

---

# 41. Existing qualified reference result

The workshop documentation MAY state the existing release evidence as a **reference expectation**, clearly separate from the current execution:

At the current 1.5° test setup, the recorded Kaggle T4 run observed frozen → adapted mean RMSE ratio:

```text
6 h  : 1.568 → 0.904
12 h : 1.328 → 0.849
18 h : 1.237 → 0.867
24 h : 1.294 → 0.969
```

Variables beating persistence:

```text
6 h  : 1/9 → 6/9
12 h : 1/9 → 8/9
18 h : 1/9 → 7/9
24 h : 2/9 → 6/9
```

The notebook MUST recompute these values rather than use them as canned outputs.

---

# 42. Lead-time error growth

For each selected variable plot:

```text
lead time
vs
RMSE
```

for:

- persistence;
- frozen;
- adapted.

Recommended headline variables:

```text
2t
msl
z500
u
```

This illustrates autoregressive error propagation.

---

# 43. Spatial forecast maps

At one deterministic test origin, display:

### 2 m temperature

```text
analysis
persistence
frozen forecast
adapted forecast
frozen absolute error
adapted absolute error
```

### Mean sea-level pressure

same structure.

### 500-hPa geopotential

same structure.

Use the 24-hour lead for the main map comparison.

---

# 44. Spatial interpretation

Maps SHOULD retain:

- latitude;
- longitude;
- physical units.

Avoid decorative interpolation that suggests resolution finer than 1.5°.

The notebook MUST NOT visually upscale these fields in a way that implies native 0.25° detail.

---

# 45. Latitude weighting exercise

Include a small exercise:

> Why would unweighted RMSE over a latitude-longitude grid over-emphasize polar regions?

Then show:

- unweighted RMSE;
- cosine-latitude-weighted RMSE

for one field.

The weighted metric remains canonical.

---

# 46. Persistence exercise

Before displaying Aurora results ask:

> Which fields do you expect persistence to remain competitive on at 6 hours?

Participants can reason about:

- slowly evolving pressure/geopotential patterns;
- temperature's diurnal cycle;
- wind variability.

This remains a non-blocking reflection prompt.

---

# 47. Domain-shift exercise

Ask:

> Why might a 0.25° model perform poorly at 1.5° even though all variable names and units are correct?

Expected mechanisms:

- physical length scale per grid cell;
- token receptive field;
- smoothed gradients;
- reduced extremes;
- different spectral content;
- coarse orography;
- regridding effects.

---

# 48. New-origin forecasting

After test evaluation, use the final available two analyses of the chosen window or another designated new-origin sample.

Generate:

```text
+6 h
+12 h
+18 h
+24 h
```

without consulting future targets in the workflow.

If future truth is not supplied:

```text
evaluation status = not-measurable
```

The purpose is to distinguish:

- retrospective forecast verification; and
- actual forecast production.

---

# 49. Adapter artifact

Export:

```text
adapter.safetensors
manifest.json
```

Canonical artifact format:

```text
org.valcorza.aurora-earth-system.adapter.v1
```

The manifest MUST contain:

- base model ID;
- base revision;
- converted base-model SHA-256;
- static-field SHA-256;
- adaptation scope;
- LoRA tensor names;
- tensor count;
- trainable parameter count;
- epochs;
- learning rate;
- seed;
- best epoch;
- loss history;
- file byte size;
- adapter SHA-256.

---

# 50. Fresh reload verification

The notebook MUST:

```text
destroy adapted model
→ reconstruct fresh base Aurora
→ verify base files
→ load adapter from artifact
→ rerun fixed forecast origin
→ compare predictions
```

Require maximum absolute difference within an explicit numerical tolerance.

The currently qualified pipeline has demonstrated exact reload parity, but the workshop must verify its own execution.

---

# 51. BYOD

BYOD input:

**NetCDF**

Expected coordinates:

```text
time
latitude
longitude
level
```

Surface fields:

```text
2t
10u
10v
msl
```

Atmospheric fields:

```text
t
u
v
q
z
```

Static:

```text
static_lsm
static_z
static_slt
```

or the corresponding accepted canonical representation.

---

# 52. BYOD workflow

Because the notebook is `E2E`, user data MUST support:

```text
load NetCDF
→ validate
→ define window roles
→ persistence
→ frozen forecast
→ LoRA adaptation
→ validation selection
→ test evaluation
→ adapter export
→ fresh reload
```

A user supplying only one short window MAY run inference/evaluation but does not satisfy the full workshop E2E adaptation branch.

For full BYOD adaptation, the notebook SHOULD require explicit role assignment or multiple windows.

---

# 53. BYOD controls

Recommended form fields:

```python
USE_BYOD = False
BYOD_TRAIN_1_PATH = ""
BYOD_TRAIN_2_PATH = ""
BYOD_VALIDATION_PATH = ""
BYOD_TEST_PATH = ""
```

When all paths are provided, the notebook MUST read them directly without opening an upload dialog.

Interactive upload MAY be offered only when `USE_BYOD=True` and no paths are supplied.

---

# 54. BYOD privacy

The notebook MUST state:

> User-supplied atmospheric analyses are processed inside the selected notebook runtime and are not sent to DIMER workers or APIs. A hosted notebook remains an external compute environment. Do not upload confidential, embargoed, proprietary, security-sensitive, or operational forecast data unless authorized.

---

# 55. Runtime environment

Use the live qualified pins:

```text
Python 3.12
torch==2.14.0
microsoft-aurora==2.0.1
timm==1.0.29
einops==0.8.2
xarray==2026.7.0
netCDF4==1.7.4
numcodecs==0.17.0
numpy==2.5.3
safetensors==0.8.0
huggingface-hub==1.32.0
```

The environment MUST install without requiring a manual restart.

---

# 56. Supply-chain workflow

The canonical standalone notebook SHOULD preserve the current pipeline trust boundary:

```text
download pinned source
→ byte/digest verify
→ static pickle audit
→ restricted one-time conversion
→ SafeTensors
→ only SafeTensors thereafter
```

Checkpoint audit allowed globals:

```text
collections.OrderedDict
torch._utils._rebuild_tensor_v2
torch.FloatStorage
```

Static-data allowed NumPy globals remain restricted to the known static-field representation.

---

# 57. Converted-files optimization

If the notebook can directly acquire trusted, digest-pinned converted SafeTensors from an approved upstream/DIMER-hosted public location, it MAY bypass source-pickle conversion.

However:

- the source provenance MUST remain documented;
- converted SHA-256s MUST match;
- default execution MUST remain credential-free.

Do not silently substitute different Aurora weights.

---

# 58. Resource envelope

Expected workshop footprint:

### Model source

approximately:

```text
464 MB
```

### Converted files

approximately:

```text
464 MB
```

### Sample data

approximately:

```text
196 MB
```

The existing clean T4 execution downloaded approximately:

```text
1.1 GB total
```

including packages/model/data assets.

Target wall-clock envelope:

**approximately 5 minutes on a T4**, subject to network/cache conditions.

The notebook SHOULD report actual wall times for:

- acquisition;
- conversion;
- frozen forecast;
- adaptation;
- test evaluation;
- reload.

---

# 59. Computational comparison

Produce:

| Stage | Time | Peak VRAM if available |
|---|---:|---:|
| Model load | | |
| Frozen 24 h rollout | | |
| LoRA training | | |
| Adapted 24 h rollout | | |
| Reload verification | | |

Also report:

```text
total parameters
trainable LoRA parameters
trainable percentage
adapter size
```

---

# 60. Workshop anatomy

Recommended notebook structure:

```text
0. Workshop overview
1. Runtime and scientific scope
2. Weather vs climate
3. Aurora architecture and model identity
4. Earth-system state primer
5. Acquire and verify WeatherBench 2 / ERA5
6. Validate the four windows
7. Explore atmospheric fields
8. Latitude weighting
9. Persistence baseline
10. Frozen Aurora rollout
11. Diagnose 1.5° domain shift
12. LoRA adaptation setup
13. Train on January + July 2019
14. Validate on April 2020
15. Select and freeze adapter
16. Independent October 2021 test
17. Lead-time skill curves
18. Spatial error maps
19. Variable-level analysis
20. Runtime / parameter-efficiency analysis
21. Forecast from a new origin
22. Export adapter and provenance
23. Fresh reload verification
24. BYOD
25. Optional lora+heads experiment
26. Optional longer rollout
27. Interpretation and operational limits
28. Troubleshooting
29. Glossary
```

---

# 61. Optional experiment — LoRA vs LoRA+heads

Disabled by default.

Compare:

```text
LoRA only
```

against:

```text
LoRA + encoder token embeddings + decoder heads
```

Use validation only for selection.

The notebook MUST NOT use test metrics to choose the adaptation scope.

Results SHOULD be presented as:

> adaptation-scope sensitivity

rather than part of the canonical test result.

---

# 62. Optional experiment — longer rollout

Aurora supports up to:

```text
8 × 6 h = 48 h
```

A disabled optional section MAY extend the forecast to 48 hours if enough reference analyses are available.

The canonical evaluation remains 24 hours because the built-in windows are deliberately small and the evidence weakens rapidly with lead and sample size.

---

# 63. Optional experiment — latitude bands

Break RMSE into broad geographic zones:

```text
90–60°N
60–30°N
30°N–30°S
30–60°S
60–90°S
```

This SHOULD be labelled diagnostic only.

One two-day test window cannot establish regional forecast skill.

---

# 64. Operational limits

The notebook MUST NOT present its outputs as sufficient for:

- weather warnings;
- aviation decisions;
- maritime routing;
- emergency management;
- energy dispatch;
- agricultural advisories;
- public forecast issuance.

The workshop model is:

- the small debugging Aurora checkpoint;
- running at out-of-distribution 1.5° resolution;
- evaluated on one small test window;
- deterministic, without ensemble uncertainty.

---

# 65. Forecast uncertainty limitation

Aurora in this workflow produces one deterministic forecast.

It does not produce:

- ensemble members;
- calibrated uncertainty;
- predictive intervals;
- probability of threshold exceedance.

The absence of uncertainty estimates MUST be stated explicitly.

---

# 66. ERA5-as-reference limitation

The notebook verifies forecasts against ERA5 analyses.

ERA5 is a reanalysis, not direct observational truth.

Therefore:

> Forecast error in this workshop is error relative to ERA5, including the characteristics and biases of ERA5 itself.

No claim of observational forecast accuracy should be made.

---

# 67. Output structure

```text
outputs/
├── data/
│   ├── dataset_manifest.json
│   ├── train_2019_01.json
│   ├── train_2019_07.json
│   ├── validation_2020_04.json
│   └── test_2021_10.json
├── baseline/
│   └── persistence_metrics.json
├── frozen_model/
│   ├── validation_metrics.json
│   └── forecasts/
├── adaptation/
│   ├── training_history.json
│   └── validation_metrics.json
├── frozen/
│   └── frozen_experiment.json
├── test/
│   ├── comparison.csv
│   ├── variable_metrics.csv
│   ├── lead_metrics.csv
│   └── forecasts/
├── future/
│   └── new_origin_forecast/
├── artifacts/
│   └── aurora_lora/
│       ├── adapter.safetensors
│       └── manifest.json
├── figures/
├── provenance/
│   ├── model_manifest.json
│   └── experiment_manifest.json
└── workshop_summary.json
```

---

# 68. Provenance

`experiment_manifest.json` SHOULD record:

```text
notebook_spec
notebook_profile
notebook_mode
workshop_revision
timestamp

model:
  id
  revision
  checkpoint_identity
  source_digests
  converted_digests
  parameter_count
  architecture

dataset:
  source = ERA5 via WeatherBench 2
  resolution = 1.5°
  object_digests
  window_ids
  dataset_digest
  attribution

state_contract:
  surface_variables
  atmospheric_variables
  static_variables
  pressure_levels
  units
  history_steps
  timestep_hours

adaptation:
  type = LoRA
  rank
  tensor_count
  parameter_count
  epochs
  learning_rate
  seed
  best_epoch
  loss_scales

evaluation:
  reference = ERA5 analysis
  baseline = persistence
  metric = latitude-weighted RMSE
  skill_semantics = model_rmse / persistence_rmse
  leads = [6,12,18,24]
```

---

# 69. Notebook metadata

```json
{
  "dimer": {
    "notebook_spec": "2.1",
    "notebook_profile": "E2E",
    "notebook_mode": "WORKSHOP",
    "standalone": true,
    "capability": "weather-and-earth-system-forecasting",
    "carrier": "Aurora short-range global forecasting and LoRA adaptation workshop",
    "dataset": "ERA5 via WeatherBench 2 at 1.5 degrees",
    "canonical_runtime": "NVIDIA Tesla T4",
    "worker_required": false,
    "credentials_required": false,
    "clean_runtime_evidence": "pending"
  }
}
```

---

# 70. Release acceptance

| Requirement | Required |
|---|---:|
| Notebook Spec 2.1 | PASS |
| `E2E` / `WORKSHOP` declaration | PASS |
| Fresh T4 `Run all` | PASS |
| No Git clone | PASS |
| No DIMER runtime source fetch | PASS |
| No DIMER services | PASS |
| No credentials | PASS |
| Model asset digests | PASS |
| Static-field asset digests | PASS |
| SafeTensors conversion/verification | PASS |
| 57 WB2 objects verified | PASS |
| Four windows reconstructed | PASS |
| Data-contract refusal probes | PASS |
| Persistence baseline | PASS |
| Frozen Aurora 6–24 h rollout | PASS |
| Latitude-weighted RMSE | PASS |
| LoRA adaptation actually runs | PASS |
| Validation-only epoch selection | PASS |
| Freeze-before-test | PASS |
| Independent October 2021 test | PASS |
| Per-variable metrics | PASS |
| Per-lead metrics | PASS |
| Spatial error maps | PASS |
| New-origin forecast | PASS |
| Adapter export | PASS |
| Fresh reload parity | PASS |
| BYOD positive case | PASS |
| BYOD invalid-input refusal | PASS |
| Provenance export | PASS |

---

# 71. Suggested registry entry

```markdown
| Notebook | Profile | Mode | Capability | Runtime | Sample | BYOD | Run-all | Status |
|---|---|---|---|---|---|---|---|---|
| `DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb` | `E2E` | `WORKSHOP` | Aurora short-range global weather forecasting + LoRA adaptation | T4 | four real ERA5 / WeatherBench 2 windows at 1.5° | NetCDF | pending | candidate |
```

---

# 72. Implementation principle

The notebook's central story should be:

> **A foundation model is not automatically skillful simply because it is pretrained.**

For this tutorial:

```text
pretrained Aurora
+ severe resolution shift
→ frozen model loses to a simple persistence forecast
```

Then:

```text
small amount of representative gridded data
+ 540k-parameter LoRA
→ adapted model recovers substantial short-range skill
```

That is a much more useful workshop than presenting only the successful adapted result.

The lesson is not:

> "Aurora beats persistence."

It is:

> **Forecast models must be evaluated against meaningful baselines in the actual deployment domain, and parameter-efficient adaptation can help when the input distribution differs from pretraining.**