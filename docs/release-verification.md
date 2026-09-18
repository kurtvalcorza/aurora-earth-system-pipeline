# Release verification

`tutorials/aurora_earth_system_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the upstream
  package's default revision for the small checkpoint is the one allowed second hash);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `AuroraPipeline.from_pretrained(weights_dir=..., device=..., use_lora=True, report=print)` so both pickle audits
  and the conversion are printed before the model loads, `fetch_sample_dataset` from the pinned cache path,
  `load_byod_dataset`, `validate_dataset`, `write_window_netcdf`, `validate_inputs` with the refusal probes,
  `pipe.predict` with the determinism assertion, `persistence_only`, `pipe.evaluate` on the frozen model and on the
  validation and test windows after adaptation with the skill assertions, `pipe.adapt` with its explicit
  hyperparameters, `pipe.predict` from a new origin, `pipe.save_artifact`, `AuroraPipeline.from_artifact` and the
  reload-parity assertion, and the provenance fields `served_from_pickle: False`, `remote_code_executed: False`
  and the data base URL), the five expected `outputs/` paths, the learner-facing statements (both assets are
  pickles unpickled once, the data is not the native resolution, out-of-distribution, persistence and the diurnal
  cycle, latitude-weighted RMSE, split by period, Copernicus attribution) and the gated-off BYOD default;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path,
  a mutable `revision='main'`, direct `huggingface_hub` / `safetensors` / `urllib` / `aurora` / `Blosc` /
  `rollout` / `Unpickler` use or `torch.load(` / `pickle.load` **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load` or `torch.load(` without `weights_only=True` anywhere, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; crafted pickles, temporary manifests, synthetic
windows and an injected fetcher, no weights and no model library — `tests/test_model_backed.py` is skipped there).
These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/aurora-0.25-small/` or the data cache `weights/wb2-era5-1p5deg/` (the standalone path writes
   the manifest itself, stages all three listed files from the Hub, audits and converts both pickles, and fetches
   the 57 pinned WeatherBench 2 objects, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `ORIGIN = 1`, `ROLLOUT_STEPS = 4`, `MAX_LEAD_STEPS = 4`, `EPOCHS = 6`,
   `LEARNING_RATE = 1e-3`, `TRAINABLE = 'lora'`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `microsoft-aurora==2.0.1`, `timm==1.0.29`, `einops==0.8.2`,
   `xarray==2026.7.0`, `netCDF4==1.7.4`, `numcodecs==0.17.0`, `numpy==2.5.3`, `safetensors==0.8.0`,
   `huggingface-hub==1.32.0` (an interpreter restart after the install is expected where the runtime's preinstalled
   torch, numpy or xarray differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `AuroraPipeline`, `audit_pickle`, `convert_model`,
     `build_model`, `verify_snapshot`, `verify_converted`, `stage_missing_files`, `validate_inputs`,
     `window_digest`, `fetch_object`, `fetch_sample_window`, `fetch_sample_dataset`, `validate_dataset`,
     `load_byod_dataset`, `write_window_netcdf`, `lat_weighted_rmse`, `forecast_metrics`, `persistence_only`)
     with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 3 entries fetched from `microsoft/aurora` at the immutable revision and `verify_snapshot`
     reporting 3 verified files;
   - the model cell printing the **conversion record** with both static audits (checkpoint globals
     `collections.OrderedDict` / `torch._utils._rebuild_tensor_v2` / `torch.FloatStorage`, static globals
     `numpy.core.numeric._frombuffer` / `numpy.dtype`, 0 violations, audit digests `e7b998d0…` / `aeec283f…`)
     and both converted files (`fc03b5fc…` 451,230,408 bytes; `9bd430b6…` 12,459,136 bytes), then the load report
     on the chosen device with source "converted from the manifest-verified source pickles";
   - the dataset manifest with 4 windows on a (121, 240) grid, 8 steps each, 24 forecast origins, the time span
     2018-12-31 to 2021-10-01, digest `4ee4be11…`, the written `outputs/aurora_earth_system_sample_window.nc`
     (about 24 MB), and four refusals (wrong levels, odd longitude count, implausible temperature, irregular time
     spacing);
   - a 4-step roll-out from origin 1 of the test window with shape (120, 240), valid times 2021-09-30 12 UTC to
     2021-10-01 06 UTC, and a repeated call bit-identical (the cell asserts both);
   - the persistence baseline and the frozen model's error per lead (on the sample: 6-hour mean skill ≈ 1.57 with
     1 of 9 variables beating persistence);
   - `pipe.adapt` printing epoch 0 as the frozen model, 540,672 trainable of 113,338,256 parameters, 12 training
     samples, and a six-epoch history with validation loss falling from ≈ 0.18 to ≈ 0.05;
   - `pipe.evaluate` on the test window with the three-way comparison and
     `outputs/aurora_earth_system_evaluation_report.json` written (the cell asserts the adapted 6-hour mean skill is
     below the frozen model's — on the sample ≈ 0.90 versus ≈ 1.57 — and that more variables beat persistence);
   - a 4-step forecast from a new origin with `outputs/aurora_earth_system_forecast.json` written;
   - `pipe.save_artifact` writing `outputs/aurora_earth_system_adapter/{adapter.safetensors,manifest.json}` (80
     tensors, about 2.1 MB), and `AuroraPipeline.from_artifact` reloading it with identical forecast fields (the
     cell asserts both differences below 1e-6);
   - `outputs/aurora_earth_system_result.json` written with `NOTEBOOK_SOURCE`, the model identity, the provenance
     block (`served_from_pickle: false`, `remote_code_executed: false`, both audits, both converted digests, the
     data base URL), the runtime versions, the summary and the reload parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directory and the data cache were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `aurora_earth_system_colab.ipynb` | __LOCAL_ROW__ | | | |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/aurora_earth_system_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/aurora_earth_system_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| __LOCAL_EXEC__ | | | | | |

## Current status

The notebook source is complete and passes all static checks, including the generator parity checks (`--check` OK).
The repository stays at **Candidate** until a Colab or fresh-container run of the exact release revision is recorded
above.
