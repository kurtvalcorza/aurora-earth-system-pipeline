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
| `aurora_earth_system_colab.ipynb` (`E2E`) | `4a828fc` / `42fb3883` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-aurora-earth-system` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 68 files, 1123 MB staged into a clean cache (Hub snapshot + the 57 pinned WeatherBench 2 objects); comparison mean skill vs persistence (RMSE ratio, lower is better) frozen → adapted: 6 h 1.568 → 0.904 (1 → 6 of 9 variables beating persistence), 12 h 1.328 → 0.849 (1 → 8), 18 h 1.237 → 0.867 (1 → 7), 24 h 1.294 → 0.969 (2 → 6); reload parity {max_abs_surf_diff: 0, max_abs_atmos_diff: 0}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-aurora-earth-system/v1/evidence/` in the workspace |
| `aurora_earth_system_colab.ipynb` | `6017ca5` / `b7e27bf3` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/aurora_earth_system_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/aurora_earth_system_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `4a828fc` / `42fb3883` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-aurora-earth-system` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 283.5 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 68 files, 1123 MB staged into a clean cache (Hub snapshot + the 57 pinned WeatherBench 2 objects); comparison mean skill vs persistence (RMSE ratio, lower is better) frozen → adapted: 6 h 1.568 → 0.904 (1 → 6 of 9 variables beating persistence), 12 h 1.328 → 0.849 (1 → 8), 18 h 1.237 → 0.867 (1 → 7), 24 h 1.294 → 0.969 (2 → 6); reload parity {max_abs_surf_diff: 0, max_abs_atmos_diff: 0}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-aurora-earth-system/v1/evidence/` in the workspace |
| 2026-09-18 | `6017ca5` / `b7e27bf3` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cpu`, `microsoft-aurora 2.0.1`) | Default sample path (stage → verify → **static audits + conversion of both pickles in the notebook** → strict rebuild with LoRA → pinned-data assembly from the cache → validate → refusal probes → 4-step roll-out + determinism → persistence + frozen evaluation at 4 leads → LoRA adapt → evaluate → new-origin forecast → export → reload); the three Hub files and the 57 data objects were pre-staged, so `stage_missing_files` fetched 0 of 3 entries, `verify_snapshot` verified 3, every data object was served from the cache after its digest check, and `convert_model` produced the pinned digests (`fc03b5fc…`, `9bd430b6…`) | 238.3 s | **PASSED** — 11/11 code cells; audits 0 violations; test window (October 2021, 121 × 240): frozen 6 / 12 / 18 / 24 h mean skill 1.569 / 1.328 / 1.238 / 1.295 with 1 / 1 / 1 / 2 of 9 variables beating persistence; LoRA adaptation 540,672 params, 12 samples, 6 epochs, 194.6 s, validation loss 0.185 → 0.052; **adapted 0.897 / 0.839 / 0.858 / 0.957 with 6 / 8 / 8 / 6 of 9**; `2t` 6 h 2.666 → 2.453 → 1.805 K, `msl` 269 → 415 → 229 Pa, `z500` 249 → 411 → 284 m²/s²; adapter 2,173,200 B (80 tensors); reload parity 0.0 / 0.0. Pre-flight; hosted clean-runtime run still required |

## Current status

**Release-grade.** The `E2E` notebook blob `42fb3883` (committed at `4a828fc`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 283.5 s, 68 files, 1123 MB fetched (Hub snapshot + the 57 pinned WeatherBench 2 objects) and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

## Supplemental weather and Earth-system forecasting workshop — `tutorials/DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb`

This entry applies only to the supplemental workshop notebook, not the primary tutorial executions above.

### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb) and authorized merging PR #10 (merge commit `1f768ce`). The file is archived byte-for-byte, SHA-256 `b166e2b78cd8385520b24beb3af3e1016c25b9a25380a811642c50ffdd0c8588`. All 22 code cells have execution counts, 38 saved outputs and zero saved errors. Code-cell sources match commit `8fba5673599ac403eae964f3269748ce053905c7`, tutorial blob `9db1e28127f6bec5886b3608fa8c15df8c6b647a`, apart from Colab-inserted `# @title` lines. Later commits on `main` that touch the notebook (`2bf00ef` (AI User Disclosure)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: Default E2E path: `microsoft/aurora` AuroraSmallPretrained at revision `a96afd7ee6d6`, ERA5 via WeatherBench2 at an out-of-distribution 1.5° grid, frozen versus LoRA-adapted rollouts. BYOD was not exercised.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, microsoft-aurora 2.0.1, xarray 2026.7.0, NumPy 2.1.3, CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Code cells 2–22 carry counts 2 to 22 in order; the first code cell (notebook controls) carries 23, so it was re-executed after the run completed. Runtime freshness and absence of other manual reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): Independent test mean RMSE ratio against persistence, frozen → adapted: 1.568 → 0.914 (6 h), 1.328 → 0.866 (12 h), 1.237 → 0.885 (18 h), 1.294 → 0.991 (24 h); variables beating persistence (of 9), frozen → adapted: 1 → 6, 1 → 8, 1 → 8, 2 → 6. LoRA adapter: 540,672 trainable parameters, best epoch 6, 2,173,200 bytes (SHA-256 `0585b94cfdcf…`); fresh reload parity max absolute difference 0.0 (PASS).

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (FULL/BYOD) or REL12 qualification gates, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).

### Kaggle T4 execution of revision `62f41b1` (PR #12) — 2026-09-28

Executed notebook: [`execution-evidence/2026-09-28/DIMER_Weather_and_Earth_System_Forecasting_Workshop_62f41b1_kaggle-t4-activity.ipynb`](execution-evidence/2026-09-28/DIMER_Weather_and_Earth_System_Forecasting_Workshop_62f41b1_kaggle-t4-activity.ipynb), SHA-256 `db73a098865b243ca08c0c636bd0d9c26f4ae7b51d381f5af02693918c8ba3ea`, archived byte for byte with the executor's [run summary](execution-evidence/2026-09-28/DIMER_Weather_and_Earth_System_Forecasting_Workshop_62f41b1_kaggle-t4-activity.run_summary.json) (SHA-256 `ae6080a6217cbcd471784ecc49eeabc85418e2d96ee5b4b9254409f06bf0e1a1`).

Source match: the executor fetched the notebook from GitHub at `62f41b135146b994a4a3123c996ffc82dd094b19` and verified its Git blob `385da6db6424b92fdec83f82a0e68f08abeed02e` before execution. The only change before Run all was the form toggle `RUN_LONGER_ROLLOUT_ACTIVITY = True` (cell `dimer-weather-workshop-47`); cell ids and order match the committed notebook, and no other source line differs.

Runtime: Kaggle private kernel `kurtvalcorza/dimer-nb2-aurora-weather-pr12-activity` v1, Tesla T4, Python 3.12.13, torch 2.14.0+cu130 after the notebook's pinned install, empty Hugging Face cache at start, no repository checkout. Fresh IPython kernel via nbclient; no restart after the install cell. Run all 306.9 s; export cell re-run 0.3 s. Peak VRAM was not recorded.

Executed cells: 23 of 23 code cells, zero errors. Counts 1–23 in order, except the export cell (`dimer-weather-workshop-51`), which carries 24 because it was re-run once after the completion cell; its first-run outputs are replaced by the re-run's.

Results (sample-sanity measures on the built-in data, not general model rankings):

| Lead | Frozen mean RMSE ratio | Adapted | Variables beating persistence (of 9), frozen → adapted |
|---|---|---|---|
| 6 h | 1.568 | 0.914 | 1 → 6 |
| 12 h | 1.328 | 0.867 | 1 → 8 |
| 18 h | 1.237 | 0.888 | 1 → 7 |
| 24 h | 1.294 | 0.997 | 2 → 6 |

- Section 4: the upstream checkpoint (451,339,106 bytes, SHA-256 `f80f78de…`) was audited and converted in this runtime; both converted files matched the pinned digests (`fc03b5fc5764…`, `9bd430b666d9…`).
- Frozen ratios equal the 2026-09-26 Colab run to six decimals. Adapted ratios differ in the third decimal (e.g. 24 h 0.997 vs 0.991; 18 h 7 vs 8 variables), and the adapter digest differs (`23043cf9…` vs `0585b94c…`). Correction (2026-09-28, after the Colab run below): this is not cross-runtime nondeterminism. The Colab run of the same notebook blob reproduces this run's losses, metrics and adapter digest exactly. See the next section.
- LoRA: best epoch 6, 540,672 trainable parameters, adapter 2,173,200 bytes. Freeze `experiment_sha256` `8baa7d685d57…`.
- Fresh-directory reload: max absolute difference 0.0 / 0.0 (PASS); consumer checks passed (adapter.v1 format, converted-base digests match, `lora` scope, 80 tensors).
- Longer-rollout activity (validation window, one fixed origin `2020-03-31T06:00`): 6–36 h scored; adapted ratio 0.924 / 0.899 / 0.889 / 0.985 / 0.920 / 0.941; `canonical_results_unchanged: True`.
- Export re-run: report bundle with 33 files, SHA-256 `7bd78219e2b1…`, `optional_outputs` listing the three activity files. The bundle ZIP itself was written outside `outputs/` and was not retained by the executor; the 33 files under `outputs/` were.

| Journey | Verdict |
|---|---|
| Default path (fresh runtime, Run all) | Pass |
| Section 4 audit and conversion with pinned converted digests | Pass |
| Freeze → independent test | Pass |
| Fresh-directory reload with consumer checks | Pass |
| Longer-rollout activity (36 h) | Pass |
| Export re-run after the activity | Pass |
| BYOD NetCDF (valid and invalid) | Not assessed in this run |
| Repeated Run all in a warm runtime | Not assessed in this run |

Evidence boundary: this is an agent-run Kaggle clean-room execution, not a maintainer-supplied Colab run. The executed notebook and run summary were inspected; the report ZIP bytes were not. Status remains **Candidate**, and `metadata.dimer.clean_runtime_evidence` stays `pending` as authored.

Still open: hosted BYOD NetCDF qualification (valid and invalid); consumption of an exported adapter by `AuroraPipeline.from_artifact`; the 57-object data-pin parity; peak VRAM.

### Maintainer-supplied Colab execution of `main` at `4f4fe8e` — 2026-09-28

Executed notebook: [`execution-evidence/2026-09-28/DIMER_Weather_and_Earth_System_Forecasting_Workshop_4f4fe8e_colab-default.ipynb`](execution-evidence/2026-09-28/DIMER_Weather_and_Earth_System_Forecasting_Workshop_4f4fe8e_colab-default.ipynb), SHA-256 `011fec14df140472538b1152f663b695cfe3de252f85e01666078e6160a013e7`, archived byte for byte (run id `20260928T214804Z-851`).

Source match: all 59 cell ids and their order match the notebook at `4f4fe8e` (blob `385da6db6424b92fdec83f82a0e68f08abeed02e`, unchanged since `62f41b1`). No code-cell source line differs, and every form control is at its default.

Runtime: Google Colab, Tesla T4, Python 3.13.15, torch 2.14.0+cu130, microsoft-aurora 2.0.1, xarray 2026.7.0, NumPy 2.1.3 (the same stack as the 2026-09-26 run). Peak VRAM was not recorded.

Executed cells: 23 of 23 code cells, execution counts 1–23 in order, zero errors. The artifact is consistent with a single Run all; runtime freshness is not independently established by the file.

Results: the independent-test comparison, per-variable table, training history, best epoch (6), adapter SHA-256 `23043cf9bceb…` and reload parity (0.0 / 0.0, consumer checks PASS) are identical to the Kaggle T4 run of `62f41b1` above. Section 4 re-converted the base in this runtime and verified the pinned converted digests. Freeze `experiment_sha256` `eefe721001a2…`; it differs from the Kaggle run's (`8baa7d685d57…`) because the digested record includes the measured training time (37.18 s here, 36.46 s on Kaggle). The export listed a 33-file report bundle, SHA-256 `057b5ce69959…`; the ZIP bytes were not supplied.

Comparison with the 2026-09-26 Colab run (pre-fix notebook, same software stack): frozen results are identical. The LoRA training history is identical through epoch 2 and differs from epoch 3 at about 1e-9 (e.g. epoch-3 train loss 0.068623723462224 vs 0.06862372159957886). That small drift grows to third-decimal differences in the adapted test ratios (e.g. 24 h 0.997 vs 0.991) and a different adapter digest (`23043cf9…` vs `0585b94c…`). Two runtimes (Kaggle Python 3.12, Colab Python 3.13) agree exactly on the new blob, so the drift most likely comes from the notebook change in PR #12 rather than the runtime. Which change causes it was not isolated. It is a numerical change of this size, not a change in method.

| Journey | Verdict |
|---|---|
| Default path (Run all) | Pass |
| Section 4 audit and conversion with pinned converted digests | Pass |
| Freeze → independent test | Pass |
| Fresh-directory reload with consumer checks | Pass |
| Longer-rollout activity | Not assessed in this run ("Activity not run"); covered by the Kaggle run above |
| BYOD NetCDF (valid and invalid) | Not assessed in this run |

Evidence boundary: saved outputs were inspected; execution was not independently repeated. Status remains **Candidate**, and `metadata.dimer.clean_runtime_evidence` stays `pending` as authored.

Still open: hosted BYOD NetCDF qualification (valid and invalid); consumption of an exported adapter by `AuroraPipeline.from_artifact`; the 57-object data-pin parity; peak VRAM.
