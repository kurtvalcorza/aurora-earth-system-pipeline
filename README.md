# Aurora Earth-System Pipeline

DIMER-oriented pipeline for **Aurora 0.25° small pretrained** (`microsoft/aurora`, the Earth-system foundation model of Bodnar et al.), pinned to an immutable Hugging Face revision. The repository exposes 6-hourly global weather forecasting from two gridded analyses with autoregressive roll-out, latitude-weighted RMSE per variable and lead time against the persistence baseline, a gridded-window contract with explicit ceilings, a bounded LoRA fine-tuning contract with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `microsoft/aurora`
- Revision: `a96afd7ee6d65e3bd2d476f3be798a25a56f2296`
- Source assets: `aurora-0.25-small-pretrained.ckpt` (451,339,106 bytes, SHA-256 `f80f78de…`) and `aurora-0.25-static.pickle` (12,459,115 bytes, SHA-256 `e382103f…`) — both pickles, both converted once to safetensors and never served (see below)
- Upstream weight license: MIT
- Upstream task: global atmospheric forecasting at 0.25° and a 6-hour step from ERA5-like analyses; this repository uses the **small** checkpoint (113 M parameters, published by upstream "only for debugging") for forecasting and for fine-tuning to local gridded data
- Runtime: `microsoft-aurora==2.0.1` + `torch==2.14.0` + `timm` + `einops` (+ `xarray`/`netCDF4` for NetCDF I/O and `numcodecs` for the tutorial data) — the model class comes from PyPI, **no Hub-hosted code and no served pickle**
- Repository adaptation: **E2E** (LoRA fine-tuning of the 80 upstream LoRA tensors on one-step forecasts, optionally also the token embeddings and decoder heads, with a portable safetensors adapter)

## Two things to know before you start

**Both upstream assets are pickles, and neither is served.** The checkpoint is a torch archive whose pickle references only `collections.OrderedDict` and torch's tensor-rebuild helpers; the static file is a plain pickle of three numpy arrays. `audit_pickle()` lists every global each would import without executing anything and refuses anything outside a per-file allow-list (audit digests pinned); `convert_model()` unpickles each **once** through a restricted loader — `torch.load(weights_only=True)` for the checkpoint, a `find_class` allow-list of two numpy names for the static fields — applies the upstream compatibility shim, loads `AuroraSmallPretrained` strictly, and writes `aurora-0.25-small-pretrained.safetensors` (451,230,408 bytes) and `aurora-0.25-static.safetensors` (12,459,136 bytes), whose digests are pinned in `pipeline.py`; `from_pretrained()` loads only those. The conversion is deterministic and its fidelity is measured against upstream's own regression fixture: relative mean deviations 1.2×10⁻⁷ to 3.5×10⁻⁵ per variable, within the upstream tolerances (`docs/WEIGHTS.md`).

**The tutorial runs the model six times coarser than it was trained.** The data is real ERA5 regridded to 1.5° by WeatherBench 2 (four two-day windows, 57 digest-pinned Zarr objects, about 196 MB, no credential), because a native 0.25° window is sixty times larger. That makes the frozen model's numbers an out-of-distribution measurement — it loses to persistence on eight of nine variables at 6 h — and makes the adaptation contract the point: a 540 k-parameter LoRA fine-tuning on twelve one-step forecasts brings the mean skill below persistence at every lead to 24 h on a held-out window from a different year. None of it says anything about Aurora at its native resolution.

## Quick start

```python
from aurora_earth_system_pipeline import AuroraPipeline, fetch_sample_dataset, persistence_only

pipe = AuroraPipeline.from_pretrained(use_lora=True)   # verifies the snapshot, audits + converts both pickles once, loads safetensors
windows = fetch_sample_dataset()                        # four 8-step windows of ERA5 at 1.5° from pinned WeatherBench 2 objects
test = windows["test-2021-10"]
forecast = pipe.predict(test, origin=1, steps=4)        # 6 h .. 24 h from the analysis at step 1
print(forecast["forecasts"][0]["valid_time"], forecast["forecasts"][0]["surf"]["2t"].shape)

print(persistence_only(test, max_lead_steps=4)["variables"]["z500"])
print(pipe.evaluate(test, max_lead_steps=4)["summary"])          # frozen model vs persistence, per lead
pipe.adapt([windows["train-2019-01"], windows["train-2019-07"]], windows["val-2020-04"])
print(pipe.evaluate(test, max_lead_steps=4)["summary"])          # adapted model
pipe.save_artifact("outputs/adapter")
```

`predict()`, `evaluate()` and `adapt()` take windows — `{lat, lon, levels, times, surf, atmos, static}` mappings on a global equiangular grid (17..721 latitudes spanning the poles, a multiple of 4 or one more; 32..1440 longitudes covering the full circle, a multiple of 4), exactly the 13 standard pressure levels, 3..64 analyses spaced exactly 6 h apart, SI units within plausibility ranges. The model predicts the rows the patch size covers (121 → 120 at 1.5°). Validation is structural: nothing checks that a window is dynamically consistent or real.

## Weights layout

```
weights/aurora-0.25-small/   README.md  aurora-0.25-small-pretrained.ckpt  aurora-0.25-static.pickle  dimer-base-manifest.json
                             aurora-0.25-small-pretrained.safetensors  aurora-0.25-static.safetensors  (git-ignored, converted)
weights/wb2-era5-1p5deg/     the 57 pinned WeatherBench 2 objects, cached on first fetch (git-ignored)
```

`from_pretrained()` calls `stage_missing_files()` (fetches only absent manifest entries, only at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (byte size + SHA-256 of all 3 manifest entries and of the converted files when present), converts the pickles when the safetensors are absent, and refuses on the first mismatch. With `require_source=False` digest-verified converted files are accepted without the pickles — the DIMER-hosted shape. `docs/WEIGHTS.md` records the provenance, the audits, the conversion, the fidelity check and the DIMER hosting notes.

## Sample data

`fetch_sample_dataset()` assembles four windows — `train-2019-01` (2018-12-31 to 2019-01-01), `train-2019-07`, `val-2020-04`, `test-2021-10` — of eight 6-hourly ERA5 analyses at 1.5° (240 × 121, with poles) from WeatherBench 2's public bucket: every Zarr chunk, `.zarray` descriptor and coordinate is pinned by byte size and SHA-256 in `WB2_OBJECTS`, refused on mismatch before decoding, decoded with `numcodecs` (Blosc/LZ4) and reordered into the Aurora layout. Distinct years and seasons per role, so nothing in the test window is a near-duplicate of anything trained on. ERA5 is © ECMWF / Copernicus Climate Change Service (licence to use Copernicus products). `write_window_netcdf` / `load_byod_dataset` round-trip NetCDF, the BYOD format.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors — the 80 LoRA tensors, about 2 MB; plus the embeddings and heads with `trainable="lora+heads"`) and a `manifest.json` recording the artifact format, the exact base model id and revision, the converted-base digests, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `AuroraPipeline.from_artifact(dir)` re-verifies the base files, checks the manifest and digest before deserialising, rebuilds the model with LoRA and overlays the tensors.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: crafted pickles, temporary manifests, synthetic windows and an injected fetcher, never the weights; `tests/test_model_backed.py` runs only where `microsoft-aurora` and the converted files are present (strict load, deterministic cropped forecasts, LoRA neutrality, a one-epoch adaptation and artifact round trip) and is skipped in CI.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/aurora-earth-system-pipeline/blob/main/tutorials/aurora_earth_system_colab.ipynb)

`tutorials/aurora_earth_system_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It downloads and converts the pinned pickles in the runtime (the audit and conversion records are printed before the model loads), fetches the pinned data, and runs the sample path: validation, a 24-hour roll-out, persistence and frozen-model baselines per lead, LoRA fine-tuning, held-out evaluation, a forecast from a new origin, adapter export and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Release-grade** — the `E2E` notebook blob `42fb3883` (committed at `4a828fc`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 283.5 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Licensing

- Upstream weights: MIT (`microsoft/aurora`; the `microsoft/aurora` code is MIT), staged from the pinned revision and converted, not modified, into the served safetensors.
- Tutorial data: ERA5 (© ECMWF, Copernicus Climate Change Service; licence to use Copernicus products, attribution required) as regridded and served by WeatherBench 2; fetched at run time, never committed.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
