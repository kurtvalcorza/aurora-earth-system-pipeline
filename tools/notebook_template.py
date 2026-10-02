"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E weather-forecasting workflow: the pinned Aurora small checkpoint and
static fields (both pickles) are digest-verified, statically audited and converted once into
safetensors, four two-day windows of real ERA5 at 1.5° are fetched from pinned WeatherBench 2 objects,
validated and assigned to roles, the frozen model is rolled out and scored against persistence per lead
time, a bounded LoRA fine-tuning runs in the kernel, the held-out window is scored again, and the
adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "aurora-earth-system-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/aurora_earth_system_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Faurora-ffcc4d?style=flat",
        "https://huggingface.co/microsoft/aurora",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-microsoft%2Faurora-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/microsoft/aurora",
    ),
    ("Paper", "https://img.shields.io/badge/Nature-10.1038%2Fs41586--025--09005--y-b31b1b.svg", "https://doi.org/10.1038/s41586-025-09005-y"),
]

TEMPLATE = {
    "package": "aurora_earth_system_pipeline",
    "repo_name": REPO,
    "stem": "aurora_earth_system",
    "notebook_name": "aurora_earth_system_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies (torch, microsoft-aurora, timm, "
        "einops, xarray, netCDF4, numcodecs, numpy, safetensors, huggingface-hub), stages and digest-verifies the pinned Aurora "
        "small checkpoint (451 MB) and static fields (12 MB) from the Hub, statically audits both pickles against allow-lists, "
        "converts them once into safetensors with pinned digests, rebuilds the model from the installed package and loads it "
        "strictly with zero-initialised LoRA parameters, fetches four two-day windows of real ERA5 reanalysis at 1.5° from "
        "WeatherBench 2 (about 196 MB of pinned, digest-verified Zarr chunks — no credential), validates them and assigns them "
        "to training, validation and test roles, rolls the frozen model out to 24 h and scores it per variable and lead time "
        "against the persistence baseline, runs a bounded LoRA fine-tuning on one-step forecasts, scores the held-out window "
        "again at every lead, forecasts from a new origin, exports the adapter as safetensors with a manifest, and reloads that "
        "artifact into a fresh pipeline to verify forecast parity. The default path needs no repository clone, no DIMER worker "
        "or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path "
        "takes about six minutes of model time after the downloads."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "gridded analyses as a NetCDF file (coordinates `time`, `level`, `latitude`, `longitude`; surface variables `2t`, `10u`, "
        "`10v`, `msl` as (time, lat, lon); atmospheric variables `t`, `u`, `v`, `q`, `z` on the 13 standard levels as (time, level, "
        "lat, lon); static fields `static_lsm`, `static_z`, `static_slt` as (lat, lon); at least three 6-hourly steps on a global "
        "equiangular grid). Your window becomes the training, validation and test window of the same contract — validation, "
        "baselines, adaptation, held-out evaluation, inference, artifact export and reload parity — and the notebook says when a "
        "single window makes the split degenerate. The expected schema, the grid rules and the ceilings are stated in the "
        "Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the "
        "default path."
    ),
    "pipeline_class": "AuroraPipeline",
    "model_load": "AuroraPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), use_lora=True, report=print)",
    "weights_key": "aurora-0.25-small",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "timm", "xarray", "numcodecs"],
    "title": "Aurora 0.25° small — DIMER E2E weather-forecasting fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "6-hourly global weather forecasting from two gridded analyses, lead-time evaluation against persistence, and bounded LoRA fine-tuning of the foundation model",
    "intro": (
        "Aurora is a foundation model for the Earth system (Bodnar et al., Nature 2025): a 3D Swin transformer between a "
        "Perceiver encoder and decoder that takes two consecutive global analyses — four surface variables, five atmospheric "
        "variables on 13 pressure levels, three static fields — and returns the state six hours later; rolling that step "
        "forward gives a forecast. The checkpoint here is the **0.25° small pretrained** variant (113 M parameters), which the "
        "upstream authors publish for debugging and testing; the production checkpoints are 1.3 B parameters and are not "
        "packaged here.\n\n"
        "Two things about this row are handled in the open. **Both upstream assets are pickles.** Section 3 downloads and "
        "digest-verifies them, statically lists every global each pickle would import (a state dict of tensors; three numpy "
        "arrays), refuses anything outside those allow-lists, unpickles each exactly once through a restricted loader, and writes "
        "safetensors whose digests are pinned in the carried module. The model you run is rebuilt from the installed "
        "`microsoft-aurora` package and loads those files strictly. **The data is real ERA5 at 1.5°, not the model's native "
        "0.25°.** Four two-day windows come from WeatherBench 2's public bucket as pinned, digest-verified Zarr chunks — a native "
        "window would be sixty times larger — so the frozen model runs six times coarser than it was trained, an out-of-distribution "
        "use. That is the honest setting for the adaptation contract: Section 6 shows the frozen model losing to persistence at "
        "this resolution, and Section 8 shows what a 540 k-parameter LoRA fine-tuning on twelve one-step forecasts recovers."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify two "
        "pickled upstream assets, read their static audits and see them converted into safetensors; fetch and validate real "
        "gridded reanalysis from pinned objects; roll a foundation weather model out to 24 h and read latitude-weighted RMSE per "
        "variable and lead time against persistence; run a bounded LoRA fine-tuning with explicit hyperparameters; evaluate the "
        "adapted model on an independent window at every lead; forecast from a new origin; and export a safetensors adapter that "
        "reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "the 0.25° native-resolution path and the 1.3 B-parameter production checkpoints, the wave, air-pollution and 0.1° "
        "variants, ensemble forecasting, tropical-cyclone tracking, multi-step (roll-out) fine-tuning, climatology and "
        "operational-forecast baselines, the published WeatherBench scores, and any claim that a 1.5° two-day window stands in "
        "for an operational evaluation. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is enough — one 6-hour step at 1.5° takes about 0.7 s and the default fine-tuning about four minutes — and CUDA is used automatically when present. About 2 GB of RAM is needed for the model and the four windows.",
        "- **Knowledge:** what a gridded atmospheric analysis is (pressure levels, surface fields, an equiangular grid), what a lead time and a persistence forecast are, and how RMSE is read.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint and static file are pickles. Each is digest-verified, statically audited against an allow-list (audit digests pinned) and unpickled **once** through a restricted loader to produce the safetensors the model is actually loaded from. No Hub-hosted Python module is imported; `microsoft-aurora` is installed from PyPI at a pinned version.",
        "- **Data contract:** a window is `{{lat, lon, levels, times, surf, atmos, static}}` on a global equiangular grid — latitudes spanning 90 to −90 (17..721 rows, a multiple of 4 or one more), longitudes covering 0 to 360 (32..1440 columns, a multiple of 4), exactly the 13 standard pressure levels, 3..64 analyses spaced exactly 6 h apart, SI units (K, m/s, Pa, kg/kg, m²/s²) within plausibility ranges. The model predicts the rows the patch size covers (121 → 120 at 1.5°). BYOD accepts NetCDF in the shape Section 4 writes.",
        "- **Validation is structural, not meteorological:** nothing checks that the fields are dynamically consistent, that the analysis is real, or that the resolution is one the model was trained on — a smooth random field within range is forecast without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — a proprietary analysis or an embargoed forecast dataset is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 57 pinned objects (about 196 MB) from the public WeatherBench 2 bucket `storage.googleapis.com/weatherbench2` over HTTPS, digest-verified before decoding; ERA5 is © ECMWF/Copernicus under the licence to use Copernicus products.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample windows, validation and roles\n\n"
                "The default dataset is real ERA5 reanalysis regridded to 1.5° by WeatherBench 2: four windows of eight "
                "6-hourly analyses — two for training (late December 2018 / early January 2019, and July 2019), one for "
                "validation (April 2020) and one for testing (October 2021) — distinct seasons and years, so nothing in the test "
                "window is a near-duplicate of anything trained on. `fetch_sample_dataset` retrieves each Zarr chunk, `.zarray` "
                "descriptor and coordinate from the pinned table `WB2_OBJECTS` (byte size and SHA-256 per object), refuses a "
                "mismatch before decoding, decodes Blosc/LZ4 with `numcodecs`, and reorders the arrays into the Aurora layout "
                "(latitude 90 → −90, longitude 0 → 360). `validate_dataset` checks every window and the shared grid before any "
                "model runs.\n\n"
                "Look for: four windows of 8 steps on a 121 × 240 grid, 24 forecast origins in total, a dataset digest, and a "
                "written `outputs/{stem}_sample_window.nc` (the first three steps of the test window, the NetCDF shape BYOD "
                "expects). Four refusal probes follow — wrong pressure levels, an odd longitude count, an implausible temperature "
                "field, irregular time spacing — each rejected before `torch` does anything."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    byod_window = load_byod_dataset(byod_path)[0]\n"
                "    train_windows, val_window, test_window = [byod_window], byod_window, byod_window\n"
                "    data_source = 'BYOD (' + file_name + ') -- one window plays every role, so the held-out numbers are NOT independent'\n"
                "else:\n"
                "    windows = fetch_sample_dataset(cache_dir='weights/wb2-era5-1p5deg')\n"
                "    train_windows = [windows['train-2019-01'], windows['train-2019-07']]\n"
                "    val_window, test_window = windows['val-2020-04'], windows['test-2021-10']\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n\n"
                "dataset_manifest = validate_dataset([*train_windows, val_window, test_window] if not USE_BYOD else [test_window])\n"
                "print({{'data_source': data_source, 'n_windows': dataset_manifest['n_windows'], 'grid': dataset_manifest['shape'], 'steps_per_window': dataset_manifest['n_steps'], 'forecast_origins': dataset_manifest['forecast_origins']}})\n"
                "print({{'time_span': dataset_manifest['time_span'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'test_window': validate_inputs(test_window)}})\n"
                "print({{'2t_mean_K_by_window': {{w['name']: round(float(w['surf']['2t'].mean()), 2) for w in [*train_windows, val_window, test_window]}}}})\n\n"
                "def trim_window(window, steps):\n"
                "    return {{**window, 'times': window['times'][:steps], 'surf': {{k: v[:steps] for k, v in window['surf'].items()}}, 'atmos': {{k: v[:steps] for k, v in window['atmos'].items()}}}}\n\n"
                "sample_path = write_window_netcdf(trim_window(test_window, 3), 'outputs/{stem}_sample_window.nc')\n"
                "print({{'sample_netcdf': str(sample_path), 'megabytes': round(sample_path.stat().st_size / 1e6, 1)}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'wrong pressure levels': {{**test_window, 'levels': list(range(13))}},\n"
                "    'odd longitude count': {{**test_window, 'lon': test_window['lon'][:-1], 'surf': {{k: v[..., :-1] for k, v in test_window['surf'].items()}}, 'atmos': {{k: v[..., :-1] for k, v in test_window['atmos'].items()}}, 'static': {{k: v[..., :-1] for k, v in test_window['static'].items()}}}},\n"
                "    'implausible temperature': {{**test_window, 'surf': {{**test_window['surf'], '2t': test_window['surf']['2t'] + 500.0}}}},\n"
                "    'irregular time spacing': {{**test_window, 'times': test_window['times'][:-1] + ['2030-01-01T00:00:00']}},\n"
                "}}\n"
                "for name, window in probes.items():\n"
                "    try:\n"
                "        validate_inputs(window)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Zero-shot roll-out from the frozen model\n\n"
                "`pipe.predict` takes a window, an origin (the index of the latest analysis to use; the step before it is the "
                "second history step) and a number of 6-hour steps, and rolls the model forward autoregressively: each forecast "
                "becomes the next input. Returned fields are (120, 240) — the model predicts the rows the patch size covers and "
                "drops the last latitude row, exactly as it does at 0.25° (721 → 720).\n\n"
                "The pipeline was built with `use_lora=True`: the 80 LoRA tensors are zero-initialised, so this is the pretrained "
                "model's behaviour. Look for: four valid times 6 h apart, a second call returning bit-identical fields "
                "(deterministic on CPU), and global means that stay physical. The mean surface pressure is printed in Pa; 2 m "
                "temperature in K."
            ),
            "code": (
                "import time\n\n"
                "ORIGIN = 1  # @param {{type:\"integer\"}}\n"
                "ROLLOUT_STEPS = 4  # @param {{type:\"integer\"}}\n\n"
                "t0 = time.perf_counter()\n"
                "rollout_result = pipe.predict(test_window, origin=ORIGIN, steps=ROLLOUT_STEPS)\n"
                "print({{'origin_time': rollout_result['origin_time'], 'shape': rollout_result['shape'], 'seconds': round(time.perf_counter() - t0, 2), 'adapted': rollout_result['model']['adapted']}})\n"
                "for forecast in rollout_result['forecasts']:\n"
                "    print({{'lead_hours': forecast['lead_hours'], 'valid_time': forecast['valid_time'], '2t_mean_K': round(float(forecast['surf']['2t'].mean()), 3), 'msl_mean_Pa': round(float(forecast['surf']['msl'].mean()), 1), 'z500_mean': round(float(forecast['atmos']['z'][7].mean()), 1)}})\n"
                "again = pipe.predict(test_window, origin=ORIGIN, steps=1)\n"
                "repeat_identical = bool(np.array_equal(again['forecasts'][0]['surf']['2t'], rollout_result['forecasts'][0]['surf']['2t']))\n"
                "print({{'repeat_identical': repeat_identical, 'units': rollout_result['units']}})\n"
                "assert repeat_identical\n"
                "assert rollout_result['shape'] == (dataset_manifest['shape'][0] - dataset_manifest['shape'][0] % 4, dataset_manifest['shape'][1])"
            ),
        },
        {
            "md": (
                "## 6. Persistence baseline and the frozen model's error by lead time\n\n"
                "Every number in this notebook is a latitude-weighted RMSE (cos-latitude weights with unit mean, the WeatherBench "
                "convention) against the ERA5 analysis at the valid time, averaged over every forecast origin the window allows. "
                "The **persistence** forecast carries the latest analysis forward unchanged; at 6 h it is a strong baseline, and "
                "for 2 m temperature it gets *better* again at 24 h because the diurnal cycle comes back into phase — a reminder "
                "that a baseline is not a straw man. The **skill** column is RMSE(model) / RMSE(persistence): below 1 beats "
                "persistence.\n\n"
                "Expect the frozen model, run six times coarser than its training grid, to lose to persistence on most "
                "variables at 6 h (mean skill above 1) and to close the gap with lead time as persistence decays. This is the "
                "out-of-distribution number the adaptation is read against; it says nothing about Aurora at its native "
                "resolution, where the published model is far ahead of persistence."
            ),
            "code": (
                "MAX_LEAD_STEPS = 4  # @param {{type:\"integer\"}}\n\n"
                "baseline_persistence = persistence_only(test_window, max_lead_steps=MAX_LEAD_STEPS)\n"
                "print({{'persistence_only': {{name: {{lead: round(row['persistence'], 3) for lead, row in table.items()}} for name, table in baseline_persistence['variables'].items() if name in ('2t', 'msl', 'z500')}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_window, max_lead_steps=MAX_LEAD_STEPS)\n"
                "print({{'frozen_model_seconds': round(time.perf_counter() - t0, 1), 'origins': frozen_test['n_origins'], 'metric': frozen_test['metric']}})\n"
                "for name in ('2t', '10u', 'msl', 't', 'z500'):\n"
                "    print({{name: {{lead: {{'model': round(row['model'], 3), 'persistence': round(row['persistence'], 3), 'skill': round(row['skill'], 3)}} for lead, row in frozen_test['variables'][name].items()}}}})\n"
                "print({{'frozen_summary': {{lead: {{'mean_skill': round(s['mean_skill'], 3), 'variables_beating_persistence': f\"{{s['variables_beating_persistence']}}/9\"}} for lead, s in frozen_test['summary'].items()}}}})"
            ),
        },
        {
            "md": (
                "## 7. Bounded LoRA fine-tuning\n\n"
                "`pipe.adapt` trains the 80 LoRA tensors (rank 8, 540,672 parameters — 0.5 % of the model) that the upstream "
                "architecture places on the query/key/value and output projections of every backbone attention block, and "
                "nothing else. Each training sample is one 6-hour forecast from one origin of one training window (12 samples "
                "here); the loss is the mean over the nine variables of the MSE divided by a fixed per-variable scale, so a "
                "kelvin of temperature and a pascal of pressure count alike; AdamW at a fixed learning rate, seeded shuffling, "
                "no scheduler. Epoch 0 records the frozen model (LoRA at zero), and the epoch with the lowest validation loss is "
                "kept.\n\n"
                "Watch the validation loss fall by two thirds and the 6-hour skill of 2 m temperature and mean sea-level pressure "
                "drop below 1 within a few epochs; six epochs take about four minutes on CPU. `TRAINABLE = 'lora+heads'` also "
                "unfreezes the encoder token embeddings and decoder heads (713 k parameters) — in the build record it helped the "
                "upper air (z500 below persistence at 6 h) and hurt the surface, so LoRA alone is the default."
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-3  # @param {{type:\"number\"}}\n"
                "TRAINABLE = 'lora'  # @param [\"lora\", \"lora+heads\"]\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'val' in entry:\n"
                "        row['val_skill_6h'] = {{name: round(entry['val'][name]['6h']['skill'], 3) for name in ('2t', 'msl', 'z500')}}\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_windows, val_window, epochs=EPOCHS, lr=LEARNING_RATE, trainable=TRAINABLE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'training_samples': adapt_result['n_train_samples'], 'best_epoch': adapt_result['best_epoch'], 'seconds': adapt_seconds}})\n"
                "print({{'loss_scales': adapt_result['loss_scales']}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation by lead time\n\n"
                "The test window (October 2021) was never used for training or epoch selection. The adapted model is rolled out "
                "from every origin to 24 h and scored exactly as the frozen model was in Section 6; the table puts the three "
                "numbers side by side per variable and lead. Look for the mean skill dropping below 1 at every lead and most "
                "variables beating persistence — the cell asserts that the adapted 6-hour mean skill is below the frozen model's "
                "and that more variables beat persistence than before. One two-day window from one seeded run gives no "
                "dispersion estimate; these are sample-sanity numbers that show the adaptation contract works, not a benchmark."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_window, max_lead_steps=MAX_LEAD_STEPS)\n"
                "adapted_val = pipe.evaluate(val_window, max_lead_steps=1)\n"
                "comparison = {{}}\n"
                "for name in ('2t', '10u', '10v', 'msl', 't', 'u', 'v', 'q', 'z', 'z500'):\n"
                "    comparison[name] = {{lead: {{'persistence': round(row['persistence'], 4), 'frozen': round(frozen_test['variables'][name][lead]['model'], 4), 'adapted': round(row['model'], 4), 'skill_frozen': round(frozen_test['variables'][name][lead]['skill'], 3), 'skill_adapted': round(row['skill'], 3)}} for lead, row in adapted_test['variables'][name].items()}}\n"
                "for name in ('2t', 'msl', 'z500', 'u'):\n"
                "    print({{name: comparison[name]}})\n"
                "summary = {{lead: {{'frozen_mean_skill': round(frozen_test['summary'][lead]['mean_skill'], 3), 'adapted_mean_skill': round(adapted_test['summary'][lead]['mean_skill'], 3), 'beating_persistence_frozen': frozen_test['summary'][lead]['variables_beating_persistence'], 'beating_persistence_adapted': adapted_test['summary'][lead]['variables_beating_persistence']}} for lead in adapted_test['leads']}}\n"
                "for lead, row in summary.items():\n"
                "    print({{lead: row}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digest': dataset_manifest['digest'],\n"
                "    'roles': {{'train': [w['name'] for w in train_windows], 'validation': val_window['name'], 'test': test_window['name']}},\n"
                "    'persistence_baseline': baseline_persistence,\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'summary': summary,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "assert adapted_test['summary']['6h']['mean_skill'] < frozen_test['summary']['6h']['mean_skill']\n"
                "assert adapted_test['summary']['6h']['variables_beating_persistence'] > frozen_test['summary']['6h']['variables_beating_persistence']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Forecast from a new origin, artifact export and fresh reload\n\n"
                "The adapted model forecasts 24 h from a later origin of the test window; the forecast valid times, global means "
                "and the per-lead 2 m temperature RMSE against the analysis are printed as a sanity check, not an evaluation.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the 80 LoRA tensors, about 2 MB — as `adapter.safetensors`, "
                "with a `manifest.json` recording the artifact format, the base model id and revision, the digests of the "
                "converted base files, the tensor names, the file size and SHA-256, the training configuration and the epoch "
                "history (OUT8). `AuroraPipeline.from_artifact` re-verifies the base files, checks the artifact manifest and "
                "digest **before** deserialising, rebuilds the model with LoRA and overlays the tensors — a fresh object from "
                "files, not the in-memory model (VER2). The cell asserts identical forecast fields (VER4)."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "import safetensors\n\n"
                "NEW_ORIGIN = min(test_window['n_steps'] - 1 - ROLLOUT_STEPS, 3)\n"
                "new_forecast = pipe.predict(test_window, origin=NEW_ORIGIN, steps=ROLLOUT_STEPS)\n"
                "print({{'origin_time': new_forecast['origin_time'], 'adapted': new_forecast['model']['adapted']}})\n"
                "height = new_forecast['shape'][0]\n"
                "for k, forecast in enumerate(new_forecast['forecasts'], start=1):\n"
                "    truth = test_window['surf']['2t'][NEW_ORIGIN + k, :height]\n"
                "    print({{'lead_hours': forecast['lead_hours'], 'valid_time': forecast['valid_time'], '2t_mean_K': round(float(forecast['surf']['2t'].mean()), 3), '2t_rmse_vs_analysis': round(lat_weighted_rmse(forecast['surf']['2t'], truth, new_forecast['lat']), 3), 'note': 'sanity check, not an evaluation'}})\n"
                "with open('outputs/{stem}_forecast.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'origin_time': new_forecast['origin_time'], 'shape': new_forecast['shape'], 'units': new_forecast['units'], 'forecasts': [{{'lead_hours': fc['lead_hours'], 'valid_time': fc['valid_time'], 'surf_means': {{k: float(v.mean()) for k, v in fc['surf'].items()}}}} for fc in new_forecast['forecasts']]}}, f, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = AuroraPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = pipe.predict(test_window, origin=ORIGIN, steps=2)['forecasts']\n"
                "after = reloaded.predict(test_window, origin=ORIGIN, steps=2)['forecasts']\n"
                "parity = {{'max_abs_surf_diff': max(float(np.abs(a['surf'][k] - b['surf'][k]).max()) for a, b in zip(before, after) for k in a['surf']), 'max_abs_atmos_diff': max(float(np.abs(a['atmos'][k] - b['atmos'][k]).max()) for a, b in zip(before, after) for k in a['atmos'])}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['max_abs_surf_diff'] < 1e-6 and parity['max_abs_atmos_diff'] < 1e-6\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_assets': [e for e in MANIFEST['files'] if e['path'] in (SOURCE_CKPT_NAME, SOURCE_STATIC_NAME)],\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickles_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "        'data_objects': len(WB2_OBJECTS),\n"
                "        'data_base_url': WB2_BASE_URL,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'xarray': xarray.__version__, 'numcodecs': numcodecs.__version__, 'safetensors': safetensors.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'summary': summary,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Run six times coarser than its training grid, the frozen small checkpoint loses to persistence on most variables at "
        "6 h — an honest number for an out-of-distribution use, and the one this tutorial exists to move. A LoRA fine-tuning of "
        "half a million parameters on twelve one-step forecasts brings the mean skill below 1 at every lead to 24 h, with most "
        "variables beating persistence, on a window from a different year and season than anything trained on. That is the "
        "claim: the adaptation contract moves a foundation weather model onto a grid it was not trained for from two days of "
        "data, and the artifact that carries the change is 2 MB.\n\n"
        "The test window is two days of one month, the scores come from a single seeded run with no dispersion estimate, and "
        "the small checkpoint is the one upstream publishes for testing. So a skill below 1 here says the contract works, not "
        "that this model forecasts at any published accuracy, that it is stable beyond 24 h, or that the adaptation transfers to "
        "other seasons — none of which this repository exercises. Fine-tuning on a narrow window can also erode the model "
        "elsewhere; upstream fine-tunes on years of data with roll-out training, which is out of scope here.\n\n"
        "Three things to carry to real data. **Native resolution:** at 0.25° the published model is far ahead of persistence "
        "without any adaptation; the frozen numbers here are a resolution story, not an Aurora story. **Independence:** "
        "consecutive analyses are near-duplicates — split by period, never by shuffling steps, and read the persistence column "
        "before any model number. **Static fields and units:** the model needs the land-sea mask, surface geopotential and soil "
        "type on your grid, SI units and the 13 standard levels; a mismatch is silent unless validation catches it.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify two pickled upstream assets, audit and convert them into safetensors without "
        "executing anything outside the audited allow-lists, rebuild the model from the installed package, fetch and validate "
        "digest-pinned real reanalysis, execute bounded fine-tuning, evaluate against persistence and the frozen model on an "
        "independent window at every lead, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, production fitness, or forecast skill beyond the checks "
        "shown.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE = 'lora+heads'` and compare the surface "
        "against the upper air; raise `EPOCHS` or `MAX_LEAD_STEPS`; change `ORIGIN`; or bring your own NetCDF window through "
        "BYOD and read the persistence column before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/aurora-earth-system-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/aurora-earth-system-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/aurora-earth-system-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/microsoft/aurora (revision `{MODEL_REVISION}`)\n"
        "- Bodnar, C., Bruinsma, W. P., Lucic, A., et al. (2025). A foundation model for the Earth system. Nature 641, 1180–1187: https://doi.org/10.1038/s41586-025-09005-y\n"
        "- Rasp, S., Hoyer, S., Merose, A., et al. (2024). WeatherBench 2: A benchmark for the next generation of data-driven global weather models. JAMES 16: https://doi.org/10.1029/2023MS004019\n"
        "- Hersbach, H., et al. (2020). The ERA5 global reanalysis. QJRMS 146, 1999–2049: https://doi.org/10.1002/qj.3803\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
