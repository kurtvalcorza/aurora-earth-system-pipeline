"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone carrier, isolated environment).

Only the task-specific prose and the learner cells live here. The runtime check, the carrier of the package / stage
runner / lock / manifest, the isolated install and the weight staging are produced by the generator from repository
sources, so they cannot drift from the package. Every learner cell runs one stage of ``tools/tutorial_stages.py`` with
``run_stage`` and prints what that stage wrote.

This template configures an E2E weather-forecasting workflow: the pinned Aurora small checkpoint and static fields
(both pickles) are digest-verified, statically audited and converted once into safetensors; four two-day windows of
real ERA5 at 1.5° are fetched from pinned WeatherBench 2 objects, validated against the schema and the configured
workflow, and assigned to roles; the frozen model is rolled out and scored against persistence per lead time; a
bounded LoRA fine-tuning runs; the held-out window is scored again; a new-origin forecast is exported as NetCDF; and
the adapter is exported and reloaded in a new process.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "aurora-earth-system-pipeline"
COLAB_URL = f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/aurora_earth_system_colab.ipynb"

BADGES = [
    ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", f"https://github.com/kurtvalcorza/{REPO}"),
    ("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg", COLAB_URL),
    ("Hugging Face", "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Faurora-ffcc4d?style=flat", "https://huggingface.co/microsoft/aurora"),
    ("Upstream", "https://img.shields.io/badge/Upstream-microsoft%2Faurora-181717?style=flat&logo=github&logoColor=white", "https://github.com/microsoft/aurora"),
    ("Paper", "https://img.shields.io/badge/Nature-10.1038%2Fs41586--025--09005--y-b31b1b.svg", "https://doi.org/10.1038/s41586-025-09005-y"),
]

TEMPLATE = {
    "package": "aurora_earth_system_pipeline",
    "repo_name": REPO,
    "stem": "aurora_earth_system",
    "notebook_name": "aurora_earth_system_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "weights_key": "aurora-0.25-small",
    "carried": {
        "src/aurora_earth_system_pipeline/__init__.py": "src/aurora_earth_system_pipeline/__init__.py",
        "src/aurora_earth_system_pipeline/pipeline.py": "src/aurora_earth_system_pipeline/pipeline.py",
        "src/aurora_earth_system_pipeline/metrics.py": "src/aurora_earth_system_pipeline/metrics.py",
        "src/aurora_earth_system_pipeline/samples.py": "src/aurora_earth_system_pipeline/samples.py",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/aurora-0.25-small/dimer-base-manifest.json": "weights/aurora-0.25-small/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 1.5, "environment": 10},
    "runtime_modules": ["torch", "timm", "xarray", "numcodecs"],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime checks the runtime, writes and hash-verifies the carried files, builds "
        "an isolated hash-locked Python environment (nothing is installed into the notebook kernel, so no restart is needed), "
        "stages and digest-verifies the pinned Aurora small checkpoint (451 MB) and static fields (12 MB) from the Hub, statically "
        "audits both pickles against allow-lists and converts them once into safetensors with pinned digests, fetches four "
        "two-day windows of real ERA5 reanalysis at 1.5° from WeatherBench 2 (about 196 MB of pinned, digest-verified Zarr chunks "
        "— no credential), validates them against the schema and against the configured workflow and assigns them to training, "
        "validation and test roles, rolls the frozen model out to 24 h and scores it per variable and lead time against the "
        "persistence baseline, runs a bounded LoRA fine-tuning on one-step forecasts, scores the held-out window again at every "
        "lead with the adapter rebuilt from its files, forecasts from a new origin and exports that forecast as NetCDF, reloads "
        "the adapter in another fresh process to verify forecast parity, and writes a provenance record. Every stage that runs "
        "the model builds it from the verified files in its own process. The default path needs no repository clone, DIMER "
        "worker or service, credential, upload dialog, or configuration edit (NOTEBOOK_SPEC 2.2 §5, RUN1, RUN10)."
    ),
    "byod": (
        "`USE_BYOD = True` in Section 5 replaces the pinned sample with your own gridded analyses as one NetCDF file "
        "(coordinates `time`, `level`, `latitude`, `longitude`; surface variables `2t`, `10u`, `10v`, `msl` as (time, lat, lon); "
        "atmospheric variables `t`, `u`, `v`, `q`, `z` on the 13 standard levels as (time, level, lat, lon); static fields "
        "`static_lsm`, `static_z`, `static_slt` as (lat, lon); a global equiangular grid). The window needs at least "
        "2 + max(`ROLLOUT_STEPS`, `MAX_LEAD_STEPS`) 6-hourly steps — **6 at the defaults** — and Section 5 refuses a shorter "
        "one before any model runs, naming the step count and the fix. Section 5 writes a sample file of exactly that length "
        "in this shape. Put the file's path in `BYOD_PATH` (any runtime), or leave it empty for the Google Colab upload "
        "dialog. Your window then plays the training, validation and test roles through the same stages — validation, "
        "baselines, adaptation, held-out evaluation, inference, export and reload parity — and the notebook says that one "
        "window makes the held-out numbers not independent. Files stay in this runtime."
    ),
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
        "Two things about this notebook are handled in the open. **Both upstream assets are pickles.** Section 3 downloads and "
        "digest-verifies them, statically lists every global each pickle would import (a state dict of tensors; three numpy "
        "arrays), refuses anything outside those allow-lists, unpickles each exactly once through a restricted loader, and writes "
        "safetensors whose digests are pinned in the carried module. The model you run is rebuilt from the installed "
        "`microsoft-aurora` package and loads those files strictly. **The data is real ERA5 at 1.5°, not the model's native "
        "0.25°.** Four two-day windows come from WeatherBench 2's public bucket as pinned, digest-verified Zarr chunks — a native "
        "window would be sixty times larger — so the frozen model runs six times coarser than it was trained, an out-of-distribution "
        "use. That is the honest setting for the adaptation contract: Section 7 measures the frozen model against persistence "
        "at this resolution, and Section 9 measures what a 540 k-parameter LoRA fine-tuning on twelve one-step forecasts recovers."
    ),
    "learning_objectives": (
        "after this notebook you should be able to (1) explain what Aurora takes as input (two analyses six hours apart plus "
        "static fields) and how a roll-out turns one 6-hour step into a 24-hour forecast; (2) read a latitude-weighted RMSE per "
        "variable and lead time and the skill ratio against persistence, and explain why persistence for 2 m temperature "
        "improves again at 24 h; (3) explain why a model run on a grid six times coarser than its training grid can lose to "
        "persistence, and why windows must be split by period rather than by shuffling time steps; (4) adapt a foundation "
        "model with LoRA, select the epoch on a validation window and judge the result on an independent test window; "
        "(5) export the adapter, rebuild the model from files in a new process and verify forecast parity; and (6) change "
        "one variable (the adaptation scope) and explain how the result changes by variable."
    ),
    "exclusions": (
        "the 0.25° native-resolution path and the 1.3 B-parameter production checkpoints, the wave, air-pollution and 0.1° "
        "variants, ensemble forecasting, tropical-cyclone tracking, multi-step (roll-out) fine-tuning, climatology and "
        "operational-forecast baselines, the published WeatherBench scores, and any claim that a 1.5° two-day window stands in "
        "for an operational evaluation. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh Linux x86_64 runtime — Google Colab, Kaggle, or a Linux Jupyter server — with about 12 GB of free disk and about 4 GB of RAM. A GPU is used automatically when present (the default runtime type is a T4) but is not required: every stage also runs on CPU, in float32 on both. The stages run in CPython 3.12.12 inside the isolated environment, whatever Python the notebook kernel itself uses (Colab's kernel is 3.13). Windows and macOS kernels are not supported, because the hash-locked environment is built for manylinux x86_64.",
        "- **Time (measured where stated, otherwise an estimate):** building the isolated environment and downloading about 660 MB of weights and data take a few minutes on a hosted runtime — an estimate that depends on the network. On the 2026-09-19 Kaggle T4 run of the previous, in-kernel version of this notebook the whole path took 283.5 s including downloads. On a Windows CPU workstation (a local pre-flight of this version, not a supported runtime) the six-epoch fine-tuning took about 3 minutes (192 s) and the learner stages of Sections 4–12 about 6 minutes; a hosted CPU runtime is usually slower than that workstation.",
        "- **Knowledge:** basic Python, and what a gridded atmospheric analysis is (pressure levels, surface fields, a latitude/longitude grid). Lead time, persistence, RMSE, the skill ratio and LoRA are explained where they are first used; the glossary collects them.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint and static file are pickles. Each is digest-verified, statically audited against an allow-list (audit digests pinned) and unpickled **once** through a restricted loader to produce the safetensors the model is actually loaded from. No Hub-hosted Python module is imported; `microsoft-aurora` is installed from PyPI at a locked version and digest.",
        "- **Data contract:** a window is `{{lat, lon, levels, times, surf, atmos, static}}` on a global equiangular grid — latitudes spanning 90 to −90 (17..721 rows, a multiple of 4 or one more), longitudes covering 0 to 360 (32..1440 columns, a multiple of 4), exactly the 13 standard pressure levels, analyses spaced exactly 6 h apart, SI units (K, m/s, Pa, kg/kg, m²/s²) within plausibility ranges. The schema accepts 3..64 steps, but this workflow needs at least 2 + max(`ROLLOUT_STEPS`, `MAX_LEAD_STEPS`) — **6 at the defaults** — and Section 5 enforces that. The model predicts the rows the patch size covers (121 → 120 at 1.5°). BYOD accepts NetCDF in the shape of the sample file Section 5 writes.",
        "- **Validation is structural, not meteorological:** nothing checks that the fields are dynamically consistent, that the analysis is real, or that the resolution is one the model was trained on — a smooth random field within range is forecast without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — a proprietary analysis or an embargoed forecast dataset is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub and PyPI, the default path fetches 57 pinned objects (about 196 MB) from the public WeatherBench 2 bucket `storage.googleapis.com/weatherbench2` over HTTPS, digest-verified before decoding; ERA5 is © ECMWF/Copernicus under the licence to use Copernicus products.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python, and who "
                "want to see how a pretrained weather model is evaluated against a baseline and adapted to new conditions. "
                "You should know what a gridded analysis and a pressure level are; every forecasting term after that is "
                "explained where it is first needed, and the glossary below collects them.\n\n"
                "**Running it.** In Colab, choose *Runtime → Run all*. The default path needs no edit, no upload, no account, "
                "no token and no runtime restart. Sections 1–3 build an isolated environment from hash-locked packages and "
                "download about 660 MB of verified weights and data, so they take the longest before any model runs; read "
                "ahead while they finish, or run one cell at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files in the run directory — the verified snapshot, "
                "the data cache, the settings record, the adapter and JSON records. **Every stage that runs the model builds it "
                "from the verified files**, so Sections 6 and 7 always show the frozen model and Section 8 always starts from "
                "LoRA at zero, however often you re-run them.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–13) are the forecasting workflow; each runs one stage and "
                "prints compact dictionaries for you to read. *Infrastructure cells* (Sections 1–3: the runtime check, the "
                "carried code, the isolated install and the pinned-weight staging) are collapsed and titled **Infrastructure**. "
                "You may run them without studying their implementation: they exist for reproducibility and provenance, not as "
                "prerequisite machine-learning knowledge.\n\n"
                "**Form controls.** Some learner cells start with fields that Colab renders as a form: `USE_BYOD`, `BYOD_PATH`, "
                "`ORIGIN`, `ROLLOUT_STEPS` and `MAX_LEAD_STEPS` (Section 5); `EPOCHS`, `LEARNING_RATE` and `TRAINABLE` "
                "(Section 8); and `RUN_ACTIVITY` and `ACTIVITY_TRAINABLE` in the optional activity (Section 13). Leave them at "
                "their defaults for the first run: the notes and sample answers describe the default path.\n\n"
                "**Changing a setting.** After changing a Section 5 field (data or forecast settings), re-run from Section 5 to "
                "the end: Section 5 checks the new settings against the window and clears every result computed for the old "
                "ones, and a later section that needs a cleared result says which section to run. After changing a Section 8 "
                "field, re-run from Section 8 to the end. Nothing else needs to be repeated.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. "
                "**[Evaluation practice]** — how the evidence is produced and how to read it. **[Engineering]** — "
                "reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit "
                "to an expectation; after it, **What to notice** describes normal output, and a collapsed **Check your "
                "reasoning** answer follows each checkpoint. Write your own answer first, then open it. Exact numbers can vary "
                "between CPU and GPU and between library builds (the third decimal of a skill ratio, for example), so the notes "
                "describe the shape of a normal result rather than fixed values."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Forecast** | two global analyses 6 h apart (4 surface variables, 5 atmospheric variables on 13 levels) and 3 static fields, on a 121 × 240 grid | Aurora: Perceiver encoder → 3D Swin backbone → decoder, applied repeatedly (a roll-out) | the predicted state at each later 6-hour step (120 × 240 rows) |\n"
                "| **Baseline and evaluation** | the forecasts and the later analyses of the same window | latitude-weighted RMSE per variable and lead; the same score for persistence | a table of errors and skill ratios by lead time |\n"
                "| **Adaptation** | 12 one-step forecasts from two training windows; one validation window | the 80 LoRA tensors trained with AdamW; the epoch with the lowest validation loss kept | a 2 MB adapter file holding only the LoRA tensors |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux and disk checked; a fresh run directory | the GPU (or CPU) |\n"
                "| 2. Carry the code, install the runtime | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage, audit and convert | [Engineering] | snapshot downloaded and digest-checked; pickles audited and converted | audits, converted digests |\n"
                "| 4. Confirm the runtime | [Engineering] | versions checked against the lock | device, ceilings |\n"
                "| 5. Windows, settings and roles | [Evaluation practice] | four windows fetched, validated, checked against the settings | roles, refusals |\n"
                "| 6. Zero-shot roll-out | [Concept] | the frozen model forecasts 24 h | `adapted: False`, valid times |\n"
                "| 7. Persistence and the frozen model | [Evaluation practice] | both scored per variable and lead | the score to beat |\n"
                "| 8. LoRA fine-tuning | [Concept] | six epochs on the LoRA tensors; adapter exported | the epoch history |\n"
                "| 9. Held-out evaluation | [Evaluation practice] | the adapter rebuilt from files and scored on the test window | the principal result |\n"
                "| 10. A new-origin forecast | [Concept] | a later forecast, exported as NetCDF | the exported fields |\n"
                "| 11. Fresh reload and parity | [Engineering] | another fresh process reproduces the trained model's forecast | `PASSED` |\n"
                "| 12. Provenance record | [Engineering] | one JSON linking every output | the file list |\n"
                "| 13. Optional activity | [Concept] | change the adaptation scope (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Short on time? Run all, then read Sections 7, 9 and 11 and the conclusion: they carry the "
                "principal results. The canonical path ends with Section 12; Section 13 changes nothing unless you switch it on."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Analysis / reanalysis** | The best estimate of the atmosphere's state at one time on a grid. ERA5 is a *reanalysis*: analyses recomputed for past decades with one fixed system. |\n"
                "| **Surface and atmospheric variables** | Surface: 2 m temperature `2t`, 10 m winds `10u`/`10v`, mean sea-level pressure `msl`. Atmospheric, on 13 pressure levels: temperature `t`, winds `u`/`v`, specific humidity `q`, geopotential `z`. `z500` is geopotential at 500 hPa. |\n"
                "| **Static fields** | Fields that do not change: land-sea mask, surface geopotential (terrain), soil type. |\n"
                "| **Equiangular grid / resolution** | A latitude/longitude grid with equal spacing; 1.5° here (121 × 240), while Aurora was trained at 0.25° (721 × 1440). |\n"
                "| **Window** | A run of consecutive analyses, 6 h apart; each tutorial window has eight (two days). |\n"
                "| **Origin** | The latest analysis a forecast starts from; Aurora also needs the one 6 h before it (the *history step*). |\n"
                "| **Lead time** | How far ahead a forecast is: 6 h, 12 h, 18 h, 24 h. |\n"
                "| **Roll-out** | Feeding each 6-hour forecast back in as input to reach longer leads; errors accumulate. |\n"
                "| **Persistence** | The baseline forecast \"tomorrow looks like now\": the origin analysis carried forward unchanged. |\n"
                "| **Latitude-weighted RMSE** | Root-mean-square error with each grid row weighted by the cosine of its latitude, so the many small polar cells do not dominate. |\n"
                "| **Skill ratio** | RMSE(model) / RMSE(persistence) for one variable and lead; below 1 beats persistence. The *mean skill* averages it over nine variables. |\n"
                "| **Frozen model** | The pretrained model with no adaptation (LoRA at zero). |\n"
                "| **Out-of-distribution** | Inputs unlike the training data — here a grid six times coarser than the model was trained on. |\n"
                "| **LoRA** | Low-rank adaptation: small trainable matrices added beside the attention projections; zero at the start, so the model begins as the pretrained one. |\n"
                "| **Epoch / validation / test** | One pass over the 12 training samples; the validation window picks the epoch; the test window is used once, for the final score. |\n"
                "| **Adapter** | The exported file holding only the trained LoRA tensors, with a manifest naming the base model it fits. |\n"
                "| **NetCDF** | A self-describing file format for gridded data, with named dimensions, coordinates and units. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes; a single changed byte changes it. |\n"
                "| **Hash-locked environment** | A separate Python environment built from a requirements file that pins every package to one version and one set of SHA-256 digests; the installer refuses anything else. |\n"
                "| **Stage** | One step of the workflow run as its own process by `run_stage`; it reads the files earlier stages wrote and writes its own. |\n"
                "| **BYOD** | Bring Your Own Data: an optional switch to run the same workflow on your own NetCDF window. |\n\n"
                "</details>"
            ),
        ],
    },
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their "
                "implementation; they exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with enough free disk, reports the "
                "GPU if there is one, and creates a new run directory. **Output:** the directories this run will use. Each run "
                "writes to a new directory under `outputs/{stem}/`, so an earlier export cannot be mistaken for a current result. "
                "The verified snapshot and the data cache are kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**Expected result:** one dictionary naming the GPU (for example `Tesla T4, 15360 MiB`) or `none (the stages run "
                "on CPU)`, the kernel's Python version, the run directory, the weights directory, the isolated environment's "
                "directory and the free disk. If the cell stops with a platform or disk message, see **Troubleshooting**."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the repository's code, carried so "
                "that this notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text, the files the workflow needs: the package's four modules under "
                "`src/aurora_earth_system_pipeline/` (identity constants, snapshot verification and staging, the pickle audits and "
                "conversion, validation, the pipeline class, the pinned-data fetcher and the metrics), the stage runner "
                "`tutorial_stages.py`, the hash-locked `requirements.txt` ({n_locked} packages), the snapshot manifest and the "
                "licence. It writes each file into the run directory and checks its SHA-256 against `CARRIED_HASHES`, stopping on "
                "any mismatch. The text is the repository's files byte for byte; the repository's parity test "
                "(`tests/test_notebook_parity.py`) fails whenever the two diverge, so what runs here is what the repository "
                "tests. Nothing in this cell runs a model."
            ),
            "after": (
                "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated "
                "from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer "
                "wheel, refused unless its size and SHA-256 match — creates a separate virtual environment with its own CPython "
                "3.12.12, and installs `requirements.txt` into it with `--require-hashes --only-binary :all:`: every package must "
                "be the locked version, a prebuilt wheel, and match a locked digest. The hosted runtime's own packages are never "
                "replaced, which is why no restart is needed. The cell also defines `run_stage`, `load_record` and "
                "`obtain_upload`, the helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and can take a few "
                "minutes (the CUDA build of `torch` is the largest download). A failed download or a hash mismatch stops the "
                "cell; never remove a pin or a hash to get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), "
                "the `torch`, `timm`, `xarray` and `numcodecs` versions, `cuda` (`True` on a GPU runtime, `False` on CPU — both "
                "are supported), the number of locked packages and the setup time."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage, audit and convert the model · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads about 463 MB of pinned upstream files, checks "
                "every file's size and SHA-256, audits the two pickles and converts them; you may run it without studying its "
                "implementation.\n\n"
                "The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the carried `pipeline.py` and the snapshot "
                "manifest (paths, byte sizes, SHA-256) — and the `weights` stage first checks that they agree. It installs the "
                "carried manifest into `weights/`, fetches exactly the files that are absent from the Hugging Face Hub **at the "
                "pinned revision** of `{MODEL_ID}` (never `main`), and re-hashes every file, raising on the first size or digest "
                "mismatch. It then lists, without executing anything, every global each pickle would import, refuses any outside "
                "the allow-lists, unpickles each exactly once through a restricted loader, and writes two safetensors files whose "
                "digests are pinned in the carried module. No later stage ever reads a pickle, and no remote model code is executed."
            ),
            "after": (
                "**What to notice:** the model id, revision, licence and file count; a `fetched` list (empty on a rerun, because "
                "staging only fetches absent files); the two **pickle audits** — the checkpoint imports only "
                "`collections.OrderedDict`, `torch.FloatStorage` and `torch._utils._rebuild_tensor_v2`, the static file only "
                "`numpy…_frombuffer` and `numpy.dtype`, with no violations; and the two converted files with their sizes and "
                "digests. A mismatch stops the cell with an error naming the file — see **Troubleshooting**, and never edit a "
                "manifest to get past one."
            ),
        },
    ],
    "cells": [
        # ------------------------------------------------------------------ 4. runtime
        {
            "md": (
                "## 4. Confirm the isolated runtime · [Engineering]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`. This cell runs the "
                "`runtime` stage. It compares the versions installed in the isolated environment (`torch`, `microsoft-aurora`, "
                "`timm`, `einops`, `xarray`, `netCDF4`, `numcodecs`, `numpy`, `safetensors`, `huggingface-hub`) with the versions "
                "in the carried lock and **stops if any differs**; it then prints the device the stages will use and the input "
                "ceilings the pipeline enforces.\n\n"
                "**Expected result:** `versions_match_lock: True`, the device (`cuda` on a GPU runtime, `cpu` otherwise) and the "
                "ceilings: the grid bounds, the 13 levels, 3..64 steps 6 hours apart, two history steps and roll-outs of 1..8 steps."
            ),
            "code": "run_stage('runtime')",
        },
        # ------------------------------------------------------------------ 5. data
        {
            "md": (
                "## 5. Windows, settings, validation and roles · [Evaluation practice]\n\n"
                "The default dataset is real ERA5 reanalysis regridded to 1.5° by WeatherBench 2: four windows of eight "
                "6-hourly analyses — two for training (late December 2018 / early January 2019, and July 2019), one for "
                "validation (April 2020) and one for testing (October 2021) — distinct seasons and years, so nothing in the test "
                "window is a near-duplicate of anything trained on. The `data` stage retrieves each Zarr chunk, `.zarray` "
                "descriptor and coordinate from the pinned table `WB2_OBJECTS` (byte size and SHA-256 per object), refuses a "
                "mismatch before decoding, decodes Blosc/LZ4 with `numcodecs`, and reorders the arrays into the Aurora layout "
                "(latitude 90 → −90, longitude 0 → 360). `validate_dataset` checks every window and the shared grid.\n\n"
                "**Settings, checked here.** `ORIGIN` is the forecast origin of Sections 6 and 11; `ROLLOUT_STEPS` the number of "
                "6-hour steps rolled out in Sections 6 and 10; `MAX_LEAD_STEPS` the number of lead times scored in Sections 7 and "
                "9. A window that passes the schema can still be too short for these settings, so the stage also checks the test "
                "window against them: it needs 2 + max(`ROLLOUT_STEPS`, `MAX_LEAD_STEPS`) steps (6 at the defaults) and an "
                "`ORIGIN` inside it. A shorter window is refused **here**, before any model runs, with the step count and the fix.\n\n"
                "**Pretraining overlap.** Upstream does not publish the period the small checkpoint was trained on (its model "
                "page describes it only as a smaller version of the pretrained model, and the paper's pretraining data include "
                "decades of ERA5). So overlap between these 2018–2021 analyses and the checkpoint's pretraining data **cannot be "
                "ruled out**: the frozen model may have seen these dates at 0.25°. The windows are independent of *each other* "
                "(different years and seasons), which is what the train / validation / test reading below relies on.\n\n"
                "**BYOD:** set `USE_BYOD = True` and put a NetCDF path in `BYOD_PATH` (any runtime), or leave the path empty for "
                "the Colab upload dialog. The file written below (`outputs/{stem}_sample_window.nc`) is a valid example: it is the "
                "test window cut to exactly the length your settings need.\n\n"
                "**Expected result:** four windows of 8 steps on a 121 × 240 grid, 24 forecast origins, the roles, "
                "`steps_needed_for_these_settings: 6`, `verdict: accepted`, the sample file, and five refusal probes — wrong "
                "pressure levels, an odd longitude count, an implausible temperature field, irregular time spacing, and a window "
                "one step too short for these settings — each rejected before any model runs."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "ORIGIN = 1  # @param {{type:\"integer\"}}\n"
                "ROLLOUT_STEPS = 4  # @param {{type:\"integer\"}}\n"
                "MAX_LEAD_STEPS = 4  # @param {{type:\"integer\"}}\n\n"
                "data_options = ['--origin', ORIGIN, '--rollout-steps', ROLLOUT_STEPS, '--max-lead-steps', MAX_LEAD_STEPS]\n"
                "if USE_BYOD:\n"
                "    data_options += ['--byod', obtain_upload(BYOD_PATH, '.nc', 'BYOD_PATH')]\n"
                "run_stage('data', *data_options)"
            ),
        },
        {
            "md": (
                "**Checkpoint:** consecutive analyses six hours apart are very similar. Why does the notebook split the data by "
                "*window* — whole days from different years — instead of shuffling all 32 analyses and holding out a random "
                "quarter?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Because a randomly held-out analysis would almost always sit six hours from one used in training, and the "
                "atmosphere barely changes in six hours. The model could then score well by having nearly seen the answer, and "
                "the test would measure memory rather than forecasting. Splitting by period keeps whole days, from other seasons "
                "and years, out of training — the same rule applies to your own data: split by period, never by shuffling steps.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 6. rollout
        {
            "md": (
                "## 6. Zero-shot roll-out from the frozen model · [Concept]\n\n"
                "The `rollout` stage builds the pipeline from the verified files in a new process, with `use_lora=True`: the 80 "
                "LoRA tensors are present but zero, so this is the pretrained model's behaviour. `predict` takes the test window, "
                "an origin (the index of the latest analysis to use; the step before it is the second history step) and a number "
                "of 6-hour steps, and rolls the model forward autoregressively: each forecast becomes the next input. Returned "
                "fields are (120, 240) — the model predicts the rows the patch size covers and drops the last latitude row, "
                "exactly as it does at 0.25° (721 → 720).\n\n"
                "**Predict before running:** as the roll-out goes from 6 h to 24 h, should the global-mean 2 m temperature and "
                "sea-level pressure drift far from their physical values, or stay close?"
            ),
            "code": "run_stage('rollout')",
        },
        {
            "md": (
                "**What to notice:** `adapted: False` and `lora_b_max_abs: 0.0` (no LoRA update — the frozen model), four valid "
                "times 6 h apart, global means that stay physical (2 m temperature near 280 K, sea-level pressure near 101,000 Pa), "
                "and `repeat_identical: True` — a second call returns bit-identical fields on the same device.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "They stay close. Global means are a weak check — a forecast can have the right average and the wrong weather — "
                "but a model that drifted tens of kelvin in a day would be broken, not just inaccurate. Whether the *pattern* is "
                "right is what Section 7 measures, grid point by grid point.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 7. frozen
        {
            "md": (
                "## 7. Persistence and the frozen model's error by lead time · [Evaluation practice]\n\n"
                "Every number from here on is a latitude-weighted RMSE (cos-latitude weights with unit mean, the WeatherBench "
                "convention) against the ERA5 analysis at the valid time, averaged over every forecast origin the window allows. "
                "The **persistence** forecast carries the latest analysis forward unchanged; at 6 h it is a strong baseline, and "
                "for 2 m temperature it gets *better* again at 24 h because the diurnal cycle comes back into phase — a reminder "
                "that a baseline is not a straw man. The **skill** column is RMSE(model) / RMSE(persistence): below 1 beats "
                "persistence. The `frozen` stage builds the frozen model afresh, scores it and persistence on the same origins, "
                "and writes both to `outputs/{stem}_frozen_test.json` before any adaptation, so the reference cannot be "
                "overwritten by a later step.\n\n"
                "**Predict before running:** the model runs on a grid six times coarser than it was trained on. At 6 h, will the "
                "frozen model beat persistence on most variables, or lose? And as the lead grows to 24 h, will the gap widen or "
                "close?"
            ),
            "code": "run_stage('frozen')",
        },
        {
            "md": (
                "**What to notice:** the persistence RMSE for `2t` is lower at 24 h than at 18 h (the diurnal cycle), and the "
                "frozen summary per lead: the mean skill and how many of the nine variables beat persistence.\n\n"
                "**Question tested:** at a resolution it was not trained for, is the frozen foundation model better than the "
                "simplest possible forecast?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "On the default sample the frozen model loses: in the recorded runs its 6-hour mean skill was about 1.57, with "
                "only one of nine variables beating persistence, and the gap narrowed with lead time as persistence decayed. "
                "That is a resolution story, not an Aurora story — at its native 0.25° the published model is far ahead of "
                "persistence. It is also the honest number the adaptation is read against: whatever Section 9 shows has to be "
                "compared with this, not with zero.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 8. adapt
        {
            "md": (
                "## 8. Bounded LoRA fine-tuning · [Concept]\n\n"
                "The `adapt` stage builds the frozen model afresh (it prints `lora_b_max_abs: 0.0` to show it starts from LoRA "
                "at zero) and trains the 80 LoRA tensors (rank 8, 540,672 parameters — 0.5 % of the model) that the upstream "
                "architecture places on the query/key/value and output projections of every backbone attention block, and "
                "nothing else. Each training sample is one 6-hour forecast from one origin of one training window (12 samples "
                "on the default data); the loss is the mean over the nine variables of the MSE divided by a fixed per-variable "
                "scale, so a kelvin of temperature and a pascal of pressure count alike; AdamW at a fixed learning rate, seeded "
                "shuffling, no scheduler. Epoch 0 records the frozen model, and the epoch with the lowest **validation** loss is "
                "kept — the test window plays no part. The stage then exports the kept tensors as "
                "`outputs/{stem}_adapter/adapter.safetensors` with a `manifest.json` (format, base model id, revision and "
                "converted-base digests, tensor names, size and SHA-256, training configuration and epoch history), and records "
                "the trained model's forecast for the parity check in Section 11.\n\n"
                "`TRAINABLE = 'lora+heads'` also unfreezes the encoder token embeddings and decoder heads; the optional activity "
                "in Section 13 compares the two without touching the canonical result, so leave `lora` here.\n\n"
                "**Predict before running:** will the validation loss fall every epoch? Which epoch do you expect to be kept?"
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-3  # @param {{type:\"number\"}}\n"
                "TRAINABLE = 'lora'  # @param [\"lora\", \"lora+heads\"]\n\n"
                "run_stage('adapt', '--epochs', EPOCHS, '--learning-rate', LEARNING_RATE, '--trainable', TRAINABLE)"
            ),
        },
        {
            "md": (
                "**What to notice:** epoch 0 labelled as the frozen model, the validation loss per epoch with the 6-hour "
                "validation skill of `2t`, `msl` and `z500`, 540,672 trainable parameters, 12 training samples, the kept epoch, "
                "and the adapter (80 tensors, about 2 MB). On the default data the validation loss fell from about 0.18 to about "
                "0.05 over six epochs in the recorded runs; the notebook does not promise a monotonic fall on other data or "
                "settings.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Usually it falls, and the last or a late epoch is kept — but with one training pass of twelve samples per epoch "
                "it can rise for an epoch, and epoch 0 (the frozen model) can win if adaptation does not help. That is why the "
                "epoch is chosen on the validation window and never on the test window: choosing it on the test numbers would "
                "make Section 9 a measure of the choice, not of the model.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 9. evaluate
        {
            "md": (
                "## 9. Held-out evaluation by lead time · [Evaluation practice]\n\n"
                "The test window (October 2021 on the default data) was never used for training or epoch selection. The "
                "`evaluate` stage starts a new process, rebuilds the base model from the verified files, loads the adapter from "
                "its files (manifest, scope and digest checked before anything is deserialised), rolls it out from every origin "
                "and scores it exactly as the frozen model was scored in Section 7. The table puts persistence, frozen and adapted "
                "side by side per variable and lead, and `outputs/{stem}_evaluation_report.json` records it with the settings and "
                "the training configuration.\n\n"
                "The stage then prints the **outcome** as a fact, not an assertion: whether the adapted 6-hour mean skill is below "
                "the frozen model's, and whether more variables beat persistence. Only on the default sample with the default "
                "settings, where the recorded runs show the improvement, does the stage treat its absence as a failure. With BYOD, "
                "other settings or other training choices a negative result is a legitimate finding: the notebook continues, and "
                "Sections 10–12 still export everything.\n\n"
                "**Predict before running:** will the adapted model beat the frozen model at every lead? Will it beat persistence "
                "on most variables?"
            ),
            "code": "run_stage('evaluate')",
        },
        {
            "md": (
                "**What to notice:** per lead, `frozen_mean_skill` against `adapted_mean_skill` and the counts of variables "
                "beating persistence; per variable, which ones improved most.\n\n"
                "**Question tested:** on a window from another year and season, does a 540 k-parameter LoRA fine-tuning on two "
                "days of data move the foundation model past persistence at this resolution?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "On the default sample, yes: in the recorded runs the 6-hour mean skill went from about 1.57 to about 0.90, and "
                "the number of variables beating persistence from 1 to 6, with the mean skill below 1 at every lead to 24 h "
                "(third-decimal differences between CPU and GPU are normal). That shows the adaptation contract works on this "
                "grid. It is one two-day window and one seeded run with no dispersion estimate, so it does not show that the "
                "model forecasts at any published accuracy, that the gain holds in other seasons, or that it lasts beyond 24 h. "
                "If your run did not improve — with BYOD, or other settings — read the persistence column first: a window where "
                "persistence is already very good leaves little to gain, and a single window plays every role in BYOD, so its "
                "numbers are not independent.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ 10. forecast
        {
            "md": (
                "## 10. A forecast from a new origin · [Concept]\n\n"
                "The `forecast` stage rebuilds the adapted model from its files and forecasts `ROLLOUT_STEPS` × 6 h from a later "
                "origin of the test window (index 3, or as late as the window allows). Where the window still holds the analysis "
                "at a valid time, the 2 m temperature RMSE against it is printed — a sanity check, not an evaluation. The full "
                "forecast is exported as `outputs/{stem}_forecast.nc`: every surface variable as (`lead_hours`, `latitude`, "
                "`longitude`) and every atmospheric variable as (`lead_hours`, `level`, `latitude`, `longitude`), with the "
                "`valid_time`, `level`, `latitude` and `longitude` coordinates, units on each variable, and the origin time and "
                "model identity as attributes. `outputs/{stem}_forecast.json` lists the fields and the per-lead sanity numbers.\n\n"
                "**Expected result:** the origin time, `adapted: True`, four lead lines, and a NetCDF file of about 30 MB with "
                "nine variables."
            ),
            "code": "run_stage('forecast')",
        },
        {
            "md": (
                "**What to notice:** the RMSE grows with lead time even for the adapted model — errors accumulate along a "
                "roll-out. The NetCDF file is the reusable output: open it with `xarray.open_dataset` in any environment to plot "
                "or verify the fields."
            ),
        },
        # ------------------------------------------------------------------ 11. reload
        {
            "md": (
                "## 11. Fresh reload and forecast parity · [Engineering]\n\n"
                "Sections 9 and 10 already loaded the adapter in new processes. The `reload` stage checks that this is faithful: "
                "it re-verifies the base files and the adapter's manifest and digest, rebuilds the pipeline from the files in "
                "another fresh process, forecasts two steps from `ORIGIN`, and compares every field with the forecast the trained "
                "model produced in Section 8 (`rtol=1e-5`, `atol=1e-6`). A mismatch stops the notebook.\n\n"
                "**Expected result:** `reload_verification: PASSED` and maximum absolute differences at or near 0."
            ),
            "code": "run_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** parity shows that the export is complete and loads safely — not that the forecast is good. "
                "A saved model reproduces its errors just as faithfully as its skill."
            ),
        },
        # ------------------------------------------------------------------ 12. bundle
        {
            "md": (
                "## 12. Export the provenance record · [Engineering]\n\n"
                "The `bundle` stage writes `outputs/{stem}_result.json`, which links the run's records — the input manifest, "
                "frozen roll-out and scores, training history, evaluation report, forecast and reload parity — with the "
                "notebook's source revision, the model id, revision and licence, the provenance block (both source assets, both "
                "audit digests, both converted digests, `served_from_pickle: false`, `remote_code_executed: false`, the 57 data "
                "objects and their base URL), the runtime versions, and the SHA-256 of every output file.\n\n"
                "**Expected result:** a list of the run's output files, including `{stem}_input_manifest.json`, "
                "`{stem}_sample_window.nc`, `{stem}_frozen_test.json`, `{stem}_evaluation_report.json`, `{stem}_forecast.nc`, "
                "`{stem}_adapter/adapter.safetensors`, `{stem}_reload_parity.json` and `{stem}_result.json`."
            ),
            "code": "run_stage('bundle')",
        },
        # ------------------------------------------------------------------ 13. activity
        {
            "md": (
                "## 13. Optional activity: change one thing — the adaptation scope · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This activity is off by default and changes nothing "
                "the canonical path produced: with `RUN_ACTIVITY = False` the next cell only prints how to switch it on. When on, "
                "the `activity` stage builds the frozen model afresh and repeats Section 8 with one change — "
                "`ACTIVITY_TRAINABLE = 'lora+heads'`, which also trains the encoder token embeddings and decoder heads (about "
                "713 k more parameters) — with the same epochs, learning rate, seed and data. It scores the test window and "
                "compares each variable's first-lead skill with the canonical LoRA-only result. It writes only under "
                "`outputs/activity/` and stops if a canonical output changed. It takes about as long as Section 8.\n\n"
                "**Change one thing:** the adaptation scope. Set `RUN_ACTIVITY = True` and run the cell.\n\n"
                "**Predict before running:** will unfreezing the heads help everywhere, or help some variables and hurt others? "
                "Surface variables (`2t`, `10u`, `10v`, `msl`) or upper-air variables (`t`, `u`, `v`, `q`, `z`, `z500`)? Write it down."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_TRAINABLE = 'lora+heads'  # @param [\"lora+heads\"]\n\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', '--trainable', ACTIVITY_TRAINABLE)\n"
                "else:\n"
                "    print({{'activity': 'skipped (optional)', 'to_run': 'set RUN_ACTIVITY = True, then run this cell'}})"
            ),
        },
        {
            "md": (
                "**Observe:** per variable, the canonical and activity skill and `better_with_activity`; then the mean skill and "
                "how many surface and upper-air variables improved.\n\n"
                "**Explain:** did the result match your prediction? Which group gained, which lost?\n\n"
                "**Question tested:** is more trainable capacity always better when adapting on two days of data?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Not on every run, and the evidence here is thin. In the local CPU pre-flight of this notebook (a Windows "
                "workstation, not a supported runtime) `lora+heads` improved the 6-hour skill of all ten reported variables, "
                "with the mean skill going from about 0.90 to about 0.80; an earlier build of this repository, with another "
                "training set-up, found it helped the upper air (`z500`) but hurt the surface. The heads map the model's internal "
                "state to each variable, so 713 k extra parameters trained on twelve samples can fit this test period better — "
                "or fit some variables' errors at the expense of others. One seeded run on one window cannot tell those apart, "
                "and a larger adapter is also a larger change to carry and to check. LoRA alone stays the default because it is "
                "the smaller, scoped adapter (80 tensors, about 2 MB) that the export and reload contract is built around. "
                "Whatever your run shows is an observation about this run, not a failed activity; because only the scope "
                "changed, any difference is caused by it.\n\n"
                "</details>"
            ),
        },
        # ------------------------------------------------------------------ troubleshooting
        {
            "md": (
                "## Troubleshooting · [Engineering]\n\n"
                "| Observation | Action |\n"
                "|---|---|\n"
                "| Section 1 stops: not Linux x86_64 | Use Google Colab, Kaggle or a Linux Jupyter server; the locked environment is built for manylinux x86_64. |\n"
                "| Section 1 stops: not enough disk | Start a fresh runtime, or delete earlier `outputs/{stem}/` run directories. |\n"
                "| Section 2: download, `uv` or hash failure | Retry once on a stable connection. Never remove a pin or a hash; a hash mismatch means the file is not the locked one. |\n"
                "| Section 3: Hugging Face download fails or a digest mismatches | Retry; delete `weights/aurora-0.25-small/` and rerun Section 3 if a partial file remains. Never edit the manifest. |\n"
                "| Section 3: a pickle audit or a converted digest fails | Stop: the files are not the pinned ones. Do not bypass the audit. |\n"
                "| Section 5: a WeatherBench 2 object fails to download or mismatches | Retry; the 196 MB fetch resumes from the cache in `weights/wb2-era5-1p5deg/`. A digest mismatch is refused, never decoded. |\n"
                "| Section 5: `test window has N 6-hourly steps; … need at least M` | Your window is too short for the settings: supply a longer window, or lower `ROLLOUT_STEPS` / `MAX_LEAD_STEPS` as the message says, then re-run from Section 5. |\n"
                "| Section 5: `ORIGIN = … must be an integer in 1..N` | Choose an origin inside the window. |\n"
                "| Section 5: levels, longitude count, range or time-spacing refusal | Fix the NetCDF as the message says: the 13 standard levels, a longitude count divisible by 4, SI units, analyses exactly 6 h apart. |\n"
                "| `The upload dialog needs Google Colab` | Outside Colab, put the file on the machine and set `BYOD_PATH` to its path. |\n"
                "| A stage fails | The cell repeats the stage's error message; the full log is in the run directory under `logs/<stage>.log`. Fix the cause and rerun from that section. |\n"
                "| \"… is missing: run the stage that writes it\" | A later cell ran before an earlier one, or a Section 5 re-run cleared it. Run from Section 5 to the end. |\n"
                "| \"the adapter was trained before the data or settings changed\" | Re-run from Section 8. |\n"
                "| Out of memory | Use the default settings; a CPU runtime needs about 4 GB of RAM for the model and the four windows. |\n"
                "| Section 9 reports that the adapted model does not beat the frozen model | A legitimate result outside the default configuration: read the persistence column first. On the default configuration the stage stops, because the recorded runs improved; keep the logs and report it. |\n"
                "| Reload parity fails | Do not use the export. Rerun from Section 8; if it persists, keep the logs and report it. |\n\n"
                "Nothing is installed into the kernel, so no runtime restart is ever needed."
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Run six times coarser than its training grid, the frozen small checkpoint loses to persistence on most variables at "
        "6 h on the default sample — an honest number for an out-of-distribution use, and the one this notebook exists to move. "
        "On the default sample, a LoRA fine-tuning of half a million parameters on twelve one-step forecasts brought the mean "
        "skill below 1 at every lead to 24 h in the recorded runs, with most variables beating persistence, on a window from a "
        "different year and season than anything trained on; your run reports its own outcome in Section 9, and with other data "
        "or settings that outcome may be negative. The claim is about the contract: the adaptation moves a foundation weather "
        "model onto a grid it was not trained for from two days of data, and the artifact that carries the change is 2 MB.\n\n"
        "The test window is two days of one month, the scores come from a single seeded run with no dispersion estimate, and "
        "the small checkpoint is the one upstream publishes for testing. So a skill below 1 here says the contract works, not "
        "that this model forecasts at any published accuracy, that it is stable beyond 24 h, or that the adaptation transfers to "
        "other seasons — none of which this repository exercises. Fine-tuning on a narrow window can also erode the model "
        "elsewhere; upstream fine-tunes on years of data with roll-out training, which is out of scope here. Overlap between the "
        "2018–2021 sample and the checkpoint's pretraining data cannot be ruled out (Section 5), so the frozen numbers may "
        "include dates the model saw at 0.25°. CPU and GPU runs agree to the second or third decimal of a skill ratio, not "
        "bit for bit.\n\n"
        "Three things to carry to real data. **Native resolution:** at 0.25° the published model is far ahead of persistence "
        "without any adaptation; the frozen numbers here are a resolution story, not an Aurora story. **Independence:** "
        "consecutive analyses are near-duplicates — split by period, never by shuffling steps, and read the persistence column "
        "before any model number. **Static fields and units:** the model needs the land-sea mask, surface geopotential and soil "
        "type on your grid, SI units and the 13 standard levels; a mismatch is silent unless validation catches it.\n\n"
        "Successful execution proves that the recorded repository revision's package and stage runner, carried in this "
        "standalone notebook, can build a hash-locked environment without touching the kernel, acquire and digest-verify two "
        "pickled upstream assets, audit and convert them into safetensors without executing anything outside the audited "
        "allow-lists, rebuild the model from the installed package, fetch and validate digest-pinned real reanalysis, execute "
        "bounded fine-tuning, evaluate against persistence and the frozen model on an independent window at every lead, and "
        "emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish "
        "benchmark superiority, production fitness, or forecast skill beyond the checks shown.\n\n"
        "## Conclude with evidence · [Evaluation practice]\n\n"
        "Complete this paragraph with the numbers your run printed:\n\n"
        "> On the [data source] test window, scored at [number] origins, the frozen model's 6-hour mean skill against "
        "persistence was [value] with [n]/9 variables beating persistence. After LoRA fine-tuning on [number] one-step samples "
        "(best epoch [value], chosen on the validation window), the adapted model's 6-hour mean skill was [value] with [n]/9 "
        "variables beating persistence; at 24 h it was [value]. The reloaded adapter reproduced the trained model's forecast "
        "within [max difference]. With `lora+heads` (optional activity), [n] surface and [n] upper-air variables improved. "
        "These results show [what they do show] and do not show [one thing they cannot show].\n\n"
        "**Transfer:** switch on BYOD (Section 5) with a NetCDF window of your own region's period — at least the length "
        "Section 5 asks for, nothing confidential on a hosted runtime — and compare the persistence column with the default "
        "sample's before reading any model number. Which assumption from Section 5 is hardest to keep with one window of "
        "your own data?\n\n"
        "**AI Assistance Disclosure:** this notebook's code and explanations were developed with generative AI assistance under "
        "maintainer direction. The maintainer remains responsible for reviewing implementation, validating results and making "
        "release decisions.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/aurora-earth-system-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/aurora-earth-system-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/aurora-earth-system-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/{MODEL_ID} (revision `{MODEL_REVISION}`)\n"
        "- Upstream model descriptions (no training period is given for the small checkpoint): https://microsoft.github.io/aurora/models.html\n"
        "- Bodnar, C., Bruinsma, W. P., Lucic, A., et al. (2025). A foundation model for the Earth system. Nature 641, 1180–1187: https://doi.org/10.1038/s41586-025-09005-y\n"
        "- Rasp, S., Hoyer, S., Merose, A., et al. (2024). WeatherBench 2: A benchmark for the next generation of data-driven global weather models. JAMES 16: https://doi.org/10.1029/2023MS004019\n"
        "- Hersbach, H., et al. (2020). The ERA5 global reanalysis. QJRMS 146, 1999–2049: https://doi.org/10.1002/qj.3803\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
