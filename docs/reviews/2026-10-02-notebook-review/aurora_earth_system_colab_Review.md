# Aurora Earth-System E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/aurora-earth-system-pipeline`  
**Notebook:** `tutorials/aurora_earth_system_colab.ipynb`  
**Reviewed commit:** `4aa17112f41cabaa2b69d202607460ba9353ad22` (`main`, confirmed with `gh api repos/kurtvalcorza/aurora-earth-system-pipeline/commits/main`)  
**Notebook Git blob:** `42fb3883e2fce53ff8b3f31cae624631d2d3304b` (unchanged since `4a828fc`; the notebook's only commit on `main` is `60e76b1`)  
**Finding prefix:** `AUR`

## Executive assessment

As a reference pipeline the notebook is careful work. It carries its three package modules byte for byte (generator `--check` exits 0). It pins the model to an immutable Hub revision and digest-verifies it. Both upstream pickles are statically audited and converted once into safetensors. It fetches 57 digest-pinned ERA5 objects with no credential and says plainly that the frozen model runs out of distribution at 1.5°. Persistence and the frozen model are scored on the same origins as the adapted model, the three data roles come from different years, and the adapter reloads from files with parity asserted. Its explanations of persistence, the diurnal cycle and latitude weighting are some of the best in the fleet.

Four problems stand in the way of the release-grade label. First, the one clean-runtime run of this blob (Kaggle T4) needed a manual restart after the install cell, and the repository records it as PASSED (AUR-M1). Second, every rerun the notebook asks for (BYOD "re-run from that cell", and the optional experiments) reuses the already-adapted pipeline. The sections headed "frozen model" then score the adapted model, and the next fine-tuning run stacks on top of the previous one while still labelling epoch 0 "frozen model (LoRA zero-initialised)" (AUR-M2). Third, the documented BYOD minimum of three steps, and the 3-step NetCDF that Section 4 writes as the BYOD example, both pass validation and then fail in Section 6 and Section 9 at the default settings (AUR-M3). Fourth, the notebook declares `GUIDED` but has none of the guided layer and no structured learner activity (AUR-M4).

The adaptation code itself is not shown to be wrong on the default path. The problems are the `Run all` contract, the validity of any non-default run, and the BYOD and learning promises.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, and the opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`, `tutorials/README.md`, References) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main`. The 2.2 guided-layer items (GDL1–GDL15) are `SHOULD`s. Per spec §33, a 2.0/2.1 notebook does not become non-conformant just because it lacks them |
| Intended audience | Stated knowledge prerequisite: gridded analyses, pressure levels, lead time, persistence, RMSE. Python/notebook level is not stated |
| Supported runtime | "Google Colab or Jupyter, Python 3.12", CPU sufficient, CUDA used when present, float32, ~2 GB RAM |
| Promised outcomes | Pinned install; carried modules; staged and digest-verified Hub snapshot; static pickle audits and safetensors conversion; four pinned ERA5 windows with validation and four refusals; deterministic 24 h roll-out; persistence and frozen latitude-weighted RMSE by lead; bounded LoRA fine-tuning (80 tensors, 540,672 params); three-way held-out comparison; new-origin forecast; safetensors adapter export with fresh-reload parity; BYOD NetCDF through every stage |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/aurora_earth_system_pipeline/` @ `262c5f2f` |

### Evidence actually obtained

- **Source inspection:** all 25 cells (11 code), the three carried modules (`pipeline.py` `adapt` / `evaluate` / `predict` / `_batch` read in full), the generator and template, `tutorials/README.md`, `docs/release-verification.md`, `STATUS.md`, `MODEL_CARD.md`.
- **Documented execution evidence:** Kaggle T4, 2026-09-19, commit `4a828fc` / blob `42fb3883`, **the same blob as the reviewed revision**. Outcome: 11/11 cells, "1 restart after install cell". The archived `executed-pass1.ipynb` (workspace `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-aurora-earth-system/v1/evidence/`) records the pass-1 failure: `Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.` A local Windows pre-flight with pins pre-installed (2026-09-18, an earlier blob `b7e27bf3`) is labelled not-promotion evidence and is not used here.
- **Direct execution (this review):** `run_probes.py`, Windows, Python 3.12.14, torch 2.13.0+cpu, NumPy 2.5.3, CPU. **No Aurora model, no weights, no network.** The probes cover notebook JSON parse and compile of all 11 code cells; the git blob; generator `--check` (exit 0); the carried `validate_inputs`, `persistence_only`, `AuroraPipeline.evaluate`, `predict`, and `adapt`, run against synthetic 17 × 32 windows. They use a **stand-in** `aurora` module and a one-parameter model (persistence + a trainable `lora_shift`). The stand-in exercises the real carried control flow, not Aurora's numerics.
- **Not verified:** any Colab run of this notebook (the Colab runs in the release record are of the separate workshop notebook); whether a fresh Colab kernel also triggers the restart; the CPU default runtime on any hosted platform; any real model forward pass in this review; the BYOD NetCDF load (`xarray`/`netCDF4` not installed in the review environment) and BYOD through adaptation and reload; learner understanding.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | Strong on the default path: immutable revision, digest checks, audited pickle conversion, strict loading, a transactional `adapt` that rolls back on failure, and reload parity from files. Defects: the install guard turns a fresh-runtime `Run all` into a two-pass run (AUR-M1); `adapt` and the "frozen" sections are not idempotent across the reruns the notebook prescribes (AUR-M2); the stated BYOD minimum is not the workflow's real minimum (AUR-M3). |
| Scientific / experimental validity | Good default design: roles from 2019 / 2020 / 2021, epoch selected on validation, test used once, persistence and frozen scored on the same origins, and an honest out-of-distribution framing. Weaknesses: any rerun silently replaces the frozen reference with the adapted model (AUR-M2); pretraining overlap of the 2019–2021 ERA5 sample is not addressed (AUR-m1); a result-dependent `assert` turns a legitimate negative result into a crash (AUR-m2). |
| Promise fulfilment | Every listed default stage ran in the documented Kaggle run. Not met: the `Run all` promise (AUR-M1); the BYOD promise at the stated minimum (AUR-M3); "Google Colab or Jupyter, Python 3.12" and the CPU timings, which no hosted run supports (AUR-m3, AUR-m4). |
| Learner experience | Clear "Look for" and "Expect" notes before most stages, and strong conceptual prose on persistence and resolution. Missing: how-to-use guidance, a roadmap, glossary, predictions, checkpoints, worked answers, a structured experiment, and troubleshooting. 74k characters of carried code are not marked as infrastructure (AUR-M4, AUR-m5). |
| Spec conformance | Unmet applicable `MUST`s: RUN1/RUN10/ENV6/REL2/REL11 and §27 release-grade marking (AUR-M1); DAT13/DAT14 (AUR-M2); DAT12/DAT19/VAL6 and REL12 (AUR-M3); DAT9 (AUR-m1); UX12 (AUR-m3). `SHOULD` gaps: UX5, UX8, GDL1–GDL15, GDL11, SRC11, EXE2, OUT1/OUT2. |

## 3. Prioritized findings

### AUR-M1 — Major: fresh-runtime `Run all` stops at the install cell and needs a manual restart, yet the notebook is registered Release-grade

**Cell/section:** Section 1, install cell (cell 3). Generated from `_INSTALL_GUARD` in `tools/build_notebook.py` (lines 47–69).

**Observed issue:** The guard records which distributions are already imported, then runs `pip install` of the pins (`numpy==2.5.3`, `torch==2.14.0`, …). If a loaded distribution changed, it raises `RuntimeError(... 'Restart the runtime, then rerun from the top.')`. On the documented fresh Kaggle T4 image, NumPy 2.0.2 and cuda-bindings were already loaded, so pass 1 failed at this cell and the executor restarted. `docs/release-verification.md` step 4 accepts this ("an interpreter restart after the install is expected where the runtime's preinstalled torch, numpy or xarray differ from the pins"). `STATUS.md`, `tutorials/README.md` and the release record nevertheless mark the blob **Release-grade** on the strength of that run. No Colab run of this notebook exists, although `docs/release-verification.md` names Colab as "the runtime the tutorial is written for".

**Consequence:** A learner on a fresh hosted runtime that preloads NumPy gets an error on the first `Run all` and must restart and rerun by hand. Spec §5 says such a notebook "is not `Run all` conformant". The release label overstates what the evidence shows.

**Evidence:** Documented execution: `executed-pass1.ipynb`, cell 3 error output (quoted above); the "1 restart after install cell" annotation in `docs/release-verification.md` lines 128, 140 and 145 and in `STATUS.md` line 3. Source inspection of cell 3. Colab behaviour: **not verified**.

**Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Note that this repository's workshop notebook still installs into the kernel (its Kaggle run avoided the restart, but the guard remains), so take the template from the reference notebooks above, not from it. Implement it in `tools/build_notebook.py` and regenerate. Remove the "restart is expected" sentence from release-verification step 4, and return the registry to **Candidate** until a one-pass run is recorded.

**Acceptance check:** A fresh Colab runtime (and a fresh Kaggle image) runs the regenerated notebook with **Run all** once, with no restart and no `RuntimeError` from cell 3, and the run is recorded with blob, runtime and outcome. Until then, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` do not say Release-grade.

**Spec:** RUN1, RUN10, ENV6, REL2, REL11, §27 (MUST).

### AUR-M2 — Major: the reruns the notebook prescribes score the adapted model as "frozen" and stack a second fine-tuning on the first

**Cell/section:** Section 4 BYOD instruction ("set `USE_BYOD = True` in Section 4 and re-run from that cell", opening cell and `tools/notebook_template.py` line 64); "Optional experiments" in Interpretation (template line 406: change `TRAINABLE`, raise `EPOCHS`, change `ORIGIN`, BYOD); Sections 5–8 (cells 15, 17, 19, 21); `AuroraPipeline.adapt` (`src/aurora_earth_system_pipeline/pipeline.py` 736–875).

**Observed issue:** `pipe` is built once, in Section 3. `adapt()` starts from the model's **current** LoRA state. `initial_state` is used only to roll back on an exception, and nothing resets the LoRA tensors at entry. It still writes the literal epoch-0 note `"frozen model (LoRA zero-initialised)"`. After the default run, rerunning from Section 4 as instructed therefore gives these results:
- Section 5, "Zero-shot roll-out from the frozen model", rolls out the adapted model (its own output shows `'adapted': True`, contradicting the heading);
- Section 6's `frozen_test` is the adapted model's error, so the Section 8 "three-way comparison" becomes adapted-versus-re-adapted;
- Section 7 fine-tunes on top of the previous adapter, under the "zero-initialised" label;
- the `TRAINABLE = 'lora+heads'` comparison the notebook suggests is confounded by the LoRA training already applied.

**Consequence:** A learner who follows the BYOD instructions, or tries any suggested experiment, gets "frozen" and "adapted" numbers that do not mean what the headings and the exported `evaluation_report.json` say. Nothing on screen flags this except one `adapted` boolean. The conclusion the notebook teaches ("what LoRA recovers over the frozen model") becomes invalid for every non-default run, BYOD included.

**Evidence:** Source inspection of `adapt` (no reset; `initial_state` appears twice, both in the `except` path). Direct execution with the stand-in (probe P3): in the first pass, Section 5 `adapted=False` and the frozen 6 h mean skill is 1.000. After `adapt`, a rerun from Section 4 on the same `pipe` gives Section 5 `adapted=True` and a "frozen" 6 h mean skill of 0.424. The second `adapt` starts from `lora_shift=0.031`, yet epoch 0 still reads "frozen model (LoRA zero-initialised)", and the epoch-0 validation loss is 0.00020 against 0.00110 in the first run. Real-model behaviour is inferred from the same control flow, **not executed**.

**Recommended correction:** Make each pass start from the frozen model. Options: have Section 4 (or Section 5) rebuild `pipe` with `AuroraPipeline.from_pretrained(...)` from the already-verified files whenever the data or experiment changes; or add a `reset_adapter()` that restores the zero-initialised LoRA (and heads) and clears `self.adapter`, and call it before Sections 5 and 7. Alternatively, have `adapt()` refuse to run when `self.adapter is not None`. Also make the epoch-0 note truthful, and tell learners which section to rerun from for each experiment.

**Acceptance check:** After a complete default run, rerunning from Section 4 (default or BYOD) prints `'adapted': False` in Section 5. The default rerun reproduces the first run's frozen Section 6 numbers. Section 7's epoch 0 reports the same validation loss as the first run's epoch 0. A test that calls `adapt` twice in a row on a stand-in model shows that the second call starts from zero LoRA, or raises.

**Spec:** DAT13, DAT14 (MUST: BYOD uses the same semantics and reaches adaptation/evaluation validly); UX7, GDL10.

### AUR-M3 — Major: the documented BYOD minimum (three steps, "the shape Section 4 writes") passes validation and then fails in Sections 6 and 9

**Cell/section:** Opening cell and Prerequisites ("at least three 6-hourly steps"; "BYOD accepts NetCDF in the shape Section 4 writes"); cell 13 writes `trim_window(test_window, 3)` as the BYOD example; Section 6 (`MAX_LEAD_STEPS = 4`); Section 9 (`NEW_ORIGIN = min(n_steps - 1 - ROLLOUT_STEPS, 3)`, template line 332).

**Observed issue:** `validate_inputs` and `validate_dataset` accept any window of 3 or more steps. At the default form values, though, `persistence_only` and `evaluate(max_lead_steps=4)` need at least 6 steps, and `NEW_ORIGIN` is negative for windows shorter than 6 steps (−2 for 3 steps). So `pipe.predict(origin=NEW_ORIGIN)` raises `origin must be in 1..2`. The 3-step example file the notebook writes for BYOD cannot complete the BYOD path it is offered for. The failures arrive after validation has passed, as range errors from inside evaluation and inference. The release record has no BYOD positive or negative run.

**Consequence:** A user who follows the stated contract, or uploads the notebook's own example file, sees validation pass and then hits an error several sections later with no guidance on what to change. The BYOD promise ("validation, baselines, adaptation, held-out evaluation, inference, artifact export and reload parity") is not delivered for a valid-by-contract input.

**Evidence:** Direct execution with the stand-in (probe P2), at notebook defaults (`ORIGIN=1`, `ROLLOUT_STEPS=4`, `MAX_LEAD_STEPS=4`):

| steps | `validate_inputs` | Section 6 `persistence_only` / `evaluate` | Section 9 `NEW_ORIGIN` → `predict` |
|---|---|---|---|
| 3 | accepted | `window too short` / `6 are needed for 4 lead steps` | −2 → `origin must be in 1..2` |
| 4 | accepted | fails (same) | −1 → fails |
| 5 | accepted | fails (same) | 0 → fails |
| 6 | accepted | ok | 1 → ok |
| 8 | accepted | ok | 3 → ok |

These failures depend only on the window's length, not on model numerics. The NetCDF round-trip itself was **not executed** (no `xarray` here).

**Recommended correction:** Validate the BYOD window against the workflow actually configured, before any model work. In the BYOD branch of cell 13, require `n_steps ≥ HISTORY_STEPS + max(MAX_LEAD_STEPS, ROLLOUT_STEPS)` (6 at the defaults), or derive `MAX_LEAD_STEPS` / `ROLLOUT_STEPS` / `NEW_ORIGIN` from the window and print what was reduced (VAL7). Raise an error that names the step count and the fix. State the real minimum in the opening cell and Prerequisites (template lines 64–71 and 115). Write a sample BYOD file that completes the path, and record one BYOD positive and one negative run.

**Acceptance check:** With `USE_BYOD = True`, the NetCDF that Section 4 writes either completes Sections 5–9 at the defaults, or is rejected in Section 4 by a message naming the required step count. A 5-step window is rejected in Section 4, not in Section 6. The stated minimum equals the enforced one. `docs/release-verification.md` records one BYOD run of each kind.

**Spec:** DAT12, DAT19, VAL6, VAL7, REL12 (MUST); UX10.

### AUR-M4 — Major: declared `GUIDED`, but the notebook has no learner activity or guided scaffolding

**Cell/section:** Whole notebook; opening (`tools/notebook_template.py`), Interpretation "Optional experiments" (one sentence, template line 406).

**Observed issue:** The opening states "Learning objectives", but they are pipeline steps ("install the pinned runtime; inspect the carried pipeline… modules; stage and digest-verify…"), not observable learner outcomes. The notebook has none of the following: a **How to use this notebook** section, a roadmap, a glossary (for analysis, pressure level, lead time, persistence, skill ratio, LoRA, roll-out), predictions before principal results, interpretation checkpoints with worked answers, a Predict → Change one thing → Run → Observe → Explain activity, a troubleshooting section (Hub download, the 196 MB data fetch, RAM, the install restart, BYOD), or an evidence-based conclusion template. The "Optional experiments" line lists four changes but gives no question, no rerun instructions (see AUR-M2) and no guidance on what to observe. Probe P1 finds 0 occurrences of troubleshoot, glossary, "check your reasoning", "what to notice", "how to use this notebook" or "roadmap". In its favour, most stages do have "Look for" or "Expect" notes, and the prose on persistence, the diurnal cycle and resolution is strong.

**Consequence:** The notebook works as a reference script for someone who already knows gridded NWP verification. A learner new to the task can run it, but is never asked to predict, interpret or diagnose anything, so the `GUIDED` promise and UX5/UX8 are not delivered.

**Evidence:** Source inspection; probe P1 keyword counts.

**Recommended correction:** In `tools/notebook_template.py`, add the guided layer as in the spec's §25.13 reference notebook. Rewrite the objectives as observable actions (e.g. "explain why persistence for `2t` improves at 24 h", "compare frozen and adapted skill per lead"), and add how-to-use, a roadmap, an Input → Model → Output contract and a glossary. Ask for a prediction before Sections 6 and 8, with collapsible sample answers. Make one bounded **Predict → Change → Run → Observe → Explain** activity (e.g. `TRAINABLE = 'lora+heads'`) with exact rerun instructions; this depends on AUR-M2. Add troubleshooting and a conclusion scaffold.

**Acceptance check:** The regenerated notebook contains how-to-use, a roadmap, a glossary, at least two prediction prompts with collapsible sample answers, one Predict → Change → Run → Observe → Explain activity that runs without breaking `Run all`, a troubleshooting section, and a conclusion template. Alternatively, it is re-declared `REFERENCE` with that choice recorded in the registry.

**Spec:** UX5, UX8, UX9 (SHOULD); GDL1–GDL15 (SHOULD).

### AUR-m1 — Minor: pretraining overlap of the ERA5 sample is not addressed

**Cell/section:** Section 4 markdown, Interpretation and limits.

**Observed issue:** `MODEL_CARD.md` line 93 states that the small checkpoint was "trained on ERA5 alone". The sample windows are ERA5 from 2018-12 to 2021-10 (regridded by WeatherBench 2). The notebook says the test window is independent of the *training windows* but never discusses whether those analyses fall in the checkpoint's pretraining period. The words "overlap" and "pretraining" do not appear in the markdown (probe P1).

**Consequence:** Frozen and adapted numbers may partly reflect analyses the base model has already seen. That bears on the "held-out" reading, even though the 1.5° regridding differs from the 0.25° training grid.

**Evidence:** Source inspection; MODEL_CARD.md. This review did **not verify** the checkpoint's pretraining date range.

**Recommended correction:** State the upstream pretraining period for the small checkpoint (with a citation). If it includes 2018–2021, add a DAT9 limitation in Section 4 and in Interpretation.

**Acceptance check:** The notebook states whether pretraining overlap with the sample windows can be ruled out, citing a source for the period.

**Spec:** DAT9 (MUST).

### AUR-m2 — Minor: result-dependent `assert`s and fixed-result prose turn legitimate variation into a crash or a contradiction

**Cell/section:** Cell 21 (template lines 311–312); Section 7 markdown ("Watch the validation loss fall by two thirds…"); Interpretation ("brings the mean skill below 1 at every lead to 24 h").

**Observed issue:** Cell 21 asserts that the adapted 6 h mean skill beats the frozen model and that more variables beat persistence. These are empirical outcomes, not invariants. With BYOD, `lora+heads`, other epochs or another learning rate, a negative result raises `AssertionError`. Under **Run all**, execution then stops before Section 9, so no artifact, forecast or `result.json` is written. The prose hard-codes the default outcome as the conclusion.

**Consequence:** A learner's legitimate experiment looks like a broken notebook, and a negative result cannot be explored or exported. GDL8 asks for expectations without hard-coded results.

**Evidence:** Source inspection. On the default sample, the recorded runs satisfy both asserts (Kaggle 6 h 1.568 → 0.904, 1 → 6 variables).

**Recommended correction:** Keep the asserts for default-path verification only (e.g. `if not USE_BYOD and EPOCHS == 6 and TRAINABLE == 'lora'`), or turn them into a printed verdict. Rephrase the Section 7 and Interpretation text as expectations for the default sample, with guidance for when the adapted model does not win.

**Acceptance check:** With `TRAINABLE = 'lora+heads'` or `USE_BYOD = True`, a run where the adapted model does not beat the frozen model still reaches Section 9 and writes all five outputs, and the cell prints which outcome occurred.

**Spec:** GDL8, GDL14 (SHOULD); RUN9.

### AUR-m3 — Minor: runtime and timing claims are not tied to an environment

**Cell/section:** Prerequisites and opening cell (template line 115: "Python 3.12", "one 6-hour step … about 0.7 s and the default fine-tuning about four minutes", "about six minutes of model time"); Section 5 ("deterministic on CPU"); Section 7 ("six epochs take about four minutes on CPU").

**Observed issue:** The CPU timings match the local Windows pre-flight harness (`docs/release-verification.md` line 141, 194.6 s adaptation), which the repository itself calls "not a supported runtime". The notebook does not name that environment. Current Colab runs Python 3.13.15 (workshop evidence in the same file), not the stated 3.12. The only hosted run was on CUDA, so the CPU default runtime has no hosted evidence.

**Consequence:** A learner cannot tell whether the wait they see is normal, and the stated supported runtime does not match the one they get.

**Evidence:** Source inspection; `docs/release-verification.md` lines 140–141 and 210.

**Recommended correction:** Label the timings as measured on the named environment (or as estimates), give the measured hosted wall time (283.5 s on Kaggle T4, including downloads), and replace "Python 3.12" with the tested range.

**Acceptance check:** Every timing in the notebook names its environment or is labelled an estimate, and the stated Python version matches a recorded run.

**Spec:** UX12 (MUST); ENV3, ENV4.

### AUR-m4 — Minor: BYOD depends on `google.colab` and has no location field, although Jupyter is a stated runtime

**Cell/section:** Cell 13 (`from google.colab import files; files.upload()`; template line 148).

**Observed issue:** The only BYOD input route is the Colab upload widget. There is no `BYOD_PATH` form field, so on Jupyter or Kaggle (both named as supported runtimes or executors) the BYOD branch fails with `ModuleNotFoundError`, and a CI executor cannot drive it.

**Consequence:** BYOD works only on Colab, and it cannot be verified non-interactively (see AUR-M3, REL12).

**Evidence:** Source inspection.

**Recommended correction:** Add `BYOD_PATH = ''  # @param {type:"string"}` and read from it when set. Fall back to the upload widget only on Colab.

**Acceptance check:** With `USE_BYOD = True` and `BYOD_PATH` pointing at a valid NetCDF, the branch runs in plain Jupyter without importing `google.colab`.

**Spec:** EXE2, EXE1 (SHOULD); DAT16.

### AUR-m5 — Minor: 74k characters of carried module code are not marked as infrastructure; model construction suppresses every warning

**Cell/section:** Cells 5, 7, 9 (carried `pipeline.py`, `metrics.py`, `samples.py`); `build_model` (`pipeline.py` 321–327).

**Observed issue:** Three code cells totalling 74,449 characters sit between the install and the first learner-facing stage. They have no "Infrastructure" title and no `cellView: form` (probe P1: 0 code cells collapsed). The Section 2 note says "Nothing in these cells runs a model yet", but it does not say the learner may skip studying them. `build_model` wraps `AuroraSmallPretrained(...)` in `warnings.simplefilter("ignore")`, which hides every warning category raised during construction.

**Consequence:** Infrastructure dominates the first screenfuls and looks like prerequisite reading, and upstream deprecation or numerical warnings are hidden from the learner and from the release run.

**Evidence:** Source inspection; probe P1.

**Recommended correction:** Have the generator title these cells `# @title Infrastructure: …` with `cellView: form` (and keep the parity test aware of the title line). Narrow the suppression to the specific known warning by category and message.

**Acceptance check:** The carried cells are labelled Infrastructure and collapsed in Colab, and `build_model` filters only named warnings.

**Spec:** GDL11, GDL12, SRC11 (SHOULD).

### AUR-m6 — Minor: the new-origin forecast is exported only as global means

**Cell/section:** Cell 23, `outputs/aurora_earth_system_forecast.json`.

**Observed issue:** The forecast export holds origin time, shape, units and, per lead, only the global mean of each surface variable. No gridded field, atmospheric variable or coordinate is written. The NetCDF writer already present (`write_window_netcdf`) is not used for the forecast.

**Consequence:** The headline inference output cannot be reused, plotted or verified downstream. Completion and transfer stop at a printed sanity check.

**Evidence:** Source inspection.

**Recommended correction:** Also write the forecast fields as NetCDF with `time`, `level`, `latitude`, `longitude` coordinates, and document the fields.

**Acceptance check:** `outputs/` contains a forecast file holding every variable at every lead with its coordinates, and the notebook names its fields.

**Spec:** OUT1, OUT2, OUT4, INF3.

### Suggestions

- **AUR-S1:** After the 2.2 migration, update the declared `notebook_spec` (metadata, opening, `NOTEBOOK_SOURCE`, References, registry) from 2.0.
- **AUR-S2:** Note in the notebook that CPU and CUDA runs differ slightly (the pre-flight run gave 0.897 at 6 h, Kaggle T4 gave 0.904). This variability is currently recorded only in `docs/release-verification.md` (ENV8).
- **AUR-S3:** Document or enforce the bounds of the `MAX_LEAD_STEPS`, `ROLLOUT_STEPS` and `ORIGIN` form fields relative to the window length, so the suggested "raise `MAX_LEAD_STEPS`" experiment fails early with a clear message (`MAX_LEAD_STEPS = 8` needs 10 steps on an 8-step window).
- **AUR-S4:** Persist the Section 6 frozen table to its own JSON before adaptation, so a rerun cannot overwrite the reference used in the comparison.

## 4. Readiness decision

**Needs revision.**

- **Open Majors:** AUR-M1 (`Run all` needs a restart), AUR-M2 (reruns invalidate the frozen/adapted comparison), AUR-M3 (the BYOD minimum fails mid-workflow), AUR-M4 (no guided layer).
- **Unmet applicable `MUST`s:** RUN1/RUN10/ENV6/REL2/REL11 and §27 (AUR-M1); DAT13/DAT14 (AUR-M2); DAT12/DAT19/VAL6/REL12 (AUR-M3); DAT9 (AUR-m1); UX12 (AUR-m3).
- **Remaining gates:** a one-pass Colab `Run all` of the regenerated blob; one BYOD positive and one negative run; a rerun-from-Section-4 check showing frozen numbers reproduce.
- The current **Release-grade** label in `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` is not supported under spec §5 and §27.

## 5. Verified versus inferred

- **Verified by direct execution (CPU, stand-in, no model):** notebook parse and compile; blob `42fb3883`; generator `--check` exit 0; the BYOD step-count failures (AUR-M3, P2); `adapt` stacking and the stale "frozen" sections across reruns, using the real carried control flow (AUR-M2, P3/P4).
- **Verified from documented execution:** the default path completes on Kaggle T4 only after one restart (AUR-M1).
- **Inferred:** that the real Aurora model shows the same rerun contamination (same code path, not executed); that Colab also triggers the install restart (not verified); that pretraining overlaps the sample period (AUR-m1, period not checked).
- **Only Kurt or a learner can confirm:** whether the learner-facing gaps (AUR-M4) block the intended audience in practice.
- **Most likely to be wrong:** AUR-M1 on Colab specifically. If a fresh Colab kernel does not preload NumPy or cuda-bindings at versions that differ from the pins, the guard may not fire there. The restart is documented only on Kaggle.

*Probe ZIP:* `aurora_earth_system_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`). Run `python run_probes.py` from the repository root with NumPy and torch; it needs no `aurora` package, no weights and no network.
