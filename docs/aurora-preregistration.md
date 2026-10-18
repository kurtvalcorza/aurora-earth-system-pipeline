# Aurora tutorial: pre-registration of the validation grid and the fresh test (2026-10-19)

Applies to `tutorials/aurora_earth_system_colab.ipynb` (stage runner `tools/tutorial_stages.py`). Committed alone,
before any code that implements it. Owner rulings: R2-D1 (fresh test) and R2-D2 (grid), Kurt, 2026-10-18.
Review: tutorial review row 2, findings R2-M1 and R2-M2.

## 1. Why

The default recipe (lr 1e-3, 6 epochs, rank-8 LoRA on attention, `LOSS_SCALES`, `trainable='lora'`) was developed
on, and gated on, the October 2021 window. That window is therefore the **exposed development test**: it is still
reported, labelled as such, and is never called independent. Validation selection never bound (best epoch 6 of 6 in
every recorded run). This document fixes, before any run, how the recipe is re-selected on validation and how a
never-scored fresh test is scored.

## 2. Data (all WeatherBench 2 ERA5 1.5°, the store pinned in `samples.WB2_BASE_URL`)

Every object has its byte size and SHA-256 pinned in `samples.WB2_OBJECTS`; a mismatch is refused before decoding.
Window dates are checked against the store's time coordinate (`time/.zarray` and `time/3`, both pinned; `time/3`
holds indices 70158..93543, which covers every window used here).

| Role | Windows (chunk index; chunk k = analyses 8k..8k+7) |
|---|---|
| train | train-2019-01 (10957), train-2019-07 (11048) |
| validation | val-2020-04 (11185), val-2021-01 (11330), val-2021-04 (11375), val-2021-07 (11420) |
| exposed development test | test-2021-10 (11459) |
| fresh test | fresh-2022-01..12: 11512, 11528, 11542, 11557, 11572, 11588, 11603, 11618, 11634, 11649, 11664, 11679 |

- New windows are chosen by a deterministic rule: the chunk that contains the 15th of the month at 00:00.
- Fresh pool: 12 windows x 9 variable chunks = 108 objects, 587,041,847 bytes. Digest of the sorted
  `key size sha256` lines: `7acb15d01bf0b2472d73c48906a7a9735370b928b29aecf01fbd440b88a46f1b`. The fixer
  downloaded these bytes only to hash them (in memory, discarded); they were never decoded or scored.
- Disjointness: the fresh windows are at least 3 months after the exposed test, 6 months after the last validation
  window and 2.5 years after training. Monthly spacing keeps them close to synoptically independent; seasonal
  climatology is shared with training, which is not leakage.
- DAT9 (R2-m2). Aurora's documented pretraining (Bodnar et al., arXiv 2405.13063, App. C) uses HRES-0.25
  2016-2020, GFS 2015-2020, GEFS reforecasts 2000-2019 and IFS-ENS 2018-2020, with 2020 as its pretraining
  validation year; ERA5 1979-2020 is inferred (its table row did not render). The 0.25° fine-tune uses HRES
  2016-2021. Upstream does not document the small checkpoint's period. So training (2019) and val-2020-04 fall
  inside the large model's pretraining period; the 2021 validation windows and the exposed test fall outside it;
  2022 is the paper's own test year and outside pretraining and the 0.25° fine-tune.

## 3. Fixed settings (all of them enter the `preregistered` check)

- data source `sample` (no BYOD), windows and pins above, fresh-pool digest above;
- `origin 1`, `rollout_steps 4`, `max_lead_steps 4` (3 forecast origins per window);
- `trainable 'lora'` (rank 8, upstream), `LOSS_SCALES` as in `pipeline.py`, one-step (6 h) objective, AdamW,
  fixed learning rate, seed 0, one origin per optimiser step, float32;
- determinism: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`,
  `torch.use_deterministic_algorithms(True)` (no `warn_only`), `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA
  initialises, `torch`/`numpy`/`random` seeded with 0, no DataLoader workers. An op without a deterministic CUDA
  kernel stops the run with the op named; it is never worked around with `warn_only`;
- device: a Tesla T4 GPU. Any other device is exploratory;
- the grid below, run complete, and its selection.

## 4. Grid (R2-D2) and selection

| Config | Learning rate | Epochs |
|---|---|---|
| C0 (default) | 1e-3 | 6 |
| C1 | 3e-4 | 6 |
| C2 | 1e-3 | up to 30, patience 3 |

- Epoch choice inside a config: lowest validation loss (the one-step training loss, mean over the four validation
  windows); strict `<`, so the earliest epoch wins ties and epoch 0 (the frozen model) stays eligible. C2 stops after
  3 consecutive epochs without a strictly lower validation loss.
- Selection metric: the 6 h nine-variable mean RMSE ratio against persistence (`summary['6h']['mean_skill']` at
  `max_lead_steps 4`), averaged over the four validation windows. Lower is better.
- Keep the default: C0 is selected unless C1 or C2 is lower by at least **0.02** (a tolerance, not a significance
  threshold). If both qualify, the lower wins; an exact tie goes to the lower config index.
- Keep frozen: the selected config replaces the frozen model only if its selection metric is lower than the frozen
  model's by at least **0.02**; otherwise the frozen model is the selected system.
- The record states whether selection bound (best epoch < budget) per config.
- Grid ON when a GPU is present, OFF on CPU. A CPU run trains the recorded selection (C0 until a hosted record
  exists) and is exploratory. A grid hash (SHA-256 of the printed grid rows) is printed so two runs can be compared.

## 5. Reading rule for the fresh test (R2-D1)

- Primary endpoint: per fresh window, Δ = adapted − frozen 6 h nine-variable mean RMSE ratio against persistence
  (3 origins per window). The estimate is the mean of Δ over the 12 windows.
- Interval: 95 % t-interval over the 12 window-level Δ (df 11, t = 2.201); the window is the cluster. A window
  bootstrap (10,000 resamples, seed 0) is reported alongside, as secondary.
- Verdict: **"improved"** if the upper bound is < 0 and the mean Δ ≤ −0.02; **"worse"** if the lower bound is > 0;
  otherwise **"no detectable difference"**. "Beats persistence" is claimed only if the upper bound of the adapted
  ratio's 95 % t-interval is < 1. Every other number (other leads, per variable, counts) is descriptive.
- Power (planning estimate). The between-window SD of Δ is **assumed** to be 0.15-0.30, because only one, exposed,
  window has ever been scored. Then the expected half-width is 0.095-0.19, and the minimum detectable effect at 80 %
  power and two-sided α = 0.05 is 0.13-0.27 (3.08 x SE). The exposed window showed Δ ≈ −0.67.

## 6. Score-once

Per the fleet `SCORE_ONCE_RULE` (items 1-8):

- The fresh test is scored only when every setting in §3 and §4 holds, the grid is complete and matches, the model
  state equals the selected system, and every fresh object resolves and parses against its pin. Otherwise the run is
  exploratory: validation and the exposed development test only, written to `*_exploratory_<n>.json`.
- `DIMER_MASK_FRESH_TEST=1` never fetches the fresh windows (used for CPU checks and the masked hosted runs).
- A look counts only after scoring completes and the report is written atomically; the ledger is then appended
  atomically. A failed fetch, a partial grid, an interrupted write or a corrupt ledger is not a look and never
  relabels a run.
- In-notebook label of the first pre-registered scoring: "pre-registered evaluation (this runtime)". Later identical
  runs: "reproduction" (`_reproduction_<n>`). Which run is THE evaluation of record is stated in
  `docs/release-verification.md`: the first hosted T4 run under these settings, after two masked hosted runs that
  must print the same grid hash. `RECORDED_EVALUATION` stays None until then.
- Nothing is changed in response to a fresh-test number.
