"""Stage runner for the standalone Aurora Earth-system tutorial (NOTEBOOK_SPEC 2.5 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage data [--byod PATH] [--origin 1] ...

Nothing is installed into the notebook kernel, so a hosted runtime's preloaded packages are never replaced and no
restart is needed. Each stage is a separate process and starts from files only: the verified snapshot and the pinned
data cache under ``--weights``, the settings and roles written by ``data``, the adapter artifact written by ``adapt``
and the JSON records of earlier stages. Every stage that runs the model builds it from the pinned files, so a stage
re-run after a change always starts from the frozen model (LoRA at zero) and never from an earlier adaptation.
Learner-facing exports go to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On failure a stage writes
``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook re-raises in the kernel.

Stages: weights → runtime → data → rollout → frozen → adapt → evaluate → forecast → reload → bundle, plus the optional
``activity``. The package API does the work; this runner only sequences it and adds the tutorial's own checks (the
workflow-length check of a window against the configured settings, the validation grid, the score-once rules for
the fresh test, the activity).

Pre-registration: docs/aurora-preregistration.md. The October 2021 window is the *exposed development test* (the
default recipe was developed and gated on it). The 12-window 2022 fresh test is scored only under every
pre-registered setting (sample data, default settings and form fields, the complete GPU grid and its selection,
determinism flags, a Tesla T4, the pinned fresh pool) and never when ``DIMER_MASK_FRESH_TEST=1``; anything else is
exploratory and writes ``*_exploratory_<n>.json``.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

STEM = "aurora_earth_system"
SNAPSHOT_KEY = "aurora-0.25-small"
DATA_CACHE = "wb2-era5-1p5deg"
LOCK = "requirements.txt"
ADAPTER_DIR = f"{STEM}_adapter"
NEW_ORIGIN_MAX = 3
PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-6
REPORTED = ("2t", "10u", "10v", "msl", "t", "u", "v", "q", "z", "z500")
SHOWN = ("2t", "10u", "msl", "t", "z500")
DEFAULT_SETTINGS = {"origin": 1, "rollout_steps": 4, "max_lead_steps": 4}
DEFAULT_TRAINING = {"epochs": 6, "learning_rate": 1e-3, "trainable": "lora"}
# ---- pre-registered settings (docs/aurora-preregistration.md) -------------------------------------------------
SEED = 0
TRAIN_WINDOWS = ("train-2019-01", "train-2019-07")
VALIDATION_POOL = ("val-2020-04", "val-2021-01", "val-2021-04", "val-2021-07")
EXPOSED_TEST = "test-2021-10"
GRID: dict[str, dict[str, Any]] = {
    "C0": {"learning_rate": 1e-3, "epochs": 6, "patience": None},
    "C1": {"learning_rate": 3e-4, "epochs": 6, "patience": None},
    "C2": {"learning_rate": 1e-3, "epochs": 30, "patience": 3},
}
DEFAULT_CONFIG = "C0"
# Margins are a tolerance, not a significance threshold: keep the default (C0) and keep the frozen model unless beaten.
SELECTION_MARGIN = 0.02
KEEP_FROZEN_MARGIN = 0.02
VERDICT_MIN_EFFECT = 0.02
BOOTSTRAP_RESAMPLES = 10_000
PREREGISTERED_GPU = "Tesla T4"
DETERMINISM = {
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "use_deterministic_algorithms": True,
    "warn_only": False,
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "sdpa_backend": "math",
    "seed": SEED,
}
MASK_ENV = "DIMER_MASK_FRESH_TEST"
# The hosted run of record, once it exists (docs/release-verification.md). Until then a CPU run trains DEFAULT_CONFIG.
RECORDED_SELECTION: str | None = None
RECORDED_EVALUATION: dict[str, Any] | None = None
LOOK_PREREGISTERED = "pre-registered evaluation (this runtime)"
LOOK_REPRODUCTION = "reproduction"
LOOK_EXPLORATORY = "not scored (exploratory)"
LOOK_MASKED = "not scored (masked: DIMER_MASK_FRESH_TEST=1)"
FRESH_LEDGER = "fresh_test_ledger.json"
FRESH_MAIN = f"{STEM}_fresh_test_preregistered.json"
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.200985, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}
VERSION_CHECKS = ("torch", "microsoft-aurora", "timm", "einops", "xarray", "netcdf4", "numcodecs", "numpy", "safetensors", "huggingface-hub")
# Files each stage writes. A stage first removes its own files and those of every stage that reads them
# (DEPENDANTS), so no record from an earlier configuration can be read as a result of the current one.
STAGE_FILES: dict[str, tuple[str, ...]] = {
    "data": ("data.json", f"{STEM}_input_manifest.json", f"{STEM}_sample_window.nc"),
    "rollout": (f"{STEM}_frozen_rollout.json",),
    "frozen": (f"{STEM}_frozen_test.json",),
    "adapt": ("adapt.json", "reference_forecast.npz", f"{STEM}_training_history.json", ADAPTER_DIR, "grid"),
    "evaluate": (f"{STEM}_evaluation_report.json",),
    "forecast": (f"{STEM}_forecast.json", f"{STEM}_forecast.nc"),
    "reload": (f"{STEM}_reload_parity.json",),
    "bundle": (f"{STEM}_result.json",),
    "activity": ("activity",),
}
DEPENDANTS: dict[str, tuple[str, ...]] = {
    "data": tuple(STAGE_FILES),
    "rollout": ("rollout",),
    "frozen": ("frozen", "evaluate", "bundle", "activity"),
    "adapt": ("adapt", "evaluate", "forecast", "reload", "bundle", "activity"),
    "evaluate": ("evaluate", "bundle", "activity"),
    "forecast": ("forecast", "bundle"),
    "reload": ("reload", "bundle"),
    "bundle": ("bundle",),
}


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, the snapshot and data cache under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot(self) -> Path:
        return self.weights / SNAPSHOT_KEY

    @property
    def data_cache(self) -> Path:
        return self.weights / DATA_CACHE

    def write_state(self, name: str, value: Any) -> Path:
        return atomic_write_text(self.state / name, json.dumps(value, indent=2))

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 5, or from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        return atomic_write_text(self.out / name, json.dumps(value, indent=2, ensure_ascii=False, default=_jsonable))

    def read_output(self, name: str, needed_by: str) -> Any:
        path = self.out / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 5, or from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def invalidate(self, stage: str) -> list[str]:
        """Remove the files of ``stage`` and of every stage that depends on them (state and outputs)."""
        removed = []
        for later in DEPENDANTS[stage]:
            for name in STAGE_FILES[later]:
                for base in (self.state, self.out):
                    path = base / name
                    if path.is_dir():
                        shutil.rmtree(path)
                        removed.append(name)
                    elif path.is_file():
                        path.unlink()
                        removed.append(name)
        return removed


class Refusal(ValueError):
    """A refused learner input (DAT20): the notebook prints the message without a traceback and stops the cell."""


def atomic_write_text(path: Path, text: str) -> Path:
    """Write ``text`` to a temporary file beside ``path`` and rename it into place: a reader sees the old file or the
    whole new one, never a truncated one (R2-m4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover - numpy is in the lock
        pass
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_versions(text: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", text, re.M)}


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------------------
# the workflow-length contract (AUR-M3): a window must be long enough for the configured stages
# --------------------------------------------------------------------------------------------------


def required_steps(rollout_steps: int, max_lead_steps: int, history_steps: int = 2) -> int:
    """Analyses a window needs so that every stage completes at these settings.

    Section 7 scores every origin up to ``max_lead_steps`` ahead, which needs ``history_steps + max_lead_steps``
    analyses; Section 10 forecasts ``rollout_steps`` ahead from a later origin, which needs at least
    ``history_steps + rollout_steps``.
    """
    return history_steps + max(rollout_steps, max_lead_steps)


def new_origin(n_steps: int, rollout_steps: int) -> int:
    """The later forecast origin of Section 10: three steps in, or as late as the window allows."""
    return min(n_steps - 1 - rollout_steps, NEW_ORIGIN_MAX)


def check_workflow(n_steps: int, settings: dict[str, int], *, max_rollout: int = 8, history_steps: int = 2, label: str = "window") -> dict[str, int]:
    """Refuse settings or a window length that a later stage could not complete, naming the fix (VAL6/VAL7)."""
    origin, rollout, lead = settings["origin"], settings["rollout_steps"], settings["max_lead_steps"]
    for name, value in (("ROLLOUT_STEPS", rollout), ("MAX_LEAD_STEPS", lead)):
        if not isinstance(value, int) or not 1 <= value <= max_rollout:
            raise ValueError(f"{name} = {value!r}; it must be an integer in 1..{max_rollout}")
    need = required_steps(rollout, lead, history_steps)
    if n_steps < need:
        raise ValueError(
            f"{label} has {n_steps} 6-hourly steps; ROLLOUT_STEPS = {rollout} and MAX_LEAD_STEPS = {lead} need at least "
            f"{need} ({history_steps} history steps + {max(rollout, lead)} lead steps). Supply a longer window, or lower "
            f"ROLLOUT_STEPS / MAX_LEAD_STEPS so that {history_steps} + max(ROLLOUT_STEPS, MAX_LEAD_STEPS) <= {n_steps}"
        )
    if not isinstance(origin, int) or not history_steps - 1 <= origin <= n_steps - 1:
        raise ValueError(f"ORIGIN = {origin!r}; on this {n_steps}-step {label} it must be an integer in {history_steps - 1}..{n_steps - 1}")
    return {"required_steps": need, "new_origin": new_origin(n_steps, rollout)}


def trim_window(window: dict[str, Any], steps: int) -> dict[str, Any]:
    return {
        **window,
        "times": window["times"][:steps],
        "surf": {k: v[:steps] for k, v in window["surf"].items()},
        "atmos": {k: v[:steps] for k, v in window["atmos"].items()},
    }


def is_default(data: dict[str, Any], training: dict[str, Any]) -> bool:
    """Sample data, default settings and default form fields: the path on which the grid and the fresh test run."""
    return data["source"] == "sample" and data["settings"] == DEFAULT_SETTINGS and training == DEFAULT_TRAINING


# --------------------------------------------------------------------------------------------------
# determinism (pre-registered; docs/aurora-preregistration.md §3)
# --------------------------------------------------------------------------------------------------


def seed_all(seed: int = SEED) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_determinism() -> None:
    """Set every pre-registered determinism flag. ``CUBLAS_WORKSPACE_CONFIG`` is set before torch is imported, so before
    CUDA initialises. Scaled-dot-product attention is restricted to the math backend: the memory-efficient backend
    (the one a T4 would pick) has a non-deterministic backward. No ``warn_only``: an op with no deterministic kernel
    raises, and ``main`` stops the stage naming it."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISM["CUBLAS_WORKSPACE_CONFIG"]
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
        if hasattr(torch.backends.cuda, name):
            getattr(torch.backends.cuda, name)(False)
    torch.backends.cuda.enable_math_sdp(True)
    seed_all()


def determinism_state() -> dict[str, Any]:
    """The flags actually in effect in this process (recomputed, never taken from an earlier record)."""
    import torch

    others = [getattr(torch.backends.cuda, n)() for n in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "cudnn_sdp_enabled") if hasattr(torch.backends.cuda, n)]
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "sdpa_backend": "math" if torch.backends.cuda.math_sdp_enabled() and not any(others) else "other",
        "seed": SEED,
    }


def current_runtime() -> dict[str, Any]:
    """Device, GPU name and determinism flags of this process."""
    import torch

    device = device_name()
    return {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "determinism": determinism_state()}


def fresh_test_masked() -> bool:
    return os.environ.get(MASK_ENV, "").strip() not in ("", "0")


# --------------------------------------------------------------------------------------------------
# grid selection and the reading rule (pure functions; docs/aurora-preregistration.md §4-5)
# --------------------------------------------------------------------------------------------------


def grid_table_hash() -> str:
    return hashlib.sha256(json.dumps({"grid": GRID, "default": DEFAULT_CONFIG, "margins": [SELECTION_MARGIN, KEEP_FROZEN_MARGIN], "validation": VALIDATION_POOL, "train": TRAIN_WINDOWS, "seed": SEED}, sort_keys=True).encode("utf-8")).hexdigest()


def select_config(metrics: dict[str, float], frozen_metric: float) -> dict[str, Any]:
    """Keep C0 unless a challenger is lower by >= SELECTION_MARGIN (lowest wins; exact tie -> lower index); then keep
    the frozen model unless the chosen config is lower than it by >= KEEP_FROZEN_MARGIN. Values are compared at 6 dp."""
    order = list(GRID)
    m = {k: round(float(v), 6) for k, v in metrics.items()}
    base = m[DEFAULT_CONFIG]
    challengers = [c for c in order if c != DEFAULT_CONFIG and c in m and m[c] <= round(base - SELECTION_MARGIN, 6)]
    config = min(challengers, key=lambda c: (m[c], order.index(c))) if challengers else DEFAULT_CONFIG
    frozen = round(float(frozen_metric), 6)
    beats_frozen = m[config] <= round(frozen - KEEP_FROZEN_MARGIN, 6)
    return {"config": config, "selected": config if beats_frozen else "frozen", "beats_frozen_by_margin": beats_frozen, "metrics": m, "frozen_metric": frozen}


def grid_hash(rows: list[dict[str, Any]], frozen_metric: float, selected: str) -> str:
    """SHA-256 of the printed grid rows (rounded as printed), the frozen metric and the selection: two runs that print
    the same numbers print the same hash."""
    printed = [{k: row[k] for k in ("config", "learning_rate", "epochs_budget", "patience", "epochs_run", "best_epoch", "val_losses", "val_metric")} for row in rows]
    return hashlib.sha256(json.dumps({"rows": printed, "frozen": round(frozen_metric, 4), "selected": selected}, sort_keys=True).encode("utf-8")).hexdigest()


def reading_rule(deltas: list[float], adapted: list[float]) -> dict[str, Any]:
    """Primary endpoint: mean window-level delta (adapted - frozen 6 h nine-variable mean RMSE ratio), 95 % t-interval
    over windows; window bootstrap (seed 0) alongside; verdict per the pre-registration."""
    import numpy as np

    d = np.asarray(deltas, dtype=float)
    a = np.asarray(adapted, dtype=float)
    n = len(d)
    t = T_975[n - 1]
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    low, high = mean - t * se, mean + t * se
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))].mean(axis=1)
    a_mean, a_se = float(a.mean()), float(a.std(ddof=1) / np.sqrt(n))
    if high < 0 and mean <= -VERDICT_MIN_EFFECT:
        verdict = "improved"
    elif low > 0:
        verdict = "worse"
    else:
        verdict = "no detectable difference"
    return {
        "n_windows": n,
        "mean_delta": round(mean, 4),
        "ci95_t": [round(low, 4), round(high, 4)],
        "ci95_bootstrap": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "verdict": verdict,
        "adapted_ratio_mean": round(a_mean, 4),
        "adapted_ratio_ci95_t": [round(a_mean - t * a_se, 4), round(a_mean + t * a_se, 4)],
        "beats_persistence": bool(a_mean + t * a_se < 1),
    }


# --------------------------------------------------------------------------------------------------
# score-once state: ledger, look labels, run-numbered files (fail-closed; SCORE_ONCE_RULE)
# --------------------------------------------------------------------------------------------------


def read_ledger(run: Run) -> list[dict[str, Any]]:
    """The fresh-test ledger. A missing, empty or corrupt ledger never crashes a run: a corrupt one is set aside and
    the look count falls back to the complete report files, which are what make a look."""
    path = run.state / FRESH_LEDGER
    if not path.is_file():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(entries, list) and all(isinstance(e, dict) and "label" in e for e in entries):
            return entries
    except (OSError, ValueError):
        pass
    path.replace(path.with_name(f"{FRESH_LEDGER}.corrupt-{int(time.time())}"))
    print({"fresh_test_ledger": "unreadable; set aside, looks counted from the complete report files"})
    return []


def _complete_fresh_reports(run: Run) -> list[Path]:
    found = []
    for path in sorted(run.out.glob(f"{STEM}_fresh_test_*.json")):
        if not (path.name == FRESH_MAIN or path.name.startswith(f"{STEM}_fresh_test_reproduction_")):
            continue
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("test_look") in (LOOK_PREREGISTERED, LOOK_REPRODUCTION):
                found.append(path)
        except (OSError, ValueError):
            continue
    return found


def _next_free(run: Run, kind: str) -> Path:
    n = 1
    while (run.out / f"{STEM}_fresh_test_{kind}_{n}.json").exists():
        n += 1
    return run.out / f"{STEM}_fresh_test_{kind}_{n}.json"


def next_look(run: Run) -> dict[str, Any]:
    """Label and file of the next completed pre-registered scoring. The main file only for the first look in a runtime
    when no record exists; otherwise a run-numbered reproduction file. Never overwrites an existing file."""
    prior = max(len(read_ledger(run)), len(_complete_fresh_reports(run)))
    if RECORDED_EVALUATION is None and prior == 0 and not (run.out / FRESH_MAIN).exists():
        return {"label": LOOK_PREREGISTERED, "path": run.out / FRESH_MAIN, "prior_looks": prior}
    return {"label": LOOK_REPRODUCTION, "path": _next_free(run, "reproduction"), "prior_looks": prior}


def record_look(run: Run, look: dict[str, Any], report: dict[str, Any]) -> None:
    """A look counts only once its report is written (atomically); the ledger is appended after that, atomically."""
    atomic_write_text(look["path"], json.dumps(report, indent=2, ensure_ascii=False, default=_jsonable))
    entries = read_ledger(run)
    entries.append({"label": look["label"], "file": look["path"].name, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "grid_hash": report.get("grid_hash"), "selected_system": report.get("selected_system")})
    atomic_write_text(run.state / FRESH_LEDGER, json.dumps(entries, indent=2))


def settings_differing(data: dict[str, Any], adapt: dict[str, Any], runtime: dict[str, Any], artifact_sha256: str | None) -> list[str]:
    """Every pre-registered setting this run does not hold, recomputed from files and the live process (Section 9 never
    trusts a flag an earlier stage left behind). Empty = pre-registered."""
    from aurora_earth_system_pipeline import FRESH_POOL_DIGEST, fresh_pool_digest

    differing = []
    if data.get("source") != "sample":
        differing.append("data source (BYOD)")
    if data.get("settings") != DEFAULT_SETTINGS:
        differing.append("Section 5 settings (ORIGIN / ROLLOUT_STEPS / MAX_LEAD_STEPS)")
    if data.get("roles", {}).get("train") != list(TRAIN_WINDOWS) or data.get("roles", {}).get("validation") != list(VALIDATION_POOL):
        differing.append("window roles")
    if adapt.get("training") != DEFAULT_TRAINING:
        differing.append("Section 8 form fields (EPOCHS / LEARNING_RATE / TRAINABLE)")
    if adapt.get("seed") != SEED:
        differing.append("seed")
    grid = adapt.get("grid") or {}
    if not grid.get("ran"):
        differing.append("grid not run (CPU or changed settings)")
    elif not grid.get("complete") or [r.get("config") for r in grid.get("rows", [])] != list(GRID):
        differing.append("grid incomplete")
    elif grid.get("table_hash") != grid_table_hash():
        differing.append("grid table differs from the pre-registration")
    else:
        recomputed = select_config({r["config"]: r["val_metric_exact"] for r in grid["rows"]}, grid["frozen_metric_exact"])
        if recomputed["selected"] != adapt.get("selected_system"):
            differing.append("selected system differs from the pre-registered rule")
    for record, label in ((runtime, "this stage"), (adapt.get("runtime") or {}, "training")):
        if record.get("device") != "cuda" or PREREGISTERED_GPU not in str(record.get("gpu") or ""):
            differing.append(f"device in {label} (not a {PREREGISTERED_GPU})")
        if record.get("determinism") != DETERMINISM:
            differing.append(f"determinism flags in {label}")
    if fresh_pool_digest() != FRESH_POOL_DIGEST:
        differing.append("fresh-pool pins")
    if adapt.get("selected_system") != "frozen" and (artifact_sha256 is None or artifact_sha256 != adapt.get("artifact_sha256")):
        differing.append("model state differs from the selected system")
    return differing


def fresh_preflight(run: Run) -> dict[str, Any]:
    """Without downloading any fresh-test chunk: every fresh object is pinned, every variable's `.zarray` resolves and
    parses as the Blosc layout the decoder handles, and every window's dates match the pinned time coordinate."""
    from aurora_earth_system_pipeline import FRESH_TEST_WINDOWS, WB2_OBJECTS, check_window_times, fetch_object
    from aurora_earth_system_pipeline.samples import WB2_ATMOS, WB2_SURF, _object_keys

    for array in [*WB2_SURF.values(), *WB2_ATMOS.values()]:
        meta = json.loads(fetch_object(f"{array}/.zarray", cache_dir=run.data_cache))
        if meta.get("compressor", {}).get("id") != "blosc" or meta.get("filters") or meta.get("order") != "C" or meta.get("dtype") != "<f4":
            raise ValueError(f"{array}/.zarray: unexpected Zarr layout {meta.get('compressor')}, {meta.get('dtype')}")
    dates = {}
    for name, chunk in FRESH_TEST_WINDOWS.items():
        missing = [k for k in _object_keys(chunk) if k not in WB2_OBJECTS]
        if missing:
            raise ValueError(f"{name}: unpinned objects {missing}")
        dates[name] = check_window_times(chunk, cache_dir=run.data_cache)[0].strftime("%Y-%m-%dT%H:%M")
    return {"windows": len(dates), "objects": 9 * len(dates), "first_times": dates, "downloaded_fresh_chunks": 0}


# --------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------


def load_windows(run: Run, stage: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The settings record and the role windows the ``data`` stage chose (train, validation list, exposed development
    test), rebuilt from the cache or the BYOD file. Never the fresh test."""
    from aurora_earth_system_pipeline import fetch_sample_dataset, load_byod_dataset

    data = run.read_state("data.json", stage)
    if data["source"] == "byod":
        window = load_byod_dataset(data["byod_path"])[0]
        return data, [window], [window], window
    windows = fetch_sample_dataset(cache_dir=run.data_cache)
    roles = data["roles"]
    return data, [windows[name] for name in roles["train"]], [windows[name] for name in roles["validation"]], windows[roles["test"]]


def frozen_pipeline(run: Run) -> Any:
    """A new pipeline from the verified files: the pretrained model with its 80 LoRA tensors at zero."""
    from aurora_earth_system_pipeline import AuroraPipeline

    pipe = AuroraPipeline.from_pretrained(weights_dir=run.snapshot, device=device_name(), use_lora=True)
    if pipe.adapter is not None or lora_b_max_abs(pipe) != 0.0:
        raise RuntimeError("a freshly built pipeline is not the frozen model; refusing to continue")
    return pipe


def adapted_pipeline(run: Run, stage: str) -> tuple[Any, dict[str, Any]]:
    """A new pipeline rebuilt from the base files and the exported adapter, checked against the current data."""
    from aurora_earth_system_pipeline import AuroraPipeline

    adapt = run.read_state("adapt.json", stage)
    data = run.read_state("data.json", stage)
    if adapt["dataset_digest"] != data["digest"] or adapt["settings"] != data["settings"]:
        raise RuntimeError("the adapter was trained before the data or settings changed; re-run from Section 8")
    if adapt.get("selected_system") == "frozen":
        return frozen_pipeline(run), adapt
    if artifact_sha256(run) != adapt.get("artifact_sha256"):
        raise RuntimeError("the exported adapter is not the one Section 8 selected; re-run from Section 8")
    pipe = AuroraPipeline.from_artifact(run.out / ADAPTER_DIR, weights_dir=run.snapshot, device=device_name())
    return pipe, adapt


def artifact_sha256(run: Run) -> str | None:
    """SHA-256 of the exported adapter's tensor file, from its manifest after re-hashing the file, or None."""
    path = run.out / ADAPTER_DIR / "manifest.json"
    if not path.is_file():
        return None
    entry = json.loads(path.read_text(encoding="utf-8"))["files"][0]
    tensor = run.out / ADAPTER_DIR / entry["path"]
    return entry["sha256"] if tensor.is_file() and sha256_file(tensor) == entry["sha256"] else None


def lora_b_max_abs(pipe: Any) -> float:
    """Largest |lora_B| entry: 0.0 exactly when every LoRA update (B @ A) is zero, i.e. the pretrained model."""
    values = [float(p.detach().abs().max()) for name, p in pipe.model.named_parameters() if name.endswith("lora_B")]
    return max(values) if values else 0.0


def surface_means(forecast: dict[str, Any]) -> dict[str, float]:
    return {"2t_mean_K": round(float(forecast["surf"]["2t"].mean()), 3), "msl_mean_Pa": round(float(forecast["surf"]["msl"].mean()), 1), "z500_mean": round(float(forecast["atmos"]["z"][7].mean()), 1)}


def forecast_dataset(result: dict[str, Any], levels: Any, source: str) -> Any:
    """A ``predict`` result as an xarray Dataset (AUR-m6): every surface variable as (lead_hours, latitude, longitude),
    every atmospheric variable as (lead_hours, level, latitude, longitude), with valid times, coordinates and units."""
    import numpy as np
    import xarray as xr

    forecasts = result["forecasts"]
    surf = {name: (("lead_hours", "latitude", "longitude"), np.stack([f["surf"][name] for f in forecasts])) for name in forecasts[0]["surf"]}
    atmos = {name: (("lead_hours", "level", "latitude", "longitude"), np.stack([f["atmos"][name] for f in forecasts])) for name in forecasts[0]["atmos"]}
    dataset = xr.Dataset(
        {**surf, **atmos},
        coords={
            "lead_hours": np.array([f["lead_hours"] for f in forecasts], dtype=np.int64),
            "valid_time": ("lead_hours", np.array([np.datetime64(f["valid_time"]) for f in forecasts])),
            "level": np.array(levels, dtype=np.int64),
            "latitude": np.array(result["lat"], dtype=np.float64),
            "longitude": np.array(result["lon"], dtype=np.float64),
        },
        attrs={"origin_time": result["origin_time"], "model": f"{result['model']['id']}@{result['model']['revision']}", "adapted": int(result["model"]["adapted"]), "source": source},
    )
    for name in dataset.data_vars:
        dataset[name].attrs["units"] = result["units"][name]
    return dataset


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifest, fetch absent files at the pinned revision, verify, audit, convert."""
    from aurora_earth_system_pipeline import (
        MANIFEST_NAME,
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        PICKLE_AUDIT_SHA256,
        SOURCE_CKPT_NAME,
        SOURCE_STATIC_NAME,
        audit_pickle,
        convert_model,
        stage_missing_files,
        verify_converted,
        verify_snapshot,
    )
    from aurora_earth_system_pipeline.pipeline import CKPT_ALLOWED_GLOBALS, STATIC_ALLOWED_GLOBALS

    carried = json.loads((run.root / "weights" / SNAPSHOT_KEY / MANIFEST_NAME).read_text(encoding="utf-8"))
    if (carried["modelId"], carried["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity pinned in the carried pipeline.py")
    run.snapshot.mkdir(parents=True, exist_ok=True)
    (run.snapshot / MANIFEST_NAME).write_text(json.dumps(carried, indent=2), encoding="utf-8")
    print({"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "files": len(carried["files"]), "total_bytes": carried["totalBytes"]})
    fetched = stage_missing_files(run.snapshot, allow_download=True)
    print({"weights_dir": str(run.snapshot), "fetched": fetched})
    verified = verify_snapshot(run.snapshot)
    print({"verified_files": len(verified["files"]), "converted_present": verified["converted"]})
    if verified["converted"]:
        audits = {
            SOURCE_CKPT_NAME: audit_pickle(run.snapshot / SOURCE_CKPT_NAME, allowed=CKPT_ALLOWED_GLOBALS),
            SOURCE_STATIC_NAME: audit_pickle(run.snapshot / SOURCE_STATIC_NAME, allowed=STATIC_ALLOWED_GLOBALS),
        }
        for name, audit in audits.items():
            if audit["audit_sha256"] != PICKLE_AUDIT_SHA256[name]:
                raise ValueError(f"{name}: pickle audit digest {audit['audit_sha256']} != pinned {PICKLE_AUDIT_SHA256[name]}")
        converted = verify_converted(run.snapshot)["files"]
        conversion = "converted files already present; re-audited and digest-verified"
    else:
        record = convert_model(run.snapshot)
        audits, converted, conversion = record["audits"], record["converted"], f"converted in this runtime in {record['seconds']} s"
    for name, audit in audits.items():
        print({"pickle_audit": name, "globals": audit["globals"], "violations": audit["violations"], "audit_sha256": audit["audit_sha256"][:16] + "..."})
    for entry in converted:
        print({"converted": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"][:16] + "..."})
    print({"conversion": conversion, "served_from_pickle": False})
    run.write_state(
        "weights.json",
        {
            "source_assets": [e for e in carried["files"] if e["path"] in (SOURCE_CKPT_NAME, SOURCE_STATIC_NAME)],
            "pickle_audit_sha256": {name: audit["audit_sha256"] for name, audit in audits.items()},
            "converted": converted,
            "conversion": conversion,
            "fetched": fetched,
        },
    )


def stage_runtime(run: Run) -> None:
    """Section 4: versions in the isolated environment against the carried lock; device; input ceilings."""
    import importlib.metadata

    import torch

    from aurora_earth_system_pipeline import INPUT_SCHEMA

    locked = lock_versions((run.root / LOCK).read_text(encoding="utf-8"))
    installed = {name: importlib.metadata.version(name) for name in VERSION_CHECKS}
    mismatched = {name: {"installed": installed[name], "locked": locked.get(name)} for name in VERSION_CHECKS if locked.get(name) != installed[name].split("+")[0]}
    if mismatched:
        raise RuntimeError(f"the isolated environment does not match the carried lock: {mismatched}")
    device = device_name()
    record = {
        "python": platform.python_version(),
        "versions": installed,
        "versions_match_lock": True,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "float": "float32",
    }
    run.write_output("runtime.json", record)
    print({k: v for k, v in record.items() if k != "versions"})
    print({"versions": installed})
    print({"ceilings": {k: INPUT_SCHEMA[k] for k in ("grid", "levels", "time_steps", "timestep_hours", "history_steps", "rollout_steps")}})


def stage_data(run: Run) -> None:
    """Section 5: windows (pinned sample or BYOD), validation against the schema AND the configured workflow, roles,
    a sample NetCDF that completes the workflow at these settings, refusal probes."""
    from aurora_earth_system_pipeline import (
        FRESH_TEST_WINDOWS,
        HISTORY_STEPS,
        INPUT_SCHEMA,
        MAX_ROLLOUT_STEPS,
        SAMPLE_LABEL_SOURCE,
        fetch_sample_dataset,
        load_byod_dataset,
        validate_dataset,
        validate_inputs,
        write_window_netcdf,
    )

    byod_path = None
    settings = {"origin": run.options.origin, "rollout_steps": run.options.rollout_steps, "max_lead_steps": run.options.max_lead_steps}
    try:
        if run.options.byod:
            # Copy the file into the run's state first: the learner may point BYOD_PATH at the sample file this stage
            # rewrites below, and every later stage re-reads the window from the accepted copy.
            original = Path(run.options.byod).resolve()
            if not original.is_file():
                raise FileNotFoundError(f"BYOD file not found: {original.name}. Set BYOD_PATH to a NetCDF file on this machine, or leave it empty to upload one in Google Colab")
            byod_path = run.state / "byod_incoming" / original.name
            byod_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, byod_path)
            # DAT20: once the file is copied (it may be the sample file this stage wrote), the run is invalidated before
            # anything is validated, so a refusal can never leave the earlier configuration's results to be read as current.
            run.invalidate("data")
            test_window = load_byod_dataset(byod_path)[0]
            train_windows, val_windows = [test_window], [test_window]
            source, data_source = "byod", f"BYOD ({byod_path.name}) -- one window plays every role, so the held-out numbers are NOT independent; the fresh test is never scored"
            dataset_manifest = validate_dataset([test_window])
            roles = {"train": [test_window["name"]], "validation": [test_window["name"]], "test": test_window["name"], "fresh_test": []}
        else:
            windows = fetch_sample_dataset(cache_dir=run.data_cache)
            train_windows = [windows[name] for name in TRAIN_WINDOWS]
            val_windows, test_window = [windows[name] for name in VALIDATION_POOL], windows[EXPOSED_TEST]
            source, data_source = "sample", SAMPLE_LABEL_SOURCE
            dataset_manifest = validate_dataset([*train_windows, *val_windows, test_window])
            roles = {"train": list(TRAIN_WINDOWS), "validation": list(VALIDATION_POOL), "test": EXPOSED_TEST, "test_role": "exposed development test", "fresh_test": list(FRESH_TEST_WINDOWS)}
        checked_test = validate_inputs(test_window)
        workflow = check_workflow(checked_test["n_steps"], settings, max_rollout=MAX_ROLLOUT_STEPS, history_steps=HISTORY_STEPS, label="test window")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if byod_path is None and not run.options.byod:
            raise
        # DAT20: a refused BYOD file invalidates the run (earlier results are removed, so none can be read as current)
        # and stops the cell with one message, no traceback.
        cleared = run.invalidate("data")
        raise Refusal(f"BYOD file refused: {exc}. The results of the earlier configuration were cleared ({len(cleared)} files); fix the file, or set USE_BYOD = False to return to the sample, then re-run Section 5") from exc
    removed = run.invalidate("data")
    if byod_path is not None:
        accepted = run.state / "byod" / byod_path.name
        accepted.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(byod_path, accepted)
        byod_path = accepted
    print({"data_source": data_source, "n_windows": dataset_manifest["n_windows"], "grid": dataset_manifest["shape"], "steps_per_window": dataset_manifest["n_steps"], "forecast_origins": dataset_manifest["forecast_origins"]})
    print({"time_span": dataset_manifest["time_span"], "digest": dataset_manifest["digest"][:16] + "..."})
    print({"roles": roles})
    print({"settings": settings, "steps_needed_for_these_settings": workflow["required_steps"], "test_window_steps": checked_test["n_steps"], "new_origin_section_10": workflow["new_origin"], "verdict": "accepted"})
    print({"2t_mean_K_by_window": {w["name"]: round(float(w["surf"]["2t"].mean()), 2) for w in [*train_windows, *val_windows, test_window]}})
    print({"exposed_development_test": test_window["name"], "note": "the default recipe was developed on this window; it is reported, never called independent", "fresh_test": f"{len(roles['fresh_test'])} pinned windows, not fetched here (Section 9 fetches them only for a pre-registered scoring)"})
    sample_path = write_window_netcdf(trim_window(test_window, workflow["required_steps"]), run.out / f"{STEM}_sample_window.nc")
    print({"sample_netcdf": str(sample_path), "steps": workflow["required_steps"], "megabytes": round(sample_path.stat().st_size / 1e6, 1), "note": "the shortest window that completes every stage at these settings"})
    print({"validation": INPUT_SCHEMA["validation"]})
    short = trim_window(test_window, workflow["required_steps"] - 1)
    probes = {
        "wrong pressure levels": lambda: validate_inputs({**test_window, "levels": list(range(13))}),
        "odd longitude count": lambda: validate_inputs({**test_window, "lon": test_window["lon"][:-1], "surf": {k: v[..., :-1] for k, v in test_window["surf"].items()}, "atmos": {k: v[..., :-1] for k, v in test_window["atmos"].items()}, "static": {k: v[..., :-1] for k, v in test_window["static"].items()}}),
        "implausible temperature": lambda: validate_inputs({**test_window, "surf": {**test_window["surf"], "2t": test_window["surf"]["2t"] + 500.0}}),
        "irregular time spacing": lambda: validate_inputs({**test_window, "times": test_window["times"][:-1] + ["2030-01-01T00:00:00"]}),
        f"{workflow['required_steps'] - 1} steps for these settings": lambda: check_workflow(validate_inputs(short)["n_steps"], settings, max_rollout=MAX_ROLLOUT_STEPS, history_steps=HISTORY_STEPS, label="window"),
    }
    for name, probe in probes.items():
        try:
            probe()
            print({"probe": name, "verdict": "accepted"})
        except (TypeError, ValueError) as exc:
            print({"probe": name, "rejected": str(exc)[:240]})
    record = {
        "source": source,
        "byod_path": str(byod_path) if byod_path else None,
        "data_source": data_source,
        "settings": settings,
        "roles": roles,
        "digest": dataset_manifest["digest"],
        "n_steps": dataset_manifest["n_steps"],
        "test_steps": checked_test["n_steps"],
        **workflow,
    }
    run.write_state("data.json", record)
    run.write_output(f"{STEM}_input_manifest.json", {**{k: v for k, v in dataset_manifest.items() if k != "windows"}, **record})
    if removed:
        print({"cleared_results_of_an_earlier_configuration": sorted(set(removed) - {"data.json", f"{STEM}_input_manifest.json", f"{STEM}_sample_window.nc"})})


def stage_rollout(run: Run) -> None:
    """Section 6: the frozen model rolled out from ORIGIN, with the determinism check."""
    import numpy as np

    run.invalidate("rollout")
    data, _train, _val, test_window = load_windows(run, "rollout")
    settings = data["settings"]
    pipe = frozen_pipeline(run)
    started = time.perf_counter()
    result = pipe.predict(test_window, origin=settings["origin"], steps=settings["rollout_steps"])
    print({"origin_time": result["origin_time"], "shape": result["shape"], "seconds": round(time.perf_counter() - started, 2), "adapted": result["model"]["adapted"], "lora_b_max_abs": lora_b_max_abs(pipe), "device": pipe.device})
    leads = []
    for forecast in result["forecasts"]:
        row = {"lead_hours": forecast["lead_hours"], "valid_time": forecast["valid_time"], **surface_means(forecast)}
        leads.append(row)
        print(row)
    again = pipe.predict(test_window, origin=settings["origin"], steps=1)
    repeat_identical = bool(np.array_equal(again["forecasts"][0]["surf"]["2t"], result["forecasts"][0]["surf"]["2t"]))
    print({"repeat_identical": repeat_identical, "units": result["units"]})
    if not repeat_identical:
        raise RuntimeError("two identical forecast calls returned different fields")
    height = len(test_window["lat"]) - len(test_window["lat"]) % 4
    if tuple(result["shape"]) != (height, len(test_window["lon"])):
        raise RuntimeError(f"forecast shape {result['shape']} != ({height}, {len(test_window['lon'])})")
    run.write_output(f"{STEM}_frozen_rollout.json", {"model": result["model"], "origin_time": result["origin_time"], "shape": result["shape"], "leads": leads, "repeat_identical": repeat_identical})


def stage_frozen(run: Run) -> None:
    """Section 7: persistence and the frozen model scored per variable and lead time on the test window."""
    from aurora_earth_system_pipeline import persistence_only

    run.invalidate("frozen")
    data, _train, _val, test_window = load_windows(run, "frozen")
    lead = data["settings"]["max_lead_steps"]
    persistence = persistence_only(test_window, max_lead_steps=lead)
    print({"persistence_only": {name: {k: round(row["persistence"], 3) for k, row in table.items()} for name, table in persistence["variables"].items() if name in ("2t", "msl", "z500")}})
    pipe = frozen_pipeline(run)
    started = time.perf_counter()
    frozen_test = pipe.evaluate(test_window, max_lead_steps=lead)
    print({"frozen_model_seconds": round(time.perf_counter() - started, 1), "origins": frozen_test["n_origins"], "metric": frozen_test["metric"], "adapted": False})
    for name in SHOWN:
        print({name: {k: {"model": round(row["model"], 3), "persistence": round(row["persistence"], 3), "skill": round(row["skill"], 3)} for k, row in frozen_test["variables"][name].items()}})
    print({"frozen_summary": {k: {"mean_skill": round(s["mean_skill"], 3), "variables_beating_persistence": f"{s['variables_beating_persistence']}/9"} for k, s in frozen_test["summary"].items()}})
    run.write_output(f"{STEM}_frozen_test.json", {"dataset_digest": data["digest"], "settings": data["settings"], "persistence_baseline": persistence, "frozen_test": frozen_test})


def _training_options(run: Run) -> dict[str, Any]:
    return {"epochs": run.options.epochs, "learning_rate": run.options.learning_rate, "trainable": run.options.trainable}


def _progress(entry: dict[str, Any]) -> None:
    row = {"epoch": entry["epoch"], "train_loss": None if entry["train_loss"] is None else round(entry["train_loss"], 4), "val_loss": round(entry["val_loss"], 4)}
    if "val" in entry:
        row["val_skill_6h"] = {name: round(entry["val"][name]["6h"]["skill"], 3) for name in ("2t", "msl", "z500")}
    if "note" in entry:
        row["note"] = entry["note"]
    print(row, flush=True)


def validation_metric(pipe: Any, windows: list[dict[str, Any]], lead_steps: int) -> dict[str, Any]:
    """The selection metric: first-lead (6 h) nine-variable mean RMSE ratio against persistence, per validation window
    and averaged over windows (lower is better)."""
    per_window = {}
    for window in windows:
        summary = pipe.evaluate(window, max_lead_steps=lead_steps)["summary"]
        per_window[window["name"]] = float(summary[next(iter(summary))]["mean_skill"])
    return {"per_window": per_window, "mean": sum(per_window.values()) / len(per_window)}


def stage_adapt(run: Run) -> None:
    """Section 8: the pre-registered validation grid (GPU) or the recorded selection (CPU) or the learner's own
    configuration (exploratory); each config trains a freshly built frozen pipeline; selection on validation only."""
    import numpy as np

    run.invalidate("adapt")  # Section 8 reset: no adapter, grid row or record of an earlier run survives
    data, train_windows, val_windows, test_window = load_windows(run, "adapt")
    training = _training_options(run)
    runtime = current_runtime()
    lead = data["settings"]["max_lead_steps"]
    default_path = is_default(data, training)
    if default_path and runtime["device"] == "cuda":
        mode, configs = "grid", list(GRID)
    elif default_path:
        chosen = RECORDED_SELECTION or DEFAULT_CONFIG
        mode, configs = f"recorded selection {chosen} (grid off on CPU; exploratory)", [chosen]
    else:
        mode, configs = "exploratory: your own configuration (validation and the exposed development test only)", ["custom"]
    print({"mode": mode, "device": runtime["device"], "gpu": runtime["gpu"], "determinism": runtime["determinism"], "validation_windows": [w["name"] for w in val_windows]})
    seed_all()
    frozen = frozen_pipeline(run)
    frozen_val = validation_metric(frozen, val_windows, lead)
    del frozen
    print({"frozen_validation_metric": round(frozen_val["mean"], 4), "per_window": {k: round(v, 4) for k, v in frozen_val["per_window"].items()}})
    rows, results, started = [], {}, time.perf_counter()
    for name in configs:
        params = GRID[name] if name in GRID else {"learning_rate": training["learning_rate"], "epochs": training["epochs"], "patience": None}
        seed_all()
        pipe = frozen_pipeline(run)
        print({"config": name, **params, "trainable": training["trainable"], "starts_from_lora_b_max_abs": lora_b_max_abs(pipe)})
        result = pipe.adapt(train_windows, val_windows, epochs=params["epochs"], lr=params["learning_rate"], trainable=training["trainable"], seed=SEED, patience=params["patience"], progress=_progress)
        metric = validation_metric(pipe, val_windows, lead)
        target = run.state / "grid" / name
        pipe.save_artifact(target / ADAPTER_DIR, metadata={"tutorial": STEM, "config": name, "data_source": data["data_source"], "dataset_digest": data["digest"]})
        reference = pipe.predict(test_window, origin=data["settings"]["origin"], steps=2)["forecasts"]
        np.savez(target / "reference_forecast.npz", **{f"{group}/{k}/{n}": f[group][n] for k, f in enumerate(reference) for group in ("surf", "atmos") for n in f[group]})
        row = {
            "config": name,
            "learning_rate": params["learning_rate"],
            "epochs_budget": params["epochs"],
            "patience": params["patience"],
            "epochs_run": result["epochs_run"],
            "best_epoch": result["best_epoch"],
            "selection_bound": result["best_epoch"] < params["epochs"],
            "val_losses": [round(h["val_loss"], 4) for h in result["history"]],
            "val_metric": round(metric["mean"], 4),
            "val_metric_exact": metric["mean"],
            "val_metric_per_window": {k: round(v, 4) for k, v in metric["per_window"].items()},
        }
        rows.append(row)
        results[name] = result
        print({k: v for k, v in row.items() if k != "val_metric_exact"}, flush=True)
        del pipe
    if mode == "grid":
        selection = select_config({r["config"]: r["val_metric_exact"] for r in rows}, frozen_val["mean"])
        selected = selection["selected"]
    else:
        selection, selected = None, configs[0]
    complete = mode == "grid" and [r["config"] for r in rows] == list(GRID)
    digest = grid_hash(rows, frozen_val["mean"], selected)
    grid = {"ran": mode == "grid", "complete": complete, "table_hash": grid_table_hash(), "rows": rows, "frozen_metric": round(frozen_val["mean"], 4), "frozen_metric_exact": frozen_val["mean"], "selection": selection, "grid_hash": digest}
    print({"selected_system": selected, "rule": f"keep {DEFAULT_CONFIG} unless beaten by >= {SELECTION_MARGIN}; keep frozen unless beaten by >= {KEEP_FROZEN_MARGIN}" if mode == "grid" else "no grid in this mode"})
    print({"grid_hash": digest, "compare": "two runs with the same settings must print the same grid_hash"})
    seconds = round(time.perf_counter() - started, 1)
    if selected == "frozen":
        sha, adaptation = None, {"note": "the frozen model was kept (no config beat it on validation by the margin)"}
        seed_all()
        reference = frozen_pipeline(run).predict(test_window, origin=data["settings"]["origin"], steps=2)["forecasts"]
        np.savez(run.state / "reference_forecast.npz", **{f"{group}/{k}/{n}": f[group][n] for k, f in enumerate(reference) for group in ("surf", "atmos") for n in f[group]})
    else:
        shutil.copytree(run.state / "grid" / selected / ADAPTER_DIR, run.out / ADAPTER_DIR)
        shutil.copyfile(run.state / "grid" / selected / "reference_forecast.npz", run.state / "reference_forecast.npz")
        sha = artifact_sha256(run)
        adaptation = {k: v for k, v in results[selected].items() if k not in ("history", "trainable_names")}
        manifest = json.loads((run.out / ADAPTER_DIR / "manifest.json").read_text(encoding="utf-8"))
        print({"artifact": ADAPTER_DIR, "config": selected, "format": manifest["format"], "tensors": len(manifest["tensors"]), "bytes": manifest["files"][0]["bytes"], "sha256": manifest["files"][0]["sha256"][:16] + "..."})
    run.write_output(f"{STEM}_training_history.json", {"training": training, "mode": mode, "grid": grid, "selected_system": selected, "histories": {n: r["history"] for n, r in results.items()}, "seconds": seconds})
    run.write_state("adapt.json", {"training": training, "mode": mode, "seed": SEED, "settings": data["settings"], "dataset_digest": data["digest"], "runtime": runtime, "grid": grid, "selected_system": selected, "artifact_sha256": sha, "frozen_validation": frozen_val, "seconds": seconds, "adaptation": adaptation})


def _comparison(frozen_test: dict[str, Any], adapted_test: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    comparison = {
        name: {
            k: {"persistence": round(row["persistence"], 4), "frozen": round(frozen_test["variables"][name][k]["model"], 4), "adapted": round(row["model"], 4), "skill_frozen": round(frozen_test["variables"][name][k]["skill"], 3), "skill_adapted": round(row["skill"], 3)}
            for k, row in adapted_test["variables"][name].items()
        }
        for name in REPORTED
    }
    summary = {
        k: {"frozen_mean_skill": round(frozen_test["summary"][k]["mean_skill"], 3), "adapted_mean_skill": round(adapted_test["summary"][k]["mean_skill"], 3), "beating_persistence_frozen": frozen_test["summary"][k]["variables_beating_persistence"], "beating_persistence_adapted": adapted_test["summary"][k]["variables_beating_persistence"]}
        for k in adapted_test["leads"]
    }
    return comparison, summary


def score_fresh(frozen: Any, selected: Any, windows: dict[str, dict[str, Any]], lead: int) -> dict[str, Any]:
    """Frozen and selected system on every fresh window; per-window first-lead ratios and the reading rule."""
    rows = {}
    for name, window in windows.items():
        f = frozen.evaluate(window, max_lead_steps=lead)["summary"]
        a = selected.evaluate(window, max_lead_steps=lead)["summary"]
        first = next(iter(a))
        rows[name] = {"frozen_ratio": round(float(f[first]["mean_skill"]), 4), "adapted_ratio": round(float(a[first]["mean_skill"]), 4), "delta": round(float(a[first]["mean_skill"]) - float(f[first]["mean_skill"]), 4)}
    rule = reading_rule([r["delta"] for r in rows.values()], [r["adapted_ratio"] for r in rows.values()])
    return {"per_window": rows, "reading_rule": rule}


def stage_evaluate(run: Run) -> None:
    """Section 9: the selected system on the exposed development test (always) and on the fresh test (only when every
    pre-registered check passes, recomputed here); never raises on any test outcome."""
    from aurora_earth_system_pipeline import MODEL_ID, MODEL_KEY, MODEL_REVISION, fetch_fresh_test

    run.invalidate("evaluate")
    data, _train, _val, test_window = load_windows(run, "evaluate")
    frozen = run.read_output(f"{STEM}_frozen_test.json", "evaluate")
    if frozen["dataset_digest"] != data["digest"] or frozen["settings"] != data["settings"]:
        raise RuntimeError("the frozen scores were computed for other data or settings; re-run from Section 7")
    pipe, adapt = adapted_pipeline(run, "evaluate")
    lead = data["settings"]["max_lead_steps"]
    adapted_test = pipe.evaluate(test_window, max_lead_steps=lead)
    frozen_test = frozen["frozen_test"]
    comparison, summary = _comparison(frozen_test, adapted_test)
    print({"exposed_development_test": test_window["name"], "selected_system": adapt["selected_system"], "note": "the default recipe was developed on this window; not an independent estimate"})
    for name in ("2t", "msl", "z500", "u"):
        print({name: comparison[name]})
    for k, row in summary.items():
        print({k: row})
    first = adapted_test["leads"][0]
    outcome = {
        "lead": first,
        "window": "exposed development test",
        "adapted_mean_skill_below_frozen": summary[first]["adapted_mean_skill"] < summary[first]["frozen_mean_skill"],
        "more_variables_beat_persistence": summary[first]["beating_persistence_adapted"] > summary[first]["beating_persistence_frozen"],
        "default_configuration": is_default(data, adapt["training"]),
    }
    runtime = current_runtime()
    differing = settings_differing(data, adapt, runtime, artifact_sha256(run))
    preregistered = not differing
    masked = fresh_test_masked()
    report = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY},
        "data_source": data["data_source"],
        "dataset_digest": data["digest"],
        "settings": data["settings"],
        "training": adapt["training"],
        "mode": adapt["mode"],
        "selected_system": adapt["selected_system"],
        "grid_hash": adapt["grid"]["grid_hash"],
        "roles": data["roles"],
        "runtime": runtime,
        "preregistered": preregistered,
        "settings_differing": differing,
        "validation": {"frozen": adapt["frozen_validation"], "grid": adapt["grid"]},
        "persistence_baseline": frozen["persistence_baseline"],
        "exposed_development_test": {"frozen": frozen_test, "adapted": adapted_test, "comparison": comparison, "summary": summary, "outcome": outcome},
        "summary": summary,
        "outcome": outcome,
        "adaptation": adapt["adaptation"],
        "adaptation_seconds": adapt["seconds"],
        "evidence": "the exposed development test is one window the recipe was developed on; the fresh test is scored only under the pre-registered settings (docs/aurora-preregistration.md); see docs/release-verification.md for the recorded evaluation and its reproductions",
    }
    if masked or not preregistered:
        report["test_look"] = LOOK_MASKED if masked and preregistered else LOOK_EXPLORATORY
        path = _next_free(run, "exploratory")
        report["fresh_test_file"] = path.name
        atomic_write_text(path, json.dumps({k: report[k] for k in ("test_look", "preregistered", "settings_differing", "mode", "selected_system", "grid_hash", "validation", "outcome", "runtime")}, indent=2, ensure_ascii=False, default=_jsonable))
        run.write_output(f"{STEM}_evaluation_report.json", report)
        print({"fresh_test": report["test_look"], "settings_differing": differing, "record": path.name})
        return
    preflight = fresh_preflight(run)
    print({"fresh_test_preflight": preflight})
    look = next_look(run)
    try:
        windows = fetch_fresh_test(cache_dir=run.data_cache)  # every object fetched, digest-checked and decoded first
    except Exception as exc:
        report["test_look"] = f"not scored (fresh-test fetch failed: {type(exc).__name__}); not a look"
        run.write_output(f"{STEM}_evaluation_report.json", report)
        raise RuntimeError(f"the fresh-test windows could not be fetched or decoded ({exc}); nothing was scored and no look was recorded") from exc
    seed_all()
    fresh = score_fresh(frozen_pipeline(run), pipe, windows, lead)
    fresh_report = {"test_look": look["label"], "preregistered": True, "settings_differing": [], "selected_system": adapt["selected_system"], "grid_hash": adapt["grid"]["grid_hash"], "runtime": runtime, "fresh_test": fresh, "prior_looks_in_this_runtime": look["prior_looks"], "record_note": "which run is THE evaluation of record is stated in docs/release-verification.md"}
    record_look(run, look, fresh_report)  # the look counts from here: report written, then the ledger
    report["test_look"] = look["label"]
    report["fresh_test_file"] = look["path"].name
    report["fresh_test"] = fresh
    run.write_output(f"{STEM}_evaluation_report.json", report)
    for name, row in fresh["per_window"].items():
        print({name: row})
    print({"fresh_test": look["label"], **fresh["reading_rule"], "record": look["path"].name})


def stage_forecast(run: Run) -> None:
    """Section 10: a forecast from a later origin, exported as gridded NetCDF (every variable, lead and coordinate)."""
    from aurora_earth_system_pipeline import LEVELS, lat_weighted_rmse

    run.invalidate("forecast")
    data, _train, _val, test_window = load_windows(run, "forecast")
    pipe, _adapt = adapted_pipeline(run, "forecast")
    origin, steps = data["new_origin"], data["settings"]["rollout_steps"]
    result = pipe.predict(test_window, origin=origin, steps=steps)
    print({"origin_index": origin, "origin_time": result["origin_time"], "adapted": result["model"]["adapted"], "shape": result["shape"]})
    height = result["shape"][0]
    leads = []
    for k, forecast in enumerate(result["forecasts"], start=1):
        row = {"lead_hours": forecast["lead_hours"], "valid_time": forecast["valid_time"], "2t_mean_K": round(float(forecast["surf"]["2t"].mean()), 3)}
        if origin + k < len(test_window["times"]):
            row["2t_rmse_vs_analysis"] = round(lat_weighted_rmse(forecast["surf"]["2t"], test_window["surf"]["2t"][origin + k, :height], result["lat"]), 3)
        row["note"] = "sanity check on the exposed development test, not an evaluation"
        leads.append(row)
        print(row)
    dataset = forecast_dataset(result, LEVELS, data["data_source"])
    nc_path = run.out / f"{STEM}_forecast.nc"
    dataset.to_netcdf(nc_path)
    fields = {name: {"dims": list(dataset[name].dims), "units": result["units"][name]} for name in dataset.data_vars}
    run.write_output(f"{STEM}_forecast.json", {"origin_time": result["origin_time"], "origin_index": origin, "shape": result["shape"], "units": result["units"], "netcdf": nc_path.name, "fields": fields, "leads": leads})
    print({"forecast_netcdf": str(nc_path), "megabytes": round(nc_path.stat().st_size / 1e6, 1), "variables": sorted(dataset.data_vars), "dims": dict(dataset.sizes)})


def stage_reload(run: Run) -> None:
    """Section 11: rebuild from the exported files in another new process; forecast parity with the trained model."""
    import numpy as np

    run.invalidate("reload")
    data, _train, _val, test_window = load_windows(run, "reload")
    manifest = json.loads((run.out / ADAPTER_DIR / "manifest.json").read_text(encoding="utf-8"))
    print({"artifact": str(run.out / ADAPTER_DIR), "format": manifest["format"], "tensors": len(manifest["tensors"]), "bytes": manifest["files"][0]["bytes"], "sha256": manifest["files"][0]["sha256"][:16] + "..."})
    pipe, _adapt = adapted_pipeline(run, "reload")
    reloaded = pipe.predict(test_window, origin=data["settings"]["origin"], steps=2)["forecasts"]
    reference = np.load(run.state / "reference_forecast.npz")
    diffs: dict[str, float] = {"surf": 0.0, "atmos": 0.0}
    within = True
    for lead, forecast in enumerate(reloaded):
        for group in ("surf", "atmos"):
            for name, field in forecast[group].items():
                expected = reference[f"{group}/{lead}/{name}"]
                diffs[group] = max(diffs[group], float(np.abs(field - expected).max()))
                within = within and bool(np.allclose(field, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL))
    parity = {"max_abs_surf_diff": diffs["surf"], "max_abs_atmos_diff": diffs["atmos"], "rtol": PARITY_RTOL, "atol": PARITY_ATOL, "reload_verification": "PASSED" if within else "FAILED", "reloaded_best_epoch": pipe.adapter["best_epoch"] if pipe.adapter else 0, "compared": "2 forecast steps from ORIGIN, every variable, against the trained model in the adapt stage"}
    run.write_output(f"{STEM}_reload_parity.json", parity)
    print({"reload_parity": parity})
    if not within:
        raise RuntimeError(f"reloaded adapter does not reproduce the trained model's forecast: {parity}")


def stage_bundle(run: Run) -> None:
    """Section 12: one result record linking every output, with provenance and SHA-256 of each file."""
    from aurora_earth_system_pipeline import MODEL_ID, MODEL_KEY, MODEL_LICENSE, MODEL_REVISION, WB2_BASE_URL, WB2_OBJECTS

    run.invalidate("bundle")
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8"))
    weights = run.read_state("weights.json", "bundle")
    data = run.read_state("data.json", "bundle")
    report = run.read_output(f"{STEM}_evaluation_report.json", "bundle")
    manifest_path = run.out / ADAPTER_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    payload = {
        "notebook_source": source,
        "repository_revision": source["revision"],
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "model_license": MODEL_LICENSE},
        "provenance": {
            "source_assets": weights["source_assets"],
            "pickle_audit_sha256": weights["pickle_audit_sha256"],
            "converted": weights["converted"],
            "pickles_unpickled_once_for_conversion": True,
            "served_from_pickle": False,
            "remote_code_executed": False,
            "data_objects": len(WB2_OBJECTS),
            "data_base_url": WB2_BASE_URL,
        },
        "runtime": run.read_output("runtime.json", "bundle"),
        "environment": "isolated hash-locked uv environment; nothing installed into the notebook kernel",
        "data_source": data["data_source"],
        "settings": data["settings"],
        "training": report["training"],
        "summary": report["summary"],
        "outcome": report["outcome"],
        "selected_system": report["selected_system"],
        "grid_hash": report["grid_hash"],
        "test_look": report["test_look"],
        "preregistered": report["preregistered"],
        "settings_differing": report["settings_differing"],
        "artifact": {"dir": ADAPTER_DIR, "sha256": manifest["files"][0]["sha256"], "bytes": manifest["files"][0]["bytes"]} if manifest else None,
        "reload_parity": run.read_output(f"{STEM}_reload_parity.json", "bundle"),
    }
    files = sorted(p for p in run.out.rglob("*") if p.is_file() and "activity" not in p.relative_to(run.out).parts and p.name != f"{STEM}_result.json")
    payload["files"] = {p.relative_to(run.out).as_posix(): sha256_file(p) for p in files}
    run.write_output(f"{STEM}_result.json", payload)
    print({"outputs_directory": str(run.out)})
    for path in sorted(p for p in run.out.rglob("*") if p.is_file()):
        print(f"  - {path.relative_to(run.out).as_posix()} ({path.stat().st_size / 1024:.1f} KB)")


def stage_activity(run: Run) -> None:
    """Section 13 (optional): change one thing -- the adaptation scope -- and compare per variable; writes only under
    outputs/activity/ and checks that no canonical output changed."""
    data, train_windows, val_windows, test_window = load_windows(run, "activity")
    report = run.read_output(f"{STEM}_evaluation_report.json", "activity")
    canonical = {p.relative_to(run.out).as_posix(): sha256_file(p) for p in sorted(run.out.rglob("*")) if p.is_file() and "activity" not in p.relative_to(run.out).parts}
    training = {**report["training"], "trainable": run.options.trainable}
    if report.get("selected_system") in GRID:
        training.update(epochs=GRID[report["selected_system"]]["epochs"], learning_rate=GRID[report["selected_system"]]["learning_rate"])
    if training["trainable"] == report["training"]["trainable"]:
        raise ValueError(f"the activity changes the adaptation scope: choose a TRAINABLE other than the canonical {report['training']['trainable']!r}")
    pipe = frozen_pipeline(run)
    print({"changed_variable": "adaptation scope (TRAINABLE)", "canonical": report["training"]["trainable"], "activity": training["trainable"], "unchanged": {k: v for k, v in training.items() if k != "trainable"}, "starts_from_lora_b_max_abs": lora_b_max_abs(pipe)})
    print({"test_look": LOOK_EXPLORATORY, "scores": "validation and the exposed development test only; the fresh test is never scored by the activity"})
    result = pipe.adapt(train_windows, val_windows, epochs=training["epochs"], lr=training["learning_rate"], trainable=training["trainable"], seed=SEED, progress=_progress)
    test = pipe.evaluate(test_window, max_lead_steps=data["settings"]["max_lead_steps"])
    first = test["leads"][0]
    canonical_test = report["exposed_development_test"]["adapted"]
    rows = {}
    for name in REPORTED:
        canon = canonical_test["variables"][name][first]["skill"]
        changed = test["variables"][name][first]["skill"]
        rows[name] = {"skill_canonical": round(canon, 3), "skill_activity": round(changed, 3), "better_with_activity": changed < canon}
        print({name: rows[name]})
    record = {
        "changed_variable": "adaptation scope",
        "canonical_trainable": report["training"]["trainable"],
        "activity_trainable": training["trainable"],
        "n_trainable": result["n_trainable"],
        "best_epoch": result["best_epoch"],
        "lead": first,
        "per_variable": rows,
        "test_look": LOOK_EXPLORATORY,
        "window": "exposed development test",
        "mean_skill": {"canonical": round(canonical_test["summary"][first]["mean_skill"], 3), "activity": round(test["summary"][first]["mean_skill"], 3)},
        "surface_better": sum(rows[n]["better_with_activity"] for n in ("2t", "10u", "10v", "msl")),
        "upper_air_better": sum(rows[n]["better_with_activity"] for n in ("t", "u", "v", "q", "z", "z500")),
    }
    out = run.out / "activity"
    out.mkdir(exist_ok=True)
    tag = training["trainable"].replace("+", "_plus_")
    atomic_write_text(out / f"{STEM}_activity_{tag}.json", json.dumps(record, indent=2))
    after = {p.relative_to(run.out).as_posix(): sha256_file(p) for p in sorted(run.out.rglob("*")) if p.is_file() and "activity" not in p.relative_to(run.out).parts}
    if after != canonical:
        raise RuntimeError("the activity changed a canonical output; it must write only under outputs/activity/")
    print({k: v for k, v in record.items() if k != "per_variable"})


STAGES = {
    "weights": stage_weights,
    "runtime": stage_runtime,
    "data": stage_data,
    "rollout": stage_rollout,
    "frozen": stage_frozen,
    "adapt": stage_adapt,
    "evaluate": stage_evaluate,
    "forecast": stage_forecast,
    "reload": stage_reload,
    "bundle": stage_bundle,
    "activity": stage_activity,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the pinned snapshot and the data cache")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--byod", default="", help="data: a BYOD NetCDF window instead of the pinned sample")
    parser.add_argument("--origin", type=int, default=DEFAULT_SETTINGS["origin"], help="data: forecast origin of Sections 6 and 11")
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_SETTINGS["rollout_steps"], help="data: 6-hour steps of the roll-outs")
    parser.add_argument("--max-lead-steps", type=int, default=DEFAULT_SETTINGS["max_lead_steps"], help="data: lead steps scored in Sections 7 and 9")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING["epochs"], help="adapt: training epochs")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAINING["learning_rate"], help="adapt: AdamW learning rate")
    parser.add_argument("--trainable", default=DEFAULT_TRAINING["trainable"], choices=("lora", "lora+heads"), help="adapt/activity: adaptation scope")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        if options.stage not in ("data", "weights"):
            apply_determinism()
        STAGES[options.stage](run)
    except Refusal as exc:  # DAT20: one actionable message, no traceback
        error_file.write_text(json.dumps({"stage": options.stage, "type": "Refusal", "message": str(exc), "refusal": True}), encoding="utf-8")
        print(f"REFUSED ({options.stage}): {exc}", flush=True)
        return 3
    except RuntimeError as exc:
        if "deterministic" not in str(exc):
            raise_into = exc
        else:
            # An op with no deterministic kernel: STOP and name it (never warn_only; docs/aurora-preregistration.md §3).
            message = f"STOP: an operation has no deterministic implementation on this device and was not worked around: {exc}"
            error_file.write_text(json.dumps({"stage": options.stage, "type": "NondeterministicOperation", "message": message}), encoding="utf-8")
            print(message, flush=True)
            return 2
        traceback.print_exc()
        message = str(raise_into) or repr(raise_into)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(raise_into).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(raise_into).__name__}: {message}", flush=True)
        return 2
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
