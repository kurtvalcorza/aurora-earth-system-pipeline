"""Stage runner for the standalone Aurora Earth-system tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

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
workflow-length check of a window against the configured settings, the default-path expectation, the activity).
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
VERSION_CHECKS = ("torch", "microsoft-aurora", "timm", "einops", "xarray", "netcdf4", "numcodecs", "numpy", "safetensors", "huggingface-hub")
# Files each stage writes. A stage first removes its own files and those of every stage that reads them
# (DEPENDANTS), so no record from an earlier configuration can be read as a result of the current one.
STAGE_FILES: dict[str, tuple[str, ...]] = {
    "data": ("data.json", f"{STEM}_input_manifest.json", f"{STEM}_sample_window.nc"),
    "rollout": (f"{STEM}_frozen_rollout.json",),
    "frozen": (f"{STEM}_frozen_test.json",),
    "adapt": ("adapt.json", "reference_forecast.npz", f"{STEM}_training_history.json", ADAPTER_DIR),
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
        path = self.state / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 5, or from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
        return path

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
    """The default configuration, on which the notebook's stated expectations were measured."""
    return data["source"] == "sample" and data["settings"] == DEFAULT_SETTINGS and training == DEFAULT_TRAINING


# --------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------


def load_windows(run: Run, stage: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """The settings record and the role windows the ``data`` stage chose, rebuilt from the cache or the BYOD file."""
    from aurora_earth_system_pipeline import fetch_sample_dataset, load_byod_dataset

    data = run.read_state("data.json", stage)
    if data["source"] == "byod":
        window = load_byod_dataset(data["byod_path"])[0]
        return data, [window], window, window
    windows = fetch_sample_dataset(cache_dir=run.data_cache)
    roles = data["roles"]
    return data, [windows[name] for name in roles["train"]], windows[roles["validation"]], windows[roles["test"]]


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
    pipe = AuroraPipeline.from_artifact(run.out / ADAPTER_DIR, weights_dir=run.snapshot, device=device_name())
    return pipe, adapt


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
    if run.options.byod:
        # Copy the file into the run's state first: the learner may point BYOD_PATH at the sample file this stage
        # rewrites below, and every later stage re-reads the window from the accepted copy.
        original = Path(run.options.byod).resolve()
        if not original.is_file():
            raise FileNotFoundError(f"BYOD file not found: {original}")
        byod_path = run.state / "byod_incoming" / original.name
        byod_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, byod_path)
    settings = {"origin": run.options.origin, "rollout_steps": run.options.rollout_steps, "max_lead_steps": run.options.max_lead_steps}
    if byod_path is not None:
        test_window = load_byod_dataset(byod_path)[0]
        train_windows, val_window = [test_window], test_window
        source, data_source = "byod", f"BYOD ({byod_path.name}) -- one window plays every role, so the held-out numbers are NOT independent"
        dataset_manifest = validate_dataset([test_window])
        roles = {"train": [test_window["name"]], "validation": test_window["name"], "test": test_window["name"]}
    else:
        windows = fetch_sample_dataset(cache_dir=run.data_cache)
        train_windows = [windows["train-2019-01"], windows["train-2019-07"]]
        val_window, test_window = windows["val-2020-04"], windows["test-2021-10"]
        source, data_source = "sample", SAMPLE_LABEL_SOURCE
        dataset_manifest = validate_dataset([*train_windows, val_window, test_window])
        roles = {"train": [w["name"] for w in train_windows], "validation": val_window["name"], "test": test_window["name"]}
    checked_test = validate_inputs(test_window)
    workflow = check_workflow(checked_test["n_steps"], settings, max_rollout=MAX_ROLLOUT_STEPS, history_steps=HISTORY_STEPS, label="test window")
    # Accepted: only now are the earlier configuration's results cleared, so a refused input leaves them intact.
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
    print({"2t_mean_K_by_window": {w["name"]: round(float(w["surf"]["2t"].mean()), 2) for w in [*train_windows, val_window, test_window]}})
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


def stage_adapt(run: Run) -> None:
    """Section 8: bounded fine-tuning of a freshly built frozen pipeline; adapter export; reference forecast."""
    import numpy as np

    run.invalidate("adapt")
    data, train_windows, val_window, test_window = load_windows(run, "adapt")
    training = _training_options(run)
    pipe = frozen_pipeline(run)
    print({"starts_from": "the pretrained model, built from the verified files in this process", "adapted": pipe.adapter is not None, "lora_b_max_abs": lora_b_max_abs(pipe), "device": pipe.device})
    started = time.perf_counter()
    result = pipe.adapt(train_windows, val_window, epochs=training["epochs"], lr=training["learning_rate"], trainable=training["trainable"], progress=_progress)
    seconds = round(time.perf_counter() - started, 1)
    print({"trainable_parameters": result["n_trainable"], "total_parameters": result["n_total"], "training_samples": result["n_train_samples"], "best_epoch": result["best_epoch"], "seconds": seconds, "device": pipe.device})
    print({"loss_scales": result["loss_scales"]})
    artifact = pipe.save_artifact(run.out / ADAPTER_DIR, metadata={"tutorial": STEM, "data_source": data["data_source"], "dataset_digest": data["digest"]})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    print({"artifact": str(artifact), "format": manifest["format"], "tensors": len(manifest["tensors"]), "bytes": manifest["files"][0]["bytes"], "sha256": manifest["files"][0]["sha256"][:16] + "..."})
    reference = pipe.predict(test_window, origin=data["settings"]["origin"], steps=2)["forecasts"]
    arrays = {f"{group}/{lead}/{name}": f[group][name] for lead, f in enumerate(reference) for group in ("surf", "atmos") for name in f[group]}
    np.savez(run.state / "reference_forecast.npz", **arrays)
    run.write_output(f"{STEM}_training_history.json", {"training": training, "history": result["history"], "best_epoch": result["best_epoch"], "seconds": seconds})
    run.write_state("adapt.json", {"training": training, "settings": data["settings"], "dataset_digest": data["digest"], "seconds": seconds, "adaptation": {k: v for k, v in result.items() if k not in ("history", "trainable_names")}})


def stage_evaluate(run: Run) -> None:
    """Section 9: the adapter, rebuilt from files in a new process, scored on the held-out window; three-way table."""
    from aurora_earth_system_pipeline import MODEL_ID, MODEL_KEY, MODEL_REVISION

    run.invalidate("evaluate")
    data, train_windows, val_window, test_window = load_windows(run, "evaluate")
    frozen = run.read_output(f"{STEM}_frozen_test.json", "evaluate")
    if frozen["dataset_digest"] != data["digest"] or frozen["settings"] != data["settings"]:
        raise RuntimeError("the frozen scores were computed for other data or settings; re-run from Section 7")
    pipe, adapt = adapted_pipeline(run, "evaluate")
    lead = data["settings"]["max_lead_steps"]
    adapted_test = pipe.evaluate(test_window, max_lead_steps=lead)
    adapted_val = pipe.evaluate(val_window, max_lead_steps=1)
    frozen_test = frozen["frozen_test"]
    comparison = {
        name: {
            k: {"persistence": round(row["persistence"], 4), "frozen": round(frozen_test["variables"][name][k]["model"], 4), "adapted": round(row["model"], 4), "skill_frozen": round(frozen_test["variables"][name][k]["skill"], 3), "skill_adapted": round(row["skill"], 3)}
            for k, row in adapted_test["variables"][name].items()
        }
        for name in REPORTED
    }
    for name in ("2t", "msl", "z500", "u"):
        print({name: comparison[name]})
    summary = {
        k: {"frozen_mean_skill": round(frozen_test["summary"][k]["mean_skill"], 3), "adapted_mean_skill": round(adapted_test["summary"][k]["mean_skill"], 3), "beating_persistence_frozen": frozen_test["summary"][k]["variables_beating_persistence"], "beating_persistence_adapted": adapted_test["summary"][k]["variables_beating_persistence"]}
        for k in adapted_test["leads"]
    }
    for k, row in summary.items():
        print({k: row})
    first = adapted_test["leads"][0]
    outcome = {
        "lead": first,
        "adapted_mean_skill_below_frozen": summary[first]["adapted_mean_skill"] < summary[first]["frozen_mean_skill"],
        "more_variables_beat_persistence": summary[first]["beating_persistence_adapted"] > summary[first]["beating_persistence_frozen"],
        "default_configuration": is_default(data, adapt["training"]),
    }
    report = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY},
        "data_source": data["data_source"],
        "dataset_digest": data["digest"],
        "settings": data["settings"],
        "training": adapt["training"],
        "roles": data["roles"],
        "persistence_baseline": frozen["persistence_baseline"],
        "frozen_test": frozen_test,
        "validation_metrics": adapted_val,
        "test_metrics": adapted_test,
        "comparison": comparison,
        "summary": summary,
        "outcome": outcome,
        "adaptation": adapt["adaptation"],
        "adaptation_seconds": adapt["seconds"],
        "evidence": "tutorial sample-sanity numbers from one window per role and one seeded run; not a benchmark",
    }
    run.write_output(f"{STEM}_evaluation_report.json", report)
    verdict = "the adapted model beats the frozen model at " + first if outcome["adapted_mean_skill_below_frozen"] else "the adapted model does NOT beat the frozen model at " + first
    print({"outcome": verdict, **outcome, "report": f"outputs/{STEM}_evaluation_report.json"})
    if outcome["default_configuration"] and not (outcome["adapted_mean_skill_below_frozen"] and outcome["more_variables_beat_persistence"]):
        # Default-path verification only: on the default sample and settings the recorded runs improve both numbers.
        raise RuntimeError(f"default configuration, but the expected improvement did not occur: {summary[first]}")


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
        row["note"] = "sanity check, not an evaluation"
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
    parity = {"max_abs_surf_diff": diffs["surf"], "max_abs_atmos_diff": diffs["atmos"], "rtol": PARITY_RTOL, "atol": PARITY_ATOL, "reload_verification": "PASSED" if within else "FAILED", "reloaded_best_epoch": pipe.adapter["best_epoch"], "compared": "2 forecast steps from ORIGIN, every variable, against the trained model in the adapt stage"}
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
    manifest = json.loads((run.out / ADAPTER_DIR / "manifest.json").read_text(encoding="utf-8"))
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
        "artifact": {"dir": ADAPTER_DIR, "sha256": manifest["files"][0]["sha256"], "bytes": manifest["files"][0]["bytes"]},
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
    data, train_windows, val_window, test_window = load_windows(run, "activity")
    report = run.read_output(f"{STEM}_evaluation_report.json", "activity")
    canonical = {p.relative_to(run.out).as_posix(): sha256_file(p) for p in sorted(run.out.rglob("*")) if p.is_file() and "activity" not in p.relative_to(run.out).parts}
    training = {**report["training"], "trainable": run.options.trainable}
    if training["trainable"] == report["training"]["trainable"]:
        raise ValueError(f"the activity changes the adaptation scope: choose a TRAINABLE other than the canonical {report['training']['trainable']!r}")
    pipe = frozen_pipeline(run)
    print({"changed_variable": "adaptation scope (TRAINABLE)", "canonical": report["training"]["trainable"], "activity": training["trainable"], "unchanged": {k: v for k, v in training.items() if k != "trainable"}, "starts_from_lora_b_max_abs": lora_b_max_abs(pipe)})
    result = pipe.adapt(train_windows, val_window, epochs=training["epochs"], lr=training["learning_rate"], trainable=training["trainable"], progress=_progress)
    test = pipe.evaluate(test_window, max_lead_steps=data["settings"]["max_lead_steps"])
    first = test["leads"][0]
    rows = {}
    for name in REPORTED:
        canon = report["test_metrics"]["variables"][name][first]["skill"]
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
        "mean_skill": {"canonical": round(report["test_metrics"]["summary"][first]["mean_skill"], 3), "activity": round(test["summary"][first]["mean_skill"], 3)},
        "surface_better": sum(rows[n]["better_with_activity"] for n in ("2t", "10u", "10v", "msl")),
        "upper_air_better": sum(rows[n]["better_with_activity"] for n in ("t", "u", "v", "q", "z", "z500")),
    }
    out = run.out / "activity"
    out.mkdir(exist_ok=True)
    tag = training["trainable"].replace("+", "_plus_")
    (out / f"{STEM}_activity_{tag}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
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
        STAGES[options.stage](run)
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
