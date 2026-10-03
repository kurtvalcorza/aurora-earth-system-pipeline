"""Regression tests for the Notebook Review Framework v1 findings on tutorials/aurora_earth_system_colab.ipynb
(AUR-M1..M4, AUR-m1..m6). They run within CI's install budget (numpy, numcodecs, xarray, netCDF4; no torch, no
aurora): the stage runner's data, validation and export logic is exercised in-process on synthetic windows, the kernel's
own helper cells are executed from the generated notebook, and the remaining properties are checked on the generated
source. Model-running stages are covered by the local pre-flight recorded in docs/release-verification.md, not here.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from aurora_earth_system_pipeline import DEFAULT_WEIGHTS_DIR, AuroraPipeline, write_window_netcdf
from conftest import synthetic_window

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_review_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stages = _load("tutorial_stages")
build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _markdown() -> str:
    return "\n".join(_source(c) for c in _notebook()["cells"] if c["cell_type"] == "markdown")


def _kernel_code() -> list[str]:
    return [_source(c) for c in _notebook()["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]


def _run(tmp_path: Path, **options) -> stages.Run:
    defaults = {"byod": "", "origin": 1, "rollout_steps": 4, "max_lead_steps": 4, "epochs": 6, "learning_rate": 1e-3, "trainable": "lora"}
    defaults.update(options)
    return stages.Run(tmp_path / "run", tmp_path / "weights", argparse.Namespace(**defaults))


def _nc(tmp_path: Path, steps: int, name: str = "byod_window.nc") -> Path:
    return write_window_netcdf(synthetic_window(height=17, width=32, steps=steps, name=name[:-3]), tmp_path / name)


# --- AUR-M1: one-pass Run all, no kernel install ------------------------------------------------------------


def test_m1_kernel_installs_nothing_and_never_asks_for_a_restart():
    code = "\n".join(_kernel_code())
    assert "sys.executable" not in code and "'-m', 'pip'" not in code and "pip install" not in code
    assert "Restart the runtime" not in code and "packages_distributions" not in code
    assert "'--require-hashes', '--only-binary', ':all:'" in code
    assert build.GENERATOR_VERSION == "build_notebook.py/3.0" and build.NOTEBOOK_SPEC == "2.2"
    md = _markdown().lower()
    assert "restart" not in md.replace("no runtime restart", "").replace("no restart", "")


def test_m1_release_record_is_candidate_and_names_the_two_passes():
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    row = next(line for line in registry.splitlines() if line.startswith("| `aurora_earth_system_colab.ipynb`"))
    assert "**Candidate**" in row and "Release-grade" not in row
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "an interpreter restart after the install is expected" not in record
    assert record.count("**2 passes — not a one-pass `Run all`**") == 2
    assert "| **PASSED** — 11/11 code cells ok (1 restart after install cell)" not in record


# --- AUR-M2: every model stage starts from the frozen model ----------------------------------------------------


def test_m2_adapt_refuses_an_already_adapted_pipeline(forbid_model_imports):
    pipe = AuroraPipeline(model=None, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR, source="test", use_lora=True, adapter={"best_epoch": 6})
    with pytest.raises(ValueError, match="already adapted"):
        pipe.adapt([synthetic_window()])


def test_m2_model_stages_build_fresh_pipelines_from_files():
    text = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    assert text.count("AuroraPipeline.from_pretrained(") == 1 and text.count("AuroraPipeline.from_artifact(") == 1
    for stage in ("stage_rollout", "stage_frozen", "stage_adapt", "stage_activity"):
        body = text[text.index(f"def {stage}("):].split("\ndef ", 1)[0]
        assert "pipe = frozen_pipeline(run)" in body, stage
    for stage in ("stage_evaluate", "stage_forecast", "stage_reload"):
        body = text[text.index(f"def {stage}("):].split("\ndef ", 1)[0]
        assert "adapted_pipeline(run, " in body, stage
    frozen = text[text.index("def frozen_pipeline("):].split("\ndef ", 1)[0]
    assert "pipe.adapter is not None or lora_b_max_abs(pipe) != 0.0" in frozen


def test_m2_rerunning_section_5_clears_every_later_result(tmp_path, forbid_model_imports):
    nc = _nc(tmp_path, 6)
    run = _run(tmp_path, byod=str(nc))
    stages.stage_data(run)
    first = run.read_state("data.json", "test")
    planted = [run.state / "adapt.json", run.state / "reference_forecast.npz", run.out / "aurora_earth_system_frozen_test.json", run.out / "aurora_earth_system_evaluation_report.json", run.out / "aurora_earth_system_result.json"]
    for path in planted:
        path.write_text("{}", encoding="utf-8")
    (run.out / "aurora_earth_system_adapter").mkdir()
    (run.out / "activity").mkdir()
    stages.stage_data(run)
    assert not any(p.exists() for p in planted) and not (run.out / "aurora_earth_system_adapter").exists()
    assert not (run.out / "activity").exists(), "activity results of an earlier configuration must be cleared too"
    assert run.read_state("data.json", "test")["digest"] == first["digest"]


def test_m2_a_stale_adapter_is_refused_before_any_model_work(tmp_path, forbid_model_imports):
    run = _run(tmp_path, byod=str(_nc(tmp_path, 6)))
    stages.stage_data(run)
    data = run.read_state("data.json", "test")
    run.write_state("adapt.json", {"dataset_digest": "0" * 64, "settings": data["settings"]})
    with pytest.raises(RuntimeError, match="re-run from Section 8"):
        stages.adapted_pipeline(run, "evaluate")


# --- AUR-M3: the documented minimum is the enforced minimum ----------------------------------------------------


@pytest.mark.parametrize(("steps", "accepted"), [(3, False), (4, False), (5, False), (6, True), (8, True)])
def test_m3_workflow_length_at_the_defaults(steps, accepted):
    settings = dict(stages.DEFAULT_SETTINGS)
    if accepted:
        workflow = stages.check_workflow(steps, settings)
        assert workflow["required_steps"] == 6 and workflow["new_origin"] >= 1
    else:
        with pytest.raises(ValueError, match=rf"has {steps} 6-hourly steps; .* need at least 6 .*lower ROLLOUT_STEPS / MAX_LEAD_STEPS"):
            stages.check_workflow(steps, settings)


def test_m3_settings_out_of_range_are_refused_with_the_field_name():
    with pytest.raises(ValueError, match="MAX_LEAD_STEPS = 9"):
        stages.check_workflow(64, {"origin": 1, "rollout_steps": 4, "max_lead_steps": 9})
    with pytest.raises(ValueError, match="ORIGIN = 7"):
        stages.check_workflow(6, {"origin": 7, "rollout_steps": 4, "max_lead_steps": 4})
    assert stages.required_steps(2, 1) == 4 and stages.check_workflow(4, {"origin": 1, "rollout_steps": 2, "max_lead_steps": 1})["new_origin"] == 1


def test_m3_a_short_byod_window_is_refused_in_section_5(tmp_path, forbid_model_imports):
    run = _run(tmp_path, byod=str(_nc(tmp_path, 5)))
    with pytest.raises(ValueError, match="has 5 6-hourly steps; ROLLOUT_STEPS = 4 and MAX_LEAD_STEPS = 4 need at least 6"):
        stages.stage_data(run)
    assert not (run.state / "data.json").exists(), "a refused window must not leave a usable settings record"
    # After an accepted run, a refused window leaves that run's records (and its sample file) untouched.
    good = _run(tmp_path, byod=str(_nc(tmp_path, 6, "good.nc")))
    stages.stage_data(good)
    before = good.read_state("data.json", "test")
    with pytest.raises(ValueError, match="need at least 6"):
        stages.stage_data(_run(tmp_path, byod=str(_nc(tmp_path, 5, "short.nc"))))
    assert good.read_state("data.json", "test") == before
    assert (good.out / "aurora_earth_system_sample_window.nc").is_file()


def test_m3_the_sample_file_section_5_writes_is_a_valid_byod_input(tmp_path, forbid_model_imports, capsys):
    run = _run(tmp_path, byod=str(_nc(tmp_path, 8)))
    stages.stage_data(run)
    sample = run.out / "aurora_earth_system_sample_window.nc"
    import xarray as xr

    with xr.open_dataset(sample) as ds:
        assert ds.sizes["time"] == 6
    # BYOD_PATH may point at the very file the stage rewrites: it is copied into the run state first.
    second = _run(tmp_path, byod=str(sample))
    stages.stage_data(second)
    record = second.read_state("data.json", "test")
    assert record["source"] == "byod" and record["test_steps"] == 6 and Path(record["byod_path"]).is_file()
    out = capsys.readouterr().out
    assert "'5 steps for these settings', 'rejected'" in out and "'verdict': 'accepted'" in out


def test_m3_documented_minimum_matches_the_enforced_one():
    assert stages.required_steps(stages.DEFAULT_SETTINGS["rollout_steps"], stages.DEFAULT_SETTINGS["max_lead_steps"]) == 6
    md = _markdown()
    assert "**6 at the defaults**" in md and "at least three 6-hourly steps" not in md and "3..64 analyses" not in md


# --- AUR-M4: guided layer ---------------------------------------------------------------------------------------


def test_m4_guided_layer_is_present():
    md = _markdown()
    for marker in ("## How to use this notebook", "## Roadmap", "<summary><strong>Glossary</strong>", "## Troubleshooting", "## Conclude with evidence", "**Predict → Change one thing → Run → Observe → Explain.**"):
        assert marker in md, marker
    assert md.count("**Predict before running:**") >= 5 and md.count("<summary>Check your reasoning") >= 6
    assert re.search(r"after this notebook you should be able to \(1\) explain", md)
    activity = next(s for s in _kernel_code() if "RUN_ACTIVITY = False" in s)
    assert "if RUN_ACTIVITY:" in activity and "run_stage('activity', '--trainable', ACTIVITY_TRAINABLE)" in activity


def test_m4_activity_refuses_the_canonical_scope(tmp_path, forbid_model_imports):
    run = _run(tmp_path, byod=str(_nc(tmp_path, 6)), trainable="lora")
    stages.stage_data(run)
    run.write_output("aurora_earth_system_evaluation_report.json", {"training": dict(stages.DEFAULT_TRAINING)})
    with pytest.raises(ValueError, match="choose a TRAINABLE other than the canonical 'lora'"):
        stages.stage_activity(run)


# --- AUR-m1 .. AUR-m6 -------------------------------------------------------------------------------------------


def test_m1_pretraining_overlap_is_addressed_with_a_source():
    md = _markdown()
    assert "**Pretraining overlap.**" in md and "**cannot be ruled out**" in md
    assert "https://microsoft.github.io/aurora/models.html" in md


def test_m2_outcome_is_printed_and_only_the_default_configuration_must_improve():
    kernel = "\n".join(_kernel_code())
    assert "assert " not in kernel, "learner cells must not assert empirical outcomes"
    data = {"source": "sample", "settings": dict(stages.DEFAULT_SETTINGS)}
    assert stages.is_default(data, dict(stages.DEFAULT_TRAINING))
    assert not stages.is_default({**data, "source": "byod"}, dict(stages.DEFAULT_TRAINING))
    assert not stages.is_default(data, {**stages.DEFAULT_TRAINING, "trainable": "lora+heads"})
    md = _markdown()
    assert "Watch the validation loss fall by two thirds" not in md and "a negative result is a legitimate finding" in md


def test_m3_timings_name_their_environment():
    md = _markdown()
    assert "six epochs take about four minutes on CPU" not in md and "Google Colab or Jupyter, Python 3.12)" not in md
    assert "283.5 s" in md and "Windows CPU workstation (a local pre-flight of this version, not a supported runtime)" in md
    assert "an estimate" in md and "CPython 3.12.12 inside the isolated environment" in md


def _kernel_helpers(tmp_path: Path) -> dict:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    carrier = next(s for s in code if s.startswith("# @title Infrastructure: write and verify the carried"))
    install = next(s for s in code if s.startswith("# @title Infrastructure: install the locked runtime"))
    run_root = tmp_path / "run"
    run_root.mkdir()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH", None)
    namespace = {"ROOT": run_root, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": env, "Path": Path}
    exec("import hashlib\nimport json\nimport subprocess\n" + carrier, namespace)  # noqa: S102 - the notebook's own cell
    exec(install[install.index("def run_stage(") :], namespace)  # noqa: S102
    return namespace


def test_m4_byod_path_works_without_google_colab(tmp_path, monkeypatch):
    ns = _kernel_helpers(tmp_path)
    nc = _nc(tmp_path, 6)
    assert ns["obtain_upload"](str(nc), ".nc", "BYOD_PATH") == nc.resolve()
    with pytest.raises(FileNotFoundError, match="BYOD_PATH"):
        ns["obtain_upload"](str(tmp_path / "missing.nc"), ".nc", "BYOD_PATH")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(RuntimeError, match="set BYOD_PATH to its path"):
        ns["obtain_upload"]("", ".nc", "BYOD_PATH")
    code = next(s for s in _kernel_code() if "BYOD_PATH = ''" in s)
    assert "from google.colab" not in code


def test_m3_and_m4_short_byod_fails_in_the_kernel_with_the_stage_message(tmp_path):
    ns = _kernel_helpers(tmp_path)
    nc = _nc(tmp_path, 5)
    with pytest.raises(RuntimeError, match=r"Stage 'data' failed \(exit 2\): ValueError: test window has 5 6-hourly steps"):
        ns["run_stage"]("data", "--byod", nc)


def test_m5_infrastructure_cells_are_titled_and_collapsed_and_warnings_are_not_hidden():
    for cell in _notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = _source(cell)
        infrastructure = source.startswith("# @title Infrastructure: ")
        assert infrastructure == (cell["metadata"].get("cellView") == "form"), source[:60]
    pipeline = (ROOT / "src" / "aurora_earth_system_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    body = pipeline[pipeline.index("def build_model("):].split("\ndef ", 1)[0]
    assert "simplefilter" not in body and "catch_warnings" not in body


def test_m6_forecast_export_holds_every_variable_lead_and_coordinate(tmp_path):
    rng = np.random.default_rng(0)
    lat, lon = np.linspace(90, -87, 16).tolist(), np.linspace(0, 337.5, 16).tolist()
    forecasts = [
        {
            "lead_hours": 6 * k,
            "valid_time": f"2021-10-01T{6 * (k - 1):02d}:00:00",
            "surf": {n: rng.random((16, 16)).astype(np.float32) for n in ("2t", "10u", "10v", "msl")},
            "atmos": {n: rng.random((13, 16, 16)).astype(np.float32) for n in ("t", "u", "v", "q", "z")},
        }
        for k in (1, 2, 3)
    ]
    units = {"2t": "K", "10u": "m/s", "10v": "m/s", "msl": "Pa", "t": "K", "u": "m/s", "v": "m/s", "q": "kg/kg", "z": "m²/s²"}
    result = {"forecasts": forecasts, "lat": lat, "lon": lon, "units": units, "origin_time": "2021-09-30T18:00:00", "model": {"id": "microsoft/aurora", "revision": "r", "adapted": True}}
    from aurora_earth_system_pipeline import LEVELS

    dataset = stages.forecast_dataset(result, LEVELS, "test")
    path = tmp_path / "forecast.nc"
    dataset.to_netcdf(path)
    import xarray as xr

    with xr.open_dataset(path) as ds:
        assert sorted(ds.data_vars) == sorted(units)
        assert dict(ds.sizes) == {"lead_hours": 3, "latitude": 16, "longitude": 16, "level": 13}
        assert ds["t"].dims == ("lead_hours", "level", "latitude", "longitude") and ds["2t"].attrs["units"] == "K"
        assert list(ds["level"].values) == list(LEVELS) and "valid_time" in ds.coords
        np.testing.assert_array_equal(ds["z"].values[2], forecasts[2]["atmos"]["z"])
