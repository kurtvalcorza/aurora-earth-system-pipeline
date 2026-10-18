"""Score-once, grid-selection, reading-rule and determinism tests for the Aurora tutorial (review row 2: R2-M1, R2-M2,
R2-m1, R2-m4, R2-m8, R2-m9; docs/aurora-preregistration.md; SCORE_ONCE_RULE items 1-8). No torch, no aurora: model
stages run against a stub pipeline, and the fresh-test fetch is a stub that records whether it was called.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import aurora_earth_system_pipeline as pkg
from aurora_earth_system_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("_score_once_stages", ROOT / "tools" / "tutorial_stages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stages = _load()
T4 = {"device": "cuda", "gpu": "Tesla T4", "determinism": dict(stages.DETERMINISM)}


class _StubPipe:
    """evaluate() returns a first-lead mean ratio taken from ``ratios`` by window name (default 1.0)."""

    def __init__(self, ratios: dict[str, float] | None = None, default: float = 1.0) -> None:
        self.ratios, self.default, self.adapter = ratios or {}, default, None

    def evaluate(self, window, max_lead_steps=1):
        x = self.ratios.get(window["name"], self.default)
        return {"leads": ["6h"], "summary": {"6h": {"mean_skill": x, "variables_beating_persistence": 3}}, "variables": {n: {"6h": {"model": x, "persistence": 1.0, "skill": x}} for n in stages.REPORTED}}


def _grid_rows(metrics: dict[str, float]) -> list[dict]:
    return [{"config": c, "learning_rate": stages.GRID[c]["learning_rate"], "epochs_budget": stages.GRID[c]["epochs"], "patience": stages.GRID[c]["patience"], "epochs_run": 6, "best_epoch": 6, "val_losses": [0.1], "val_metric": round(m, 4), "val_metric_exact": m} for c, m in metrics.items()]


def _adapt_record(**over) -> dict:
    metrics = {"C0": 0.80, "C1": 0.85, "C2": 0.79}
    rows = _grid_rows(metrics)
    sel = stages.select_config(metrics, 1.5)
    record = {
        "training": dict(stages.DEFAULT_TRAINING), "mode": "grid", "seed": 0, "settings": dict(stages.DEFAULT_SETTINGS), "dataset_digest": "d" * 64,
        "runtime": T4, "grid": {"ran": True, "complete": True, "table_hash": stages.grid_table_hash(), "rows": rows, "frozen_metric": 1.5, "frozen_metric_exact": 1.5, "selection": sel, "grid_hash": stages.grid_hash(rows, 1.5, sel["selected"])},
        "selected_system": sel["selected"], "artifact_sha256": "a" * 64, "frozen_validation": {"mean": 1.5, "per_window": {}}, "seconds": 1.0, "adaptation": {},
    }
    record.update(over)
    return record


def _data_record(**over) -> dict:
    record = {"source": "sample", "data_source": "sample", "settings": dict(stages.DEFAULT_SETTINGS), "digest": "d" * 64, "roles": {"train": list(stages.TRAIN_WINDOWS), "validation": list(stages.VALIDATION_POOL), "test": stages.EXPOSED_TEST, "fresh_test": list(sm.FRESH_TEST_WINDOWS)}}
    record.update(over)
    return record


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A run whose Section 9 uses stubs; returns (run, calls) where calls['fetch'] counts fresh-test fetches."""
    run = stages.Run(tmp_path / "run", tmp_path / "weights", argparse.Namespace(byod="", origin=1, rollout_steps=4, max_lead_steps=4, epochs=6, learning_rate=1e-3, trainable="lora"))
    calls = {"fetch": 0, "runtime": dict(T4), "fail_fetch": False}
    exposed = {"name": stages.EXPOSED_TEST}

    def fetch(**_kw):
        calls["fetch"] += 1
        if calls["fail_fetch"]:
            raise ValueError("pinned digest mismatch")
        return {name: {"name": name} for name in sm.FRESH_TEST_WINDOWS}

    monkeypatch.setattr(stages, "load_windows", lambda run, stage: (run.read_state("data.json", stage), [], [], exposed))
    monkeypatch.setattr(stages, "adapted_pipeline", lambda run, stage: (_StubPipe(default=0.8), run.read_state("adapt.json", stage)))
    monkeypatch.setattr(stages, "frozen_pipeline", lambda run: _StubPipe(default=1.2))
    monkeypatch.setattr(stages, "current_runtime", lambda: calls["runtime"])
    monkeypatch.setattr(stages, "artifact_sha256", lambda run: "a" * 64)
    monkeypatch.setattr(stages, "fresh_preflight", lambda run: {"windows": 12, "downloaded_fresh_chunks": 0})
    monkeypatch.setattr(stages, "seed_all", lambda seed=0: None)
    monkeypatch.setattr(pkg, "fetch_fresh_test", fetch)
    monkeypatch.delenv(stages.MASK_ENV, raising=False)
    stub = _StubPipe(default=1.2).evaluate(exposed)
    run.write_output(f"{stages.STEM}_frozen_test.json", {"dataset_digest": "d" * 64, "settings": dict(stages.DEFAULT_SETTINGS), "persistence_baseline": {}, "frozen_test": stub})
    run.write_state("data.json", _data_record())
    run.write_state("adapt.json", _adapt_record())
    return run, calls


def _report(run) -> dict:
    return json.loads((run.out / f"{stages.STEM}_evaluation_report.json").read_text(encoding="utf-8"))


# --- R2-m1 / R2-D2: selection rule ----------------------------------------------------------------------------


def test_selection_keeps_c0_unless_beaten_by_the_margin_and_keeps_frozen_unless_beaten():
    assert stages.select_config({"C0": 0.80, "C1": 0.781, "C2": 0.79}, 1.5)["selected"] == "C0"
    assert stages.select_config({"C0": 0.80, "C1": 0.78, "C2": 0.79}, 1.5)["selected"] == "C1"
    assert stages.select_config({"C0": 0.80, "C1": 0.70, "C2": 0.70}, 1.5)["selected"] == "C1"  # exact tie -> lower index
    assert stages.select_config({"C0": 0.80, "C1": 0.90, "C2": 0.90}, 0.81)["selected"] == "frozen"
    assert stages.select_config({"C0": 0.80, "C1": 0.90, "C2": 0.90}, 0.82)["selected"] == "C0"


def test_grid_is_the_preregistered_three_configs_and_the_hash_tracks_printed_numbers():
    assert stages.GRID == {"C0": {"learning_rate": 1e-3, "epochs": 6, "patience": None}, "C1": {"learning_rate": 3e-4, "epochs": 6, "patience": None}, "C2": {"learning_rate": 1e-3, "epochs": 30, "patience": 3}}
    rows = _grid_rows({"C0": 0.8, "C1": 0.85, "C2": 0.79})
    h = stages.grid_hash(rows, 1.5, "C0")
    assert h == stages.grid_hash(_grid_rows({"C0": 0.80001, "C1": 0.85, "C2": 0.79}), 1.5, "C0")  # below printed precision
    assert h != stages.grid_hash(_grid_rows({"C0": 0.8001, "C1": 0.85, "C2": 0.79}), 1.5, "C0")
    assert stages.VALIDATION_POOL == ("val-2020-04", "val-2021-01", "val-2021-04", "val-2021-07")


# --- R2-D1: fresh pool and reading rule -----------------------------------------------------------------------


def test_fresh_pool_is_twelve_pinned_2022_windows_containing_the_15th():
    assert sm.fresh_pool_digest() == sm.FRESH_POOL_DIGEST
    assert len(sm.FRESH_TEST_WINDOWS) == 12
    for name, chunk in sm.FRESH_TEST_WINDOWS.items():
        year, month = int(name[6:10]), int(name[11:13])
        start = sm.WB2_EPOCH + timedelta(hours=48 * chunk)
        assert year == 2022 and start <= datetime(year, month, 15) < start + timedelta(hours=48)
        assert all(k in sm.WB2_OBJECTS for k in sm._object_keys(chunk))
    assert not set(sm.FRESH_TEST_WINDOWS.values()) & (set(sm.SAMPLE_WINDOWS.values()) | set(sm.VALIDATION_WINDOWS.values()))
    assert "time/3" in sm.WB2_OBJECTS and "time/.zarray" in sm.WB2_OBJECTS  # R2-m9
    assert sorted(pkg.fetch_sample_dataset.__code__.co_names).count("FRESH_TEST_WINDOWS") == 0


def test_reading_rule_verdicts():
    improved = stages.reading_rule([-0.3, -0.25, -0.35, -0.2] * 3, [0.8] * 12)
    assert improved["verdict"] == "improved" and improved["ci95_t"][1] < 0 and improved["beats_persistence"] is True
    assert stages.reading_rule([0.3, 0.25, 0.35, 0.2] * 3, [1.2] * 12)["verdict"] == "worse"
    assert stages.reading_rule([0.3, -0.3, 0.1, -0.1] * 3, [0.9, 0.8, 0.85, 0.95] * 3)["verdict"] == "no detectable difference"
    assert stages.reading_rule([-0.01] * 11 + [-0.011], [0.9] * 12)["verdict"] == "no detectable difference"  # below the 0.02 minimum effect


# --- R2-M2: score-once, fail-closed -----------------------------------------------------------------------------


def test_preregistered_run_scores_once_then_a_rerun_is_a_reproduction(harness):
    run, calls = harness
    stages.stage_evaluate(run)
    first = _report(run)
    assert first["preregistered"] is True and first["test_look"] == stages.LOOK_PREREGISTERED and calls["fetch"] == 1
    assert (run.out / stages.FRESH_MAIN).is_file() and first["fresh_test"]["reading_rule"]["n_windows"] == 12
    stages.stage_evaluate(run)
    again = _report(run)
    assert again["test_look"] == "reproduction" and again["fresh_test_file"] == f"{stages.STEM}_fresh_test_reproduction_1.json"
    assert json.loads((run.out / stages.FRESH_MAIN).read_text(encoding="utf-8"))["test_look"] == stages.LOOK_PREREGISTERED
    assert len(json.loads((run.state / stages.FRESH_LEDGER).read_text(encoding="utf-8"))) == 2


@pytest.mark.parametrize(
    "data_over, adapt_over, runtime_over, expected",
    [
        ({}, {"training": {"epochs": 8, "learning_rate": 1e-3, "trainable": "lora"}}, {}, "form fields"),
        ({}, {"training": {"epochs": 6, "learning_rate": 3e-4, "trainable": "lora"}}, {}, "form fields"),
        ({}, {"seed": 1}, {}, "seed"),
        ({"source": "byod"}, {}, {}, "data source"),
        ({}, {}, {"device": "cpu", "gpu": None}, "device"),
        ({}, {}, {"gpu": "NVIDIA L4"}, "device"),
        ({}, {}, {"determinism": {**stages.DETERMINISM, "warn_only": True}}, "determinism"),
    ],
)
def test_a_changed_setting_never_fetches_or_scores_the_fresh_test(harness, data_over, adapt_over, runtime_over, expected):
    run, calls = harness
    run.write_state("data.json", _data_record(**data_over))
    run.write_state("adapt.json", _adapt_record(**adapt_over))
    calls["runtime"] = {**T4, **runtime_over}
    stages.stage_evaluate(run)
    report = _report(run)
    assert calls["fetch"] == 0 and report["preregistered"] is False and report["test_look"] == stages.LOOK_EXPLORATORY
    assert any(expected in item for item in report["settings_differing"])
    assert report["fresh_test_file"] == f"{stages.STEM}_fresh_test_exploratory_1.json" and not (run.out / stages.FRESH_MAIN).exists()


def test_a_partial_grid_or_a_wrong_selection_is_not_preregistered(harness):
    run, calls = harness
    record = _adapt_record()
    record["grid"] = {**record["grid"], "complete": False, "rows": record["grid"]["rows"][:2]}
    run.write_state("adapt.json", record)
    stages.stage_evaluate(run)
    assert calls["fetch"] == 0 and "grid incomplete" in _report(run)["settings_differing"]
    run.write_state("adapt.json", _adapt_record(selected_system="C1"))
    stages.stage_evaluate(run)
    assert calls["fetch"] == 0 and any("selected system" in d for d in _report(run)["settings_differing"])


def test_masked_run_never_fetches_the_fresh_windows(harness, monkeypatch):
    run, calls = harness
    monkeypatch.setenv(stages.MASK_ENV, "1")
    stages.stage_evaluate(run)
    assert calls["fetch"] == 0 and _report(run)["test_look"] == stages.LOOK_MASKED
    monkeypatch.delenv(stages.MASK_ENV)
    stages.stage_evaluate(run)
    assert _report(run)["test_look"] == stages.LOOK_PREREGISTERED  # a masked run is not a look


def test_a_failed_fetch_is_not_a_look(harness):
    run, calls = harness
    calls["fail_fetch"] = True
    with pytest.raises(RuntimeError, match="no look was recorded"):
        stages.stage_evaluate(run)
    assert not (run.out / stages.FRESH_MAIN).exists() and not (run.state / stages.FRESH_LEDGER).exists()
    calls["fail_fetch"] = False
    stages.stage_evaluate(run)
    assert _report(run)["test_look"] == stages.LOOK_PREREGISTERED


def test_an_interrupted_report_write_is_not_a_look(harness, monkeypatch):
    run, _calls = harness
    real = os.replace

    def interrupted(src, dst):
        if Path(dst).name == stages.FRESH_MAIN:
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(stages.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        stages.stage_evaluate(run)
    assert not (run.out / stages.FRESH_MAIN).exists() and not (run.state / stages.FRESH_LEDGER).exists()
    assert not list(run.out.glob(".*tmp*"))
    monkeypatch.setattr(stages.os, "replace", real)
    stages.stage_evaluate(run)
    assert _report(run)["test_look"] == stages.LOOK_PREREGISTERED


@pytest.mark.parametrize("ledger_text", ["", "{not json", "[1, 2]"])
def test_a_corrupt_ledger_neither_crashes_nor_relabels(harness, ledger_text):
    run, _calls = harness
    (run.state / stages.FRESH_LEDGER).write_text(ledger_text, encoding="utf-8")
    stages.stage_evaluate(run)
    assert _report(run)["test_look"] == stages.LOOK_PREREGISTERED  # no completed report existed: still the first look
    (run.state / stages.FRESH_LEDGER).write_text("{not json", encoding="utf-8")
    stages.stage_evaluate(run)
    assert _report(run)["test_look"] == "reproduction"  # the completed main report still counts


def test_once_a_record_exists_a_new_runtime_writes_only_reproduction_files(harness, monkeypatch):
    run, _calls = harness
    monkeypatch.setattr(stages, "RECORDED_EVALUATION", {"run": "hosted T4", "revision": "0" * 7})
    stages.stage_evaluate(run)
    assert _report(run)["test_look"] == "reproduction" and not (run.out / stages.FRESH_MAIN).exists()


def test_section_9_never_raises_on_a_test_outcome(harness, monkeypatch):
    run, _calls = harness
    monkeypatch.setattr(stages, "adapted_pipeline", lambda run, stage: (_StubPipe(default=2.0), run.read_state("adapt.json", stage)))
    stages.stage_evaluate(run)  # worse than frozen everywhere: reported, not raised
    assert _report(run)["fresh_test"]["reading_rule"]["verdict"] == "worse"
    assert "default-path regression check" not in (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")


# --- determinism, atomic writes, DAT20 ----------------------------------------------------------------------


def test_determinism_settings_are_preregistered_and_never_warn_only():
    assert stages.DETERMINISM["warn_only"] is False and stages.DETERMINISM["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    src = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    body = src[src.index("def apply_determinism"):src.index("def determinism_state")]
    assert body.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]') < body.index("import torch")
    assert "use_deterministic_algorithms(True)" in body and "warn_only=True" not in src
    assert "torch.backends.cudnn.benchmark = False" in body and "torch.backends.cudnn.deterministic = True" in body
    main = src[src.index("def main("):]
    assert main.index("apply_determinism()") < main.index("STAGES[options.stage](run)")


def test_state_and_output_writes_are_atomic(tmp_path):
    src = (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "os.replace(tmp, path)" in src and ".write_text(json.dumps(value" not in src
    path = stages.atomic_write_text(tmp_path / "x.json", "{}")
    assert path.read_text(encoding="utf-8") == "{}" and [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_a_refused_byod_file_clears_earlier_results_and_raises_a_refusal(tmp_path):
    run = stages.Run(tmp_path / "run", tmp_path / "weights", argparse.Namespace(byod=str(tmp_path / "missing.nc"), origin=1, rollout_steps=4, max_lead_steps=4, epochs=6, learning_rate=1e-3, trainable="lora"))
    run.write_output(f"{stages.STEM}_evaluation_report.json", {"stale": True})
    with pytest.raises(stages.Refusal, match="BYOD file refused"):
        stages.stage_data(run)
    assert not (run.out / f"{stages.STEM}_evaluation_report.json").exists()


def test_changed_section_5_settings_are_not_preregistered():
    data = _data_record(settings={"origin": 2, "rollout_steps": 4, "max_lead_steps": 4})
    assert any("Section 5" in d for d in stages.settings_differing(data, _adapt_record(), T4, "a" * 64))
    assert stages.settings_differing(_data_record(), _adapt_record(), T4, "a" * 64) == []
    assert any("model state" in d for d in stages.settings_differing(_data_record(), _adapt_record(), T4, "b" * 64))
