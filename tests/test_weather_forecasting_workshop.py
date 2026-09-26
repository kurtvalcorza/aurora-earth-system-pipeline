# ruff: noqa: E501,I001
"""Static contract tests for the weather and Earth-system forecasting workshop notebook."""
from __future__ import annotations
import ast
import contextlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb"
PACKAGE = REPO / "src" / "aurora_earth_system_pipeline"


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells():
    return ["".join(cell["source"]) for cell in load()["cells"] if cell["cell_type"] == "code"]


def body():
    return "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])


def constants_from_source(source):
    """Top-level literal assignments, read without importing anything (CI has no torch)."""
    out = {}
    for node in ast.parse(source).body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
        if isinstance(target, ast.Name) and getattr(node, "value", None) is not None:
            with contextlib.suppress(ValueError):
                out[target.id] = ast.literal_eval(node.value)
    return out


def module_constants(path):
    return constants_from_source(path.read_text(encoding="utf-8"))


def notebook_constants():
    out = {}
    for cell in code_cells():
        out.update(constants_from_source(cell))
    return out


def test_generator_parity():
    subprocess.run([sys.executable, str(REPO / "tools" / "build_weather_forecasting_workshop.py"), "--check"], cwd=REPO, check=True)


def test_metadata():
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "E2E"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["worker_required"] is False
    assert meta["credentials_required"] is False
    assert meta["clean_runtime_evidence"] == "pending"


def test_code_cells_compile():
    for cell in code_cells():
        compile(cell, "cell", "exec")


def test_model_and_state_contract_match_the_pipeline():
    pipeline = module_constants(PACKAGE / "pipeline.py")
    notebook = notebook_constants()
    assert notebook["MODEL_REVISION"] == pipeline["MODEL_REVISION"]
    assert notebook["CHECKPOINT_SHA256"] == pipeline["SOURCE_CKPT_SHA256"]
    assert notebook["PARAMETER_COUNT"] == pipeline["PARAMETER_COUNT"]
    assert notebook["LORA_PARAMETERS_EXPECTED"] == pipeline["LORA_PARAMETERS"]
    assert notebook["LEVELS"] == pipeline["LEVELS"]
    assert notebook["LOSS_SCALES"] == pipeline["LOSS_SCALES"]
    for key, bounds in pipeline["RANGES"].items():
        assert notebook["RANGES"][key] == bounds


def test_sample_windows_match_the_pinned_corpus():
    samples = module_constants(PACKAGE / "samples.py")
    notebook = notebook_constants()
    assert notebook["SAMPLE_WINDOWS"] == samples["SAMPLE_WINDOWS"]
    assert samples["SAMPLE_WINDOWS"] and "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" in body()


def test_contract_present():
    text = body()
    for literal in [
        "USE_BYOD = False",
        'TRAINABLE = "lora"',
        "EPOCHS = 6",
        "LEARNING_RATE = 0.001",
        "frozen_experiment.json",
        "adapter.safetensors",
        "org.valcorza.aurora-earth-system.adapter.v1",
        "fresh reload parity failed",
        "not-measurable",
        "workshop_summary.json",
        "experiment_manifest.json",
    ]:
        assert literal in text


def test_no_runtime_repo_dependency():
    text = body()
    for forbidden in ["git clone ", "pip install -e", "raw.githubusercontent.com/kurtvalcorza", "import aurora_earth_system_pipeline", "dimer-backend"]:
        assert forbidden not in text


def test_clean_notebook():
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
