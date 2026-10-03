# ruff: noqa: E501
"""The weather workshop's uv isolated environment (2026-10-03).

Nothing is pip-installed into the notebook kernel and there is no restart guard: a pinned uv builds a managed CPython
3.12.12 environment from a hash-locked, wheel-only lock, and every code cell after the router runs in one persistent
worker in that environment. These tests read the generated notebook and the generator; the worker test starts the real
worker source with this interpreter (POSIX only, as on the notebook's Linux x86_64 runtimes and CI).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb"
LOCK = REPO / "tutorials" / "requirements-weather-workshop.lock.txt"
LOCK_INPUT = REPO / "tutorials" / "requirements-weather-workshop.in"
sys.path.insert(0, str(REPO / "tools"))

import build_weather_forecasting_workshop as builder  # noqa: E402
import weather_workshop_isolated_runtime as isolated  # noqa: E402

# The pins of the previous in-kernel install cell (main 6e62329, blob 385da6db): the environment keeps them.
PREVIOUS_PINS = [
    "torch==2.14.0",
    "microsoft-aurora==2.0.1",
    "timm==1.0.29",
    "einops==0.8.2",
    "xarray==2026.7.0",
    "netCDF4==1.7.4",
    "numcodecs==0.17.0",
    "numpy==2.5.3",
    "safetensors==0.8.0",
    "huggingface-hub==1.32.0",
    "zarr==3.4.0",
    "gcsfs==2026.7.0",
    "matplotlib>=3.9,<3.11",
    "pandas>=2.2,<3.1",
]
# SHA-256 of the fleet's verified carrier strings, copied verbatim from bart-mnli-zero-shot-classification-pipeline
# ee128d2 tools/build_notebook.py (_ISOLATED_INSTALL, _ISOLATED_ROUTER). A change here is a change of mechanism.
CARRIER_SHA256 = {
    "ISOLATED_INSTALL": "bea14b3f188812e97a63fcd0bad416fafae43ea2cb3ce75861f5923913fa9bbb",
    "ISOLATED_ROUTER": "0a3a1a302fad03007f7c638639fb94893d7d32515d85b1183f4a7d88e07d30c2",
}
INSTALL_TITLE = "# @title Infrastructure: install the locked runtime into an isolated environment"
ROUTER_TITLE = "# @title Route the remaining cells to the isolated environment"
IMPORTS_TITLE = "# @title Import the pinned runtime and choose the device"
CONTROLS_TITLE = "# @title Notebook controls"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells():
    return [("".join(c["source"]), c["id"]) for c in notebook()["cells"] if c["cell_type"] == "code"]


def code(title):
    hits = [src for src, _ in code_cells() if src.startswith(title)]
    assert len(hits) == 1, title
    return hits[0]


def code_position(title):
    return next(i for i, (src, _) in enumerate(code_cells()) if src.startswith(title))


def assigned(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned")


def test_no_kernel_pip_install_and_no_restart_guard():
    text = "\n".join(src for src, _ in code_cells())
    for forbidden in ('"-m", "pip"', "'-m', 'pip'", "pip install", "NUMPY_PRELOADED", "invalidate_caches", "Restart session", "if stale:"):
        assert forbidden not in text, forbidden
    markdown = "\n".join("".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "markdown")
    assert "One-pass pinned runtime" not in markdown and "Install one-pass pinned runtime" not in markdown
    assert "no restart is needed" in markdown


def test_the_only_installer_is_the_pinned_uv_with_hashes_and_wheels_only():
    install = code(INSTALL_TITLE)
    assert '"--require-hashes", "--only-binary", ":all:"' in install
    assert '"--index-url", "https://pypi.org/simple"' in install
    assert '"venv", "--quiet", "--managed-python", "--python", MANAGED_PYTHON' in install
    assert assigned(install, "MANAGED_PYTHON") == "3.12.12"
    assert assigned(install, "UV_URL") == isolated.UV["url"]
    assert assigned(install, "UV_BYTES") == isolated.UV["bytes"]
    assert assigned(install, "UV_SHA256") == isolated.UV["sha256"]
    assert 'platform.system() != "Linux" or platform.machine() != "x86_64"' in install
    # The uv wheel is the one the primary Colab tutorial pins.
    template = (REPO / "tools" / "notebook_template.py").read_text(encoding="utf-8")
    assert isolated.UV["url"] in template and isolated.UV["sha256"] in template and str(isolated.UV["bytes"]) in template


def test_lock_is_hashed_wheel_only_and_carried_byte_for_byte():
    lock_text = LOCK.read_text(encoding="utf-8")
    assert "--generate-hashes --only-binary :all:" in lock_text.splitlines()[1]
    assert "--python-platform x86_64-manylinux_2_28" in lock_text.splitlines()[1]
    entries = [e for e in re.split(r"\n(?=[A-Za-z0-9])", lock_text) if "==" in e.split("\n", 1)[0]]
    assert len(entries) == len(isolated.lock_packages(lock_text)) > 100
    assert all("--hash=sha256:" in e for e in entries)
    install = code(INSTALL_TITLE)
    carried = assigned(install, "LOCK_TEXT")
    assert carried == lock_text  # ast.literal_eval of the carried literal equals the repository lock
    assert assigned(install, "LOCK_SHA256") == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert assigned(install, "LOCKED_PACKAGES") == len(isolated.lock_packages(lock_text))
    assert assigned(install, "LOCK_NAME") == LOCK.name


def test_pins_are_the_previous_notebook_pins_and_the_lock_resolves_them():
    assert isolated.read_pins() == PREVIOUS_PINS
    assert assigned(code(INSTALL_TITLE), "PINS") == PREVIOUS_PINS
    resolved = isolated.resolved_pins(PREVIOUS_PINS, LOCK.read_text(encoding="utf-8"))
    for pin in PREVIOUS_PINS:
        name, _, version = pin.partition("==")
        if version:
            assert resolved[name] == version
    assert assigned(code(IMPORTS_TITLE), "LOCKED_VERSIONS") == resolved


def test_lock_check_refuses_missing_pins_broken_ranges_and_unhashed_entries():
    lock_text = LOCK.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="does not satisfy"):
        isolated.resolved_pins(["torch==2.13.0"], lock_text)
    with pytest.raises(SystemExit, match="does not satisfy"):
        isolated.resolved_pins(["pandas>=2.2,<3.0"], lock_text)
    with pytest.raises(SystemExit, match="does not satisfy"):
        isolated.resolved_pins(["not-a-package==1.0"], lock_text)
    stripped = re.sub(r"(zarr==3\.4\.0) \\\n(?:    --hash=sha256:[0-9a-f]+(?: \\)?\n)+", r"\1\n", lock_text)
    assert stripped != lock_text
    with pytest.raises(SystemExit, match="without --hash"):
        isolated.resolved_pins(["zarr==3.4.0"], stripped)


def test_learner_cells_run_in_the_isolated_python_and_only_setup_runs_in_the_kernel():
    kernel = [cid for src, cid in code_cells() if "# dimer: kernel cell" in src]
    assert kernel == ["dimer-weather-workshop-uv-02", "dimer-weather-workshop-uv-04"]
    router = code(ROUTER_TITLE)
    assert "_DIMER_ISOLATED_RUNTIME = IsolatedRuntime(ISOLATED_PYTHON)" in router
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in router
    assert 'ISOLATED_PYTHON = ISOLATED_ENV / "bin" / "python"' in code(INSTALL_TITLE)
    # Hosted-runtime traps: the worker gets Agg and no kernel Python path or Hugging Face token.
    assert 'MPLBACKEND="Agg"' in router
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        assert f'"{name}"' in router
    # Order: build the environment, start the worker, then every learner cell, starting with the controls.
    order = [code_position(t) for t in (INSTALL_TITLE, ROUTER_TITLE, CONTROLS_TITLE, IMPORTS_TITLE, "# @title Immutable model identity and data contract")]
    assert order == sorted(order) and order[:2] == [0, 1]


def test_carrier_is_the_fleet_reference_verbatim():
    for name, digest in CARRIER_SHA256.items():
        assert hashlib.sha256(getattr(isolated, name).encode("utf-8")).hexdigest() == digest, name
    assert code(ROUTER_TITLE) == isolated.ISOLATED_ROUTER


def test_imports_cell_stops_on_a_version_mismatch():
    imports = code(IMPORTS_TITLE)
    assert "if mismatched:" in imports and "raise RuntimeError" in imports
    for module in ("import numpy as np", "import pandas as pd", "import torch", "import matplotlib.pyplot as plt", "import xarray as xr", "import gcsfs", "import zarr"):
        assert module in imports
    assert 'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"' in imports


def test_no_notebook_cell_line_exceeds_2000_characters():
    longest = max(len(line) for cell in notebook()["cells"] for line in "".join(cell["source"]).split("\n"))
    assert longest <= builder.MAX_CELL_LINE == 2000


def test_generator_refuses_an_overlong_line(monkeypatch):
    monkeypatch.setitem(builder.GENERATED, "isolated_router", lambda: "x = " + repr("a" * 2100))
    with pytest.raises(SystemExit, match="over 2000 characters"):
        builder.build_notebook()


def test_existing_cell_ids_are_kept_and_metadata_records_the_revision():
    nb = notebook()
    ids = [c["id"] for c in nb["cells"]]
    assert ids[:3] == ["dimer-weather-workshop-00", "dimer-weather-workshop-01", "dimer-weather-workshop-02"]
    assert ids[3:7] == [f"dimer-weather-workshop-uv-0{i}" for i in range(1, 5)]
    assert ids[7:] == [f"dimer-weather-workshop-{i:02d}" for i in range(3, 59)]
    meta = nb["metadata"]["dimer"]
    env = meta["isolated_environment"]
    assert env["platform"] == "linux-x86_64" and env["managed_python"] == "3.12.12" and env["kernel_installs"] is False
    assert env["lock_sha256"] == hashlib.sha256(LOCK.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    assert meta["revisions"][-1]["date"] == "2026-10-03"
    assert meta["revisions"][-1]["previous_blob"] == "385da6db6424b92fdec83f82a0e68f08abeed02e"


def _runtime_namespace():
    """The router cell's worker source and IsolatedRuntime class, without its IPython registration."""
    tree = ast.parse(code(ROUTER_TITLE))
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and n.targets[0].id == "_WORKER_SOURCE") or (isinstance(n, ast.ClassDef) and n.name in {"IsolatedCellError", "IsolatedRuntime"})]
    assert len(keep) == 3
    ns = {"os": os, "sys": sys, "subprocess": subprocess, "signal": __import__("signal"), "Connection": __import__("multiprocessing.connection", fromlist=["Connection"]).Connection}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "router", "exec"), ns)
    return ns


@pytest.mark.skipif(os.name != "posix", reason="the worker passes pipe descriptors (POSIX only, like the notebook's Linux runtimes)")
def test_worker_keeps_state_streams_output_displays_values_and_raises_errors(capsys):
    ns = _runtime_namespace()
    shown = []
    runtime = ns["IsolatedRuntime"](sys.executable, display=lambda bundle, raw: shown.append(bundle))
    try:
        runtime.run("import os\nPRODUCED = {}\nEPOCHS = 6\nprint('controls', EPOCHS, os.environ['MPLBACKEND'], 'PYTHONPATH' in os.environ)")
        runtime.run("PRODUCED['a'] = EPOCHS * 2\nPRODUCED")
        with pytest.raises(ns["IsolatedCellError"], match="ZeroDivisionError"):
            runtime.run("1 / 0")
        runtime.run("print('still alive', PRODUCED['a'])")
    finally:
        runtime.close()
    out = capsys.readouterr()
    assert "controls 6 Agg False" in out.out and "still alive 12" in out.out
    assert "ZeroDivisionError" in out.err
    assert shown == [{"text/plain": "{'a': 12}"}]
