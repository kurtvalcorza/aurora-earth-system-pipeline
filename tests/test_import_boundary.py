"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; the snapshot is verified and both source pickles are
statically audited before torch or aurora are imported.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aurora_earth_system_pipeline import validate_dataset, validate_inputs
from conftest import MODEL_LIBRARIES, synthetic_window

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "aurora_earth_system_pipeline"


def test_package_import_does_not_import_model_libraries(forbid_model_imports):
    import importlib

    import aurora_earth_system_pipeline

    importlib.reload(aurora_earth_system_pipeline)


def test_no_module_level_model_imports():
    """Every torch / aurora / safetensors / huggingface_hub import is inside a function body."""
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.partition(".")[0] not in MODEL_LIBRARIES, f"{path.name} imports {name} at module level"


def test_invalid_windows_are_rejected_before_model_imports(forbid_model_imports):
    with pytest.raises(ValueError, match="levels must be exactly"):
        validate_inputs({**synthetic_window(), "levels": [1000]})
    with pytest.raises(ValueError, match="at least 2 are required"):
        validate_dataset([synthetic_window()], min_windows=2)
