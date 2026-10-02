"""Offline tests for the public validation-stage helpers and the package surface."""

from __future__ import annotations

import aurora_earth_system_pipeline as pkg
from aurora_earth_system_pipeline import HISTORY_STEPS, INPUT_SCHEMA, MAX_ROLLOUT_STEPS, TIMESTEP_HOURS, validate_inputs
from conftest import synthetic_window


def test_input_schema_names_the_contract():
    assert INPUT_SCHEMA["timestep_hours"] == TIMESTEP_HOURS == 6
    assert INPUT_SCHEMA["history_steps"] == HISTORY_STEPS == 2
    assert INPUT_SCHEMA["rollout_steps"] == [1, MAX_ROLLOUT_STEPS]
    assert "dynamically consistent" in INPUT_SCHEMA["validation"]


def test_validate_inputs_reports_grid_and_origins():
    report = validate_inputs(synthetic_window(steps=6, name="six"))
    assert report["name"] == "six" and report["forecast_origins"] == 4 and report["shape"] == (17, 32)


def test_public_surface_is_exported():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
    assert "AuroraPipeline" in pkg.__all__ and "audit_pickle" in pkg.__all__
