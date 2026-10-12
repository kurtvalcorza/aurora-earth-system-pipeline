# ruff: noqa: E501
"""Regression tests for review W2 of the weather workshop (AUR2-M1, AUR2-M2, AUR2-m1..m5, 2026-10-13).

Markdown and metadata checks on the generated notebook, plus the wording of the carrier's default-path gate.
Standard library only, matching CI's lightweight dependencies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def markdown():
    return [(i, "".join(c["source"])) for i, c in enumerate(notebook()["cells"]) if c["cell_type"] == "markdown"]


def cell_with(heading):
    hits = [t for _, t in markdown() if t.startswith(heading)]
    assert len(hits) == 1, heading
    return hits[0]


def test_m1_resolution_is_not_presented_as_the_isolated_cause():
    opening = markdown()[0][1]
    assert "does not isolate" in opening and "plausible contributor" in opening
    assert "non-integer codes" in opening and "native-resolution control" in opening
    rows = [line for _, t in markdown() for line in t.splitlines() if line.startswith("| frozen model loses to persistence")]
    assert len(rows) == 1 and "not isolated" in rows[0] and "OOD setup" not in rows[0]
    checkpoint = cell_with("## Interpretation checkpoint")
    assert "hypothesis" in checkpoint and "soil-type" in checkpoint
    assert "expected to notice that change" not in checkpoint


def test_m2_fixed_recipe_reused_window_and_non_binding_selection_are_stated():
    section12 = cell_with("# 12. LoRA adaptation")
    assert "fixed recipe, inherited and not tuned here" in section12 and "still falling" in section12
    section14 = cell_with("# 14. Freeze the experiment")
    assert "within this run only" in section14 and "re-used sample-sanity window" in section14
    assert "default-path regression check against its recorded run, not a scientific test" in section14
    assert not any(t.startswith("# 15. Independent test") for _, t in markdown())


def test_m2_carrier_gate_is_worded_as_a_default_path_regression_check():
    stages = (REPO / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "default-path regression check against the recorded run failed (not a scientific test)" in stages
    assert "expected improvement did not occur" not in stages


def test_m1_minor_quoted_numbers_are_sourced_and_timing_is_current():
    reference = cell_with("## Reference expectation")
    assert "not re-run here" in reference and "`b5c8494`" in reference and "`09775738`" in reference
    assert "clean" not in reference.lower()
    assert all(v in reference for v in ("0.914", "0.867", "0.888", "0.997"))
    text = "\n".join(t for _, t in markdown())
    assert "must receive its own clean-runtime timing" not in text and "207.8 s" in text


def test_m2_minor_diurnal_note_and_printed_2t_rows():
    reference = cell_with("## Reference expectation")
    assert "diurnal" in reference and "1.589" in reference
    test_cell = next("".join(c["source"]) for c in notebook()["cells"] if c["id"] == "dimer-weather-workshop-33")
    assert 'variable_table[variable_table["variable"] == "2t"]' in test_cell and ".to_string(index=False)" in test_cell


def test_m3_prediction_prompts_and_collapsed_answers():
    for heading in ("# 11. Frozen Aurora", "# 12. LoRA adaptation", "# 15. Test on a separate period"):
        assert "Predict before you run" in cell_with(heading), heading
    assert "Exercise 1 — persistence (answer before you run section 11)" in cell_with("# 10. Persistence first")
    exercises = cell_with("# 24. Exercises")
    assert exercises.count("<details>") == 5
    assert "<details>" in cell_with("# 8. Area weighting")


def test_m4_kind_references_and_licences():
    meta = notebook()["metadata"]["dimer"]
    assert meta["notebook_kind"] == "WORKSHOP" and meta["notebook_spec"] == "2.1"
    refs = [t for _, t in markdown() if re.search(r"^## References and licences", t, re.M)]
    assert len(refs) == 1
    for needle in ("10.1038/s41586-025-09005-y", "10.1002/qj.3803", "10.1029/2023MS004019", "Licence: MIT", "Licence to use Copernicus Products"):
        assert needle in refs[0], needle


def test_m5_environment_variables_are_documented():
    section1 = cell_with("## 1. Build the isolated runtime")
    assert "`DIMER_ISOLATED_ENV`" in section1 and "`DIMER_NOTEBOOK_CI_PREINSTALLED=1`" in section1
