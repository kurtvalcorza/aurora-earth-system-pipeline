"""DAT20: a refused BYOD input renders as exactly one line, ``Refused: <file>: <reason>``, with no traceback, and the
cell still raises so Run all stops there. A genuine stage failure keeps its error. The kernel helper cells are executed
from the generated notebook, and the data stage runs in a real subprocess, as in the review-fix tests."""
# ruff: noqa: E501

from __future__ import annotations

import sys
import types

import pytest

from test_aurora_earth_system_colab_review_fixes import _kernel_code, _kernel_helpers, _nc


def _one_line(exc: BaseException) -> str:
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    return rendered[0]


def test_a_refused_byod_window_shows_one_refused_line_naming_the_file(tmp_path, capsys):
    ns = _kernel_helpers(tmp_path)
    nc = _nc(tmp_path, 5)
    with pytest.raises(ns["StageRefused"]) as caught:
        ns["run_stage"]("data", "--byod", nc)
    line = _one_line(caught.value)
    assert line.startswith("Refused: byod_window.nc: BYOD file refused: test window has 5 6-hourly steps")
    assert "re-run Section 5" in line
    shown = capsys.readouterr().out
    assert "REFUSED" not in shown and "Refused:" not in shown and "Traceback" not in shown
    assert "REFUSED (data)" in (tmp_path / "run" / "logs" / "data.log").read_text(encoding="utf-8")


def test_the_scripted_refusal_input_a_non_netcdf_file_is_refused_by_name(tmp_path):
    ns = _kernel_helpers(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    path = ns["obtain_upload"](str(hostname), ".nc", "BYOD_PATH")
    with pytest.raises(ns["StageRefused"]) as caught:
        ns["run_stage"]("data", "--byod", path)
    assert _one_line(caught.value).startswith("Refused: hostname: BYOD file refused: ")


def test_a_directory_or_missing_byod_path_is_refused_by_name(tmp_path):
    ns = _kernel_helpers(tmp_path)
    folder = tmp_path / "etc"
    folder.mkdir()
    with pytest.raises(ns["StageRefused"]) as caught:
        ns["obtain_upload"](str(folder), ".nc", "BYOD_PATH")
    assert _one_line(caught.value).startswith(f"Refused: etc: BYOD_PATH = {str(folder)!r} is not a file on this machine")


@pytest.mark.parametrize(("uploaded", "expected"), [({}, "Refused: (no file): the upload was cancelled, empty or more than one file (got 0)"), ({"w.grib": b"x"}, "Refused: w.grib: the uploaded file must end in .nc")])
def test_a_cancelled_or_wrong_upload_is_refused_in_one_line(tmp_path, monkeypatch, uploaded, expected):
    ns = _kernel_helpers(tmp_path)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: dict(uploaded))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ns["StageRefused"]) as caught:
        ns["obtain_upload"]("", ".nc", "BYOD_PATH")
    assert _one_line(caught.value).startswith(expected)


def test_a_genuine_stage_failure_is_not_presented_as_a_refusal(tmp_path):
    ns = _kernel_helpers(tmp_path)
    with pytest.raises(RuntimeError, match="Stage 'no-such-stage' failed") as caught:
        ns["run_stage"]("no-such-stage")
    assert not isinstance(caught.value, ns["StageRefused"]) and not hasattr(caught.value, "_render_traceback_")


def test_no_learner_cell_renders_the_old_prefix():
    code = "\n".join(_kernel_code())
    assert "'REFUSED: '" not in code and "return ['Refused: ' + str(self)]" in code
