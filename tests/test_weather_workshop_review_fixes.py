# ruff: noqa: E501
"""Regression tests for the weather workshop review findings (AUR-01..AUR-05 and minor findings).

The notebook's own helper definitions are extracted from the generated cells and executed with NumPy
(and xarray/netCDF4 for the NetCDF loader), matching CI's lightweight dependencies: no torch, no Aurora.
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import io
import json
import math
import pickle
import pickletools
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb"


def cells():
    return [("".join(c["source"]), c["cell_type"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]]


def code(title):
    hits = [src for src, kind in cells() if kind == "code" and src.startswith(title)]
    assert len(hits) == 1, title
    return hits[0]


def code_index(title):
    return next(i for i, (src, kind) in enumerate(cells()) if kind == "code" and src.startswith(title))


def definitions(source, names):
    """Only the top-level defs/classes/assignments that define `names` (nothing that runs the pipeline)."""
    keep = []
    for node in ast.parse(source).body:
        defines = (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names) or (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
        if defines:
            keep.append(node)
    found = {n.name if not isinstance(n, ast.Assign) else n.targets[0].id for n in keep}
    assert found == set(names), set(names) - found
    return ast.Module(body=keep, type_ignores=[])


def namespace():
    ns = {
        "np": np, "json": json, "hashlib": hashlib, "math": math, "io": io, "pickle": pickle,
        "pickletools": pickletools, "zipfile": zipfile, "Path": Path,
        "datetime": dt.datetime, "timedelta": dt.timedelta, "RUN_ID": "test-run",
    }
    exec(code("# @title Immutable model identity and data contract"), ns)  # literal constants only
    parts = [
        ("# @title Stage, audit and convert the pinned Aurora checkpoint",
         ["sha256_file", "pickle_globals", "audit_pickle", "RestrictedUnpickler"]),
        ("# @title Data loader and validator",
         ["_netcdf_field", "_load_window_netcdf", "validate_window", "window_digest", "window_content_digest",
          "analysis_digests", "ROLE_ORDER", "validate_experiment", "eval_origins", "evaluation_preflight"]),
        ("# @title Weighted and unweighted RMSE helpers", ["lat_weights", "lat_weighted_rmse"]),
        ("# @title Common forecast evaluator",
         ["HEADLINE_LEVEL", "REPORTED", "METRIC_CONVENTION", "check_forecast_fields", "_as_datetime",
          "forecast_arrays", "forecast_metrics", "evaluate_model", "persistence_only"]),
        ("# @title Freeze experiment", ["tensor_state_digest", "experiment_sha256", "freeze_mismatches"]),
        ("# @title SafeTensors adapter export", ["build_adapter_manifest"]),
        ("# @title Fresh reload and parity", ["check_adapter_manifest", "parity_report"]),
        ("# @title Final provenance export", ["write_report_bundle"]),
    ]
    for title, names in parts:
        exec(compile(definitions(code(title), names), title, "exec"), ns)
    return ns


NS = namespace()
LEVELS = NS["LEVELS"]
H, W = 17, 32
LAT = np.linspace(90.0, -90.0, H)
LON = np.arange(W) * (360.0 / W)


def raw_window(name, start, n=8, lat=LAT, lon=LON, seed=0, static_seed=99):
    rng = np.random.default_rng(seed)
    srng = np.random.default_rng(static_seed)
    times = [(start + dt.timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(n)]
    h, w = len(lat), len(lon)
    base = {"2t": 280.0, "10u": 2.0, "10v": 1.0, "msl": 101_000.0, "t": 250.0, "u": 5.0, "v": 1.0, "q": 0.005, "z": 50_000.0}
    spread = {"2t": 5.0, "10u": 2.0, "10v": 2.0, "msl": 500.0, "t": 5.0, "u": 3.0, "v": 3.0, "q": 0.001, "z": 500.0}
    surf = {k: (base[k] + spread[k] * rng.standard_normal((n, h, w))).astype(np.float32) for k in NS["SURF_VARS"]}
    atmos = {k: (base[k] + spread[k] * rng.standard_normal((n, len(LEVELS), h, w))).astype(np.float32) for k in NS["ATMOS_VARS"]}
    static = {
        "lsm": srng.uniform(0, 1, (h, w)).astype(np.float32),
        "z": srng.uniform(0, 3000, (h, w)).astype(np.float32),
        "slt": srng.integers(0, 7, (h, w)).astype(np.float32),
    }
    return {"name": name, "lat": lat.copy(), "lon": lon.copy(), "levels": list(LEVELS), "times": times,
            "surf": surf, "atmos": atmos, "static": static}


def four_roles(**overrides):
    raw = {
        "train-1": raw_window("train-1", dt.datetime(2019, 1, 1), seed=1),
        "train-2": raw_window("train-2", dt.datetime(2019, 7, 1), seed=2),
        "validation": raw_window("validation", dt.datetime(2020, 4, 1), seed=3),
        "test": raw_window("test", dt.datetime(2021, 10, 1), seed=4),
    }
    raw.update(overrides)
    return {role: NS["validate_window"](w) for role, w in raw.items()}


# ---------------------------------------------------------------------------------------------- AUR-01
def test_disjoint_compatible_windows_pass_and_are_receipted():
    receipt = NS["validate_experiment"](four_roles())
    digests = {r["content_sha256"] for r in receipt["roles"].values()}
    assert len(digests) == 4
    assert receipt["roles"]["test"]["first_time"] == "2021-10-01T00:00:00"
    assert receipt["grid"]["shape"] == [H, W]


def test_role_time_overlap_is_refused_fully_and_partially():
    same = raw_window("validation", dt.datetime(2019, 1, 1), seed=3)
    with pytest.raises(ValueError, match="share 8 analysis time"):
        NS["validate_experiment"](four_roles(validation=same))
    partial = raw_window("train-2", dt.datetime(2019, 1, 2, 6), seed=2)  # last 3 analyses overlap train-1
    with pytest.raises(ValueError, match="share 3 analysis time"):
        NS["validate_experiment"](four_roles(**{"train-2": partial}))


def test_relabelled_copy_under_new_timestamps_is_refused():
    copy = raw_window("test", dt.datetime(2021, 10, 1), seed=1)  # train-1's fields, later times
    with pytest.raises(ValueError, match="exactly the same fields"):
        NS["validate_experiment"](four_roles(test=copy))


def test_chronological_policy_is_enforced():
    early_validation = raw_window("validation", dt.datetime(2018, 4, 1), seed=3)
    with pytest.raises(ValueError, match="every training analysis must precede validation"):
        NS["validate_experiment"](four_roles(validation=early_validation))
    early_test = raw_window("test", dt.datetime(2020, 1, 1), seed=4)
    with pytest.raises(ValueError, match="validation must precede test"):
        NS["validate_experiment"](four_roles(test=early_test))


def test_same_shape_but_different_coordinates_or_static_fields_are_refused():
    shifted = raw_window("test", dt.datetime(2021, 10, 1), lon=LON + 1.0, seed=4)
    with pytest.raises(ValueError, match="coordinates differ from train-1"):
        NS["validate_experiment"](four_roles(test=shifted))
    other_static = raw_window("test", dt.datetime(2021, 10, 1), seed=4, static_seed=7)
    with pytest.raises(ValueError, match="static field lsm differs"):
        NS["validate_experiment"](four_roles(test=other_static))


def test_regional_negative_and_irregular_grids_are_refused():
    regional = raw_window("r", dt.datetime(2019, 1, 1), lon=np.arange(W) * 1.5)
    with pytest.raises(ValueError, match="cover the full circle"):
        NS["validate_window"](regional)
    negative = raw_window("n", dt.datetime(2019, 1, 1), lon=LON - 180.0)
    with pytest.raises(ValueError, match=r"\[0, 360\)"):
        NS["validate_window"](negative)
    lat = LAT.copy()
    lat[5] += 2.0
    with pytest.raises(ValueError, match="latitudes must be uniformly spaced"):
        NS["validate_window"](raw_window("i", dt.datetime(2019, 1, 1), lat=lat))
    # the resolution label is only meaningful once the grid is global: 360 / W
    assert NS["validate_window"](raw_window("ok", dt.datetime(2019, 1, 1)))["resolution_degrees"] == 360 / W


def _to_netcdf(window, path, permute_2t=False, extra_dim=False):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    NS["xr"] = xr  # the notebook's loader uses the runtime's xarray import
    times = np.array([np.datetime64(t) for t in window["times"]], dtype="datetime64[ns]")
    data = {}
    for k in NS["SURF_VARS"]:
        arr = window["surf"][k]
        if k == "2t" and permute_2t:
            data[k] = (("longitude", "latitude", "time"), np.transpose(arr, (2, 1, 0)))
        else:
            data[k] = (("time", "latitude", "longitude"), arr)
    for k in NS["ATMOS_VARS"]:
        data[k] = (("time", "level", "latitude", "longitude"), window["atmos"][k])
    for k in NS["STATIC_VARS"]:
        if extra_dim and k == "lsm":
            data[f"static_{k}"] = (("member", "latitude", "longitude"), window["static"][k][None])
        else:
            data[f"static_{k}"] = (("latitude", "longitude"), window["static"][k])
    xr.Dataset(data, coords={"time": times, "level": list(LEVELS), "latitude": window["lat"], "longitude": window["lon"]}).to_netcdf(path, engine="netcdf4")


def test_netcdf_variables_are_transposed_by_dimension_name(tmp_path):
    # 32 analyses on a 32-longitude grid: (time, lat, lon) and (lon, lat, time) have the same shape, so a
    # loader that reads `.values` in stored order would silently swap time and longitude.
    window = raw_window("byod", dt.datetime(2019, 1, 1), n=32, seed=5)
    path = tmp_path / "permuted.nc"
    _to_netcdf(window, path, permute_2t=True)
    loaded = NS["_load_window_netcdf"](str(path))
    assert loaded["surf"]["2t"].shape == (32, H, W)
    np.testing.assert_array_equal(loaded["surf"]["2t"], window["surf"]["2t"])
    NS["validate_window"](loaded)


def test_netcdf_variable_with_other_dimensions_is_refused(tmp_path):
    window = raw_window("byod", dt.datetime(2019, 1, 1), seed=5)
    path = tmp_path / "extra.nc"
    _to_netcdf(window, path, extra_dim=True)
    with pytest.raises(ValueError, match="has dimensions"):
        NS["_load_window_netcdf"](str(path))


# ---------------------------------------------------------------------------------------------- AUR-02
def test_preflight_refuses_leads_without_reference_targets():
    roles = four_roles()
    windows = {"validation": roles["validation"], "test": roles["test"]}
    ok = NS["evaluation_preflight"](6, windows)
    assert ok["windows"]["test"]["origins"] == [1]
    assert ok["windows"]["test"]["longest_measurable_lead_hours"] == 36
    for steps in (7, 8):
        with pytest.raises(ValueError, match="longest measurable lead here is 36 h"):
            NS["evaluation_preflight"](steps, windows)


def test_short_byod_windows_are_refused_before_model_loading():
    short = NS["validate_window"](raw_window("v", dt.datetime(2020, 4, 1), n=3, seed=3))
    with pytest.raises(ValueError, match="has 3 analyses"):
        NS["evaluation_preflight"](4, {"validation": short})
    assert NS["evaluation_preflight"](1, {"validation": short})["windows"]["validation"]["origins"] == [1]


def test_unmeasurable_persistence_request_is_an_actionable_error_not_an_index_error():
    window = four_roles()["test"]
    with pytest.raises(ValueError, match="10 are needed for 8 lead steps"):
        NS["persistence_only"](window, 8)


def test_origin_selection_matches_the_previous_default():
    window = four_roles()["test"]
    for steps in range(1, 7):
        last_origin = window["n_steps"] - 1 - steps
        assert NS["eval_origins"](window, steps) == list(range(1, last_origin + 1))


# ---------------------------------------------------------------------------------------------- AUR-03
def _frozen_record():
    record = {
        "model": {"checkpoint_sha256": "c" * 64},
        "adaptation": {"trainable": "lora"},
        "selected_adapter": {"state_sha256": "a" * 64},
        "test_request": {"window_content_sha256": "t" * 64, "lead_steps": 4, "origins": [1, 2, 3]},
    }
    record["experiment_sha256"] = NS["experiment_sha256"](record)
    return record


LIVE = {"checkpoint_sha256": "c" * 64, "trainable": "lora", "adapter_state_sha256": "a" * 64,
        "test_content_sha256": "t" * 64, "lead_steps": 4}


def test_unchanged_live_state_matches_the_freeze():
    assert NS["freeze_mismatches"](_frozen_record(), **LIVE) == []


@pytest.mark.parametrize("field,value,expected", [
    ("test_content_sha256", "x" * 64, "test window data"),
    ("adapter_state_sha256", "x" * 64, "selected adapter tensors"),
    ("lead_steps", 6, "evaluation lead steps"),
    ("trainable", "lora+heads", "adaptation scope"),
    ("checkpoint_sha256", "x" * 64, "base checkpoint"),
])
def test_changed_live_state_is_refused(field, value, expected):
    problems = NS["freeze_mismatches"](_frozen_record(), **{**LIVE, field: value})
    assert any(expected in p for p in problems)


def test_an_edited_freeze_record_is_detected():
    record = _frozen_record()
    record["test_request"]["lead_steps"] = 6
    problems = NS["freeze_mismatches"](record, **{**LIVE, "lead_steps": 6})
    assert problems == ["the frozen record was edited after it was written"]


def test_adapter_state_digest_tracks_tensor_values():
    a = {"x.lora_A": np.zeros((2, 3), np.float32), "x.lora_B": np.ones(3, np.float32)}
    b = {"x.lora_B": np.ones(3, np.float32), "x.lora_A": np.zeros((2, 3), np.float32)}
    assert NS["tensor_state_digest"](a) == NS["tensor_state_digest"](b)
    b["x.lora_B"] = b["x.lora_B"].copy()
    b["x.lora_B"][0] = 1.0001
    assert NS["tensor_state_digest"](a) != NS["tensor_state_digest"](b)


def test_test_cell_verifies_the_freeze_before_any_test_forecast():
    src = code("# @title Independent test comparison")
    assert src.index("freeze_mismatches(") < src.index("evaluate_model(")
    assert "origins=TEST_ORIGINS" in src and "TEST_LEAD_STEPS = frozen_record" in src


# ---------------------------------------------------------------------------------------------- AUR-04
def test_notebook_base_and_artifact_identities_match_the_carrier():
    from aurora_earth_system_pipeline import pipeline

    assert NS["ARTIFACT_FORMAT"] == pipeline.ARTIFACT_FORMAT
    assert NS["ARTIFACT_FORMAT_VERSION"] == pipeline.ARTIFACT_FORMAT_VERSION
    assert NS["ARTIFACT_WEIGHTS_NAME"] == pipeline.ARTIFACT_WEIGHTS_NAME
    assert NS["CONVERTED_SHA256"] == pipeline.CONVERTED_SHA256
    assert NS["CONVERTED_BYTES"] == pipeline.CONVERTED_BYTES
    assert NS["PICKLE_AUDIT_SHA256"] == pipeline.PICKLE_AUDIT_SHA256
    assert (NS["STATIC_NAME"], NS["STATIC_BYTES"], NS["STATIC_SHA256"]) == (
        pipeline.SOURCE_STATIC_NAME, pipeline.SOURCE_STATIC_BYTES, pipeline.SOURCE_STATIC_SHA256)
    assert set(NS["CKPT_ALLOWED_GLOBALS"]) == set(pipeline.CKPT_ALLOWED_GLOBALS)
    assert set(NS["STATIC_ALLOWED_GLOBALS"]) == set(pipeline.STATIC_ALLOWED_GLOBALS)
    assert (NS["STATE_TENSORS"], NS["MODEL_KEY"]) == (pipeline.STATE_TENSORS, pipeline.MODEL_KEY)
    assert (NS["MIN_GRID"], NS["MAX_GRID"]) == (pipeline.MIN_GRID, pipeline.MAX_GRID)
    assert tuple(NS["ADAPTATION_MODES"]) == tuple(pipeline.ADAPTATION_MODES)


def _adapter(tmp_path, names=("a.lora_A", "a.lora_B")):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "adapter.safetensors").write_bytes(b"synthetic adapter bytes")
    adaptation = {"trainable": "lora", "n_trainable": 12, "epochs": 6, "best_epoch": 6, "learning_rate": 1e-3,
                  "seed": 0, "loss_scales": {"2t": 10.0}, "history": [{"epoch": 0}]}
    manifest = NS["build_adapter_manifest"](root / "adapter.safetensors", list(names), adaptation, "s" * 64)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def test_notebook_manifest_passes_the_carrier_consumer_static_check(tmp_path):
    from aurora_earth_system_pipeline.pipeline import AuroraPipeline

    root, manifest = _adapter(tmp_path)
    weights, mode = AuroraPipeline.check_artifact_manifest(root, json.loads((root / "manifest.json").read_text()))
    assert weights == (root / "adapter.safetensors").resolve() and mode == "lora"
    assert NS["check_adapter_manifest"](root, manifest, ["a.lora_B", "a.lora_A"])[1] == "lora"


@pytest.mark.parametrize("mutate,expected", [
    (lambda m: m["base_model"]["converted_sha256"].update({"aurora-0.25-static.safetensors": "0" * 64}), "converted-base digests"),
    (lambda m: m["base_model"].update(revision="0" * 40), "base revision"),
    (lambda m: m["files"][0].update(sha256="0" * 64), "digest/size"),
    (lambda m: m["files"].append(dict(m["files"][0])), "exactly one weights file"),
    (lambda m: m["files"][0].update(path="../adapter.safetensors"), "must be named"),
    (lambda m: m["tensors"].pop(), "tensor manifest mismatch"),
    (lambda m: m["adapter"].update(trainable="full"), "adapter.trainable"),
])
def test_invalid_manifests_still_fail(tmp_path, mutate, expected):
    root, manifest = _adapter(tmp_path)
    mutate(manifest)
    with pytest.raises(ValueError, match=expected):
        NS["check_adapter_manifest"](root, manifest, ["a.lora_A", "a.lora_B"])


def test_pickle_audit_allows_only_the_listed_globals(tmp_path):
    from collections import OrderedDict

    good = tmp_path / "state.pkl"
    good.write_bytes(pickle.dumps(OrderedDict(a=1), protocol=2))
    assert NS["audit_pickle"](good, NS["CKPT_ALLOWED_GLOBALS"])["globals"] == ["collections.OrderedDict"]
    archive = tmp_path / "ckpt.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(OrderedDict(a=1), protocol=4))
    assert NS["audit_pickle"](archive, NS["CKPT_ALLOWED_GLOBALS"])["globals"] == ["collections.OrderedDict"]
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps(Path("x"), protocol=4))
    with pytest.raises(ValueError, match="outside the allow-list"):
        NS["audit_pickle"](bad, NS["CKPT_ALLOWED_GLOBALS"])


def test_base_is_audited_and_converted_before_any_model_is_built():
    src = code("# @title Stage, audit and convert the pinned Aurora checkpoint")
    assert src.index("audit_pickle(checkpoint_path") < src.index("convert_base(CONVERTED_DIR)")
    assert "do not match the pinned DIMER converted-base digests" in src
    assert code_index("# @title Stage, audit and convert the pinned Aurora checkpoint") < code_index("# @title Model and Batch helpers")


# ---------------------------------------------------------------------------------------------- AUR-05
def _persistence_like(window, origins, steps, height=16):
    return [[{"surf": {k: window["surf"][k][o, :height].copy() for k in NS["SURF_VARS"]},
              "atmos": {k: window["atmos"][k][o, :, :height].copy() for k in NS["ATMOS_VARS"]}} for _ in range(steps)]
            for o in origins]


def test_well_formed_forecasts_are_scored():
    window = four_roles()["test"]
    result = NS["forecast_metrics"](window, [1, 2], _persistence_like(window, [1, 2], 4))
    assert result["n_origins"] == 2 and result["origins"] == [1, 2]
    assert result["summary"]["6h"]["mean_skill"] == pytest.approx(1.0)
    assert "mean of per-origin RMSEs" in result["metric"]


@pytest.mark.parametrize("case,expected", [
    ("missing", "1 forecast sets for 2 origins"),
    ("extra", "3 forecast sets for 2 origins"),
    ("duplicate", "duplicate forecast origins"),
    ("nan", "non-finite"),
    ("inf", "non-finite"),
    ("shape", "has shape"),
    ("variable", "variables"),
    ("beyond", "needs analyses up to index"),
    ("ragged", "leads, expected"),
])
def test_malformed_forecasts_are_refused(case, expected):
    window = four_roles()["test"]
    origins = [1, 2]
    preds = _persistence_like(window, origins, 4)
    if case == "missing":
        preds = preds[:1]
    elif case == "extra":
        preds = preds + preds[:1]
    elif case == "duplicate":
        origins = [1, 1]
    elif case == "nan":
        preds[1][2]["surf"]["2t"][:] = np.nan
    elif case == "inf":
        preds[0][0]["atmos"]["q"][3, 2, 1] = np.inf
    elif case == "shape":
        preds[0][0]["surf"]["msl"] = preds[0][0]["surf"]["msl"][:-1]
    elif case == "variable":
        del preds[0][0]["atmos"]["z"]
    elif case == "beyond":
        origins = [1, 4]
    elif case == "ragged":
        preds[1] = preds[1][:3]
    with pytest.raises(ValueError, match=expected):
        NS["forecast_metrics"](window, origins, preds)


def test_a_legitimately_poor_finite_forecast_is_still_a_result():
    window = four_roles()["test"]
    preds = _persistence_like(window, [1], 2)
    preds[0][0]["surf"]["2t"] = preds[0][0]["surf"]["2t"] + 40.0
    result = NS["forecast_metrics"](window, [1], preds)
    assert result["variables"]["2t"]["6h"]["skill"] > 1


def test_rmse_is_a_mean_of_per_origin_rmses_not_pooled():
    window = four_roles()["test"]
    preds = []
    for origin, error in ((1, 1.0), (2, 3.0)):
        target = {"surf": {k: window["surf"][k][origin + 1, :16].copy() for k in NS["SURF_VARS"]},
                  "atmos": {k: window["atmos"][k][origin + 1, :, :16].copy() for k in NS["ATMOS_VARS"]}}
        target["surf"]["2t"] = target["surf"]["2t"] + error
        preds.append([target])
    result = NS["forecast_metrics"](window, [1, 2], preds)
    assert result["variables"]["2t"]["6h"]["model"] == pytest.approx(2.0)


def test_undefined_persistence_ratios_are_excluded_from_the_mean():
    raw = raw_window("test", dt.datetime(2021, 10, 1), seed=4)
    raw["surf"]["10u"][:] = 3.0  # constant field: persistence RMSE 0
    window = NS["validate_window"](raw)
    preds = _persistence_like(window, [1], 1)
    preds[0][0]["surf"]["10u"] = preds[0][0]["surf"]["10u"] + 1.0
    summary = NS["forecast_metrics"](window, [1], preds)["summary"]["6h"]
    assert summary["variables_with_defined_ratio"] == 8
    assert math.isfinite(summary["mean_skill"])


class _FakeTensor(np.ndarray):
    def numpy(self):
        return np.asarray(self)


def _fake_rollout(window, origin, steps, shift_hours=0):
    start = dt.datetime.fromisoformat(window["times"][origin])
    out = []
    for k in range(1, steps + 1):
        fields = _persistence_like(window, [origin], 1)[0][0]
        out.append(SimpleNamespace(
            surf_vars={n: v[None, None].view(_FakeTensor) for n, v in fields["surf"].items()},
            atmos_vars={n: v[None, None].view(_FakeTensor) for n, v in fields["atmos"].items()},
            metadata=SimpleNamespace(time=(start + dt.timedelta(hours=6 * k + shift_hours),)),
        ))
    return out


def test_mistimed_or_truncated_rollouts_are_refused():
    window = four_roles()["test"]
    NS["forecast_arrays"](_fake_rollout(window, 1, 4), window, 1)
    with pytest.raises(ValueError, match="valid at"):
        NS["forecast_arrays"](_fake_rollout(window, 1, 4, shift_hours=6), window, 1)
    NS["forecast_from_origin"] = lambda model, w, origin, steps: _fake_rollout(w, origin, steps - 1)
    try:
        with pytest.raises(ValueError, match="rollout returned 3 leads, expected 4"):
            NS["evaluate_model"](None, window, 4)
        NS["forecast_from_origin"] = lambda model, w, origin, steps: _fake_rollout(w, origin, steps)
        assert NS["evaluate_model"](None, window, 4)["n_origins"] == 3
    finally:
        NS.pop("forecast_from_origin")


def test_reload_parity_requires_finite_matching_outputs():
    window = four_roles()["test"]
    a = _persistence_like(window, [1], 1)[0][0]
    assert NS["parity_report"](a, a, 1e-6)["status"] == "PASS"
    nan = {g: {k: np.full_like(v, np.nan) for k, v in a[g].items()} for g in a}
    with pytest.raises(ValueError, match="non-finite"):
        NS["parity_report"](nan, nan, 1e-6)
    off = {g: {k: v + 1.0 for k, v in a[g].items()} for g in a}
    with pytest.raises(RuntimeError, match="fresh reload parity failed"):
        NS["parity_report"](a, off, 1e-6)
    short = {g: {k: v[..., :-1] for k, v in a[g].items()} for g in a}
    with pytest.raises(ValueError, match="shapes"):
        NS["parity_report"](a, short, 1e-6)


# ---------------------------------------------------------------------------------------- minor findings
def test_report_bundle_contains_only_this_runs_files(tmp_path):
    root = tmp_path / "outputs"
    (root / "test").mkdir(parents=True)
    (root / "old_experiment.txt").write_text("previous run", encoding="utf-8")
    current = root / "test" / "comparison.csv"
    current.write_text("lead,ratio\n6,0.9\n", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    inventory = NS["write_report_bundle"](bundle, root, [str(current)], "run-1")
    assert [e["path"] for e in inventory] == ["test/comparison.csv"]
    with zipfile.ZipFile(bundle) as zf:
        assert sorted(zf.namelist()) == ["run_inventory.json", "test/comparison.csv"]
        assert json.loads(zf.read("run_inventory.json"))["run_id"] == "run-1"
    assert (root / "old_experiment.txt").exists()  # preserved, just not packaged


def test_spatial_maps_share_scales_and_state_their_times():
    src = code("# @title Spatial maps at the longest evaluated lead")
    assert "vmin=vmin, vmax=vmax" in src and "err_hi" in src and "valid {test_window['times'][truth_idx]}" in src


def test_future_export_carries_coordinates_times_and_scope():
    src = code("# @title Forecast beyond the available window")
    for key in ("latitude=", "longitude=", "origin_time=", "valid_times=", "units=", "scope=", "area_weighted_mean_2t_K"):
        assert key in src
    assert "FUTURE_FORECAST_STEPS" in src


def test_activity_is_off_by_default_separate_and_checked():
    src = code("# @title Optional activity: longer rollouts on one fixed set of origins")
    assert "RUN_LONGER_ROLLOUT_ACTIVITY = False" in src
    assert 'OUTPUT_ROOT / "activity"' in src and "origins=activity_origins" in src
    assert "the activity changed a canonical result" in src
    assert code_index("# @title Optional activity: longer rollouts on one fixed set of origins") < code_index("# @title Final provenance export")


def test_lead_preflight_runs_before_any_model_is_built():
    assert code_index("# @title Data loader and validator") < code_index("# @title Model and Batch helpers")
    assert "evaluation_preflight(MAX_LEAD_STEPS" in code("# @title Data loader and validator")
