"""Offline tests: identity, snapshot verification and staging, the static pickle audits, window validation
and the digests. No model library is imported."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile

import numpy as np
import pytest

from aurora_earth_system_pipeline import (
    CONVERTED_SHA256,
    CONVERTED_STATIC_NAME,
    CONVERTED_WEIGHTS_NAME,
    DEFAULT_WEIGHTS_DIR,
    INPUT_SCHEMA,
    LEVELS,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    PICKLE_AUDIT_SHA256,
    SOURCE_CKPT_NAME,
    SOURCE_CKPT_SHA256,
    SOURCE_STATIC_NAME,
    SOURCE_STATIC_SHA256,
    AuroraPipeline,
    audit_pickle,
    stage_missing_files,
    validate_inputs,
    verify_converted,
    verify_snapshot,
    window_digest,
)
from aurora_earth_system_pipeline import pipeline as pl
from conftest import synthetic_window

# --- identity -------------------------------------------------------------------------------------


def test_identity_constants_and_manifest():
    assert MODEL_ID == "microsoft/aurora" and MODEL_LICENSE == "mit" and MODEL_KEY == "aurora-0.25-small"
    assert len(MODEL_REVISION) == 40 and MODEL_REVISION.lower() == MODEL_REVISION
    manifest = json.loads((DEFAULT_WEIGHTS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == "dimer_hf_snapshot"
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    assert set(by_path) == {"README.md", SOURCE_CKPT_NAME, SOURCE_STATIC_NAME}
    assert (by_path[SOURCE_CKPT_NAME]["sha256"], by_path[SOURCE_CKPT_NAME]["bytes"]) == (SOURCE_CKPT_SHA256, pl.SOURCE_CKPT_BYTES)
    assert (by_path[SOURCE_STATIC_NAME]["sha256"], by_path[SOURCE_STATIC_NAME]["bytes"]) == (
        SOURCE_STATIC_SHA256,
        pl.SOURCE_STATIC_BYTES,
    )
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])
    assert set(CONVERTED_SHA256) == {CONVERTED_WEIGHTS_NAME, CONVERTED_STATIC_NAME}
    assert all(len(d) == 64 for d in (*CONVERTED_SHA256.values(), *PICKLE_AUDIT_SHA256.values()))
    assert LEVELS == (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


# --- snapshot verification and staging ------------------------------------------------------------


def _write_snapshot(root, *, tamper=False):
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("README.md", SOURCE_CKPT_NAME, SOURCE_STATIC_NAME):
        payload = name.encode() * 4
        (root / name).write_bytes(payload)
        files.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    if tamper:
        files[0]["sha256"] = "0" * 64
    (root / MANIFEST_NAME).write_text(
        json.dumps({"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": files}), encoding="utf-8"
    )


def test_verify_snapshot_checks_digests_and_the_pinned_source_constants(tmp_path, forbid_model_imports):
    _write_snapshot(tmp_path)
    with pytest.raises(ValueError, match="disagrees with the package constant"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_refuses_missing_manifest_or_wrong_identity(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no snapshot manifest"):
        verify_snapshot(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"modelId": "other/model", "revision": MODEL_REVISION, "files": []}))
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": []}))
    with pytest.raises(ValueError, match="does not list"):
        verify_snapshot(tmp_path)


def test_verify_converted_checks_both_files(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        verify_converted(tmp_path)
    (tmp_path / CONVERTED_WEIGHTS_NAME).write_bytes(b"x")
    (tmp_path / CONVERTED_STATIC_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="size"):
        verify_converted(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    _write_snapshot(tmp_path)
    (tmp_path / SOURCE_STATIC_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def downloader(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(relative_path.encode() * 4)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == [SOURCE_STATIC_NAME]
    assert fetched == [SOURCE_STATIC_NAME]
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []


def test_stage_refuses_manifest_for_another_model(tmp_path, forbid_model_imports):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"modelId": MODEL_ID, "revision": "0" * 40, "files": []}))
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


def test_from_pretrained_refuses_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError):
        AuroraPipeline.from_pretrained(weights_dir=tmp_path)
    _write_snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        AuroraPipeline.from_pretrained(weights_dir=tmp_path)
    (tmp_path / MANIFEST_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        AuroraPipeline.from_pretrained(weights_dir=tmp_path, require_source=False)


def test_require_source_false_never_stages_even_when_the_manifest_is_present(tmp_path, monkeypatch, forbid_model_imports):
    """A checkout keeps the committed manifest beside the converted files and no pickles: the converted-only
    path must not touch the sources at all (it used to fall into stage_missing_files whenever the manifest existed)."""
    _write_snapshot(tmp_path)
    for name in (pl.SOURCE_CKPT_NAME, pl.SOURCE_STATIC_NAME):
        (tmp_path / name).unlink()

    def refuse(*args, **kwargs):
        raise AssertionError("stage_missing_files must not run with require_source=False")

    monkeypatch.setattr(pl, "stage_missing_files", refuse)
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        AuroraPipeline.from_pretrained(weights_dir=tmp_path, require_source=False)
    with pytest.raises(AssertionError, match="must not run"):
        AuroraPipeline.from_pretrained(weights_dir=tmp_path)


def test_convert_model_refuses_a_wrong_sized_source_before_unpickling(tmp_path, forbid_model_imports):
    (tmp_path / SOURCE_CKPT_NAME).write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="size"):
        pl.convert_model(tmp_path)


# --- static pickle audits ----------------------------------------------------------------------------


def _global(module: str, name: str) -> bytes:
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _torch_like_archive(pickle_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model/data.pkl", pickle_bytes)
        archive.writestr("model/version", b"3\n")
    return buffer.getvalue()


def test_audit_accepts_a_state_dict_archive_and_refuses_extras(tmp_path, forbid_model_imports):
    ok = b"\x80\x02" + _global("collections", "OrderedDict") + _global("torch._utils", "_rebuild_tensor_v2") + b"."
    path = tmp_path / SOURCE_CKPT_NAME
    path.write_bytes(_torch_like_archive(ok))
    report = audit_pickle(path, allowed=pl.CKPT_ALLOWED_GLOBALS)
    assert report["torch_archive"] and report["globals"] == ["collections.OrderedDict", "torch._utils._rebuild_tensor_v2"]
    bad = b"\x80\x02" + _global("torch._utils", "_rebuild_tensor_v2") + _global("os", "system") + b"."
    path.write_bytes(_torch_like_archive(bad))
    with pytest.raises(ValueError, match=r"outside the allow-list: \['os.system'\]"):
        audit_pickle(path, allowed=pl.CKPT_ALLOWED_GLOBALS)


def test_audit_reads_plain_pickles_with_stack_globals(tmp_path, forbid_model_imports):
    path = tmp_path / SOURCE_STATIC_NAME
    path.write_bytes(pickle.dumps({"lsm": np.zeros((2, 2), dtype=np.float32)}, protocol=5))
    report = audit_pickle(path, allowed=pl.STATIC_ALLOWED_GLOBALS)
    assert not report["torch_archive"] and all(g.startswith("numpy") for g in report["globals"])
    path.write_bytes(b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93.")
    with pytest.raises(ValueError, match="builtins.eval"):
        audit_pickle(path, allowed=pl.STATIC_ALLOWED_GLOBALS)


def test_restricted_unpickler_refuses_anything_outside_the_allow_list(forbid_model_imports):
    payload = pickle.dumps({"lsm": np.ones((2, 2), dtype=np.float32)}, protocol=5)
    fields = pl._RestrictedUnpickler(io.BytesIO(payload), pl.STATIC_ALLOWED_GLOBALS).load()
    assert fields["lsm"].shape == (2, 2)
    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        pl._RestrictedUnpickler(io.BytesIO(pickle.dumps({"x": complex(1, 2)})), pl.STATIC_ALLOWED_GLOBALS).load()


@pytest.mark.skipif(not (DEFAULT_WEIGHTS_DIR / SOURCE_STATIC_NAME).is_file(), reason="source pickles not staged")
def test_audit_of_the_real_sources_matches_the_pinned_digests(forbid_model_imports):
    static = audit_pickle(DEFAULT_WEIGHTS_DIR / SOURCE_STATIC_NAME, allowed=pl.STATIC_ALLOWED_GLOBALS)
    assert static["audit_sha256"] == PICKLE_AUDIT_SHA256[SOURCE_STATIC_NAME] and static["globals"] == [
        "numpy.core.numeric._frombuffer",
        "numpy.dtype",
    ]
    if (DEFAULT_WEIGHTS_DIR / SOURCE_CKPT_NAME).is_file():
        ckpt = audit_pickle(DEFAULT_WEIGHTS_DIR / SOURCE_CKPT_NAME, allowed=pl.CKPT_ALLOWED_GLOBALS)
        assert ckpt["audit_sha256"] == PICKLE_AUDIT_SHA256[SOURCE_CKPT_NAME] and ckpt["torch_archive"]


# --- window validation ---------------------------------------------------------------------------------


def test_validate_inputs_accepts_and_normalises(forbid_model_imports):
    report = validate_inputs(synthetic_window())
    assert report["shape"] == (17, 32) and report["n_steps"] == 4 and report["forecast_origins"] == 2
    assert report["resolution_degrees"] == 11.25 and report["native_resolution"] is False
    assert report["flipped_latitude"] is False and report["ignored_variables"] == []
    flipped = validate_inputs(synthetic_window(ascending_lat=True))
    assert flipped["flipped_latitude"] is True
    assert window_digest(synthetic_window()) == window_digest(synthetic_window(ascending_lat=True))
    assert INPUT_SCHEMA["levels"] == list(LEVELS)


def _with(window, **changes):
    return {**window, **changes}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda w: _with(w, levels=list(range(13))), "levels must be exactly"),
        (lambda w: _with(w, lon=w["lon"][:-1]), "multiple of the patch size"),
        (lambda w: _with(w, lat=w["lat"][1:]), "must span the poles"),
        (lambda w: _with(w, times=w["times"][:-1] + ["2030-01-01T00:00:00"]), "spaced exactly 6 h"),
        (lambda w: _with(w, times=["not-a-time"] * 4), "not ISO 8601"),
        (lambda w: _with(w, times=w["times"][:2], surf=w["surf"]), "time steps"),
        (lambda w: _with(w, surf={**w["surf"], "2t": w["surf"]["2t"] + 500.0}), "plausible range"),
        (lambda w: _with(w, surf={**w["surf"], "2t": w["surf"]["2t"][:, :-1]}), "has shape"),
        (lambda w: _with(w, atmos={k: v for k, v in w["atmos"].items() if k != "q"}), "missing variable 'q'"),
        (lambda w: _with(w, static={**w["static"], "lsm": w["static"]["lsm"] * np.nan}), "non-finite"),
        (lambda w: _with(w, lon=[x * 0.9 for x in w["lon"]]), "cover the full circle"),
        (lambda w: {k: v for k, v in w.items() if k != "static"}, "missing 'static'"),
        (lambda w: _with(w, name=7), "name must be a string"),
    ],
)
def test_validate_inputs_rejections(mutate, message, forbid_model_imports):
    with pytest.raises(ValueError, match=message):
        validate_inputs(mutate(synthetic_window(height=33, width=64)))


def test_validate_inputs_rejects_grids_outside_the_ceilings(forbid_model_imports):
    with pytest.raises(ValueError, match="outside"):
        validate_inputs(synthetic_window(height=9, width=16))
    with pytest.raises(ValueError, match="multiple of 4, or one more"):
        validate_inputs(synthetic_window(height=18, width=32))
    with pytest.raises(ValueError, match="list of window mappings"):
        from aurora_earth_system_pipeline import validate_dataset

        validate_dataset(synthetic_window())


def test_predict_and_evaluate_validate_arguments_before_model_work(forbid_model_imports):
    pipe = AuroraPipeline(model=None, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR, source="test", use_lora=False)
    with pytest.raises(ValueError, match="steps must be"):
        pipe.predict(synthetic_window(), steps=0)
    with pytest.raises(ValueError, match="max_lead_steps"):
        pipe.evaluate(synthetic_window(), max_lead_steps=99)
    with pytest.raises(ValueError, match="are needed"):
        pipe.evaluate(synthetic_window(steps=3), max_lead_steps=4)
    with pytest.raises(ValueError, match="use_lora=True"):
        pipe.adapt([synthetic_window()])
    with pytest.raises(ValueError, match="trainable must be"):
        AuroraPipeline(model=None, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR, source="t", use_lora=True)._trainable("all")
