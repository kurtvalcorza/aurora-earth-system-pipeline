"""Offline tests for the dataset contract, the pinned WeatherBench 2 objects, metrics, the persistence
baseline, NetCDF I/O and artifact-manifest rejections. No model library is imported."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from aurora_earth_system_pipeline import (
    CONVERTED_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_WINDOWS,
    WB2_OBJECTS,
    AuroraPipeline,
    dataset_digest,
    fetch_object,
    forecast_metrics,
    lat_weighted_rmse,
    load_byod_dataset,
    persistence_only,
    validate_dataset,
    write_window_netcdf,
)
from aurora_earth_system_pipeline import pipeline as pl
from aurora_earth_system_pipeline import samples as sm
from conftest import synthetic_window

# --- pinned dataset objects ------------------------------------------------------------------------------


def test_pinned_objects_cover_every_sample_window():
    assert len(SAMPLE_WINDOWS) == 4 and len(WB2_OBJECTS) == 57
    for chunk in SAMPLE_WINDOWS.values():
        for var in sm.WB2_SURF.values():
            assert f"{var}/{chunk}.0.0" in WB2_OBJECTS and f"{var}/.zarray" in WB2_OBJECTS
        for var in sm.WB2_ATMOS.values():
            assert f"{var}/{chunk}.0.0.0" in WB2_OBJECTS
    for var in sm.WB2_STATIC.values():
        assert f"{var}/0.0" in WB2_OBJECTS
    for coord in ("latitude", "longitude", "level"):
        assert f"{coord}/0" in WB2_OBJECTS and f"{coord}/.zarray" in WB2_OBJECTS
    assert all(len(d) == 64 and b > 0 for b, d in WB2_OBJECTS.values())
    assert sum(b for b, _ in WB2_OBJECTS.values()) < 200_000_000


def test_fetch_object_verifies_digest_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    payload = b"zarr-chunk-bytes"
    monkeypatch.setitem(sm.WB2_OBJECTS, "fake/0.0", (len(payload), hashlib.sha256(payload).hexdigest()))
    calls = []

    def fetcher(key):
        calls.append(key)
        return payload

    assert fetch_object("fake/0.0", cache_dir=tmp_path, fetcher=fetcher) == payload
    assert fetch_object("fake/0.0", cache_dir=tmp_path, fetcher=fetcher) == payload  # served from cache
    assert calls == ["fake/0.0"]
    with pytest.raises(ValueError, match="pinned"):
        fetch_object("fake/0.0", cache_dir=tmp_path / "other", fetcher=lambda key: b"tampered-bytes!!")
    with pytest.raises(ValueError, match="not a pinned"):
        fetch_object("unknown/0", cache_dir=tmp_path, fetcher=fetcher)


def test_sample_window_from_cache_if_present(forbid_model_imports):
    cache = sm.DEFAULT_CACHE_DIR
    if not (pl.DEFAULT_WEIGHTS_DIR.parent / "wb2-era5-1p5deg" / "latitude__0").is_file():
        pytest.skip("WeatherBench 2 cache not present")
    window = sm.fetch_sample_window(
        "test-2021-10",
        cache_dir=pl.DEFAULT_WEIGHTS_DIR.parent / "wb2-era5-1p5deg",
        fetcher=lambda key: (_ for _ in ()).throw(AssertionError(key)),
    )
    assert window["shape"] == (121, 240) and window["n_steps"] == 8
    assert window["times"][0] == "2021-09-30T00:00:00" and window["lat"][0] == 90.0 and window["lon"][0] == 0.0
    assert 200.0 < float(window["surf"]["2t"].mean()) < 300.0
    del cache


# --- dataset validation -----------------------------------------------------------------------------------


def test_validate_dataset_and_digest(forbid_model_imports):
    windows = [synthetic_window(name="a"), synthetic_window(name="b", seed=1)]
    report = validate_dataset(windows)
    assert report["n_windows"] == 2 and report["shape"] == (17, 32) and report["forecast_origins"] == 4
    assert report["digest"] == dataset_digest(windows) and len(report["digest"]) == 64
    with pytest.raises(ValueError, match="duplicate window name"):
        validate_dataset([synthetic_window(name="a"), synthetic_window(name="a")])
    with pytest.raises(ValueError, match="share one grid"):
        validate_dataset([synthetic_window(name="a"), synthetic_window(name="c", width=36)])
    with pytest.raises(ValueError, match="at least 2 are required"):
        validate_dataset([synthetic_window()], min_windows=2)
    with pytest.raises(ValueError, match="at least 5"):
        validate_dataset([synthetic_window()], min_steps=5)


# --- metrics ---------------------------------------------------------------------------------------------


def test_lat_weighted_rmse_weights_the_equator_more_than_the_poles(forbid_model_imports):
    lat = [90.0, 0.0, -90.0]
    truth = np.zeros((3, 4))
    polar = np.array([[1.0] * 4, [0.0] * 4, [0.0] * 4])
    equatorial = np.array([[0.0] * 4, [1.0] * 4, [0.0] * 4])
    assert lat_weighted_rmse(polar, truth, lat) < 1e-6 < lat_weighted_rmse(equatorial, truth, lat)
    with pytest.raises(ValueError, match="shape"):
        lat_weighted_rmse(np.zeros((2, 4)), truth, lat)


def test_forecast_metrics_and_persistence_baseline(forbid_model_imports):
    window = pl._check_window(synthetic_window(steps=5))
    origins = [1, 2]
    height = 16  # 17 rows cropped to the patch size
    perfect = [
        [
            {
                "surf": {k: window["surf"][k][o + s + 1, :height] for k in pl.SURF_VARS},
                "atmos": {k: window["atmos"][k][o + s + 1, :, :height] for k in pl.ATMOS_VARS},
            }
            for s in range(2)
        ]
        for o in origins
    ]
    metrics = forecast_metrics(window, origins, perfect)
    assert metrics["leads"] == ["6h", "12h"] and metrics["grid"] == [16, 32]
    assert all(
        row["model"] == 0.0 and row["persistence"] > 0.0
        for name in metrics["variables"]
        for row in metrics["variables"][name].values()
    )
    assert metrics["summary"]["6h"]["variables_beating_persistence"] == 9
    baseline = persistence_only(synthetic_window(steps=5), max_lead_steps=2)
    assert baseline["variables"]["2t"]["6h"]["persistence"] == pytest.approx(metrics["variables"]["2t"]["6h"]["persistence"])
    assert "model" not in baseline["variables"]["2t"]["6h"]
    with pytest.raises(ValueError, match="no analysis at lead step"):
        forecast_metrics(window, [3], [perfect[0]])
    with pytest.raises(ValueError, match="same number of lead steps"):
        forecast_metrics(window, origins, [perfect[0], perfect[1][:1]])


# --- NetCDF ------------------------------------------------------------------------------------------------


def test_netcdf_round_trip_and_rejections(tmp_path, forbid_model_imports):
    window = synthetic_window(name="rt")
    path = write_window_netcdf(window, tmp_path / "rt.nc")
    back = load_byod_dataset(path)
    assert len(back) == 1 and back[0]["name"] == "rt"
    checked = pl._check_window(back[0])
    assert checked["shape"] == (17, 32) and checked["times"] == window["times"]
    for k in pl.SURF_VARS:
        assert np.array_equal(checked["surf"][k], window["surf"][k])
    assert np.array_equal(checked["static"]["z"], window["static"]["z"])
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.nc")
    (tmp_path / "data.csv").write_text("a,b\n")
    with pytest.raises(ValueError, match="NetCDF"):
        load_byod_dataset(tmp_path / "data.csv")


# --- artifacts ---------------------------------------------------------------------------------------------


class _NoModel:
    def state_dict(self):
        return {}


def _pipeline_without_model(use_lora=True):
    return AuroraPipeline(model=_NoModel(), device="cpu", weights_dir=pl.DEFAULT_WEIGHTS_DIR, source="test", use_lora=use_lora)


def test_save_artifact_requires_adaptation(forbid_model_imports):
    with pytest.raises(ValueError, match="call adapt"):
        _pipeline_without_model().save_artifact("unused")


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    manifest = {
        "format": pl.ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "converted_sha256": dict(CONVERTED_SHA256)},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["backbone.x.lora_A"],
        "adapter": {},
    }
    pipe = _pipeline_without_model()
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({**manifest, "base_model": {**manifest["base_model"], "revision": "0" * 40}})
    )
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)
    good = {**manifest, "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}]}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(good))
    with pytest.raises(ValueError, match="use_lora=True"):
        _pipeline_without_model(use_lora=False).load_artifact(tmp_path)


def test_adapt_validates_hyperparameters_before_model_work(forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt([synthetic_window()], epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt([synthetic_window()], lr=1.0)
    with pytest.raises(ValueError, match="at least one training window"):
        pipe.adapt([])
