"""Model-backed checks that run only where microsoft-aurora and the converted files are present (local
pre-flight; skipped in the lightweight CI): strict load with and without LoRA, deterministic forecasts
with the patch-size crop, a one-epoch LoRA adaptation on a synthetic window and an artifact round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aurora_earth_system_pipeline import CONVERTED_WEIGHTS_NAME, DEFAULT_WEIGHTS_DIR, AuroraPipeline
from aurora_earth_system_pipeline import pipeline as pl
from conftest import synthetic_window

pytest.importorskip("aurora")
if not (DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_NAME).is_file():
    pytest.skip("converted weights not staged", allow_module_level=True)


@pytest.fixture(scope="module")
def pipe():
    return AuroraPipeline.from_pretrained(use_lora=True)


def test_rebuilt_model_identity(pipe):
    assert pipe.source.startswith("converted") and pipe.use_lora
    assert sum(p.numel() for n, p in pipe.model.named_parameters() if "lora" not in n) == 112_797_584
    assert sum(p.numel() for n, p in pipe.model.named_parameters() if "lora" in n) == 540_672
    assert pipe.native_static_fields()["lsm"].shape == (721, 1440)
    plain = AuroraPipeline.from_pretrained(use_lora=False)
    assert not any("lora" in n for n, _ in plain.model.named_parameters())


def test_forecasts_are_deterministic_cropped_and_lora_neutral(pipe):
    window = synthetic_window(height=33, width=64, steps=4)
    first = pipe.predict(window, origin=1, steps=2)
    second = pipe.predict(window, origin=1, steps=2)
    assert first["shape"] == (32, 64) and [f["lead_hours"] for f in first["forecasts"]] == [6, 12]
    assert first["forecasts"][0]["valid_time"] == "2020-01-01T12:00:00"
    assert np.array_equal(first["forecasts"][1]["surf"]["2t"], second["forecasts"][1]["surf"]["2t"])
    plain = AuroraPipeline.from_pretrained(use_lora=False)
    zero_lora = plain.predict(window, origin=1, steps=1)
    assert np.allclose(zero_lora["forecasts"][0]["surf"]["msl"], first["forecasts"][0]["surf"]["msl"], atol=1e-3)


def test_short_lora_adaptation_and_artifact_round_trip(pipe, tmp_path):
    train = synthetic_window(height=33, width=64, steps=4, name="tr")
    val = synthetic_window(height=33, width=64, steps=3, name="va", seed=3)
    result = pipe.adapt([train], val, epochs=1, lr=1e-3)
    assert (
        result["n_trainable"] == 540_672 and result["n_train_samples"] == 2 and result["history"][0]["note"].startswith("frozen")
    )
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert len(manifest["tensors"]) == 80 and all("lora" in name for name in manifest["tensors"])
    reloaded = AuroraPipeline.from_artifact(artifact)
    a = pipe.predict(val, steps=1)["forecasts"][0]["surf"]["2t"]
    b = reloaded.predict(val, steps=1)["forecasts"][0]["surf"]["2t"]
    assert np.array_equal(a, b)


def test_converted_only_layout_with_manifest_present_loads_without_the_sources(tmp_path, monkeypatch):
    """manifest present + both converted files present + both source pickles absent: the documented layout."""
    import shutil

    for name in (pl.MANIFEST_NAME, *pl.CONVERTED_SHA256):
        shutil.copy2(pl.DEFAULT_WEIGHTS_DIR / name, tmp_path / name)
    assert not any((tmp_path / name).exists() for name in (pl.SOURCE_CKPT_NAME, pl.SOURCE_STATIC_NAME))

    def refuse(*args, **kwargs):
        raise AssertionError("stage_missing_files must not run with require_source=False")

    monkeypatch.setattr(pl, "stage_missing_files", refuse)
    pipe = AuroraPipeline.from_pretrained(weights_dir=tmp_path, require_source=False)
    assert pipe.source.startswith("converted files, pinned digests")


def test_lora_plus_heads_round_trip_and_the_scope_check_on_a_real_artifact(pipe, tmp_path):
    train = synthetic_window(height=33, width=64, steps=4, name="tr")
    val = synthetic_window(height=33, width=64, steps=3, name="va", seed=3)
    result = pipe.adapt([train], val, epochs=1, lr=1e-3, trainable="lora+heads")
    assert result["n_trainable"] > 540_672
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "heads"})
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["adapter"]["trainable"] == "lora+heads" and len(manifest["tensors"]) > 80
    reloaded = AuroraPipeline.from_artifact(artifact)
    assert reloaded.use_lora and reloaded.adapter["trainable"] == "lora+heads"
    # the same artifact re-labelled as lora-only no longer matches its tensor list and is refused
    manifest["adapter"]["trainable"] = "lora"
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        AuroraPipeline.from_artifact(artifact)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    import torch

    train = synthetic_window(height=33, width=64, steps=4, name="tx")
    before = {k: v.detach().clone() for k, v in pipe.model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        pipe.adapt([train], None, epochs=2, lr=1e-3, progress=boom)
    assert pipe.adapter is None
    assert not any(p.requires_grad for p in pipe.model.parameters())
    after = pipe.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)

