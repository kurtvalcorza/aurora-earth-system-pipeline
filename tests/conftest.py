import builtins
from datetime import datetime, timedelta

import numpy as np
import pytest

MODEL_LIBRARIES = {"torch", "aurora", "safetensors", "huggingface_hub", "timm", "einops"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_window(
    *, height: int = 17, width: int = 32, steps: int = 4, seed: int = 0, ascending_lat: bool = False, name: str = "synthetic"
):
    """A small, physically plausible, smooth window on a global grid (no model semantics)."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(90.0, -90.0, height)
    lon = np.linspace(0.0, 360.0, width, endpoint=False)
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    base = datetime(2020, 1, 1)
    times = [(base + timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(steps)]
    cosl = np.cos(np.deg2rad(lat))[:, None]
    trend = np.arange(steps, dtype=np.float32)[:, None, None] * 0.5

    def surf_field(mean, amp):
        drift = trend * amp * 0.05
        return (mean + amp * cosl + drift + rng.normal(0, amp * 0.05, (steps, height, width))).astype(np.float32)

    def atmos_field(profile, amp):
        prof = np.asarray(profile, dtype=np.float32)[None, :, None, None]
        drift = trend[:, None] * amp * 0.05
        noise = rng.normal(0, amp * 0.05, (steps, 13, height, width))
        return (prof + amp * cosl[None, None] + drift + noise).astype(np.float32)

    window = {
        "name": name,
        "lat": (lat[::-1] if ascending_lat else lat).tolist(),
        "lon": lon.tolist(),
        "levels": levels,
        "times": times,
        "surf": {
            "2t": surf_field(260.0, 30.0),
            "10u": surf_field(0.0, 5.0),
            "10v": surf_field(0.0, 5.0),
            "msl": surf_field(100_000.0, 1_500.0),
        },
        "atmos": {
            "t": atmos_field(np.linspace(215.0, 290.0, 13), 20.0),
            "u": atmos_field(np.linspace(20.0, 2.0, 13), 10.0),
            "v": atmos_field(np.zeros(13), 8.0),
            "q": atmos_field(np.linspace(1e-6, 8e-3, 13), 2e-3),
            "z": atmos_field(np.linspace(200_000.0, 1_000.0, 13), 3_000.0),
        },
        "static": {
            "lsm": (rng.random((height, width)) > 0.7).astype(np.float32),
            "z": (rng.random((height, width)) * 5_000.0).astype(np.float32),
            "slt": rng.integers(0, 7, (height, width)).astype(np.float32),
        },
    }
    if ascending_lat:
        for group in ("surf", "atmos", "static"):
            window[group] = {k: np.ascontiguousarray(v[..., ::-1, :]) for k, v in window[group].items()}
    return window
