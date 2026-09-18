"""Aurora 0.25° small pretrained (`microsoft/aurora`) DIMER pipeline: verified snapshot, 6-hourly global
weather forecasting from two analysis steps, lead-time evaluation against persistence, and bounded LoRA
fine-tuning of the foundation model to a user's gridded reference data with a portable adapter.

Aurora is a foundation model for the Earth system (Bodnar et al., Nature 2025): a 3D Swin transformer
backbone between a Perceiver encoder and decoder that maps two consecutive atmospheric states — four
surface variables, five atmospheric variables on pressure levels, three static fields — on an
equiangular latitude/longitude grid to the state six hours later. The "0.25° small pretrained"
checkpoint packaged here is the 113 M-parameter variant the upstream authors publish for debugging and
testing; the 1.3 B-parameter production checkpoints are the same architecture at larger width.

Two upstream assets are pickles. Under the fleet asset specification (§11) that is executable
serialization, so this package converts both once and serves neither:

* `aurora-0.25-small-pretrained.ckpt` is a torch archive whose pickle references only
  `collections.OrderedDict` and torch's tensor-rebuild helpers (verified statically by
  `audit_pickle`); it is loaded with `torch.load(weights_only=True)` — torch's restricted unpickler —
  adapted to the current parameter naming with the upstream compatibility shim, loaded into
  `AuroraSmallPretrained` with `strict=True`, and re-saved as safetensors.
* `aurora-0.25-static.pickle` is a plain pickle of three `numpy` arrays (land-sea mask, surface
  geopotential, soil type at 0.25°) referencing only `numpy…_frombuffer` and `numpy.dtype`; it is
  loaded with a `pickle.Unpickler` whose `find_class` allows exactly those two names and re-saved as
  safetensors.

Both converted files have pinned digests, and `from_pretrained` loads only them. Fidelity: the
converted model reproduces the upstream package's own regression output for the small model within
the upstream tolerances (docs/WEIGHTS.md).

Everything model-related is imported lazily so that snapshot verification, the pickle audits and input
validation run (and can refuse) before `torch` or `aurora` are imported (fleet RTM-001). `numpy` is
used for gridded data and is imported freely.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import pickle
import pickletools
import time
import warnings
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

MODEL_ID = "microsoft/aurora"
MODEL_REVISION = "a96afd7ee6d65e3bd2d476f3be798a25a56f2296"
MODEL_LICENSE = "mit"
MODEL_KEY = "aurora-0.25-small"
ARTIFACT_FORMAT = "org.valcorza.aurora-earth-system.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Immutable upstream source assets (both pickles; see docs/WEIGHTS.md).
SOURCE_CKPT_NAME = "aurora-0.25-small-pretrained.ckpt"
SOURCE_CKPT_BYTES = 451_339_106
SOURCE_CKPT_SHA256 = "f80f78de1524a9faba8c9053e4a8ce6a2114ec01cff7f7b4efe9377200d50621"
SOURCE_STATIC_NAME = "aurora-0.25-static.pickle"
SOURCE_STATIC_BYTES = 12_459_115
SOURCE_STATIC_SHA256 = "e382103f6b24bcf1f996cc0af217c71ff2fc66507a5221e1300b5017581bd318"
# Code-free serving files produced deterministically by `convert_model` (asset spec §11.2).
CONVERTED_WEIGHTS_NAME = "aurora-0.25-small-pretrained.safetensors"
CONVERTED_STATIC_NAME = "aurora-0.25-static.safetensors"
CONVERTED_SHA256 = {
    CONVERTED_WEIGHTS_NAME: "fc03b5fc5764e08e1f7a05f06ec4debb87fbb66fa2496ec203142221d0843370",
    CONVERTED_STATIC_NAME: "9bd430b666d9267aca34d5c7924d594c3474296c9e39d85701df389816303bc0",
}
CONVERTED_BYTES = {CONVERTED_WEIGHTS_NAME: 451_230_408, CONVERTED_STATIC_NAME: 12_459_136}
# Static-audit digests of the two source pickles (sorted global names), see `audit_pickle`.
PICKLE_AUDIT_SHA256 = {
    SOURCE_CKPT_NAME: "e7b998d087a5dcadd37713daf30b63cc571160c3180ebc138500ab662197e932",
    SOURCE_STATIC_NAME: "aeec283f2dbb5861afffe185d1ee13e6df5b6f8616a475b38310c0791085c27f",
}
# What each pickle may reference. The checkpoint is a state dict; the static file is three arrays.
CKPT_ALLOWED_GLOBALS = frozenset({"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage"})
STATIC_ALLOWED_GLOBALS = frozenset({"numpy.core.numeric._frombuffer", "numpy._core.numeric._frombuffer", "numpy.dtype"})

# Architecture and data-contract facts.
MODEL_CLASS = "AuroraSmallPretrained"
PARAMETER_COUNT = 112_797_584
STATE_TENSORS = 332
LORA_TENSORS = 80
LORA_PARAMETERS = 540_672
SURF_VARS: tuple[str, ...] = ("2t", "10u", "10v", "msl")
ATMOS_VARS: tuple[str, ...] = ("t", "u", "v", "q", "z")
STATIC_VARS: tuple[str, ...] = ("lsm", "z", "slt")
LEVELS: tuple[int, ...] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
TIMESTEP_HOURS = 6
HISTORY_STEPS = 2
PATCH_SIZE = 4
NATIVE_SHAPE = (721, 1440)
MIN_GRID = (17, 32)
MAX_GRID = NATIVE_SHAPE
MAX_STEPS_PER_WINDOW = 64
MAX_ROLLOUT_STEPS = 8
UNITS = {"2t": "K", "10u": "m/s", "10v": "m/s", "msl": "Pa", "t": "K", "u": "m/s", "v": "m/s", "q": "kg/kg", "z": "m²/s²"}
# Physical plausibility ranges (refusal only; nothing checks meteorological consistency).
RANGES = {
    "2t": (150.0, 350.0),
    "10u": (-150.0, 150.0),
    "10v": (-150.0, 150.0),
    "msl": (85_000.0, 110_000.0),
    "t": (150.0, 350.0),
    "u": (-300.0, 300.0),
    "v": (-300.0, 300.0),
    "q": (-1e-3, 0.1),
    "z": (-10_000.0, 250_000.0),
    "lsm": (0.0, 1.0),
    "slt": (0.0, 10.0),
}
LOSS_SCALES = {"2t": 10.0, "10u": 5.0, "10v": 5.0, "msl": 1000.0, "t": 10.0, "u": 15.0, "v": 15.0, "q": 4e-3, "z": 25_000.0}
_ISO = "%Y-%m-%dT%H:%M:%S"


# --------------------------------------------------------------------------------------------------
# manifest, staging, static pickle audits and conversion
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    for required in (SOURCE_CKPT_NAME, SOURCE_STATIC_NAME):
        if required not in listed:
            raise ValueError(f"manifest does not list {required}; refusing to proceed")
    pinned = {
        SOURCE_CKPT_NAME: (SOURCE_CKPT_BYTES, SOURCE_CKPT_SHA256),
        SOURCE_STATIC_NAME: (SOURCE_STATIC_BYTES, SOURCE_STATIC_SHA256),
    }
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] in pinned and (size, digest) != pinned[entry["path"]]:
            raise ValueError(f"{entry['path']}: manifest digest disagrees with the package constant")
    return manifest


def verify_converted(path: str | Path | None = None) -> dict[str, Any]:
    """Check the converted serving files (weights + static fields, safetensors) against the pinned digests."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    report: dict[str, Any] = {"files": []}
    for name, expected in CONVERTED_SHA256.items():
        file_path = root / name
        if not file_path.is_file():
            raise FileNotFoundError(f"converted file missing: {file_path}")
        size = file_path.stat().st_size
        if size != CONVERTED_BYTES[name]:
            raise ValueError(f"{name}: size {size} != pinned {CONVERTED_BYTES[name]}")
        digest = _sha256_file(file_path)
        if digest != expected:
            raise ValueError(f"{name}: sha256 {digest} != pinned {expected}")
        report["files"].append({"path": name, "bytes": size, "sha256": digest})
    return report


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed Hub file) and, when
    the converted serving files are present, those against the pinned digests."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _verify_manifest(root, MODEL_ID, MODEL_REVISION)
    converted = all((root / name).is_file() for name in CONVERTED_SHA256)
    if converted:
        verify_converted(root)
    return {**manifest, "converted": converted}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest and git-ignores
    the 451 MB checkpoint, the 12 MB static pickle and the safetensors they convert to)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _pickle_globals(data: bytes) -> dict[str, int]:
    """Every global a pickle stream would import, collected with `pickletools.genops` (no execution)."""
    found: dict[str, int] = {}
    stack: list[Any] = []
    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            found[key] = found.get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            found[key] = found.get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)
    return found


def audit_pickle(path: str | Path, *, allowed: frozenset[str]) -> dict[str, Any]:
    """Statically list the globals a pickle (plain, or inside a torch zip archive) would import and refuse
    any outside `allowed`. Executes nothing. Returns the sorted globals and their digest."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    data = file_path.read_bytes()
    found: dict[str, int] = {}
    nested = 0
    if data[:4] == b"PK\x03\x04":
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.endswith(".pkl"):
                nested += 1
                for key, count in _pickle_globals(archive.read(name)).items():
                    found[key] = found.get(key, 0) + count
    else:
        found = _pickle_globals(data)
    violations = sorted(name for name in found if name not in allowed)
    summary = {
        "file": file_path.name,
        "torch_archive": data[:4] == b"PK\x03\x04",
        "pickles": nested if nested else 1,
        "globals": sorted(found),
        "violations": violations,
        "audit_sha256": hashlib.sha256("\n".join(sorted(found)).encode("utf-8")).hexdigest(),
    }
    if violations:
        raise ValueError(f"{file_path.name}: pickle audit failed, globals outside the allow-list: {violations}")
    return summary


class _RestrictedUnpickler(pickle.Unpickler):
    """`find_class` limited to an exact allow-list of `module.name` strings."""

    def __init__(self, stream: Any, allowed: frozenset[str]) -> None:
        super().__init__(stream)
        self._allowed = allowed

    def find_class(self, module: str, name: str) -> Any:
        if f"{module}.{name}" not in self._allowed:
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        return super().find_class(module, name)


def _check_pinned_source(
    root: Path, name: str, size_expected: int, digest_expected: str, allowed: frozenset[str]
) -> dict[str, Any]:
    source = root / name
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    size = source.stat().st_size
    if size != size_expected:
        raise ValueError(f"{name}: size {size} != pinned {size_expected}")
    digest = _sha256_file(source)
    if digest != digest_expected:
        raise ValueError(f"{name}: sha256 {digest} != pinned {digest_expected}")
    audit = audit_pickle(source, allowed=allowed)
    if audit["audit_sha256"] != PICKLE_AUDIT_SHA256[name]:
        raise ValueError(f"{name}: pickle audit digest {audit['audit_sha256']} != pinned {PICKLE_AUDIT_SHA256[name]}")
    return {"path": name, "bytes": size, "sha256": digest, "audit": audit}


def build_model(*, use_lora: bool = False) -> Any:
    """Instantiate the small pretrained architecture from the installed `microsoft-aurora` package."""
    from aurora import AuroraSmallPretrained

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AuroraSmallPretrained(use_lora=use_lora)


def convert_model(path: str | Path | None = None) -> dict[str, Any]:
    """Convert both pinned pickles into safetensors, deterministically, after size, digest and static-audit
    checks. The checkpoint goes through torch's weights-only unpickler and the upstream compatibility shim
    into a strictly loaded model whose state dict is saved; the static fields go through a `find_class`
    allow-list of two numpy names. Returns both identities."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    ckpt = _check_pinned_source(root, SOURCE_CKPT_NAME, SOURCE_CKPT_BYTES, SOURCE_CKPT_SHA256, CKPT_ALLOWED_GLOBALS)
    static = _check_pinned_source(root, SOURCE_STATIC_NAME, SOURCE_STATIC_BYTES, SOURCE_STATIC_SHA256, STATIC_ALLOWED_GLOBALS)
    import numpy as np
    import torch
    from safetensors.torch import save_file

    started = time.perf_counter()
    state = torch.load(root / SOURCE_CKPT_NAME, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{SOURCE_CKPT_NAME} did not unpickle to a state dict of tensors")
    model = build_model(use_lora=False)
    adapted = model._adapt_checkpoint(dict(state))
    model.load_state_dict(adapted, strict=True)
    canonical = {k: v.contiguous() for k, v in model.state_dict().items()}
    if len(canonical) != STATE_TENSORS or sum(v.numel() for v in canonical.values()) != PARAMETER_COUNT:
        raise ValueError(
            f"converted state dict has {len(canonical)} tensors; expected {STATE_TENSORS} with {PARAMETER_COUNT} parameters"
        )
    save_file(canonical, str(root / CONVERTED_WEIGHTS_NAME), metadata={"format": "pt"})

    with open(root / SOURCE_STATIC_NAME, "rb") as fh:
        fields = _RestrictedUnpickler(fh, STATIC_ALLOWED_GLOBALS).load()
    if not isinstance(fields, dict) or set(fields) != set(STATIC_VARS):
        raise ValueError(f"{SOURCE_STATIC_NAME} did not unpickle to the three static fields {STATIC_VARS}")
    tensors = {}
    for name in STATIC_VARS:
        array = np.asarray(fields[name], dtype=np.float32)
        if array.shape != NATIVE_SHAPE or not np.all(np.isfinite(array)):
            raise ValueError(f"{SOURCE_STATIC_NAME}: field {name} has shape {array.shape} or non-finite values")
        tensors[name] = torch.from_numpy(np.ascontiguousarray(array))
    save_file(tensors, str(root / CONVERTED_STATIC_NAME), metadata={"format": "pt"})
    report = verify_converted(root)
    return {
        "sources": [{k: v for k, v in ckpt.items() if k != "audit"}, {k: v for k, v in static.items() if k != "audit"}],
        "audits": {SOURCE_CKPT_NAME: ckpt["audit"], SOURCE_STATIC_NAME: static["audit"]},
        "converted": report["files"],
        "seconds": round(time.perf_counter() - started, 2),
    }


# --------------------------------------------------------------------------------------------------
# gridded windows and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one window: {lat, lon, levels, times, surf: {2t, 10u, 10v, msl: (T, H, W)}, atmos: {t, u, v, q, z: (T, 13, H, W)}, "
        "static: {lsm, z, slt: (H, W)}} on a global equiangular grid"
    ),
    "grid": (
        f"H in {MIN_GRID[0]}..{MAX_GRID[0]} latitudes (90 → -90, with poles; H % 4 in (0, 1)), "
        f"W in {MIN_GRID[1]}..{MAX_GRID[1]} longitudes (0 → 360, W % 4 == 0)"
    ),
    "levels": list(LEVELS),
    "time_steps": [HISTORY_STEPS + 1, MAX_STEPS_PER_WINDOW],
    "timestep_hours": TIMESTEP_HOURS,
    "history_steps": HISTORY_STEPS,
    "rollout_steps": [1, MAX_ROLLOUT_STEPS],
    "units": dict(UNITS),
    "ranges": {k: list(v) for k, v in RANGES.items()},
    "validation": (
        "variable names, array shapes, grid monotonicity and spacing, 6-hour time spacing, finiteness and physical "
        "ranges only. Nothing checks that the fields are dynamically consistent, that the analysis is real, or that "
        "the resolution is one the model was trained on -- a smooth random field within range is forecast without complaint"
    ),
}


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        try:
            return datetime.strptime(value[:19], _ISO)
        except ValueError as exc:
            raise ValueError(f"time {value!r} is not ISO 8601 (YYYY-MM-DDTHH:MM:SS)") from exc
    raise ValueError(f"time {value!r} must be a datetime or an ISO 8601 string")


def _check_window(window: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one window and return a normalised copy (float32 arrays, lat 90→-90, lon 0→360)."""
    import numpy as np

    if not isinstance(window, Mapping):
        raise ValueError("window must be a mapping with lat/lon/levels/times/surf/atmos/static")
    for key in ("lat", "lon", "levels", "times", "surf", "atmos", "static"):
        if key not in window:
            raise ValueError(f"window is missing {key!r}")
    lat = np.asarray(window["lat"], dtype=np.float64)
    lon = np.asarray(window["lon"], dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("lat and lon must be finite 1-D arrays")
    height, width = len(lat), len(lon)
    if not MIN_GRID[0] <= height <= MAX_GRID[0] or not MIN_GRID[1] <= width <= MAX_GRID[1]:
        raise ValueError(f"grid {height}x{width} is outside {MIN_GRID}..{MAX_GRID}")
    if width % PATCH_SIZE:
        raise ValueError(f"the number of longitudes ({width}) must be a multiple of the patch size {PATCH_SIZE}")
    if height % PATCH_SIZE not in (0, 1):
        raise ValueError(f"the number of latitudes ({height}) must be a multiple of {PATCH_SIZE}, or one more")
    flip = False
    dlat = np.diff(lat)
    if np.all(dlat > 0):
        flip = True
        lat = lat[::-1]
        dlat = -dlat[::-1]
    if not np.all(dlat < 0) or not np.allclose(dlat, dlat[0], atol=1e-6):
        raise ValueError("lat must be strictly monotonic with uniform spacing")
    if not (abs(lat[0] - 90.0) < 1e-6 and abs(lat[-1] + 90.0) < 1e-6):
        raise ValueError("lat must span the poles: 90 to -90 (Aurora is a global model)")
    dlon = np.diff(lon)
    if not np.all(dlon > 0) or not np.allclose(dlon, dlon[0], atol=1e-6) or lon[0] < 0.0 or lon[-1] >= 360.0:
        raise ValueError("lon must be strictly increasing with uniform spacing within [0, 360)")
    if not np.isclose(lon[0] + 360.0 - lon[-1], dlon[0], atol=1e-6):
        raise ValueError("lon must cover the full circle (last step wraps to the first)")
    levels = [int(x) for x in np.asarray(window["levels"]).tolist()]
    if tuple(levels) != LEVELS:
        raise ValueError(f"levels must be exactly {LEVELS}, got {tuple(levels)}")
    times = [_parse_time(t) for t in window["times"]]
    n_steps = len(times)
    if not HISTORY_STEPS + 1 <= n_steps <= MAX_STEPS_PER_WINDOW:
        raise ValueError(f"a window needs {HISTORY_STEPS + 1}..{MAX_STEPS_PER_WINDOW} time steps, got {n_steps}")
    for earlier, later in zip(times, times[1:], strict=False):
        if later - earlier != timedelta(hours=TIMESTEP_HOURS):
            raise ValueError(f"times must be spaced exactly {TIMESTEP_HOURS} h apart ({earlier} -> {later})")

    def field_array(group: str, name: str, shape: tuple[int, ...]) -> Any:
        source = window[group]
        if not isinstance(source, Mapping) or name not in source:
            raise ValueError(f"{group} is missing variable {name!r}")
        try:
            array = np.asarray(source[name], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{group}/{name} must be a numeric array") from exc
        if array.shape != shape:
            raise ValueError(f"{group}/{name} has shape {array.shape}, expected {shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{group}/{name} contains non-finite values")
        low, high = RANGES.get(name if group != "static" else name, (-math.inf, math.inf))
        if group == "static" and name == "z":
            low, high = RANGES["z"]
        if float(array.min()) < low or float(array.max()) > high:
            raise ValueError(
                f"{group}/{name} has values outside the plausible range {(low, high)} {UNITS.get(name, '')}".rstrip()
            )
        if flip:
            array = array[..., ::-1, :]
        return np.ascontiguousarray(array)

    surf = {name: field_array("surf", name, (n_steps, height, width)) for name in SURF_VARS}
    atmos = {name: field_array("atmos", name, (n_steps, len(LEVELS), height, width)) for name in ATMOS_VARS}
    static = {name: field_array("static", name, (height, width)) for name in STATIC_VARS}
    unknown = sorted(set(window["surf"]) - set(SURF_VARS)) + sorted(set(window["atmos"]) - set(ATMOS_VARS))
    name = window.get("name")
    if name is not None and (not isinstance(name, str) or len(name) > 200):
        raise ValueError("name must be a string of at most 200 characters")
    return {
        "name": name if name is not None else f"window-{times[0].strftime('%Y%m%d%H')}",
        "lat": lat.tolist(),
        "lon": lon.tolist(),
        "levels": list(LEVELS),
        "times": [t.strftime(_ISO) for t in times],
        "surf": surf,
        "atmos": atmos,
        "static": static,
        "shape": (height, width),
        "n_steps": n_steps,
        "flipped_latitude": flip,
        "ignored_variables": unknown,
    }


def window_digest(window: Mapping[str, Any]) -> str:
    """SHA-256 over the grid, times and every field (float32 bytes) of a validated window."""
    checked = _check_window(window)
    digest = hashlib.sha256()
    digest.update(json.dumps({"lat": checked["lat"], "lon": checked["lon"], "times": checked["times"]}).encode("utf-8"))
    for group in ("surf", "atmos", "static"):
        for name in sorted(checked[group]):
            digest.update(name.encode("utf-8"))
            digest.update(checked[group][name].tobytes())
    return digest.hexdigest()


def validate_inputs(window: Mapping[str, Any]) -> dict[str, Any]:
    """Structural validation only; raises ValueError before any model library is imported."""
    checked = _check_window(window)
    return {
        "name": checked["name"],
        "shape": checked["shape"],
        "n_steps": checked["n_steps"],
        "times": [checked["times"][0], checked["times"][-1]],
        "forecast_origins": checked["n_steps"] - HISTORY_STEPS,
        "flipped_latitude": checked["flipped_latitude"],
        "ignored_variables": checked["ignored_variables"],
        "resolution_degrees": round(360.0 / checked["shape"][1], 4),
        "native_resolution": checked["shape"] == NATIVE_SHAPE,
    }


# --------------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------------


def _lat_weights(lat: Sequence[float]) -> Any:
    import numpy as np

    weights = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return weights / weights.mean()


@dataclass
class AuroraPipeline:
    """6-hourly global forecasting and bounded LoRA fine-tuning on top of the verified Aurora small model."""

    model: Any
    device: str
    weights_dir: Path
    source: str
    use_lora: bool
    adapter: dict[str, Any] | None = None
    _static_native: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str = "cpu",
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
        use_lora: bool = False,
        report: Callable[[dict[str, Any]], None] | None = None,
    ) -> AuroraPipeline:
        """Verify, convert if needed, build from the installed package and strictly load. `use_lora=True`
        adds the (zero-initialised, behaviour-preserving) LoRA parameters that `adapt` trains. With
        `require_source=False` the pickles may be absent (the DIMER-hosted case) as long as the converted
        files verify. `report` receives the audit and conversion records when a conversion happens."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if require_source or (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            snapshot = verify_snapshot(root)
            if not snapshot["converted"]:
                conversion = convert_model(root)
                if report is not None:
                    report({"conversion": conversion})
                snapshot = verify_snapshot(root)
            elif report is not None:
                report({"conversion": "converted files already present and digest-verified"})
            source = "converted from the manifest-verified source pickles"
        else:
            verify_converted(root)
            source = "converted files, pinned digests (source pickles absent)"
        import torch
        from safetensors.torch import load_file

        model = build_model(use_lora=use_lora)
        state = load_file(str(root / CONVERTED_WEIGHTS_NAME))
        result = model.load_state_dict(state, strict=not use_lora)
        if use_lora and (
            result.unexpected_keys
            or any("lora" not in k for k in result.missing_keys)
            or len(result.missing_keys) != LORA_TENSORS
        ):
            raise ValueError(
                f"unexpected state-dict layout with LoRA: missing={len(result.missing_keys)} unexpected={result.unexpected_keys}"
            )
        n_params = sum(p.numel() for n, p in model.named_parameters() if "lora" not in n)
        if n_params != PARAMETER_COUNT:
            raise ValueError(f"rebuilt model has {n_params} base parameters, expected {PARAMETER_COUNT}")
        model.to(torch.device(device)).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        static = load_file(str(root / CONVERTED_STATIC_NAME))
        return cls(model=model, device=device, weights_dir=root, source=source, use_lora=use_lora, _static_native=static)

    # ---- batches --------------------------------------------------------------------------------------

    def native_static_fields(self) -> dict[str, Any]:
        """The upstream 0.25° static fields (lsm, z, slt) as numpy arrays of shape (721, 1440), lat 90→-90."""
        return {k: v.numpy() for k, v in self._static_native.items()}

    def _batch(self, checked: Mapping[str, Any], origin: int) -> Any:
        """Model input for the forecast origin `origin` (the step index of the latest analysis)."""
        import torch
        from aurora import Batch, Metadata

        if origin < HISTORY_STEPS - 1 or origin >= checked["n_steps"]:
            raise ValueError(f"origin must be in {HISTORY_STEPS - 1}..{checked['n_steps'] - 1}")
        lo, hi = origin - HISTORY_STEPS + 1, origin + 1
        return Batch(
            surf_vars={k: torch.from_numpy(v[lo:hi][None]) for k, v in checked["surf"].items()},
            static_vars={k: torch.from_numpy(v) for k, v in checked["static"].items()},
            atmos_vars={k: torch.from_numpy(v[lo:hi][None]) for k, v in checked["atmos"].items()},
            metadata=Metadata(
                lat=torch.tensor(checked["lat"], dtype=torch.float32),
                lon=torch.tensor(checked["lon"], dtype=torch.float32),
                time=(_parse_time(checked["times"][origin]),),
                atmos_levels=tuple(LEVELS),
            ),
        )

    def _forecast(self, batch: Any, steps: int) -> list[Any]:
        """Autoregressive roll-out of `steps` × 6 h; each element is the model's Batch for that lead."""
        import torch
        from aurora import rollout

        with torch.inference_mode():
            return [pred.to("cpu") for pred in rollout(self.model, batch.to(self.device), steps=steps)]

    # ---- inference ------------------------------------------------------------------------------------

    def predict(self, window: Mapping[str, Any], *, origin: int | None = None, steps: int = 1) -> dict[str, Any]:
        """Forecast `steps` × 6 h from the analysis at `origin` (default: the last step of the window)."""
        if not isinstance(steps, int) or not 1 <= steps <= MAX_ROLLOUT_STEPS:
            raise ValueError(f"steps must be an int in 1..{MAX_ROLLOUT_STEPS}")
        checked = _check_window(window)
        if origin is None:
            origin = checked["n_steps"] - 1
        started = time.perf_counter()
        batch = self._batch(checked, origin)
        preds = self._forecast(batch, steps)
        origin_time = _parse_time(checked["times"][origin])
        forecasts = []
        for k, pred in enumerate(preds, start=1):
            forecasts.append(
                {
                    "lead_hours": k * TIMESTEP_HOURS,
                    "valid_time": (origin_time + timedelta(hours=k * TIMESTEP_HOURS)).strftime(_ISO),
                    "surf": {name: pred.surf_vars[name][0, 0].numpy() for name in SURF_VARS},
                    "atmos": {name: pred.atmos_vars[name][0, 0].numpy() for name in ATMOS_VARS},
                }
            )
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "name": checked["name"],
            "origin_time": origin_time.strftime(_ISO),
            "shape": (len(preds[0].metadata.lat), len(preds[0].metadata.lon)),
            "lat": preds[0].metadata.lat.tolist(),
            "lon": preds[0].metadata.lon.tolist(),
            "units": dict(UNITS),
            "forecasts": forecasts,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def evaluate(
        self, window: Mapping[str, Any], *, max_lead_steps: int = 1, origins: Sequence[int] | None = None
    ) -> dict[str, Any]:
        """Latitude-weighted RMSE per variable and lead time against the window's later analyses, with
        the persistence forecast (the latest analysis carried forward) scored the same way."""
        from .metrics import forecast_metrics

        if not isinstance(max_lead_steps, int) or not 1 <= max_lead_steps <= MAX_ROLLOUT_STEPS:
            raise ValueError(f"max_lead_steps must be an int in 1..{MAX_ROLLOUT_STEPS}")
        checked = _check_window(window)
        last_origin = checked["n_steps"] - 1 - max_lead_steps
        if last_origin < HISTORY_STEPS - 1:
            raise ValueError(
                f"window has {checked['n_steps']} steps; {HISTORY_STEPS + max_lead_steps} are needed "
                f"for {max_lead_steps} lead steps"
            )
        chosen = list(origins) if origins is not None else list(range(HISTORY_STEPS - 1, last_origin + 1))
        for origin in chosen:
            if not HISTORY_STEPS - 1 <= origin <= last_origin:
                raise ValueError(f"origin {origin} is outside {HISTORY_STEPS - 1}..{last_origin}")
        started = time.perf_counter()
        predictions = []
        for origin in chosen:
            preds = self._forecast(self._batch(checked, origin), max_lead_steps)
            predictions.append(
                [
                    {
                        "surf": {n: p.surf_vars[n][0, 0].numpy() for n in SURF_VARS},
                        "atmos": {n: p.atmos_vars[n][0, 0].numpy() for n in ATMOS_VARS},
                    }
                    for p in preds
                ]
            )
        metrics = forecast_metrics(checked, chosen, predictions)
        metrics["seconds"] = round(time.perf_counter() - started, 3)
        return metrics

    # ---- adaptation -----------------------------------------------------------------------------------

    def _trainable(self, mode: str) -> list[str]:
        if mode not in ("lora", "lora+heads"):
            raise ValueError("trainable must be 'lora' or 'lora+heads'")
        if not self.use_lora:
            raise ValueError("adapt() needs a pipeline built with use_lora=True")
        prefixes = ("decoder.surf_heads.", "decoder.atmos_heads.", "encoder.surf_token_embeds.", "encoder.atmos_token_embeds.")
        names = []
        for name, _param in self.model.named_parameters():
            if "lora" in name or (mode == "lora+heads" and name.startswith(prefixes)):
                names.append(name)
        return names

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Mapping[str, Any] | None = None,
        *,
        epochs: int = 6,
        lr: float = 1e-3,
        trainable: str = "lora",
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning on one-step (6 h) forecasts from every origin of every training window.

        `trainable="lora"` trains the 80 LoRA tensors (540,672 parameters) the upstream architecture
        provides in every backbone attention block, zero-initialised so epoch 0 is the pretrained model;
        `"lora+heads"` also unfreezes the encoder token embeddings and decoder heads. Loss = mean over the
        nine variables of MSE / scale², AdamW, fixed learning rate, one origin per step. The epoch with the
        lowest validation loss is kept; epoch 0 records the frozen model."""
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 0.1):
            raise ValueError("lr must be in (0, 0.1]")
        if not train:
            raise ValueError("at least one training window is required")
        names = self._trainable(trainable)
        train_checked = [_check_window(w) for w in train]
        val_checked = _check_window(val) if val is not None else None
        import torch

        torch.manual_seed(seed)
        started = time.perf_counter()
        model = self.model
        for name, param in model.named_parameters():
            param.requires_grad_(name in set(names))
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr)
        scales = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in LOSS_SCALES.items()}

        def targets(checked: Mapping[str, Any], origin: int) -> dict[str, Any]:
            height = checked["shape"][0] - (checked["shape"][0] % PATCH_SIZE)
            return {
                **{k: torch.from_numpy(v[origin + 1, :height]).to(self.device) for k, v in checked["surf"].items()},
                **{k: torch.from_numpy(v[origin + 1, :, :height]).to(self.device) for k, v in checked["atmos"].items()},
            }

        def loss_of(pred: Any, target: Mapping[str, Any]) -> Any:
            total = 0.0
            for k in SURF_VARS:
                total = total + torch.mean(((pred.surf_vars[k][0, 0] - target[k]) / scales[k]) ** 2)
            for k in ATMOS_VARS:
                total = total + torch.mean(((pred.atmos_vars[k][0, 0] - target[k]) / scales[k]) ** 2)
            return total / (len(SURF_VARS) + len(ATMOS_VARS))

        def val_loss() -> float | None:
            if val_checked is None:
                return None
            model.eval()
            losses = []
            with torch.inference_mode():
                for origin in range(HISTORY_STEPS - 1, val_checked["n_steps"] - 1):
                    pred = model.forward(self._batch(val_checked, origin).to(self.device))
                    losses.append(float(loss_of(pred, targets(val_checked, origin))))
            return sum(losses) / len(losses)

        history: list[dict[str, Any]] = []
        best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items() if k in set(names)})
        best_epoch = 0
        entry: dict[str, Any] = {
            "epoch": 0,
            "train_loss": None,
            "val_loss": val_loss(),
            "note": "frozen model (LoRA zero-initialised)",
        }
        if val_checked is not None:
            entry["val"] = self.evaluate(val_checked)["variables"]
        history.append(entry)
        best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
        if progress:
            progress(entry)
        samples = [(w, origin) for w in train_checked for origin in range(HISTORY_STEPS - 1, w["n_steps"] - 1)]
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(samples), generator=generator).tolist()
            losses = []
            for index in order:
                checked, origin = samples[index]
                pred = model.forward(self._batch(checked, origin).to(self.device))
                loss = loss_of(pred, targets(checked, origin))
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                losses.append(float(loss.detach()))
            model.eval()
            entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": val_loss()}
            if val_checked is not None:
                entry["val"] = self.evaluate(val_checked)["variables"]
            history.append(entry)
            if progress:
                progress(entry)
            if entry["val_loss"] is None or entry["val_loss"] < best_val:
                best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in set(names)}
                best_epoch = epoch
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable": trainable,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "loss_scales": dict(LOSS_SCALES),
            "n_train_windows": len(train_checked),
            "n_train_samples": len(samples),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted tensors (LoRA, plus heads when trained) as safetensors with a manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "converted_sha256": dict(CONVERTED_SHA256),
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest and digest, then overwrite exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        if base.get("converted_sha256") != CONVERTED_SHA256:
            raise ValueError("artifact records different converted-base digests")
        entry = manifest["files"][0]
        weights_path = root / entry["path"]
        digest = _sha256_file(weights_path)
        if digest != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        if any("lora" in name for name in manifest["tensors"]) and not self.use_lora:
            raise ValueError("this adapter carries LoRA tensors; build the pipeline with use_lora=True")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != manifest["tensors"]:
            raise ValueError("artifact tensor names differ from its manifest")
        state = self.model.state_dict()
        for key, value in tensors.items():
            if key not in state:
                raise ValueError(f"artifact tensor {key} is not part of the model")
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, model has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str = "cpu",
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
    ) -> AuroraPipeline:
        manifest = json.loads((Path(artifact_dir) / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        use_lora = any("lora" in name for name in manifest.get("tensors", []))
        pipeline = cls.from_pretrained(
            device=device,
            weights_dir=weights_dir,
            allow_download=allow_download,
            require_source=require_source,
            use_lora=use_lora,
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
