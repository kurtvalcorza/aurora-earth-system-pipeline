"""Tutorial dataset, validation, extended-NetCDF I/O for the Aurora pipeline.

The default dataset is **real ERA5 reanalysis** at 1.5° (240 × 121, with poles), served by WeatherBench 2
(Rasp et al., 2024) from a public Google Cloud Storage bucket as a Zarr v2 store. Four two-day windows
(eight 6-hourly analyses each) are fetched: two for training (January and July 2019), one for validation
(April 2020) and one for testing (October 2021) — distinct seasons and years, so nothing in the test
window is a near-duplicate of anything trained on. Every object fetched (chunk, `.zarray`, coordinate)
has its byte size and SHA-256 pinned in `WB2_OBJECTS`; a mismatch is refused before decoding, and the
Zarr chunks are decoded with `numcodecs` (Blosc/LZ4) directly, without a Zarr or xarray dependency on
the download path. About 196 MB is transferred.

Why 1.5° and not the model's native 0.25°: a native-resolution window is 60× larger (about 430 MB per
atmospheric variable and window), far beyond a tutorial's budget, and the small checkpoint is the one
upstream publishes for testing. Running the 0.25° model on a 1.5° grid is therefore an out-of-distribution
use, and the tutorial says so: the frozen model's error is measured honestly against persistence, and the
adaptation contract is what moves it. ERA5 is produced by ECMWF/Copernicus (licence to use Copernicus
products, attribution required); WeatherBench 2 redistributes it unchanged apart from regridding.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .pipeline import ATMOS_VARS, HISTORY_STEPS, STATIC_VARS, SURF_VARS, TIMESTEP_HOURS, _check_window, window_digest

WB2_BASE_URL = (
    "https://storage.googleapis.com/weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
WB2_EPOCH = datetime(1959, 1, 1)
WB2_CHUNK_STEPS = 8
WB2_SURF = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
WB2_ATMOS = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
WB2_STATIC = {"lsm": "land_sea_mask", "z": "geopotential_at_surface", "slt": "soil_type"}
# Chunk index -> role. Chunk k holds analyses 8k .. 8k+7 (6-hourly from 1959-01-01 00:00).
SAMPLE_WINDOWS: dict[str, int] = {"train-2019-01": 10957, "train-2019-07": 11048, "val-2020-04": 11185, "test-2021-10": 11459}
SAMPLE_LABEL_SOURCE = "ERA5 via WeatherBench 2 (1.5°, GCS bucket weatherbench2)"
DEFAULT_CACHE_DIR = Path("weights") / "wb2-era5-1p5deg"
MIN_WINDOWS = 1

WB2_OBJECTS: dict[str, tuple[int, str]] = {
    "latitude/.zarray": (317, "1576147d9e73e403a5558d7030613586fe94e0ca1c7c9e9e14fb393b0724c8aa"),
    "longitude/.zarray": (317, "eb41dc00d27b72577623ec2ddccb7693006f14cdc9f06a81f2a93b94fa9dd10e"),
    "level/.zarray": (314, "8217857a6c10d13b295ab6ffd491092a75ac976d5af0a4e8d8f5964a61c56f84"),
    "10m_u_component_of_wind/.zarray": (369, "6372c239b61980a3807f4f7138539f9cf97e5b08eed8ab5d2f6b357a79827907"),
    "10m_u_component_of_wind/10957.0.0": (817882, "1eaf8efe88fe1c2eb8f4f7fae0c23dc587312ff1255ffe04af6afd259aacffc0"),
    "10m_u_component_of_wind/11048.0.0": (817202, "ec90cb193bdbcd316c9ed37ae70cf9e680129fb361a9ac947ff78876d6eb8bb5"),
    "10m_u_component_of_wind/11185.0.0": (813825, "3aee927c30dc6101a09cfc7f970d9edca936cf1c522033d1c89cb374cbf49fbd"),
    "10m_u_component_of_wind/11459.0.0": (817758, "16e05fa2291d35775036e2f2feebfc7541dbaf56ac96562218460e82f9c62069"),
    "10m_v_component_of_wind/.zarray": (369, "6372c239b61980a3807f4f7138539f9cf97e5b08eed8ab5d2f6b357a79827907"),
    "10m_v_component_of_wind/10957.0.0": (814637, "6bd14583e1f00014aeca0324c006860c66a994557595fa27d5d44aef8b03fdc5"),
    "10m_v_component_of_wind/11048.0.0": (818216, "ec5e60a9db34773b23b1a5a79bf9ff66148c10016febddbecb02e2cf2a29f8d5"),
    "10m_v_component_of_wind/11185.0.0": (819793, "0ed0783d4abdcdc85ce456f38700761af89f7b3f1a93b31fe537f140b1f2b365"),
    "10m_v_component_of_wind/11459.0.0": (817993, "efe882d2300adada80b870cb89c540690f6f64c92705deabfb084cc11ed95680"),
    "2m_temperature/.zarray": (369, "6372c239b61980a3807f4f7138539f9cf97e5b08eed8ab5d2f6b357a79827907"),
    "2m_temperature/10957.0.0": (615801, "13dd6303a537adec10cf5bd9446e2f595c9b62a1c26f586233978c647173a0db"),
    "2m_temperature/11048.0.0": (611125, "3e139f475ccb6f905344723e961c0a15c9fc037d67c21a511804300de4b50917"),
    "2m_temperature/11185.0.0": (616959, "3d0b090fe11d580a5b8a7dc9b6573505bddf3a6705b4d65f34ebec467300a354"),
    "2m_temperature/11459.0.0": (609456, "9fdffb77031f8372544c68e43d9431b3c26b86b0396ce13ff91a552d6c9bfb38"),
    "geopotential/.zarray": (393, "61cfdce172da056ef17cf80bd9e30d89ddff268f848813b21b488df8e8da270d"),
    "geopotential/10957.0.0.0": (7685214, "1e978f48ee0d2808ad357d2ba6d82804b5b3e20867c874e3ec1223150da33f8a"),
    "geopotential/11048.0.0.0": (7651568, "8136af69c379816caca6986615d52abfc983d6f06821f8555d12c5f61ca0b877"),
    "geopotential/11185.0.0.0": (7683052, "c6d241d6c2d04949b1cd0333634f934f82ad51d0fbe2d569e5ea1f11cd62c3d2"),
    "geopotential/11459.0.0.0": (7670669, "b55a98729b3b4ad906c4d272a82e509f850152449d882586e98c8290d75d1ecc"),
    "geopotential_at_surface/.zarray": (343, "757b13432c66096257f584133a71e5308134547aef9dbc6505ef553b2d2e8d67"),
    "geopotential_at_surface/0.0": (107690, "2df9d20c03bceb688bd1522a7c5fb154c15e01b4375c9dfe3f7fdb5348edbcbe"),
    "land_sea_mask/.zarray": (343, "757b13432c66096257f584133a71e5308134547aef9dbc6505ef553b2d2e8d67"),
    "land_sea_mask/0.0": (52875, "dcd229c9d152d5cdf422defe2bb1d2c425cfc52832078cb1b0600875f9d9a5ef"),
    "latitude/0": (528, "0099a8a0cfec2eb3e7d3c1a5a0b00b063c86ba0fe98bdc379bfe269af0207875"),
    "level/0": (120, "defe6a82a653349d9bdc64bbb6cf092e3d680d913a7f5ae7a584289df37edb79"),
    "longitude/0": (845, "127c574174be6960bd1abe3d7f6259b9b9a85fad751df91ae1e6c0c469bb0beb"),
    "mean_sea_level_pressure/.zarray": (369, "6372c239b61980a3807f4f7138539f9cf97e5b08eed8ab5d2f6b357a79827907"),
    "mean_sea_level_pressure/10957.0.0": (553842, "23530ea6b4aa7a081b6f344fce9e5aab388f695ce485ada5af172807a69231b3"),
    "mean_sea_level_pressure/11048.0.0": (555456, "727cc3a840d305eb4cc4da60d0c8cadad0634358394d21c10259c26dbf0e20dd"),
    "mean_sea_level_pressure/11185.0.0": (555386, "2c1c4870239f435dd7d053a333059167b7e41bc6a481e6f52c61ad6bbea1df62"),
    "mean_sea_level_pressure/11459.0.0": (554365, "5ce8ae2cb07ee901da9784060a533946c6ae3ce3f80030c02e0789e2f596dabd"),
    "soil_type/.zarray": (343, "757b13432c66096257f584133a71e5308134547aef9dbc6505ef553b2d2e8d67"),
    "soil_type/0.0": (47405, "7300007695fff5062fa5981260d5253cc068a1e5b6f3eab40ae08bd270b2aaeb"),
    "specific_humidity/.zarray": (393, "61cfdce172da056ef17cf80bd9e30d89ddff268f848813b21b488df8e8da270d"),
    "specific_humidity/10957.0.0.0": (9745439, "72b787c6bfbc5da541e1d0e0792365df85d99878d930e1042a068a21cd27963d"),
    "specific_humidity/11048.0.0.0": (9803016, "dacee098673741378a78ad021b093e2138e200c6caf7d312d51f3faa80b2c575"),
    "specific_humidity/11185.0.0.0": (9704312, "7eee37cd5002da29d49b52b8237c0440415a219b8bb729d7e07714477e82bfdf"),
    "specific_humidity/11459.0.0.0": (9832381, "9e73a10c98d9da9cc50be7c5bb89e93159bdcdbe989bb1ea98880c55e97cf17c"),
    "temperature/.zarray": (393, "61cfdce172da056ef17cf80bd9e30d89ddff268f848813b21b488df8e8da270d"),
    "temperature/10957.0.0.0": (7806723, "de314034a300736ff2224f92f8c32f64c66791adfed6cfa4388d6ea99adb744b"),
    "temperature/11048.0.0.0": (7833418, "5f1c3f9545df154d8b55aea5412aa5f4c813e85eb97f9ecdba4ecc7cd30ce38b"),
    "temperature/11185.0.0.0": (7860352, "3c94aa8bd87565bcc925c9bd334db7bfcaac16c1362a857bafe8ac9cdb2c5aad"),
    "temperature/11459.0.0.0": (7808917, "3541ff21a48367c0f4234f5636d95852be6063ccef60d6899a124ae6d42feb18"),
    "u_component_of_wind/.zarray": (393, "61cfdce172da056ef17cf80bd9e30d89ddff268f848813b21b488df8e8da270d"),
    "u_component_of_wind/10957.0.0.0": (10334472, "dba4740ec5d28cb60d2e062480897987851387cd5e0464c85dc06b2a074e9f5d"),
    "u_component_of_wind/11048.0.0.0": (10374965, "7c31a0e5421dcca225d6731eb20e4b62e07cc3d0e4bcc75353d4618e1bc849ee"),
    "u_component_of_wind/11185.0.0.0": (10308600, "b1396833d45d942c4f963305622647cf51b555abfc475f11b3c1516c473dddb8"),
    "u_component_of_wind/11459.0.0.0": (10376135, "c365e4e8fdcb00626693219ed03e32310b3a2bfcb83a0fadf7bfff28414042d2"),
    "v_component_of_wind/.zarray": (393, "61cfdce172da056ef17cf80bd9e30d89ddff268f848813b21b488df8e8da270d"),
    "v_component_of_wind/10957.0.0.0": (10483018, "aa5a30b21d57b56d17bab0f173f4c085cf87f26220def8d099868681a9c30efd"),
    "v_component_of_wind/11048.0.0.0": (10541142, "8fc0d4390f04c7a56875c395ebcd123c32d500682cb527bfe887359cf0cd6219"),
    "v_component_of_wind/11185.0.0.0": (10516124, "5184bd1c8e64b582d74e6efd645b8cd5747431f8caa50e1fab86e5beb14955cc"),
    "v_component_of_wind/11459.0.0.0": (10519849, "1a43b22672a16c5d08ef5a93a13abfdaf659edd22fa365c3a12fb1fdec90b007"),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_object(key: str, *, cache_dir: str | Path | None = None, fetcher: Any = None) -> bytes:
    """Return the bytes of one pinned WeatherBench 2 object, from the cache or the bucket, digest-verified."""
    if key not in WB2_OBJECTS:
        raise ValueError(f"{key} is not a pinned WeatherBench 2 object")
    size, digest = WB2_OBJECTS[key]
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / key.replace("/", "__")
    if local.is_file():
        data = local.read_bytes()
        if len(data) == size and _sha256(data) == digest:
            return data
    if fetcher is not None:
        data = fetcher(key)
    else:
        with urllib.request.urlopen(f"{WB2_BASE_URL}/{key}", timeout=180) as response:  # noqa: S310 (pinned https URL)
            data = response.read()
    if len(data) != size or _sha256(data) != digest:
        raise ValueError(f"{key}: fetched {len(data)} bytes with sha256 {_sha256(data)[:16]}…, pinned {size} / {digest[:16]}…")
    local.write_bytes(data)
    return data


def _decode(key: str, array_key: str, **kwargs: Any) -> Any:
    """Decode one Zarr v2 chunk (Blosc) into a numpy array shaped like its `.zarray` chunks."""
    import numpy as np
    from numcodecs import Blosc

    meta = json.loads(fetch_object(array_key, **kwargs))
    if meta.get("compressor", {}).get("id") != "blosc" or meta.get("filters") or meta.get("order") != "C":
        raise ValueError(f"{array_key}: unexpected Zarr encoding {meta.get('compressor')}")
    raw = fetch_object(key, **kwargs)
    return np.frombuffer(Blosc().decode(raw), dtype=meta["dtype"]).reshape(meta["chunks"])


def fetch_sample_window(name: str, *, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, Any]:
    """Assemble one named tutorial window (lat 90→-90, lon 0→360, 8 analyses) from pinned objects."""
    if name not in SAMPLE_WINDOWS:
        raise ValueError(f"window must be one of {sorted(SAMPLE_WINDOWS)}")
    import numpy as np

    kw = {"cache_dir": cache_dir, "fetcher": fetcher}
    chunk = SAMPLE_WINDOWS[name]
    lat = _decode("latitude/0", "latitude/.zarray", **kw)
    lon = _decode("longitude/0", "longitude/.zarray", **kw)
    levels = _decode("level/0", "level/.zarray", **kw)
    first = chunk * WB2_CHUNK_STEPS  # the store's time axis is uniform: hours since 1959-01-01 in steps of 6 (verified at build)
    times = [WB2_EPOCH + timedelta(hours=TIMESTEP_HOURS * (first + j)) for j in range(WB2_CHUNK_STEPS)]
    # WB2 stores (time, [level,] longitude, latitude) with latitude ascending; Aurora wants (…, lat, lon), lat descending.
    surf = {
        k: np.ascontiguousarray(_decode(f"{v}/{chunk}.0.0", f"{v}/.zarray", **kw).transpose(0, 2, 1)[:, ::-1, :])
        for k, v in WB2_SURF.items()
    }
    atmos = {
        k: np.ascontiguousarray(_decode(f"{v}/{chunk}.0.0.0", f"{v}/.zarray", **kw).transpose(0, 1, 3, 2)[:, :, ::-1, :])
        for k, v in WB2_ATMOS.items()
    }
    static = {k: np.ascontiguousarray(_decode(f"{v}/0.0", f"{v}/.zarray", **kw).T[::-1, :]) for k, v in WB2_STATIC.items()}
    window = {
        "name": name,
        "lat": lat[::-1].tolist(),
        "lon": lon.tolist(),
        "levels": [int(x) for x in levels],
        "times": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
        "surf": surf,
        "atmos": atmos,
        "static": static,
        "source": SAMPLE_LABEL_SOURCE,
    }
    return _check_window(window) | {"source": SAMPLE_LABEL_SOURCE}


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, dict[str, Any]]:
    """All four tutorial windows, keyed by role name."""
    return {name: fetch_sample_window(name, cache_dir=cache_dir, fetcher=fetcher) for name in SAMPLE_WINDOWS}


def validate_dataset(
    windows: Sequence[Mapping[str, Any]], *, min_windows: int = MIN_WINDOWS, min_steps: int = HISTORY_STEPS + 1
) -> dict[str, Any]:
    """Structural validation of a list of windows; raises ValueError before any model import."""
    if isinstance(windows, Mapping) or not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise ValueError("windows must be a list of window mappings")
    if len(windows) < min_windows:
        raise ValueError(f"{len(windows)} windows; at least {min_windows} are required")
    checked = []
    names: set[str] = set()
    shape = None
    for window in windows:
        c = _check_window(window)
        if c["n_steps"] < min_steps:
            raise ValueError(f"{c['name']}: {c['n_steps']} steps; at least {min_steps} are required")
        if c["name"] in names:
            raise ValueError(f"duplicate window name {c['name']!r}")
        names.add(c["name"])
        if shape is None:
            shape = c["shape"]
        elif c["shape"] != shape:
            raise ValueError(f"{c['name']}: grid {c['shape']} differs from {shape}; all windows must share one grid")
        checked.append(c)
    return {
        "windows": checked,
        "n_windows": len(checked),
        "shape": shape,
        "n_steps": [c["n_steps"] for c in checked],
        "forecast_origins": sum(c["n_steps"] - HISTORY_STEPS for c in checked),
        "time_span": [checked[0]["times"][0], checked[-1]["times"][-1]],
        "digest": dataset_digest(checked),
    }


def dataset_digest(windows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(window_digest(w) for w in windows).encode("utf-8")).hexdigest()


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read a NetCDF file into one window: coordinates `latitude`/`longitude`/`level`/`time`, surface
    variables as (time, lat, lon), atmospheric variables as (time, level, lat, lon), static fields as
    (lat, lon), all under their Aurora short names (`2t`, `10u`, `10v`, `msl`, `t`, `u`, `v`, `q`, `z`,
    `lsm`, `z_static` or `z` in `static_*`, `slt`) — the shape `write_window_netcdf` produces."""
    import numpy as np
    import xarray as xr

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    if file_path.suffix.lower() not in (".nc", ".nc4", ".netcdf"):
        raise ValueError("BYOD datasets must be NetCDF (.nc)")
    with xr.open_dataset(file_path) as ds:
        for coord in ("latitude", "longitude", "level", "time"):
            if coord not in ds.coords and coord not in ds.variables:
                raise ValueError(f"NetCDF is missing coordinate {coord!r}")
        times = [datetime.strptime(str(np.datetime_as_string(t, unit="s")), "%Y-%m-%dT%H:%M:%S") for t in ds["time"].values]
        window = {
            "name": file_path.stem,
            "lat": ds["latitude"].values.tolist(),
            "lon": ds["longitude"].values.tolist(),
            "levels": [int(x) for x in ds["level"].values],
            "times": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "surf": {k: np.asarray(ds[k].values) for k in SURF_VARS if k in ds},
            "atmos": {k: np.asarray(ds[k].values) for k in ATMOS_VARS if k in ds},
            "static": {k: np.asarray(ds[f"static_{k}"].values) for k in STATIC_VARS if f"static_{k}" in ds},
        }
    return [window]


def write_window_netcdf(window: Mapping[str, Any], path: str | Path) -> Path:
    """Write a validated window as NetCDF in the shape `load_byod_dataset` reads."""
    import numpy as np
    import xarray as xr

    c = _check_window(window)
    coords = {
        "time": np.array([np.datetime64(t) for t in c["times"]]),
        "level": np.array(c["levels"], dtype=np.int64),
        "latitude": np.array(c["lat"], dtype=np.float64),
        "longitude": np.array(c["lon"], dtype=np.float64),
    }
    data = {}
    for k, v in c["surf"].items():
        data[k] = (("time", "latitude", "longitude"), v)
    for k, v in c["atmos"].items():
        data[k] = (("time", "level", "latitude", "longitude"), v)
    for k, v in c["static"].items():
        data[f"static_{k}"] = (("latitude", "longitude"), v)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(data, coords=coords, attrs={"source": str(window.get("source", "")), "timestep_hours": TIMESTEP_HOURS}).to_netcdf(
        out
    )
    return out
