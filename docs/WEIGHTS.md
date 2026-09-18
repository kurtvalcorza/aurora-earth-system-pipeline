# Weight provenance, the pickle audits, the conversion, the fidelity check and DIMER hosting

This repository pins **one** snapshot with its own `dimer-base-manifest.json`. Two of its three entries are pickles, which this pipeline audits and converts but never serves.

## Aurora 0.25° small pretrained weights and static fields

- Upstream: `microsoft/aurora`
- Immutable revision: `a96afd7ee6d65e3bd2d476f3be798a25a56f2296` (2026-07-23, "Add Aurora 1.5"); the small checkpoint was first published in commit `0be7e57c685dac86b78c4a19a3ab149d13c6a3dd` (2024-08-21, "Upload 8 files"), which is also the `default_checkpoint_revision` of `AuroraSmallPretrained` in the upstream package; the file bytes are identical at both revisions.
- Source formats: `aurora-0.25-small-pretrained.ckpt` — torch zip archive (`aurora-0.25-small-pretrained/data.pkl`, 312 entries) holding a 308-tensor float32 state dict (112,797,584 parameters) with the historical `net.` prefix; `aurora-0.25-static.pickle` — protocol-5 pickle of a dict of three float32 (721, 1440) numpy arrays `z` (surface geopotential), `lsm` (land-sea mask), `slt` (soil type), latitude 90 → −90.
- Upstream weight license: MIT (`license: mit` in the pinned upstream README front matter and in the Hub repository metadata; the `microsoft/aurora` code is MIT).
- Local layout: `weights/aurora-0.25-small/` holds the 3 manifest entries (`aurora-0.25-small-pretrained.ckpt`, `aurora-0.25-static.pickle`, upstream `README.md`; 463,799,373 bytes total) with byte size and SHA-256 for each, plus the two converted files described below. `verify_snapshot()` in `src/aurora_earth_system_pipeline/pipeline.py` checks the manifest entries, asserts both source digests against the package constants, and checks the converted files against their pinned digests when present.
- Cross-checks: the manifest's checkpoint digest `f80f78de1524a9faba8c9053e4a8ce6a2114ec01cff7f7b4efe9377200d50621` and static digest `e382103f6b24bcf1f996cc0af217c71ff2fc66507a5221e1300b5017581bd318` equal the `oid sha256` of the Hub LFS pointers at the pinned revision.

## What each pickle would execute, and how it is audited

Under the fleet asset specification (§11) a pickle is executable serialization. `audit_pickle()` disassembles a file with `pickletools.genops` — every `.pkl` inside a torch zip archive, or the plain stream — collects every `GLOBAL` / `STACK_GLOBAL` it would import, and refuses anything outside a per-file allow-list, executing nothing:

| File | Globals found | Allow-list | Audit SHA-256 |
|---|---|---|---|
| `aurora-0.25-small-pretrained.ckpt` | `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage` | exactly those three | `e7b998d087a5dcadd37713daf30b63cc571160c3180ebc138500ab662197e932` |
| `aurora-0.25-static.pickle` | `numpy.core.numeric._frombuffer`, `numpy.dtype` | those two plus the numpy-2 spelling `numpy._core.numeric._frombuffer` | `aeec283f2dbb5861afffe185d1ee13e6df5b6f8616a475b38310c0791085c27f` |

Both audits report 0 violations and their digests are pinned in `PICKLE_AUDIT_SHA256`; `convert_model()` refuses a file whose audit digest differs. Tests craft a torch archive carrying `os.system`, a plain pickle carrying `builtins.eval` (as a `STACK_GLOBAL`) and a pickle of a `complex` number, and assert that the audit and the restricted unpickler refuse each before anything is constructed.

An allow-list bounds what the unpickler can name; the loaders below bound what it can construct. The digest pins tie the audited bytes to the loaded bytes, and each unpickle happens once, in the operator's environment.

## The conversion (asset spec §11.2)

`convert_model()` runs, for each file in turn, size check → SHA-256 check against the package constant → static audit and audit-digest check, and only then:

- **checkpoint:** `torch.load(map_location="cpu", weights_only=True)` — torch's restricted unpickler, which constructs tensors and containers and nothing else — must return a dict of tensors; the upstream compatibility shim `Aurora._adapt_checkpoint` (`aurora.model.compat._adapt_checkpoint_pretrained`) strips the `net.` prefix and splits the id-indexed token embeddings and decoder heads into per-variable tensors (308 → 332 tensors, same 112,797,584 parameters); `AuroraSmallPretrained(use_lora=False).load_state_dict(strict=True)` must match every key; the model's state dict is saved as safetensors;
- **static fields:** a `pickle.Unpickler` whose `find_class` allows exactly the two numpy names must return a dict with keys `lsm`, `z`, `slt` of finite float32 (721, 1440) arrays; they are saved as safetensors.

Serving files (both identities recorded, `derived_from_sha256` = the source digests above):

| File | Bytes | Tensors | SHA-256 | In Git |
|---|---|---|---|---|
| `aurora-0.25-small-pretrained.safetensors` | 451,230,408 | 332 | `fc03b5fc5764e08e1f7a05f06ec4debb87fbb66fa2496ec203142221d0843370` | no (regenerated) |
| `aurora-0.25-static.safetensors` | 12,459,136 | 3 | `9bd430b666d9267aca34d5c7924d594c3474296c9e39d85701df389816303bc0` | no (regenerated) |

The conversion is deterministic: the digests were reproduced on every build run and by the executed tutorial notebook, which converts the files it downloads. `verify_converted()` checks sizes and digests; `from_pretrained()` loads the safetensors with `strict=True`, or — with `use_lora=True` — with exactly the 80 LoRA tensors missing (they keep their zero initialisation, so the model is the pretrained model until trained) and nothing unexpected, and asserts the base parameter count.

## Fidelity: upstream's own regression fixture

The Hub repository also carries `aurora-0.25-small-pretrained-test-input.pickle` (163,850,296 bytes, SHA-256 `f52e985b…`) and `aurora-0.25-small-pretrained-test-output.pickle` (81,930,296 bytes, SHA-256 `e7ea0bd4…`): the batch and the recorded output that upstream's `tests/test_model.py` uses to regression-test this exact checkpoint — a 400 × 800 grid with 7 pressure levels (50, 250, 500, 600, 700, 850, 925 hPa), ERA5 1950-01-01 06 UTC, float64. At build time (never by the pipeline) both were loaded through the same restricted unpickler (their globals: `numpy…_frombuffer`, `numpy.dtype`, `datetime.datetime`), the static fields were interpolated from the converted 0.25° file exactly as upstream's `conftest.py` does, the converted model was cast to double as upstream's fixture is, and its forward was compared with the recorded output using upstream's own statistic (mean absolute deviation divided by mean absolute reference):

| Variable | Relative mean deviation | Upstream tolerance |
|---|---|---|
| `2t` | 4.5×10⁻⁷ | 1×10⁻⁴ |
| `10u` | 2.9×10⁻⁵ | 5×10⁻³ |
| `10v` | 3.5×10⁻⁵ | 5×10⁻³ |
| `msl` | 1.2×10⁻⁷ | 1×10⁻⁴ |
| `u` | 1.1×10⁻⁵ | 5×10⁻³ |
| `v` | 1.9×10⁻⁵ | 5×10⁻³ |
| `t` | 2.0×10⁻⁷ | 1×10⁻⁴ |
| `q` | 8.8×10⁻⁶ | 5×10⁻³ |

All within tolerance; the forward took 25.6 s on CPU. The comparison is under one `microsoft-aurora` version (2.0.1); behaviour under the version that produced the checkpoint was not measured. The fixture pickles are not in the manifest and are not fetched by the pipeline or the notebook.

## Runtime facts

- The model is float32 as shipped; `predict` runs under `torch.inference_mode()` and moves results to the CPU; `adapt` trains with gradients only on the selected tensors.
- `Batch.crop` drops the last latitude row when the row count is one more than a multiple of the patch size (721 → 720 at 0.25°, 121 → 120 at 1.5°); every metric in this repository is computed on the rows the model predicts.
- The LoRA tensors (`lora_qkv`, `lora_proj` in every backbone attention block; rank 8, alpha 8, `LoRARollout` with `lora_steps=40` in `"single"` mode) are the upstream architecture's own adaptation mechanism; `adapt(trainable="lora")` trains exactly those 540,672 parameters.
- `microsoft-aurora` pulls `azure-storage-blob`, `pydantic`, `scipy`, `xarray`, `netcdf4`, `timm`, `einops` and `huggingface-hub`; the pipeline uses `aurora.Batch`, `aurora.Metadata`, `aurora.AuroraSmallPretrained`, `aurora.rollout` and the compatibility shim only.

## The tutorial data: pinned WeatherBench 2 objects

`samples.py` fetches four two-day windows of ERA5 at 1.5° (240 × 121, with poles) from the public store `https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr` — chunks 10957 (2018-12-31 00 UTC to 2019-01-01 18 UTC), 11048 (2019-07-01/02), 11185 (2020-03-31/04-01) and 11459 (2021-09-30/10-01) of `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `mean_sea_level_pressure`, `temperature`, `u_component_of_wind`, `v_component_of_wind`, `specific_humidity` and `geopotential`; the static `land_sea_mask`, `geopotential_at_surface` and `soil_type`; and the `latitude`, `longitude` and `level` coordinates with their `.zarray` descriptors — 57 objects, 195,963,943 bytes, each pinned by byte size and SHA-256 in `WB2_OBJECTS`. `fetch_object` refuses a mismatch before decoding, decodes Blosc/LZ4 with `numcodecs` only (no Zarr, xarray or cloud SDK on the download path), and caches the raw bytes under `weights/wb2-era5-1p5deg/` (git-ignored). The store's time axis is uniform (hours since 1959-01-01 in steps of 6; verified at build), so window times are computed from the chunk index. ERA5 is © ECMWF, produced by the Copernicus Climate Change Service and redistributed by WeatherBench 2 regridded; the licence to use Copernicus products requires attribution.

## Files deliberately not staged

The upstream repository at the pinned revision also carries 17 other files: the 5 GB production checkpoints (`aurora-0.25-pretrained.ckpt`, `aurora-0.25-finetuned.ckpt`, `aurora-0.25-12h-pretrained.ckpt`, `aurora-0.25-v1.5.ckpt`, `aurora-0.25-v1.5-ensemble.ckpt`, `aurora-0.25-wave.ckpt`, `aurora-0.4-air-pollution.ckpt`, `aurora-0.1-finetuned.ckpt`), their static files (`.pickle` and `.nc`), the two regression-fixture pickles used at build time as described above, and `.gitattributes`. None is listed in the manifest. The production checkpoints would go through the same audit + conversion path if a row for them were built; the fleet inventory records them as HPC-tier.

## DIMER hosting

- MIT permits use, modification, redistribution and commercial use subject to preservation of the licence and copyright notice. DIMER may host the converted safetensors in its model store under those terms; they are derived from, and recorded beside, the unmodified upstream assets.
- Upload set: `aurora-0.25-small-pretrained.safetensors` + `aurora-0.25-static.safetensors`. **The `.ckpt` and `.pickle` files must not be uploaded** — a profile that carries them would reintroduce the executable-serialization boundary this conversion removes.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no pickle on the serving path; the model class comes from `microsoft-aurora==2.0.1` on PyPI, the served state dict and static fields are safetensors, and `from_pretrained(require_source=False)` accepts the digest-verified files without the manifest or the pickles.
- Serving shape: a forecast needs the 451 MB weights, the static fields on the input grid (the converted 0.25° fields are served by `native_static_fields()`; other grids supply their own, as the tutorial does from WeatherBench 2) and two consecutive analyses; at 1.5° one 6-hour step takes about 0.7 s on CPU, at 400 × 800 in double precision 25.6 s; a native 0.25° profile should have a GPU worker. An adapted profile needs the weights plus a 2 MB LoRA adapter and must be built with `use_lora=True`.
- Two review items are open: whether the one-time restricted unpickles (in the build and, for the tutorial, in the runtime) meet the DIMER deserialization-trust bar or whether DIMER hosts only maintainer-converted files; and which resolution tier a hosted profile should serve, given that only 0.25° is in-distribution. The served artifacts are the same files either way.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
