"""Tutorial dataset, validation, extended-NetCDF I/O for the Aurora pipeline.

The default dataset is **real ERA5 reanalysis** at 1.5° (240 × 121, with poles), served by WeatherBench 2
(Rasp et al., 2024) from a public Google Cloud Storage bucket as a Zarr v2 store. Four two-day windows
(eight 6-hourly analyses each) are fetched: two for training (January and July 2019), one for validation
(April 2020) and one for testing (October 2021) — distinct seasons and years, so nothing in the test
window is a near-duplicate of anything trained on. Every object fetched (chunk, `.zarray`, coordinate)
has its byte size and SHA-256 pinned in `WB2_OBJECTS`; a mismatch is refused before decoding, and the
Zarr chunks are decoded with `numcodecs` (Blosc/LZ4) directly, without a Zarr or xarray dependency on
the download path. About 196 MB is transferred for the original four windows, about 145 MB more for the three 2021
validation windows, and about 587 MB for the 12-window 2022 fresh test, which is fetched only for a pre-registered
scoring (docs/aurora-preregistration.md).

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
# R2-D2: three more validation windows (outside Aurora's documented pretraining period).
VALIDATION_WINDOWS: dict[str, int] = {"val-2021-01": 11330, "val-2021-04": 11375, "val-2021-07": 11420}  # noqa: E501
# R2-D1: the fresh test, 12 two-day windows from 2022. Fetched only for a pre-registered scoring, never when
# DIMER_MASK_FRESH_TEST=1; see docs/aurora-preregistration.md.
FRESH_TEST_WINDOWS: dict[str, int] = {"fresh-2022-01": 11512, "fresh-2022-02": 11528, "fresh-2022-03": 11542, "fresh-2022-04": 11557, "fresh-2022-05": 11572, "fresh-2022-06": 11588, "fresh-2022-07": 11603, "fresh-2022-08": 11618, "fresh-2022-09": 11634, "fresh-2022-10": 11649, "fresh-2022-11": 11664, "fresh-2022-12": 11679}  # noqa: E501
FRESH_POOL_DIGEST = "7acb15d01bf0b2472d73c48906a7a9735370b928b29aecf01fbd440b88a46f1b"
WB2_TIME_CHUNK = 23386
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
    # R2-D2 validation windows (2021), R2-D1 fresh-test windows (2022) and the time coordinate (R2-m9),
    # pinned 2026-10-19 (docs/aurora-preregistration.md). Chunk = the chunk containing the 15th at 00:00.
    "10m_u_component_of_wind/11330.0.0": (817593, "68f37610590bc3c89770201a68153b93461f39edc501cd308b89dce7ef0e35ec"),
    "10m_u_component_of_wind/11375.0.0": (817724, "4640b9d2ac3cced5b7be1dfca0ef6e0c3d6b98ec71805dc6695d813816ac6f45"),
    "10m_u_component_of_wind/11420.0.0": (816828, "cce595cdf75ef729a49493faaced40c0c106d1e1af75e85c375e6c17f4059ccc"),
    "10m_u_component_of_wind/11512.0.0": (817715, "8fbe807f2355d92d6cad66fb64a6b243a4fb6fe380231e9d6588c81a26582df6"),
    "10m_u_component_of_wind/11528.0.0": (815659, "1a7e78776f2d466d2cdf8c2f043433ec9aa407a01fb8783a740ce3d5a325aa16"),
    "10m_u_component_of_wind/11542.0.0": (816435, "54b4614848eb9a7f516af0c6a3c8372b27b0fb29721d18b395b52e0636ea06d7"),
    "10m_u_component_of_wind/11557.0.0": (816676, "905364b32106b1d3f8f351b666d302c860ae8afdd0bd32e948751aee4f9475e7"),
    "10m_u_component_of_wind/11572.0.0": (817071, "73bafd6599824bf31de90f097cbb2662728824a4db23598a7294d29b91169cd2"),
    "10m_u_component_of_wind/11588.0.0": (816577, "28e60aa219d105c31ef821f365379a35820d278413ac235730633f945e0a5fa0"),
    "10m_u_component_of_wind/11603.0.0": (816095, "6f0912fe1cef74f429cd401ae5db8509e3c3ceaa04b5b79f49b6dd9f4b89f2cf"),
    "10m_u_component_of_wind/11618.0.0": (819049, "8f64c25d52edd69f40a41b36a8f0af064cb782b77352bcbdf083a08d554edfac"),
    "10m_u_component_of_wind/11634.0.0": (816116, "ea5d31c47015fba9659aea54bd5c9a0013dbaee092dc5a116c80dd76f110cfba"),
    "10m_u_component_of_wind/11649.0.0": (814483, "3e539dbcbcd4aff379a41ee60813d87918329458766e3d3e9866f16524b1b904"),
    "10m_u_component_of_wind/11664.0.0": (818323, "0d290858aab4a88a5f88fbf30b9e878c049f6a5426315f248a515272e9627605"),
    "10m_u_component_of_wind/11679.0.0": (815461, "317409f6964d6e5d25ea8cd411aa66da0f12491607e1f75a57d854c0b24956a6"),
    "10m_v_component_of_wind/11330.0.0": (817563, "5d3ecb6b0ecb7191659e2dcff8cbace59ca81e0d06e378040ebf99a3a248dcb8"),
    "10m_v_component_of_wind/11375.0.0": (820580, "9d8751a41299733ddeb254eae9666877f28b66e075a634d5a42b14dd6eefb959"),
    "10m_v_component_of_wind/11420.0.0": (817830, "d5e977cced4fa490b29d76d05f50fb6958b88f058baf166ebb30bc2b8fd90405"),
    "10m_v_component_of_wind/11512.0.0": (814663, "e93dd57f7ff2abc7bc016f1c9af2cbcba72aad88558dfe02716e46cd823265bd"),
    "10m_v_component_of_wind/11528.0.0": (817791, "700f04c8e98c8a43be900032bcca076e5b7f86f2062381dcfcb822e152d08a0f"),
    "10m_v_component_of_wind/11542.0.0": (821543, "3a0585e60d4d4b8a9eaacce6a9ac872f829f8f6c9fa84cc0f57e22a8dbb1e258"),
    "10m_v_component_of_wind/11557.0.0": (817173, "417bf2db1bbfe1b6b32bd7994ec5228276c1b88dea23162066f77abf7758221b"),
    "10m_v_component_of_wind/11572.0.0": (820334, "1b8706b0a102da830577226256e57fdfbff1576ee38f825b64e4acf814f377cb"),
    "10m_v_component_of_wind/11588.0.0": (817466, "3710695a9fba67e97098f1d34fbebf6ede466ece549db4340815da3481fcbb0d"),
    "10m_v_component_of_wind/11603.0.0": (817019, "047aa20e31af3ae1112274e8bbab15b3d82391dc578116c58fde2cfec543790e"),
    "10m_v_component_of_wind/11618.0.0": (819788, "6a0c6d4a497a229be9ff664db7b74fe3302f4b734bb0c49131bc294705495744"),
    "10m_v_component_of_wind/11634.0.0": (816206, "831f2f1bb37ae958e194a79b128324b3aace4bd4eac2b3537be99b6f6a765f51"),
    "10m_v_component_of_wind/11649.0.0": (818002, "27a4c6ee728f1378412cf0f1b4d5a1c2717f20bdf12a813420ae20e9e350834b"),
    "10m_v_component_of_wind/11664.0.0": (818447, "5008e62ce7f2f61cdb9d4b8bd0cef9529f94b5454d90a0fc29f9132e3edb7c60"),
    "10m_v_component_of_wind/11679.0.0": (818823, "bd764e2696f240da5750a94e4161a8b380fe95ae29992763235601baf5be5f1f"),
    "2m_temperature/11330.0.0": (617265, "e976db4f8a1ddd77ac19b07ccb80e4f50e4d499882e9921200b04dbf341a2f15"),
    "2m_temperature/11375.0.0": (616611, "25972144b2e022e3a1e203f1299418ef683f3c657aede34f82ab68164a8d28a7"),
    "2m_temperature/11420.0.0": (611012, "fb5542152127a3ede461a646a02ebe0c09923fca32fdfaeaa7e852751298331e"),
    "2m_temperature/11512.0.0": (616966, "293a9ebac2ac06690938ec0790664931092544bafe23f6b27995002a8555c66d"),
    "2m_temperature/11528.0.0": (619541, "f1ddf541fdad53cb262a50f12341ece726db6eb55566fd204abe548d83331598"),
    "2m_temperature/11542.0.0": (618087, "6abdadfd9a5ac708013d545ec791a06cdd633946fc6493bd52175a17504d7dea"),
    "2m_temperature/11557.0.0": (615373, "7217368d704b6a25a8b5c80f8adbb1b8b2bcb3cb994a8ecce08b4abfb741ed27"),
    "2m_temperature/11572.0.0": (613475, "f4534b1d7576be1fcbece9dc48dc6e40802f0b1fefb1746c5937eac7b300e9c9"),
    "2m_temperature/11588.0.0": (608722, "0247c5bf1d672a6d18a0a8fe8927d47762aa83883c669ec7127e8d2f237a623e"),
    "2m_temperature/11603.0.0": (611095, "62c30b417813be02c78b90bd53e568d9308f9aa68fd937cce73cc885300b2e77"),
    "2m_temperature/11618.0.0": (610072, "7601478a3d89f6ecb18e9d1c6971389151adc559cd46f1366b94bf6deb4648e1"),
    "2m_temperature/11634.0.0": (607131, "6e0828d047a0e45cd9123b36697149565af1ff4856c3218ae46fe0a117ae7971"),
    "2m_temperature/11649.0.0": (609776, "ea7ca6d9f0dbaa7d2223326b494997b7553da34969bdae9dfce70cff4c649b8c"),
    "2m_temperature/11664.0.0": (619015, "379d8c8223698f00fb0be208d5c645cd32e5ecee487df72cd13b7946733d3b15"),
    "2m_temperature/11679.0.0": (614969, "706ce168aa28ad3e8934e9a0525d600f1d985ff330d74156d5eca0cb50dc9d13"),
    "geopotential/11330.0.0.0": (7680113, "bbdee66f45fb3f97e988f7bbf1e4f76f515d1e43f0e8fad43689e5db5d0666f5"),
    "geopotential/11375.0.0.0": (7660723, "f393f2fbf0368eecb5f852dc64024ced79065756b316e22c4bc0490233dd29c1"),
    "geopotential/11420.0.0.0": (7663345, "8220d5ef3ddf942f2f08476c76edeb04d837d5aed3bef75d0519aef22997f7b9"),
    "geopotential/11512.0.0.0": (7680056, "058bacdf7416f74fdcebafc3677d92c513df1dc01b8334b9c04700d29a9551b4"),
    "geopotential/11528.0.0.0": (7660738, "e3ecc76555f3f079696cf1b7f49bbe524e64a64ccc3ae8752d29a8301c342cd1"),
    "geopotential/11542.0.0.0": (7718848, "f005b69bd86c395cd08796a9cbb715b04d780c749f9d3d4856879ec37f38f9a4"),
    "geopotential/11557.0.0.0": (7660146, "560b10d0d60e61e9329491b9089eac816dcf70a9a998f0b3bfb2c2250a9a2d53"),
    "geopotential/11572.0.0.0": (7666708, "ef110dc81cc0a4cc332b703bb11ab14dbec195960128dfda6d9594722a6a038b"),
    "geopotential/11588.0.0.0": (7635338, "0012a9d458d116ec37e5f4c5c7c8c1d5bfc54ff69afc29161ff7667c8231bf7f"),
    "geopotential/11603.0.0.0": (7689878, "ea6a4d493543961a20c8ce76b540fe7e8d67e0f6e5a3e616d261b2e29d127051"),
    "geopotential/11618.0.0.0": (7649899, "eb0f5e3abe8ff356ffdb76f9812a171de1f31a79be95c527ad18faacec76c4ff"),
    "geopotential/11634.0.0.0": (7657368, "71bbcbfda4d7a140ce7ca6171bec6b0e98793ac211d73d014b26953740782bd5"),
    "geopotential/11649.0.0.0": (7665396, "18fc5023a2fb69dcc5e75224570a80c71bcb5269b790fb8018ad75ef6af62371"),
    "geopotential/11664.0.0.0": (7688149, "67d681b49fc9e5b95da2991f491e5bd1e7ee209d6be2e8d583c503c6bc54a14e"),
    "geopotential/11679.0.0.0": (7686720, "6f09d374b4836d12f6cde3189537d12da04e879821965e198a41e41011f924be"),
    "mean_sea_level_pressure/11330.0.0": (555020, "589bd48d4e10833abe11016b91c3394c6d4a27fd0fde001d1c155920d41e9fa9"),
    "mean_sea_level_pressure/11375.0.0": (553253, "1469a942441047ec949c2faa32a022d0ba6080f25fc977546a1d4a6c8db07fb3"),
    "mean_sea_level_pressure/11420.0.0": (555831, "62b98d43f8053f3c689ca8eeb605ed57a7082fed3f01a06059e7bda052a87acf"),
    "mean_sea_level_pressure/11512.0.0": (554746, "b7f27f1105064b2457d7610eac1cb7ba03bb6da06a09c8982c570e63dc7bcba3"),
    "mean_sea_level_pressure/11528.0.0": (553427, "d49f8716766b514ba95aced75e8eb01e033765610edcdb9f4b7ca5375c9da964"),
    "mean_sea_level_pressure/11542.0.0": (555996, "a05f8a14d42f87175714b95a52a7a94802842d1043aa0f0be5e66d77fe05e507"),
    "mean_sea_level_pressure/11557.0.0": (554966, "868185f927f3b25854b607934e2467cc9eb351be0ca3526c97cd37db92b17416"),
    "mean_sea_level_pressure/11572.0.0": (556426, "a8e941b6163a2d973e7a8fedd81f9eb5e8b43e76cf14620871e5aa92acdb6bde"),
    "mean_sea_level_pressure/11588.0.0": (555825, "9a599f1d18d4382be68c5ccf1c4726f9591fa4e6c111da48bceec9c92d4e52d3"),
    "mean_sea_level_pressure/11603.0.0": (555775, "111df4a00e05dbb076107ca874822870a6e26e1332dabcfd2e7243de6a006c69"),
    "mean_sea_level_pressure/11618.0.0": (554772, "174252e1fa9619db37b33f343264405aa0e753d424eee9a1e97042d1fff00094"),
    "mean_sea_level_pressure/11634.0.0": (555398, "acc5c3404aadf01236688015cf37c75346b06bf36422c0801eb4891b976b8516"),
    "mean_sea_level_pressure/11649.0.0": (555001, "3b86ecb2c6cfe6767dd939856028b0745ed8a48c63dd54c51a6af09c9d1af33b"),
    "mean_sea_level_pressure/11664.0.0": (554568, "8d7fcdf46edaee3dc7f97c92b6f087ea02b341e1236de1e645eca8450154b4f3"),
    "mean_sea_level_pressure/11679.0.0": (553922, "46e86bcc399028f7ae1717c15b4a0444f6f027da56445a1b768159d01f8a5850"),
    "specific_humidity/11330.0.0.0": (9734396, "3876429b2e8d518edfac770b7cbbf0e6f90ba04f464909554da5a3ba53ae9341"),
    "specific_humidity/11375.0.0.0": (9744798, "41b4f3cbee5870b4f17231a85d5c245b4cdd24eb2ab2a6e58a52809828407f20"),
    "specific_humidity/11420.0.0.0": (9758589, "ef4c3874daab06ec8433e71079402daded88810a744b7e8cdabfddde040b010f"),
    "specific_humidity/11512.0.0.0": (9751013, "4078ed32520684a3d0acae5cb76cb41df791d65af8a01f850e5a8ee360de8e96"),
    "specific_humidity/11528.0.0.0": (9700117, "d820b074a2e0c2560cd42e69d2c3a88d99fbbc4811574e68eaa7dae4e0fa38af"),
    "specific_humidity/11542.0.0.0": (9719502, "71bd4d1e1ee54751a9ae1790e2686810b3588c2ecfa42c2a24b8224a3b752c88"),
    "specific_humidity/11557.0.0.0": (9694842, "10704cc117bdb4aed8925e04afa494bb1896978f9021325bd58f4991a6fe894a"),
    "specific_humidity/11572.0.0.0": (9738258, "2caacfc9228408435b99636f5d500fa1f7dea2f204602f8d739e04a725b2f456"),
    "specific_humidity/11588.0.0.0": (9733189, "77d8c03967485e470ee1667863cbc4e72aa2cc5f62c42fac6e4202ef4b10a6b8"),
    "specific_humidity/11603.0.0.0": (9759965, "a885deafb125585dc4e5de8e4a5760f330b39369aa57118fc12ebb5fca977723"),
    "specific_humidity/11618.0.0.0": (9783740, "2e81ee070ab8943e50030577111776dec590846f341b20f38d0e4be60272b1b8"),
    "specific_humidity/11634.0.0.0": (9812212, "ffc3de42c269012028d75eef34a65f5954a9809623ceaebcc909b63b2935cc23"),
    "specific_humidity/11649.0.0.0": (9822075, "d1be5f75b6ca8fc5b820c82bbfd9e88cb1d3e619ca97aa28d3ffe610888b1a7a"),
    "specific_humidity/11664.0.0.0": (9806983, "e9a28aae6fb4d50ed5d1c80207316f6ef33c2a5ff9d7719bd52185fbf6551982"),
    "specific_humidity/11679.0.0.0": (9762542, "8387fc2539b3b9e1a8845405ae558c5584f458f2617ddeafa61f63e1bcf0f0bb"),
    "temperature/11330.0.0.0": (7830505, "77b8e6e3db316df2dde047ef758e116e02517f7e2f3ac5c57b5268ec1b350353"),
    "temperature/11375.0.0.0": (7805692, "fc2088d2f28fb7dd7fd0ac7a6a416c617201b4bf54a15eb3eba7562741fc95a9"),
    "temperature/11420.0.0.0": (7809867, "c86f5fd590f005252a167edf6f9082045e6dfbe8ddd50c8c4d7184d67c651284"),
    "temperature/11512.0.0.0": (7807372, "26ad7ccb43c6b4242b318f0df69a3f93021390982ad8ec71533e892c846e17ed"),
    "temperature/11528.0.0.0": (7818730, "47ebed5c10e43e721c83103bf97eb096fd1926a439c486f902ec712eb4293c28"),
    "temperature/11542.0.0.0": (7888493, "31748155d4b8ad37a24e652a2b2a3ea6623cda2ca8b0cfd6392450f6a5513c64"),
    "temperature/11557.0.0.0": (7841003, "6215fb27cb7256ed7c4712fca44249df14233523d616cc69f18ae2c83788638a"),
    "temperature/11572.0.0.0": (7817921, "a9090444b485e73c20c9039bf1fbd8cc7d9fb991d2cc86d5ca4a66a905d3fef4"),
    "temperature/11588.0.0.0": (7793307, "0236633ed3752af4f534c702a08cae45ca2ee3d3811c9dba51832b4e51858cef"),
    "temperature/11603.0.0.0": (7837688, "2b161c5b37044de129ffc347cd19ffd4a5e01e302a5140126bda33b2683df6cc"),
    "temperature/11618.0.0.0": (7824149, "62328b248b656281e1c0f517554d37cc02f35579d853de6eb524e9f67bad3d4f"),
    "temperature/11634.0.0.0": (7825719, "dc0ff913f8f828f45ca0c20a0dc0bcefcd9c5bfeadd12287dcf1f4b52a77aad6"),
    "temperature/11649.0.0.0": (7811857, "d88499f37bc1db03de128b7ec68f9daebe0eb6cb520b1aae14d206fe584b96cc"),
    "temperature/11664.0.0.0": (7821499, "b1199842d257d0a6af175952e6a85817e5cfe24b21086f5b51ef688c481e842e"),
    "temperature/11679.0.0.0": (7807484, "f6b7ec1e1adbe9c04742bff6e812d7f01a0d7926fa33934bb2dd2dd9a01a2fcd"),
    "time/.zarray": (320, "1833a4abfd22df610af0b9a463b51a342adeb8344f8cdc5409b051d8cbeb4abd"),
    "time/3": (3374, "a7f6e4cac0dc83bb0426d1dcc31bc5b1d369d3a0e3380e30b7d8ca7c58ba8000"),
    "u_component_of_wind/11330.0.0.0": (10361303, "b9ac787ba1e26302471ec3227cbff8c03d193cb11db8be894e174731819ef804"),
    "u_component_of_wind/11375.0.0.0": (10314537, "e716ccd45336840487ca995814ec6e69de6a90cd3750fd2d0f778a4db303fb29"),
    "u_component_of_wind/11420.0.0.0": (10375443, "119a9c31cb6b2fd32fe3e50a5db949d7032206ce3870c01564832c8cc95977a9"),
    "u_component_of_wind/11512.0.0.0": (10348338, "94563280d8d65b0544f56020f7a5cfb7f0291b973245161d58172dab2fa86e59"),
    "u_component_of_wind/11528.0.0.0": (10349913, "9b274e6b391dff252638cb58505b74caa21268c683b589a88dba8fa1d8d0fe37"),
    "u_component_of_wind/11542.0.0.0": (10363102, "b0946ea8dc1becb20b49e5ea1e457fdfa087b501e75b92aed3996ac8f5faffad"),
    "u_component_of_wind/11557.0.0.0": (10350532, "b4892b30cf710a7ed1a4131d192e7222da7ccd0eaf18331d8c85392ec783f2b6"),
    "u_component_of_wind/11572.0.0.0": (10347987, "5dafbb1f280de10412a93186745a001adb5f7de0c6df2570720ef35bd9378db1"),
    "u_component_of_wind/11588.0.0.0": (10374009, "d9c786cc0740fc1362172afed5233c6f3305a423731735d94ca2ab2032654727"),
    "u_component_of_wind/11603.0.0.0": (10360964, "c08927a9851d1229e59151d10c8057cfee4e4b8b6cbc22afdbb67ec5493ea77f"),
    "u_component_of_wind/11618.0.0.0": (10383938, "bec8fffa484e237ac16ff49b6f15e67241ae1e1f84114a477e2e708808246b34"),
    "u_component_of_wind/11634.0.0.0": (10375729, "ce3cdfc150829383c8991da1795f9abc026df118dfe6fb60e01ad1c638f3ea85"),
    "u_component_of_wind/11649.0.0.0": (10345621, "0b63f33acbbb38ef012ac61726bce618933610e5c9a26bf76aeb0037cf6efce6"),
    "u_component_of_wind/11664.0.0.0": (10327428, "86ac056c609064b4e37b0fe96e22e01445e2b84d6ac0ebf95699b49e775fb2bb"),
    "u_component_of_wind/11679.0.0.0": (10337246, "f6f256714d4b57a34fc3e4d749ec105904f2e2434828cf74d75f74905e598561"),
    "v_component_of_wind/11330.0.0.0": (10481799, "8c8d4f15b873b0eb96ca61909e7b0e0ef09a3d2d97dc4f7d3ca78ed440f1c74b"),
    "v_component_of_wind/11375.0.0.0": (10516421, "1543e0fe831a49165d90284644d108f8a339e4783da89a00feed1d9a21742c06"),
    "v_component_of_wind/11420.0.0.0": (10546819, "f63c08d1e75beba73c81578afa810f7e2b095a3ee22907fe24ebabff63757c99"),
    "v_component_of_wind/11512.0.0.0": (10466954, "786f9403d4694a6a035dbcf1d03c0bbc626bb7932af8f6dc318e58d7954366a9"),
    "v_component_of_wind/11528.0.0.0": (10518230, "a1c43644a1a0b92c02d46474c6377f1a1f1537fda28e7664384d570366eee197"),
    "v_component_of_wind/11542.0.0.0": (10520329, "f6952ab7e8101bf9c9ebff2a431f9304c51b9b8e3c65afa42731cdb75ab126a4"),
    "v_component_of_wind/11557.0.0.0": (10514140, "1d2bd8c7ce4937d31136c90443d471e6442bb1a68ec9da7da1603047ed2c8eee"),
    "v_component_of_wind/11572.0.0.0": (10499776, "a45481ce679200eded2045fd47c59e4be6be0d7c6063109c933a9d07d849456f"),
    "v_component_of_wind/11588.0.0.0": (10518743, "c1025331420bab8b838d6ddf7c1e32d671c45931a9e518ee17d8a4c7c8d26947"),
    "v_component_of_wind/11603.0.0.0": (10505124, "c86172504054f573ac06d88c34a46ee5474e9498f70cd390c56f816e30813675"),
    "v_component_of_wind/11618.0.0.0": (10535158, "a868870749076094d73ef7aca656a7f30354e9b97d62bd94817798432cc40141"),
    "v_component_of_wind/11634.0.0.0": (10542817, "eedeff3ae6e182f74f80b173957aefa4ebb73c1f7e45d9238d303027918468e8"),
    "v_component_of_wind/11649.0.0.0": (10508974, "691edca127d7a788d35146850202afc9fac3a248bf76c7fee750e8c1f9d1384b"),
    "v_component_of_wind/11664.0.0.0": (10473559, "444da348f10d042d5d1774c15b1ec067adcc5e6579e25e89835ddda6521378ef"),
    "v_component_of_wind/11679.0.0.0": (10492373, "5d3648277990a9c029764777f1d9efbc7d3b23fdba8f636aab5c4113227ac94c"),
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


ALL_WINDOWS: dict[str, int] = {**SAMPLE_WINDOWS, **VALIDATION_WINDOWS, **FRESH_TEST_WINDOWS}


def _object_keys(chunk: int) -> list[str]:
    return [f"{v}/{chunk}.0.0" for v in WB2_SURF.values()] + [f"{v}/{chunk}.0.0.0" for v in WB2_ATMOS.values()]


def fresh_pool_digest() -> str:
    """SHA-256 of the sorted `key size sha256` lines of every fresh-test object (docs/aurora-preregistration.md)."""
    keys = sorted(k for chunk in FRESH_TEST_WINDOWS.values() for k in _object_keys(chunk))
    return hashlib.sha256("\n".join(f"{k} {WB2_OBJECTS[k][0]} {WB2_OBJECTS[k][1]}" for k in keys).encode("utf-8")).hexdigest()


def check_window_times(chunk: int, *, cache_dir: str | Path | None = None, fetcher: Any = None) -> list[datetime]:
    """R2-m9: the window's analysis times read from the store's pinned time coordinate, checked against the chunk
    arithmetic (hours since 1959-01-01, 6-hourly); a re-indexed store is refused."""
    import numpy as np
    from numcodecs import Blosc

    meta = json.loads(fetch_object("time/.zarray", cache_dir=cache_dir, fetcher=fetcher))
    if meta.get("dtype") != "<i8" or meta.get("chunks") != [WB2_TIME_CHUNK] or meta.get("compressor", {}).get("id") != "blosc":
        raise ValueError(f"time/.zarray: unexpected encoding {meta}")
    first = chunk * WB2_CHUNK_STEPS
    block, offset = divmod(first, WB2_TIME_CHUNK)
    if block != 3 or offset + WB2_CHUNK_STEPS > WB2_TIME_CHUNK:
        raise ValueError(f"chunk {chunk} is outside the pinned time block time/3")
    hours = np.frombuffer(Blosc().decode(fetch_object("time/3", cache_dir=cache_dir, fetcher=fetcher)), dtype="<i8")
    stored = [int(h) for h in hours[offset:offset + WB2_CHUNK_STEPS]]
    expected = [TIMESTEP_HOURS * (first + j) for j in range(WB2_CHUNK_STEPS)]
    if stored != expected:
        raise ValueError(f"chunk {chunk}: store times {stored} != chunk arithmetic {expected}; the store was re-indexed")
    return [WB2_EPOCH + timedelta(hours=h) for h in stored]


def fetch_sample_window(name: str, *, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, Any]:
    """Assemble one named window (lat 90→-90, lon 0→360, 8 analyses) from pinned objects."""
    if name not in ALL_WINDOWS:
        raise ValueError(f"window must be one of {sorted(ALL_WINDOWS)}")
    import numpy as np

    kw = {"cache_dir": cache_dir, "fetcher": fetcher}
    chunk = ALL_WINDOWS[name]
    lat = _decode("latitude/0", "latitude/.zarray", **kw)
    lon = _decode("longitude/0", "longitude/.zarray", **kw)
    levels = _decode("level/0", "level/.zarray", **kw)
    times = check_window_times(chunk, **kw)
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
    """The development windows (train, validation, exposed development test), keyed by name. Never the fresh test."""
    names = [*SAMPLE_WINDOWS, *VALIDATION_WINDOWS]
    return {name: fetch_sample_window(name, cache_dir=cache_dir, fetcher=fetcher) for name in names}


def fetch_fresh_test(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, dict[str, Any]]:
    """The 12 fresh-test windows. Call only for a pre-registered scoring: every object is fetched (or read from the
    cache), size/SHA-256 checked and decoded first, so a failure is refused before any window is scored."""
    import os

    if os.environ.get("DIMER_MASK_FRESH_TEST", "").strip() not in ("", "0"):
        raise PermissionError("DIMER_MASK_FRESH_TEST is set: the fresh test is masked and is never fetched")
    if fresh_pool_digest() != FRESH_POOL_DIGEST:
        raise ValueError("the fresh-test pins do not match the pre-registered pool digest")
    return {name: fetch_sample_window(name, cache_dir=cache_dir, fetcher=fetcher) for name in FRESH_TEST_WINDOWS}


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
