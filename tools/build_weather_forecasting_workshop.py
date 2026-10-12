# ruff: noqa: E501,I001
"""Generate the DIMER weather and Earth-system forecasting workshop notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import weather_workshop_isolated_runtime as isolated
from weather_forecasting_workshop_source import CELLS

NOTEBOOK_NAME = "DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb"

# Cells generated from repository files rather than written in the source module (2026-10-03 uv isolated environment).
GENERATED = {
    "isolated_install": isolated.install_cell,
    "isolated_router": isolated.router_cell,
    "isolated_imports": isolated.imports_cell,
}
MAX_CELL_LINE = 2000


def cell_ids():
    """Explicit ids where the source gives one; otherwise the next number after the last numbered id. The cells that
    existed before 2026-10-03 therefore keep their ids, and the four uv cells carry their own."""
    ids, number = [], 0
    for cell in CELLS:
        explicit = cell.get("id")
        if explicit:
            ids.append(explicit)
            match = re.fullmatch(r"dimer-weather-workshop-(\d+)", explicit)
            if match:
                number = int(match.group(1)) + 1
        else:
            ids.append(f"dimer-weather-workshop-{number:02d}")
            number += 1
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate workshop cell ids")
    return ids


def isolated_metadata():
    lock_text = isolated.read_lock()
    return {
        "mechanism": "uv isolated environment; every code cell after the router runs in one persistent worker there",
        "platform": "linux-x86_64",
        "managed_python": isolated.MANAGED_PYTHON,
        "uv": isolated.UV["version"],
        "lock": isolated.LOCK,
        "lock_sha256": hashlib.sha256(lock_text.encode("utf-8")).hexdigest(),
        "locked_packages": len(isolated.lock_packages(lock_text)),
        "install_flags": ["--require-hashes", "--only-binary", ":all:"],
        "kernel_installs": False,
    }


def build_notebook():
    rendered = []
    for cell_id, cell in zip(cell_ids(), CELLS, strict=True):
        source = GENERATED[cell["generated"]]() if "generated" in cell else cell["source"]
        too_long = [len(line) for line in source.split("\n") if len(line) > MAX_CELL_LINE]
        if too_long:
            raise SystemExit(f"{cell_id}: {len(too_long)} line(s) over {MAX_CELL_LINE} characters; split the literal")
        base = {
            "id": cell_id,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell["kind"] == "markdown":
            rendered.append({"cell_type": "markdown", **base})
        else:
            rendered.append({"cell_type": "code", "execution_count": None, "outputs": [], **base})
    return {
        "cells": rendered,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "dimer": {
                "canonical_runtime": "NVIDIA Tesla T4",
                "capability": "weather-and-earth-system-forecasting",
                "carrier": "Aurora short-range global forecasting and LoRA adaptation workshop",
                "clean_runtime_evidence": "pending",
                "credentials_required": False,
                "dataset": "ERA5 via WeatherBench2 at 1.5 degrees",
                "notebook_kind": "WORKSHOP",
                "evaluation_window": {
                    "description": "temporally separate, development-reused evaluation window",
                    "built_in": "2021-09-30T00:00:00..2021-10-01T18:00:00 (chunk label test-2021-10)",
                    "development_reuse": "scored in earlier runs of this notebook; used to develop the primary tutorial's default recipe",
                    "pretrained_checkpoint_exposure": "unknown",
                },
                "notebook_mode": "WORKSHOP",
                "notebook_profile": "E2E",
                "notebook_spec": "2.1",
                "release_status": "candidate",
                "standalone": True,
                "worker_required": False,
                "generated_from": {
                    "repository": "kurtvalcorza/aurora-earth-system-pipeline",
                    "source": "tools/weather_forecasting_workshop_source.py",
                    "generator": "tools/build_weather_forecasting_workshop.py",
                    "isolated_runtime": "tools/weather_workshop_isolated_runtime.py",
                    "lock": isolated.LOCK,
                },
                "isolated_environment": isolated_metadata(),
                "revisions": [
                    {
                        "date": "2026-10-03",
                        "previous_blob": "385da6db6424b92fdec83f82a0e68f08abeed02e",
                        "change": (
                            "uv isolated environment: the in-kernel pip install and its restart guard are replaced by a "
                            "pinned uv, a managed CPython 3.12.12 and a hash-locked wheel-only install; every later code "
                            "cell runs in one persistent worker there. Linux x86_64 only. Learner cells unchanged."
                        ),
                    },
                    {
                        "date": "2026-10-13",
                        "previous_blob": "09775738e2214172975f9c0f30d25a6bb29ab48d",
                        "change": (
                            "Review W2 (AUR2) fixes: the 1.5 degree comparison is described as a resolution/data-product shift "
                            "that does not isolate resolution; the fixed inherited recipe, the re-used October 2021 test window "
                            "and the non-binding epoch selection are stated; the quoted reference numbers name their run; "
                            "prediction prompts, collapsed exercise answers, a diurnal-cycle note with a printed 2 m temperature "
                            "table, documented environment variables, notebook_kind and References with licences. Default-path "
                            "computation unchanged; the test cell prints one extra table from values it already computed."
                        ),
                    },
                    {
                        "date": "2026-10-13",
                        "previous_blob": "1eed1127b5dfa3230ba8f6a06ebab626540a1505",
                        "change": (
                            "Peer-review follow-up: a fresh zero-update LoRA model before every adaptation (a re-run no longer "
                            "continues training); the test is a temporally separate, development-reused evaluation window, with "
                            "development reuse and pretrained-checkpoint exposure (unknown) exported separately; the built-in "
                            "times are checked against the store's time coordinate and the exact dates replace month labels; "
                            "the soil-type count is computed from the loaded field; macro-average naming and the defined-ratio "
                            "denominator; selection-objective, diurnal, control-design, leakage and training-vocabulary wording."
                        ),
                    },
                ],
            },
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "workshop_revision": "0.1.0-candidate",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def serialized():
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.out or repo / "tutorials" / NOTEBOOK_NAME
    content = serialized()
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != content:
            raise SystemExit(f"STALE: {out}; regenerate the workshop notebook")
        print(f"OK: {out}")
        return 0
    out.write_text(content, encoding="utf-8")
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
