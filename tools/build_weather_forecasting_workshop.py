# ruff: noqa: E501,I001
"""Generate the DIMER weather and Earth-system forecasting workshop notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from weather_forecasting_workshop_source import CELLS

NOTEBOOK_NAME = "DIMER_Weather_and_Earth_System_Forecasting_Workshop.ipynb"

def build_notebook():
    rendered = []
    for index, cell in enumerate(CELLS):
        base = {
            "id": f"dimer-weather-workshop-{index:02d}",
            "metadata": {},
            "source": cell["source"].splitlines(keepends=True),
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
                },
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
