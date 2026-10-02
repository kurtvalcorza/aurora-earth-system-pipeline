"""Forecast verification: latitude-weighted RMSE per variable and lead time, the persistence baseline,
and a summary skill ratio.

RMSE is weighted by cos(latitude) normalised to unit mean, the WeatherBench convention, so the poles do
not dominate an equiangular grid. Atmospheric variables are averaged over all 13 pressure levels, and
geopotential at 500 hPa (`z500`) is reported on its own because it is the customary headline. The
**persistence** forecast carries the latest analysis forward unchanged; at 6 h it is a strong baseline
that any forecast model has to beat, and it decays with lead time. The skill ratio is
RMSE(model) / RMSE(persistence): below 1 the model beats persistence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import ATMOS_VARS, LEVELS, PATCH_SIZE, SURF_VARS, TIMESTEP_HOURS, _lat_weights

HEADLINE_LEVEL = 500
REPORTED = (*SURF_VARS, *ATMOS_VARS, "z500")


def lat_weighted_rmse(pred: Any, truth: Any, lat: Sequence[float]) -> float:
    """Latitude-weighted RMSE over (H, W) or (L, H, W) arrays; `lat` matches the H axis."""
    import numpy as np

    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} != truth shape {truth.shape}")
    weights = _lat_weights(lat)
    weights = weights[:, None] if pred.ndim == 2 else weights[None, :, None]
    return float(np.sqrt(np.mean((pred - truth) ** 2 * weights)))


def forecast_metrics(
    window: Mapping[str, Any], origins: Sequence[int], predictions: Sequence[Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Score roll-outs from `origins` of a validated window. `predictions[i][k]` is the forecast from
    `origins[i]` at lead step k+1 with `surf` / `atmos` numpy fields (H', W) where H' = H rounded down to
    the patch size, the rows the model actually predicts."""
    if len(origins) != len(predictions) or not origins:
        raise ValueError("origins and predictions must be non-empty and equal in length")
    height = window["shape"][0] - (window["shape"][0] % PATCH_SIZE)
    lat = window["lat"][:height]
    n_leads = len(predictions[0])
    level_index = LEVELS.index(HEADLINE_LEVEL)
    per_lead: dict[str, dict[str, dict[str, float]]] = {name: {} for name in REPORTED}
    for k in range(n_leads):
        lead = f"{(k + 1) * TIMESTEP_HOURS}h"
        errors: dict[str, list[float]] = {name: [] for name in REPORTED}
        persist: dict[str, list[float]] = {name: [] for name in REPORTED}
        for origin, preds in zip(origins, predictions, strict=True):
            if len(preds) != n_leads:
                raise ValueError("every origin must carry the same number of lead steps")
            target = origin + k + 1
            if target >= window["n_steps"]:
                raise ValueError(f"origin {origin} has no analysis at lead step {k + 1}")
            for name in SURF_VARS:
                truth = window["surf"][name][target, :height]
                errors[name].append(lat_weighted_rmse(preds[k]["surf"][name], truth, lat))
                persist[name].append(lat_weighted_rmse(window["surf"][name][origin, :height], truth, lat))
            for name in ATMOS_VARS:
                truth = window["atmos"][name][target, :, :height]
                errors[name].append(lat_weighted_rmse(preds[k]["atmos"][name], truth, lat))
                persist[name].append(lat_weighted_rmse(window["atmos"][name][origin, :, :height], truth, lat))
            truth = window["atmos"]["z"][target, level_index, :height]
            errors["z500"].append(lat_weighted_rmse(preds[k]["atmos"]["z"][level_index], truth, lat))
            persist["z500"].append(lat_weighted_rmse(window["atmos"]["z"][origin, level_index, :height], truth, lat))
        for name in REPORTED:
            model = sum(errors[name]) / len(errors[name])
            baseline = sum(persist[name]) / len(persist[name])
            per_lead[name][lead] = {
                "model": model,
                "persistence": baseline,
                "skill": model / baseline if baseline > 0 else math.nan,
            }
    leads = [f"{(k + 1) * TIMESTEP_HOURS}h" for k in range(n_leads)]
    summary = {
        lead: {
            "mean_skill": sum(per_lead[name][lead]["skill"] for name in REPORTED if name != "z500") / (len(REPORTED) - 1),
            "variables_beating_persistence": sum(
                1 for name in REPORTED if name != "z500" and per_lead[name][lead]["skill"] < 1.0
            ),
        }
        for lead in leads
    }
    return {
        "n_origins": len(origins),
        "leads": leads,
        "grid": [height, window["shape"][1]],
        "variables": per_lead,
        "summary": summary,
        "metric": "latitude-weighted RMSE (cos-lat weights, unit mean); atmospheric variables averaged over the 13 levels",
        "units": "as the variables: K, m/s, Pa, kg/kg, m²/s²",
    }


def persistence_only(window: Mapping[str, Any], *, max_lead_steps: int = 1) -> dict[str, Any]:
    """The persistence baseline alone (no model): the same table with the model column omitted."""
    import numpy as np

    from .pipeline import HISTORY_STEPS, _check_window

    checked = _check_window(window)
    last_origin = checked["n_steps"] - 1 - max_lead_steps
    origins = list(range(HISTORY_STEPS - 1, last_origin + 1))
    if not origins:
        raise ValueError("window too short for the requested lead")
    height = checked["shape"][0] - (checked["shape"][0] % PATCH_SIZE)
    predictions = [
        [
            {
                "surf": {n: np.asarray(checked["surf"][n][origin, :height]) for n in SURF_VARS},
                "atmos": {n: np.asarray(checked["atmos"][n][origin, :, :height]) for n in ATMOS_VARS},
            }
            for _ in range(max_lead_steps)
        ]
        for origin in origins
    ]
    metrics = forecast_metrics(checked, origins, predictions)
    metrics["variables"] = {
        name: {lead: {"persistence": v["persistence"]} for lead, v in leads.items()}
        for name, leads in metrics["variables"].items()
    }
    metrics.pop("summary")
    metrics["baseline"] = "persistence (latest analysis carried forward)"
    return metrics
