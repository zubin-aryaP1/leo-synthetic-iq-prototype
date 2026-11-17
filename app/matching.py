"""Illustrative Doppler-track matching to synthetic reference curves.

No SGP4 or real satellite identification: all labels begin with SIM-.
"""
from __future__ import annotations

import numpy as np
from app.simulator import reference_frequency

CANDIDATES = ("SIM-LEO-A", "SIM-LEO-B", "SIM-LEO-C")


def match(track):
    obs = np.asarray(track.observations[-400:], dtype=float)
    if len(obs) < 12 or obs[-1, 0] - obs[0, 0] < 2.0:
        return None
    t, freq = obs[:, 0], obs[:, 1]
    ranked = []
    for name in CANDIDATES:
        pred = np.array([reference_frequency(float(x), name) for x in t])
        offset = float(np.mean(freq - pred))
        residual = freq - pred - offset
        rmse = float(np.sqrt(np.mean(residual**2)))
        ranked.append({"satellite": name, "rmse_hz": round(rmse, 1),
                       "fitted_offset_hz": round(offset, 1)})
    ranked.sort(key=lambda item: item["rmse_hz"])
    best = ranked[0]
    margin = ranked[1]["rmse_hz"] - best["rmse_hz"]
    duration = t[-1] - t[0]
    if best["rmse_hz"] > 500.0 or abs(best["fitted_offset_hz"]) > 2_000.0:
        confidence = "no reliable match"
    elif duration >= 7 and best["rmse_hz"] < 130 and margin > 130:
        confidence = "strong synthetic match"
    else:
        confidence = "tentative synthetic match"
    return {**best, "confidence": confidence, "duration_s": round(float(duration), 1),
            "runner_up_margin_hz": round(float(margin), 1),
            "predicted_frequency_hz": round(float(reference_frequency(float(t[-1]), best["satellite"]) + best["fitted_offset_hz"]),1),
            "synthetic_only": True}
