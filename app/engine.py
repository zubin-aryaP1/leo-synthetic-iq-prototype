"""Synthetic source -> DSP -> tracker -> candidate matching, independent of API/DB."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from uuid import uuid4

from app.simulator import SyntheticIQ, SCENARIOS, reference_frequency
from app.processing import FrequencyTracker, detect
from app.matching import match


class DemoEngine:
    def __init__(self, scenario="clean", seed=77):
        self.seed = seed
        self.reset(scenario)

    def reset(self, scenario="clean"):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.source = SyntheticIQ(scenario=scenario, seed=self.seed)
        self.tracker = FrequencyTracker()
        self.run_id = str(uuid4())
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.running = False
        self.last = {
            "run_id": self.run_id, "scenario": scenario, "simulated": True,
            "elapsed_seconds": 0., "sample_rate_sps": self.source.sample_rate,
            "detections": [], "tracks": [], "best_match": None,
            "spectrum_db_relative": [], "reference": [], "dropped_events": 0,
        }
        self.plot = deque(maxlen=600)
        return self.last

    def step(self):
        iq, hidden = self.source.next_block()
        detections, spectrum = detect(iq, self.source.sample_rate)
        t = hidden["simulation_time_s"]
        tracks = self.tracker.step(detections, t)
        items = []
        rows = []
        for tr in tracks[:8]:
            matched = match(tr)
            item = {
                "id": tr.id,
                "frequency_hz": round(float(tr.state[0]), 1),
                "frequency_rate_hz_s": round(float(tr.state[1]), 1),
                "snr_db": round(float(tr.snr_db), 1),
                "observations": len(tr.observations),
                "match": matched,
            }
            items.append(item)
            rows.append({"run_id": self.run_id, "elapsed_s": t, **item})
        def rank(item):
            m = item.get("match")
            if not m or m["confidence"] == "no reliable match":
                return (1, float("inf"))
            return (0 if m["confidence"] == "strong synthetic match" else 0.5, m["rmse_hz"])
        plausible = [x for x in items if x.get("match") and x["match"]["confidence"] != "no reliable match"]
        best = min(plausible, key=rank) if plausible else None
        observed = best["frequency_hz"] if best else None
        predicted = best["match"]["predicted_frequency_hz"] if best else None
        self.plot.append({"t": round(t,2),"observed":observed,"predicted":predicted})
        self.last = {
            "run_id": self.run_id, "scenario": self.source.scenario,
            "simulated": True, "elapsed_seconds": round(t, 2),
            "sample_rate_sps": self.source.sample_rate,
            "sample_count": len(iq),
            "detections": detections,
            "tracks": items, "best_match": best,
            "spectrum_db_relative": spectrum,
            "plot": list(self.plot),
            "running": self.running,
        }
        return self.last, rows
