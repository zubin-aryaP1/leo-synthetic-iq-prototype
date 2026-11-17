"""Run from project root: python -m benchmarks.compare_detector path/to/satellite_tracker_api.py"""
import argparse
import runpy

import numpy as np

from app.processing import detect
from app.simulator import SCENARIOS, SyntheticIQ


def compare(reference_path):
    reference = runpy.run_path(str(reference_path), run_name="reference_tracker")
    for scenario in SCENARIOS:
        source = SyntheticIQ(scenario=scenario, seed=77)
        config = reference["DetectorConfig"](
            sample_rate_hz=source.sample_rate, epoch_duration_s=source.epoch_seconds,
            cfar_excluded_lower_hz=None, cfar_excluded_upper_hz=None,
        )
        baseline = reference["Detector"](config)
        errors = {"reference": [], "prototype": []}
        visible = 0
        for _ in range(400):
            iq, truth = source.next_block()
            if scenario == "dropout" and 9 <= truth["simulation_time_s"] < 10.5:
                continue
            visible += 1
            expected = truth["injected_frequency_hz"]
            detections, _ = detect(iq, source.sample_rate)
            outputs = (baseline.process_block(iq).detected_frequency_offsets_hz,
                       [d["frequency_hz"] for d in detections])
            for name, frequencies in zip(errors, outputs):
                nearest = min((abs(f - expected) for f in frequencies), default=np.inf)
                if nearest <= 100:
                    errors[name].append(nearest)
        for name, values in errors.items():
            rmse = np.sqrt(np.mean(np.square(values))) if values else np.nan
            print(f"{scenario:14} {name:10} {len(values)}/{visible} hits, RMSE {rmse:.2f} Hz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_path")
    compare(parser.parse_args().reference_path)
