"""Score one labeled target per recording using measured, confirmed tracks."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from app.processing import FrequencyTracker, NFFT, detect


def evaluation_events(path, labels, tolerance):
    meta = json.loads(path.with_suffix(".sigmf-meta").read_text())
    info = meta["global"]
    captures = meta["captures"]
    if (info["core:datatype"] != "cf32_le" or info.get("core:num_channels", 1) != 1
            or len(captures) != 1 or captures[0]["core:sample_start"] != 0
            or captures[0].get("core:header_bytes", 0) or info.get("core:offset", 0)
            or info.get("core:trailing_bytes", 0)):
        raise ValueError("Expected single-channel headerless cf32_le, one capture starting at sample zero.")
    rate = float(info["core:sample_rate"])
    size = round(rate * 0.1)
    if size < NFFT or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Need at least 4096 samples per epoch and a positive tolerance.")
    raw = np.memmap(path, dtype="<c8", mode="r")
    epochs = len(raw) // size
    if not epochs or any(i < 0 or i >= epochs for i in labels):
        raise ValueError("Empty recording or label outside complete epochs.")
    tracker = FrequencyTracker()
    errors = []
    seen = set()
    observed = switches = 0
    last_epoch = last_id = None
    for i in range(epochs):
        start = i * size
        detections, _ = detect(raw[start:start + size], rate)
        tracks = tracker.step(detections, (start + size / 2) / rate)
        measured = [t for t in tracks if t.missed == 0]
        observed += bool(measured)
        seen.update(t.id for t in measured)
        if i in labels and measured:
            closest = min(measured, key=lambda t: abs(t.state[0] - labels[i]))
            error = float(closest.state[0] - labels[i])
            if abs(error) <= tolerance:
                errors.append(error)
                switches += last_epoch == i - 1 and last_id != closest.id
                last_epoch, last_id = i, closest.id
        if i % max(1, epochs // 100) == 0 or i == epochs - 1:
            yield dict(processed=i+1, epochs=epochs, tracks=len(measured), distinct=len(seen))
    score = None
    if labels:
        score = dict(labeled=len(labels), matched=len(errors), missed=len(labels)-len(errors),
                     tolerance_hz=tolerance, consecutive_id_switches=switches,
                     matched_rmse_hz=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)
    yield dict(epochs=epochs, epoch_seconds=size/rate, observed_epochs=observed,
                distinct_confirmed_tracks=len(seen), accuracy=score)


def evaluate(path, labels, tolerance):
    for event in evaluation_events(path, labels, tolerance):
        pass
    return event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--reference", type=Path, help="Single-target CSV: epoch,offset_hz")
    parser.add_argument("--tolerance-hz", type=float, default=100.0)
    args = parser.parse_args()
    labels = {}
    if args.reference:
        with args.reference.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                i, freq = int(row["epoch"]), float(row["offset_hz"])
                if i in labels or not math.isfinite(freq):
                    parser.error("Expected unique epochs and finite reference frequencies.")
                labels[i] = freq
        if not labels:
            parser.error("Reference CSV has no labels.")
    print(json.dumps(evaluate(args.recording, labels, args.tolerance_hz), indent=2))
