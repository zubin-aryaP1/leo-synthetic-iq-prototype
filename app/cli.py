"""Create/replay synthetic SigMF without USRP hardware.

Usage:
  python -m app.cli generate --seconds 12 --out recordings/synthetic
  python -m app.cli replay recordings/synthetic.sigmf-data
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from app.simulator import SyntheticIQ, SCENARIOS
from app.processing import detect, FrequencyTracker
from app.matching import match


def generate(args):
    if not 0.2 <= args.seconds <= 120:
        raise SystemExit("Duration must be 0.2–120 seconds (to limit disk use).")
    source = SyntheticIQ(scenario=args.scenario)
    data = Path(args.out).with_suffix(".sigmf-data")
    meta = Path(args.out).with_suffix(".sigmf-meta")
    data.parent.mkdir(parents=True, exist_ok=True)
    num_epochs = int(round(args.seconds / source.epoch_seconds))
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with data.open("wb") as out:
        for _ in range(num_epochs):
            iq, _ = source.next_block()
            iq.astype("<c8", copy=False).tofile(out)
    meta_obj = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": source.sample_rate,
            "core:version": "1.0.0",
            "core:description": "SYNTHETIC GENERATED I/Q; NOT a real satellite recording",
            "core:recorder": "leo-synthetic-prototype",
            "core:hw": "simulated complex64 source; no USRP",
        },
        "captures": [{"core:sample_start": 0, "core:frequency": 2_575_000_000,
                      "core:datetime": timestamp}],
        "annotations": [],
        "synthetic_info": {"scenario": args.scenario, "signal": "SIM-LEO-A",
                           "sample_rate_sps": source.sample_rate, "epoch_seconds": source.epoch_seconds,
                           "samples_written": num_epochs * source.n_per_epoch,
                           "frequency_reference": "2.575 GHz illustrative IF center; not real Ku-band RF"},
    }
    meta.write_text(json.dumps(meta_obj, indent=2))
    print(f"Generated SYNTHETIC I/Q: {data} ({data.stat().st_size / 1e6:.2f} MB)")
    print(f"Metadata: {meta}")


def replay(args):
    data = Path(args.path)
    meta = data.with_suffix(".sigmf-meta")
    if not data.exists() or not meta.exists():
        raise SystemExit("Need both .sigmf-data and adjacent .sigmf-meta files")
    info = json.loads(meta.read_text())
    if info["global"].get("core:datatype") != "cf32_le":
        raise SystemExit("This minimal replay supports only cf32_le")
    sample_rate = float(info["global"]["core:sample_rate"])
    epoch_s = 0.1
    epoch_n = int(sample_rate * epoch_s)
    raw = np.memmap(data, dtype="<c8", mode="r")
    tracker = FrequencyTracker()
    last = []
    for i, offset in enumerate(range(0, len(raw) - epoch_n + 1, epoch_n)):
        samples = raw[offset:offset + epoch_n]
        detections, _ = detect(samples, sample_rate)
        last = tracker.step(detections, i * epoch_s)
    print(f"Replayed: {data} ({len(raw) / sample_rate:.1f} seconds)")
    for track in last[:8]:
        m = match(track)
        print(f"Track {track.id}: {track.state[0]:.1f} Hz, hits={track.hits}, match={m}")
    if not last:
        print("No confirmed tracks in final epoch")


def main():
    parser = argparse.ArgumentParser(description="Generate and replay synthetic (not real) SigMF I/Q")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--seconds", type=float, default=12.0)
    g.add_argument("--scenario", choices=SCENARIOS, default="clean")
    g.add_argument("--out", default="recordings/synthetic")
    r = sub.add_parser("replay")
    r.add_argument("path")
    args = parser.parse_args()
    (generate if args.command == "generate" else replay)(args)


if __name__ == "__main__":
    main()
