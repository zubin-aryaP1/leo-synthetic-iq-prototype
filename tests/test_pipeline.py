import math
from app.simulator import SyntheticIQ
from app.processing import detect, FrequencyTracker
from app.matching import match


def test_synthetic_iq_detector_tracks_and_matching():
    source = SyntheticIQ(seed=5)
    tracker = FrequencyTracker()
    for _ in range(190):
        iq, hidden = source.next_block()
        det, spectrum = detect(iq, source.sample_rate)
        assert iq.dtype.name == "complex64"
        assert len(spectrum) == 256
        active = tracker.step(det, hidden["simulation_time_s"])
    matches = [(tr, match(tr)) for tr in active]
    moving = [(tr, m) for tr, m in matches if m and m["satellite"] == "SIM-LEO-A"
              and m["confidence"] == "strong synthetic match"]
    assert moving, matches
    assert moving[0][1]["rmse_hz"] < 130
    # High-frequency stationary interferer must not be declared reliable satellite.
    for tr, m in matches:
        if tr.state[0] > 16000:
            assert m is None or m["confidence"] == "no reliable match"


def test_dropout_does_not_crash():
    source = SyntheticIQ(scenario="dropout")
    tracker = FrequencyTracker()
    for _ in range(114):
        iq, meta = source.next_block()
        d, _ = detect(iq, source.sample_rate)
        tracks = tracker.step(d, meta["simulation_time_s"])
    assert isinstance(tracks, list)

def test_joint_assignment_keeps_both_track_ids():
    tracker = FrequencyTracker()

    def peaks(*frequencies):
        return [{"frequency_hz": f, "snr_db": 25.0} for f in frequencies]

    tracker.step(peaks(0, 250), 0.0)
    # The closest pair (250, 150) would leave the 0 Hz track without a valid partner.
    tracker.step(peaks(150, 400), 0.1)
    assert set(tracker.tracks) == {1, 2}
    assert all(track.hits == 2 for track in tracker.tracks.values())


def test_frequency_estimate_between_fft_bins():
    import numpy as np
    fs = 100000
    frequency = 1234.5
    iq = np.exp(2j * np.pi * frequency * np.arange(10000) / fs)
    detections, _ = detect(iq, fs)
    assert abs(detections[0]["frequency_hz"] - frequency) < 1.0
