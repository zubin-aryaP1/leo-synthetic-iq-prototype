"""Independent prototype FFT/CA-CFAR frequency detection and Kalman tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
import numpy as np

NFFT = 4096


def detect(iq: np.ndarray, sample_rate: float):
    """Return sorted detections and a 256-bin, normalized display spectrum.

    CA-CFAR is approximate here (sliding 2-sided linear-power reference cells).
    Power/SNR are relative estimates, not calibrated receiver dBm.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < NFFT:
        raise ValueError("I/Q block shorter than FFT")
    chunks = min(2, len(iq) // NFFT)
    # Center the integration on the epoch midpoint used for frequency evaluation.
    start = (len(iq) - chunks * NFFT) // 2
    windows = iq[start:start + chunks * NFFT].reshape(chunks, NFFT)
    fft = np.fft.fftshift(np.fft.fft(windows * np.hanning(NFFT), axis=1), axes=1)
    power = (np.abs(fft) ** 2).mean(axis=0)
    train, guard = 18, 3
    kernel = np.zeros(2 * (train + guard) + 1, dtype=np.float64)
    kernel[:train] = 1.0
    kernel[-train:] = 1.0
    # Estimate noise from adjacent frequency cells, excluding the cells close to a candidate.
    noise = np.convolve(power, kernel, mode="same") / (2 * train)
    alpha = 2 * train * (1e-5 ** (-1.0 / (2 * train)) - 1.0)  # CA-CFAR, Pfa target 1e-5
    # A minimum of ~10 dB above reference guards against spurious noise peaks.
    snr = 10 * np.log10(np.maximum(power, 1e-12) / np.maximum(noise, 1e-12))
    is_peak = (power > np.roll(power, 1)) & (power > np.roll(power, -1))
    above = (power > alpha * noise) & (snr > 9.0) & is_peak
    above[:train+guard] = False
    above[-(train+guard):] = False
    candidate_idx = np.flatnonzero(above)
    strongest = sorted(candidate_idx, key=lambda k: power[k], reverse=True)
    selected = []
    for k in strongest:
        # Adjacent signal sidelobes describe a single signal, not new tracks.
        if any(abs(k - existing) < 9 for existing in selected):
            continue
        selected.append(int(k))
        if len(selected) >= 12:
            break
    hz = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / sample_rate))
    detections = []
    for k in selected:
        # A local quadratic fit to the Hann peak estimates the carrier between FFT bins.
        left, center, right = np.log(np.maximum(power[k-1:k+2], 1e-30))
        curvature = left - 2 * center + right
        fraction = np.clip(0.5 * (left - right) / curvature, -0.5, 0.5) if curvature < 0 else 0.0
        frequency = hz[k] + fraction * sample_rate / NFFT
        detections.append({"frequency_hz": round(float(frequency), 2), "snr_db": round(float(snr[k]), 1)})
    # Downsample spectrum by max pooling to preserve narrow peaks.
    pooled = (10 * np.log10(np.maximum(power, 1e-12))).reshape(256, NFFT // 256).max(axis=1)
    normalized = (pooled - float(np.median(pooled))).clip(-12, 60)
    return detections, [round(float(x), 2) for x in normalized]


@dataclass
class _Track:
    id: int
    state: np.ndarray
    covariance: np.ndarray
    last_t: float
    hits: int = 1
    missed: int = 0
    snr_db: float = 0.0
    observations: list = field(default_factory=list)

    def predict(self, t):
        dt = max(0.001, t - self.last_t)
        f = np.array([[1., dt], [0., 1.]])
        sigma_acc = 160.0
        q = sigma_acc ** 2 * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q
        self.last_t = t

    def update(self, freq, snr_db, t):
        h = np.array([[1., 0.]])
        innovation = float(freq - (h @ self.state)[0])
        measurement_variance = 28.0**2
        s = float((h @ self.covariance @ h.T)[0, 0] + measurement_variance)
        k = (self.covariance @ h.T)[:, 0] / s
        self.state += k * innovation
        self.covariance = (np.eye(2) - np.outer(k, h[0])) @ self.covariance
        self.hits += 1
        self.missed = 0
        self.snr_db = snr_db
        self.observations.append((t, float(self.state[0]), float(snr_db)))
        # Keep a bounded 60-second rolling window for matching.
        if len(self.observations) > 600:
            self.observations = self.observations[-600:]


class FrequencyTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def step(self, detections: list, t: float):
        for tr in self.tracks.values():
            tr.predict(t)
            tr.missed += 1

        remaining = list(detections)
        tracks = list(self.tracks.values())
        used_detections = set()
        if tracks and remaining:
            # Dummy columns allow every track to coast. Solve all associations jointly.
            frequencies = np.array([det["frequency_hz"] for det in remaining])
            costs = np.full((len(tracks), len(remaining) + len(tracks)), 16.0)
            for row, track in enumerate(tracks):
                innovation = frequencies - track.state[0]
                variance = track.covariance[0, 0] + 28.0**2
                distance = innovation**2 / variance
                costs[row, :len(remaining)] = np.where(
                    (distance < 16.0) & (np.abs(innovation) <= 260.0), distance, np.inf
                )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if column < len(remaining):
                    det = remaining[column]
                    tracks[row].update(det["frequency_hz"], det["snr_db"], t)
                    used_detections.add(column)
        # Create tentative tracks for unmatched strong detections.
        for i, det in enumerate(remaining):
            if i in used_detections or len(self.tracks) >= 16:
                continue
            tr = _Track(
                id=self.next_id,
                state=np.array([float(det["frequency_hz"]), 0.0]),
                covariance=np.diag([75.0**2, 1000.0**2]),
                last_t=t,
                snr_db=float(det["snr_db"]),
                observations=[(t, float(det["frequency_hz"]), float(det["snr_db"]))],
            )
            self.tracks[tr.id] = tr
            self.next_id += 1
        # Keep tracks across short fades, retire after a bounded gap.
        for tid, tr in list(self.tracks.items()):
            if tr.missed > 16:
                del self.tracks[tid]
        active = [tr for tr in self.tracks.values() if tr.hits >= 4 and tr.missed < 4]
        return sorted(active, key=lambda tr: (tr.hits, tr.snr_db), reverse=True)
