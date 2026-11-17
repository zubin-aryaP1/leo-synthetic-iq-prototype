"""Deterministic synthetic complex I/Q source. No UHD or RF hardware involved."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SCENARIOS = ("clean", "noisy", "interference", "dropout", "noaa_multi")


def reference_frequency(t: float, candidate: str = "SIM-LEO-A") -> float:
    """Illustrative Doppler-shaped curves, NOT SGP4 orbital predictions."""
    if candidate == "SIM-LEO-A":
        return -7800.0 * float(np.tanh((t - 35.0) / 12.0))
    if candidate == "SIM-LEO-B":
        return -6500.0 * float(np.tanh((t - 22.0) / 15.0))
    if candidate == "SIM-LEO-C":
        return 5800.0 * float(np.tanh((t - 39.0) / 12.0))
    raise ValueError(f"Unknown synthetic candidate: {candidate}")


# Historical NOAA APT carriers (MHz). Source: https://ospo.noaa.gov/operations/poes/status.html
# Separate simulated channels compress their nominal spacing into our 100 kS/s bandwidth.
NOAA_CHANNELS = (("NOAA-15", 137.620), ("NOAA-18", 137.9125), ("NOAA-19", 137.100))


@dataclass
class SyntheticIQ:
    sample_rate: int = 100_000
    epoch_seconds: float = 0.1
    scenario: str = "clean"
    seed: int = 77

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.rng = np.random.default_rng(self.seed)
        self.index = 0
        self.channel_phases = np.zeros(len(NOAA_CHANNELS))
        self.phase = 0.0
        self.interference_phase = 0.0
        self.n_per_epoch = int(round(self.sample_rate * self.epoch_seconds))
        if self.n_per_epoch < 4096:
            raise ValueError("Use >=4096 samples per epoch")

    @property
    def elapsed(self):
        return self.index * self.epoch_seconds

    def next_block(self):
        """Return (complex64 IQ, *hidden-for-evaluation* injection metadata)."""
        t = self.elapsed
        n = self.n_per_epoch
        times = t + np.arange(n, dtype=np.float64) / self.sample_rate
        # Same trajectory family as the reference, but with independent receiver bias
        # and slow drift; the matcher has to fit the bias from its observations.
        injected = np.array([reference_frequency(float(x)) for x in times])
        injected += 470.0 + 0.8 * times
        increments = 2 * np.pi * injected / self.sample_rate
        signal_phase = self.phase + np.cumsum(increments)
        self.phase = float(signal_phase[-1] % (2 * np.pi))
        carrier = 0.60 * np.exp(1j * signal_phase)

        # A constant-frequency competing signal should not match moving LEO curves.
        interference = 0.16 * np.exp(
            1j * (self.interference_phase + 2 * np.pi * 17_000.0 * np.arange(n) / self.sample_rate)
        )
        self.interference_phase = float(
            (self.interference_phase + 2 * np.pi * 17_000.0 * n / self.sample_rate) % (2 * np.pi)
        )
        if self.scenario == "interference":
            interference *= 2.8

        noise_amp = 0.24 if self.scenario in ("clean", "dropout") else 0.64
        noise = noise_amp * (
            self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
        ) / np.sqrt(2)
        # Brief absence of the target to test tracker missed-frame handling.
        if self.scenario == "dropout" and 9.0 <= t < 10.5:
            carrier *= 0.0
        if self.scenario == "noaa_multi":
            for index, (_, mhz) in enumerate(NOAA_CHANNELS):
                # Channel placement is compressed 12.5:1. Doppler uses the original RF.
                center_hz = (mhz - 137.55) * 80_000
                # Shift NOAA-15 clear of the existing synthetic pass.
                if index == 0:
                    center_hz -= 24_000
                radial_speed = 6800 * np.tanh((times - 20 - index * 12) / 20)
                frequencies = center_hz - mhz * 1e6 * radial_speed / 299792458
                phase = self.channel_phases[index] + np.cumsum(frequencies) * (2*np.pi/self.sample_rate)
                carrier += 0.38 * np.exp(1j * phase)
                self.channel_phases[index] = phase[-1] % (2*np.pi)
        iq = (carrier + interference + noise).astype(np.complex64)
        self.index += 1
        hidden = {"simulation_time_s": t, "injected_frequency_hz": float(injected[n // 2]),
                  "scenario": self.scenario, "synthetic": True}
        return iq, hidden
