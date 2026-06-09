"""Shared test fixtures: synthetic audio so tests run fully offline."""

from __future__ import annotations

import numpy as np
import pytest

from syne.audio import from_array

SR = 22050


def _synth_track(seconds: float = 8.0, bpm: float = 120.0, sr: int = SR) -> np.ndarray:
    """A C-major-ish chord pulsed at ``bpm`` with percussive clicks on beats.

    Designed to exercise every feature group: clear tempo, a definite key,
    amplitude dynamics, and broadband click transients.
    """
    n = int(seconds * sr)
    t = np.arange(n) / sr

    # C major triad: C4, E4, G4
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.0))
    chord /= 3.0

    # amplitude envelope pulsing at the beat rate
    beat_hz = bpm / 60.0
    env = 0.5 + 0.5 * np.maximum(0.0, np.sin(2 * np.pi * beat_hz * t))
    sig = chord * env

    # percussive clicks on each beat
    beat_period = int(sr / beat_hz)
    for start in range(0, n, beat_period):
        end = min(start + 200, n)
        click = np.random.default_rng(0).standard_normal(end - start) * 0.4
        click *= np.linspace(1.0, 0.0, end - start)
        sig[start:end] += click

    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9
    return sig.astype(np.float32)


@pytest.fixture(scope="session")
def synth_clip():
    return from_array(_synth_track(), SR, source="<synthetic>")
