"""Dynamics features: RMS energy envelope, loudness, dynamic range."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from syne.audio import AudioClip


@dataclass
class DynamicsFeatures:
    rms: np.ndarray              # (n_frames,) raw RMS
    energy_level: float          # 0..1, perceptual loudness proxy
    dynamic_range: float         # 0..1, quiet-to-loud contrast


def extract(clip: AudioClip, S: np.ndarray) -> DynamicsFeatures:
    rms = librosa.feature.rms(S=S, hop_length=clip.hop_length)[0]
    energy_level = _energy_level(rms)
    dynamic_range = _dynamic_range(rms)
    return DynamicsFeatures(
        rms=rms,
        energy_level=energy_level,
        dynamic_range=dynamic_range,
    )


def _energy_level(rms: np.ndarray) -> float:
    """Median RMS mapped through a dB curve into 0..1."""
    if not np.any(rms):
        return 0.0
    med = float(np.median(rms))
    db = librosa.amplitude_to_db(np.array([med]), ref=1.0)[0]
    # map a useful loudness window (-60..0 dB) onto 0..1
    return float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))


def _dynamic_range(rms: np.ndarray) -> float:
    """Spread between loud (95th pct) and quiet (5th pct) sections, in dB."""
    if not np.any(rms):
        return 0.0
    loud = np.percentile(rms, 95)
    quiet = np.percentile(rms, 5)
    if quiet <= 0:
        quiet = max(rms[rms > 0].min(), 1e-6) if np.any(rms > 0) else 1e-6
    db_range = 20.0 * np.log10(loud / quiet)
    # ~40 dB is a very wide range; clamp into 0..1
    return float(np.clip(db_range / 40.0, 0.0, 1.0))
