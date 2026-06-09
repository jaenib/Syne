"""Tonal features: chroma, key, mode (Krumhansl-Schmuckler key finding)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from syne.audio import AudioClip

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (major / minor).
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


@dataclass
class TonalFeatures:
    chroma: np.ndarray           # (12, n_frames)
    key: str
    mode: str                    # "major" | "minor"
    key_confidence: float        # 0..1
    chroma_centroid: float       # 0..11 circular-mean pitch class


def extract(clip: AudioClip) -> TonalFeatures:
    y, sr, hop = clip.samples, clip.sample_rate, clip.hop_length
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)

    key, mode, confidence = _estimate_key(chroma)
    centroid = _chroma_centroid(chroma)

    return TonalFeatures(
        chroma=chroma,
        key=key,
        mode=mode,
        key_confidence=confidence,
        chroma_centroid=centroid,
    )


def _estimate_key(chroma: np.ndarray) -> tuple[str, str, float]:
    """Correlate the mean chroma against all 24 rotated K-K profiles."""
    profile = chroma.mean(axis=1)
    if not np.any(profile):
        return "C", "major", 0.0

    best = (-2.0, 0, "major")
    scores = []
    for mode_name, base in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        for tonic in range(12):
            rotated = np.roll(base, tonic)
            r = _pearson(profile, rotated)
            scores.append(r)
            if r > best[0]:
                best = (r, tonic, mode_name)

    best_r, tonic, mode_name = best
    scores = np.array(scores)
    # confidence: separation of the winner from the field, mapped to 0..1.
    if scores.std() > 0:
        z = (best_r - scores.mean()) / scores.std()
        confidence = float(np.clip(z / 3.0, 0.0, 1.0))
    else:
        confidence = 0.0
    return PITCH_CLASSES[tonic], mode_name, confidence


def _chroma_centroid(chroma: np.ndarray) -> float:
    """Circular mean of pitch-class energy, expressed on a 0..11 scale."""
    weights = chroma.mean(axis=1)
    if not np.any(weights):
        return 0.0
    angles = np.arange(12) * (2 * np.pi / 12)
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    ang = np.arctan2(y, x) % (2 * np.pi)
    return float(ang / (2 * np.pi) * 12)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0
