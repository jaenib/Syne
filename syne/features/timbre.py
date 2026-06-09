"""Timbre features: spectral shape descriptors and MFCCs.

Operates on a precomputed magnitude spectrogram so the STFT is only run once
for the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from syne.audio import AudioClip


@dataclass
class TimbreFeatures:
    spectral_centroid: np.ndarray    # (n_frames,) Hz
    spectral_rolloff: np.ndarray     # (n_frames,) Hz
    spectral_flatness: np.ndarray    # (n_frames,) 0..1
    mfcc: np.ndarray                 # (n_mfcc, n_frames)
    brightness: float                # 0..1, normalized centroid
    warmth: float                    # 0..1, low-band energy share
    roughness: float                 # 0..1, mean flatness (noisiness)


def extract(clip: AudioClip, S: np.ndarray) -> TimbreFeatures:
    sr, hop, n_fft = clip.sample_rate, clip.hop_length, clip.n_fft

    centroid = librosa.feature.spectral_centroid(
        S=S, sr=sr, hop_length=hop, n_fft=n_fft
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=S, sr=sr, hop_length=hop, n_fft=n_fft, roll_percent=0.85
    )[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    mfcc = librosa.feature.mfcc(y=clip.samples, sr=sr, hop_length=hop, n_mfcc=13)

    nyquist = sr / 2.0
    brightness = float(np.clip(np.mean(centroid) / nyquist, 0.0, 1.0))
    warmth = _warmth(S, sr, n_fft)
    roughness = float(np.clip(np.mean(flatness), 0.0, 1.0))

    return TimbreFeatures(
        spectral_centroid=centroid,
        spectral_rolloff=rolloff,
        spectral_flatness=flatness,
        mfcc=mfcc,
        brightness=brightness,
        warmth=warmth,
        roughness=roughness,
    )


def _warmth(S: np.ndarray, sr: int, n_fft: int) -> float:
    """Share of spectral energy below ~500 Hz (low-end emphasis)."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = S.sum()
    if total <= 0:
        return 0.0
    low = S[freqs < 500.0].sum()
    return float(np.clip(low / total, 0.0, 1.0))
