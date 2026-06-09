"""Feature extraction grouped into interpretable semantic categories.

The grouping (rhythm / tonal / timbre / dynamics / structure) follows the
"semantic-aware, interpretable feature groups" idea from SAMAT: every feature
belongs to a named perceptual group, which keeps the downstream tags
explainable. The STFT magnitude is computed once here and shared across groups.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from syne.audio import AudioClip
from syne.features import dynamics, rhythm, structure, timbre, tonal
from syne.features.dynamics import DynamicsFeatures
from syne.features.rhythm import RhythmFeatures
from syne.features.structure import StructureFeatures
from syne.features.timbre import TimbreFeatures
from syne.features.tonal import TonalFeatures


@dataclass
class FeatureBundle:
    clip: AudioClip
    frame_times: np.ndarray      # seconds, shared time grid for frame signals
    rhythm: RhythmFeatures
    tonal: TonalFeatures
    timbre: TimbreFeatures
    dynamics: DynamicsFeatures
    structure: StructureFeatures
    band_energy: np.ndarray      # (n_bands, n_frames) log-spaced band RMS


def extract_features(clip: AudioClip) -> FeatureBundle:
    """Run all feature groups and return a single bundle."""
    # Shared magnitude spectrogram — reused by timbre, dynamics, structure.
    S = np.abs(librosa.stft(clip.samples, n_fft=clip.n_fft, hop_length=clip.hop_length))

    rhythm_f = rhythm.extract(clip)
    tonal_f = tonal.extract(clip)
    timbre_f = timbre.extract(clip, S)
    dynamics_f = dynamics.extract(clip, S)
    structure_f = structure.extract(clip, S, tonal_f.chroma)
    band_energy = _band_energy(S, clip)

    n_frames = S.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=clip.sample_rate, hop_length=clip.hop_length
    )

    return FeatureBundle(
        clip=clip,
        frame_times=frame_times,
        rhythm=rhythm_f,
        tonal=tonal_f,
        timbre=timbre_f,
        dynamics=dynamics_f,
        structure=structure_f,
        band_energy=band_energy,
    )


def _band_energy(S: np.ndarray, clip: AudioClip, n_bands: int = 3) -> np.ndarray:
    """Per-frame RMS energy in ``n_bands`` log-spaced frequency bands.

    Splits the spectrum into (by default) bass / mid / treble — the spectral
    balance that a single ``brightness`` scalar otherwise throws away.
    """
    freqs = librosa.fft_frequencies(sr=clip.sample_rate, n_fft=clip.n_fft)
    fmin = max(freqs[1], 20.0)
    fmax = clip.sample_rate / 2.0
    edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
    power = S ** 2
    out = np.zeros((n_bands, S.shape[1]), dtype=np.float32)
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if np.any(mask):
            out[b] = np.sqrt(power[mask].mean(axis=0))
    return out


__all__ = ["FeatureBundle", "extract_features"]
