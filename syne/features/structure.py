"""Structure features: spectral flux (motion) and section boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from syne.audio import AudioClip


@dataclass
class StructureFeatures:
    flux: np.ndarray                     # (n_frames,) spectral flux, raw
    boundary_times: np.ndarray          # section boundaries, seconds
    boundary_novelty: np.ndarray        # 0..1 novelty at each boundary


def extract(clip: AudioClip, S: np.ndarray, chroma: np.ndarray) -> StructureFeatures:
    sr, hop = clip.sample_rate, clip.hop_length

    flux = _spectral_flux(S)
    boundary_times, novelty = _segment(clip, chroma, sr, hop)

    return StructureFeatures(
        flux=flux,
        boundary_times=boundary_times,
        boundary_novelty=novelty,
    )


def _spectral_flux(S: np.ndarray) -> np.ndarray:
    """Half-wave-rectified frame-to-frame spectral difference."""
    diff = np.diff(S, axis=1, prepend=S[:, :1])
    flux = np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2, axis=0))
    return flux


def _segment(
    clip: AudioClip, chroma: np.ndarray, sr: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative segmentation over stacked MFCC + chroma features."""
    n_frames = chroma.shape[1]
    if n_frames < 8:
        return np.array([0.0]), np.array([1.0])

    mfcc = librosa.feature.mfcc(y=clip.samples, sr=sr, hop_length=hop, n_mfcc=13)
    mfcc = librosa.util.fix_length(mfcc, size=n_frames, axis=1)
    feats = np.vstack([librosa.util.normalize(mfcc, axis=1),
                       librosa.util.normalize(chroma, axis=1)])

    # aim for ~1 section per 12 s, bounded to a sensible range
    duration = clip.duration
    k = int(np.clip(round(duration / 12.0), 2, 8))
    k = min(k, n_frames - 1)
    bounds = librosa.segment.agglomerative(feats, k)
    boundary_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)

    novelty = _boundary_novelty(feats, bounds)
    return boundary_times, novelty


def _boundary_novelty(feats: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Cosine distance across each boundary, normalized to 0..1."""
    if bounds.size == 0:
        return np.array([])
    out = []
    for b in bounds:
        lo = max(b - 4, 0)
        hi = min(b + 4, feats.shape[1])
        if b <= lo or hi <= b:
            out.append(0.0)
            continue
        before = feats[:, lo:b].mean(axis=1)
        after = feats[:, b:hi].mean(axis=1)
        denom = np.linalg.norm(before) * np.linalg.norm(after)
        cos = np.dot(before, after) / denom if denom else 1.0
        out.append((1.0 - cos) / 2.0)       # cosine in [-1,1] -> distance [0,1]
    arr = np.array(out)
    return np.clip(arr, 0.0, 1.0)
