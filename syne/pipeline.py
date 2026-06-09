"""End-to-end pipeline: audio file -> SemanticProfile.

    load_audio -> extract_features -> build timelines + run tagger -> profile

The tagger is injected so the heuristic DSP tagger can be swapped for a learned
one without touching the rest of the pipeline.
"""

from __future__ import annotations

import librosa
import numpy as np

from syne.audio import AudioClip, load_audio
from syne.features import FeatureBundle, extract_features
from syne.semantics.schema import (
    SCHEMA_VERSION,
    AudioMeta,
    Segment,
    SemanticProfile,
    Timelines,
)
from syne.tagging.base import Tagger
from syne.tagging.heuristic import HeuristicTagger
from syne.util import normalize01, to_list


def analyze_file(path: str, *, tagger: Tagger | None = None, **load_kwargs) -> SemanticProfile:
    """Analyze an audio file on disk and return its semantic profile."""
    clip = load_audio(path, **load_kwargs)
    return analyze_clip(clip, tagger=tagger)


def analyze_clip(clip: AudioClip, *, tagger: Tagger | None = None) -> SemanticProfile:
    """Analyze an already-loaded :class:`AudioClip`."""
    tagger = tagger or HeuristicTagger()
    features = extract_features(clip)

    tags = tagger.tag(features)
    timelines = _build_timelines(features)
    meta = AudioMeta(
        source=clip.source,
        duration=round(clip.duration, 4),
        sample_rate=clip.sample_rate,
        channels=clip.channels,
    )
    return SemanticProfile(
        schema_version=SCHEMA_VERSION,
        meta=meta,
        tags=tags,
        timelines=timelines,
    )


def _build_timelines(f: FeatureBundle) -> Timelines:
    n = len(f.frame_times)

    def grid(x: np.ndarray) -> np.ndarray:
        """Align a frame signal to the shared grid length."""
        return librosa.util.fix_length(np.asarray(x), size=n)

    energy = normalize01(grid(f.dynamics.rms))
    brightness = normalize01(grid(f.timbre.spectral_centroid))
    flux = normalize01(grid(f.structure.flux))
    onset = normalize01(grid(f.rhythm.onset_envelope))

    # chroma: (12, n) -> per-frame 12-vectors, each frame L1-normalized
    chroma = librosa.util.fix_length(f.tonal.chroma, size=n, axis=1)
    chroma_t = chroma.T
    sums = chroma_t.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    chroma_norm = chroma_t / sums

    # bands: (n_bands, n) -> per-frame vectors, each band robust-normalized to 0..1
    bands = librosa.util.fix_length(f.band_energy, size=n, axis=1)
    bands_norm = np.stack([normalize01(grid(b)) for b in bands], axis=1)  # (n, n_bands)

    segments = _build_segments(f)

    return Timelines(
        hop_seconds=round(f.clip.hop_seconds, 6),
        times=to_list(f.frame_times),
        energy=to_list(energy),
        brightness=to_list(brightness),
        flux=to_list(flux),
        onset_strength=to_list(onset),
        chroma=[to_list(row, decimals=4) for row in chroma_norm],
        bands=[to_list(row, decimals=4) for row in bands_norm],
        beats=to_list(f.rhythm.beat_times),
        segments=segments,
    )


def _build_segments(f: FeatureBundle) -> list[Segment]:
    bounds = np.asarray(f.structure.boundary_times, dtype=float)
    novelty = np.asarray(f.structure.boundary_novelty, dtype=float)
    duration = f.clip.duration

    starts = np.unique(np.clip(bounds, 0.0, duration))
    if starts.size == 0 or starts[0] != 0.0:
        starts = np.insert(starts, 0, 0.0)

    segments: list[Segment] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else duration
        if end - start < 1e-3:
            continue
        nov = float(novelty[i]) if i < len(novelty) else 0.0
        segments.append(
            Segment(
                start=round(float(start), 4),
                end=round(float(end), 4),
                label=f"section_{len(segments)}",
                novelty=round(nov, 4),
            )
        )
    return segments
