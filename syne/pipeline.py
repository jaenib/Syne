"""Pipeline: drive the streaming tagger and aggregate its frame stream.

Real-time-first: ``analyze_*`` run the *same* streaming tagger the live
renderer uses, then aggregate the emitted :class:`SemanticFrame` stream into a
whole-track :class:`SemanticProfile` (a recording of the stream). So the
offline path and the GIF test harness exercise the actual real-time code rather
than a parallel batch implementation.
"""

from __future__ import annotations

import numpy as np

from syne.audio import AudioClip, load_audio
from syne.semantics.schema import (
    SCHEMA_VERSION,
    AudioMeta,
    EnergyTags,
    GenreHint,
    MoodTags,
    RhythmTags,
    Segment,
    SemanticFrame,
    SemanticProfile,
    Timelines,
    TimbreTags,
    TonalTags,
    TrackTags,
)
from syne.stream import HeuristicStreamingTagger, StreamingTagger
from syne.tagging import vocab


def analyze_file(path: str, *, tagger: StreamingTagger | None = None, **load_kwargs) -> SemanticProfile:
    clip = load_audio(path, **load_kwargs)
    return analyze_clip(clip, tagger=tagger)


def analyze_clip(clip: AudioClip, *, tagger: StreamingTagger | None = None) -> SemanticProfile:
    """Stream ``clip`` through the tagger and aggregate to a profile."""
    tagger = tagger or HeuristicStreamingTagger(clip.sample_rate, hop_length=clip.hop_length)
    tagger.reset()
    frames = tagger.push(clip.samples)
    return frames_to_profile(frames, clip)


# --------------------------------------------------------------------------- #
def frames_to_profile(frames: list[SemanticFrame], clip: AudioClip) -> SemanticProfile:
    """Aggregate a SemanticFrame stream into a whole-track SemanticProfile."""
    meta = AudioMeta(
        source=clip.source,
        duration=round(clip.duration, 4),
        sample_rate=clip.sample_rate,
        channels=clip.channels,
    )
    if not frames:
        return SemanticProfile(SCHEMA_VERSION, meta, _empty_tags(), _empty_timelines(clip))

    times = np.array([f.t for f in frames])
    energy = np.array([f.energy for f in frames])
    brightness = np.array([f.brightness for f in frames])
    flux = np.array([f.flux for f in frames])
    onset = np.array([f.onset for f in frames])
    flatness = np.array([f.flatness for f in frames])
    chroma = np.array([f.chroma for f in frames])           # (n, 12)
    bands = np.array([f.bands for f in frames])             # (n, n_bands)
    beats = [f.t for f in frames if f.beat]

    timelines = Timelines(
        hop_seconds=round(clip.hop_seconds, 6),
        times=_r(times),
        energy=_r(energy), brightness=_r(brightness),
        flux=_r(flux), onset_strength=_r(onset),
        chroma=[_r(c, 4) for c in chroma],
        bands=[_r(b, 4) for b in bands],
        beats=_r(np.array(beats)),
        segments=_segments(frames, clip.duration),
    )
    tags = _aggregate_tags(frames, times, energy, brightness, flux, onset,
                           flatness, chroma, bands, beats)
    return SemanticProfile(SCHEMA_VERSION, meta, tags, timelines)


def _aggregate_tags(frames, times, energy, brightness, flux, onset, flatness,
                    chroma, bands, beats) -> TrackTags:
    # use the settled second half for stable track-level estimates
    warm = len(frames) // 2
    tempo = float(np.median([f.tempo_bpm for f in frames[warm:]] or [frames[-1].tempo_bpm]))

    # key / mode: majority vote over the settled region
    keys = [(f.key, f.mode) for f in frames[warm:]] or [(frames[-1].key, frames[-1].mode)]
    (key, mode), votes = _mode_of(keys)
    key_conf = round(votes / len(keys), 4)

    regularity = _regularity(beats)
    beat_strength = float(np.clip(onset.mean() * 1.5, 0, 1))
    valence = float(np.mean([f.valence for f in frames[warm:]]))
    arousal = float(np.mean([f.arousal for f in frames[warm:]]))
    level = float(energy.mean())
    dyn_range = float(np.clip(np.percentile(energy, 95) - np.percentile(energy, 5), 0, 1))
    danceability = float(np.clip(regularity * beat_strength * (0.5 + 0.5 * level), 0, 1))
    bright = float(brightness.mean())
    warmth = float(bands[:, 0].mean()) if bands.size else 0.0
    roughness = float(flatness.mean())

    return TrackTags(
        rhythm=RhythmTags(
            tempo_bpm=round(tempo, 2),
            tempo_category=vocab.tempo_category(tempo),
            beat_strength=round(beat_strength, 4),
            regularity=round(regularity, 4),
            time_feel=vocab.time_feel(0.0, regularity),
        ),
        tonal=TonalTags(
            key=key, mode=mode, key_confidence=key_conf,
            chroma_centroid=round(_chroma_centroid(chroma.mean(axis=0)), 4),
        ),
        mood=MoodTags(
            valence=round(valence, 4), arousal=round(arousal, 4),
            label=vocab.mood_label(valence, arousal),
            descriptors=vocab.mood_descriptors(valence, arousal, mode),
        ),
        energy=EnergyTags(
            level=round(level, 4), dynamic_range=round(dyn_range, 4),
            danceability=round(danceability, 4), label=vocab.energy_label(level),
        ),
        timbre=TimbreTags(
            brightness=round(bright, 4), warmth=round(warmth, 4),
            roughness=round(roughness, 4),
            descriptors=vocab.timbre_descriptors(bright, warmth, roughness),
        ),
        genre_hints=_genre_hints(tempo, bright, roughness, level, danceability, warmth),
    )


def _segments(frames: list[SemanticFrame], duration: float) -> list[Segment]:
    starts = [0.0] + [f.t for f in frames if f.section_change]
    segs = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else duration
        if end - s < 1e-3:
            continue
        segs.append(Segment(start=round(s, 4), end=round(end, 4),
                            label=f"section_{len(segs)}", novelty=0.5))
    return segs


def _genre_hints(tempo, bright, rough, level, dance, warm) -> list[GenreHint]:
    scored = {
        "electronic": 0.6 * dance + 0.3 * (tempo >= 115) + 0.2 * bright,
        "ambient": 0.6 * (level < 0.35) + 0.3 * warm + 0.2 * (tempo < 90),
        "rock": 0.5 * rough + 0.3 * (level > 0.6) + 0.2 * (90 <= tempo <= 160),
        "acoustic": 0.5 * (1 - rough) + 0.3 * warm,
    }
    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return [GenreHint(label=k, confidence=round(float(np.clip(v, 0, 1)) * 0.7, 4))
            for k, v in ranked if v * 0.7 >= 0.15]


# --------------------------------------------------------------------------- #
def _regularity(beats: list[float]) -> float:
    if len(beats) < 3:
        return 0.0
    ibi = np.diff(beats)
    if ibi.mean() <= 0:
        return 0.0
    return float(np.clip(1.0 - ibi.std() / ibi.mean(), 0.0, 1.0))


def _chroma_centroid(weights: np.ndarray) -> float:
    if not np.any(weights):
        return 0.0
    ang = np.arange(12) * (2 * np.pi / 12)
    a = np.arctan2(np.sum(weights * np.sin(ang)), np.sum(weights * np.cos(ang)))
    return float((a % (2 * np.pi)) / (2 * np.pi) * 12)


def _mode_of(pairs: list[tuple]) -> tuple[tuple, int]:
    counts: dict[tuple, int] = {}
    for p in pairs:
        counts[p] = counts.get(p, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0], best[1]


def _empty_tags() -> TrackTags:
    return TrackTags(
        rhythm=RhythmTags(0.0, "slow", 0.0, 0.0, "loose"),
        tonal=TonalTags("C", "major", 0.0, 0.0),
        mood=MoodTags(0.5, 0.5, "serene", []),
        energy=EnergyTags(0.0, 0.0, 0.0, "calm"),
        timbre=TimbreTags(0.0, 0.0, 0.0, []),
        genre_hints=[],
    )


def _empty_timelines(clip: AudioClip) -> Timelines:
    return Timelines(round(clip.hop_seconds, 6), [], [], [], [], [], [], [], [], [])


def _r(x, d: int = 5) -> list[float]:
    return [round(float(v), d) for v in np.asarray(x).ravel()]
