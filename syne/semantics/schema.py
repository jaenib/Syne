"""Semantic data contract for Syne.

This module defines the ``SemanticProfile`` — the single, stable hand-off
between the *tagging* pipeline and any future *renderer*. The renderer should
only ever read from this structure (or its serialized JSON form), never from
the raw audio or the internal feature objects.

The profile is intentionally split into two layers:

* **Track-level tags** (``tags``)  — global, interpretable descriptors for the
  whole piece, organized into semantic groups (rhythm, tonal, mood, energy,
  timbre, genre hints). This mirrors the "semantic-aware, interpretable
  grouping" idea from the SAMAT project.
* **Timelines** (``timelines``)    — time-varying signals sampled on a regular
  grid plus event lists (beats, segments). These let a renderer morph geometry
  *in sync* with the music rather than only reacting to one global mood.

Everything is plain data (dataclasses) so it serializes cleanly to JSON and is
trivial to consume from another language or process.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Real-time atomic unit: the instantaneous semantics at one analysis hop.
# A live tagger emits a stream of these; the renderer consumes them frame by
# frame. A SemanticProfile (below) is just a recording of such a stream.
# --------------------------------------------------------------------------- #
@dataclass
class SemanticFrame:
    t: float                     # timestamp (seconds, causal)
    # instantaneous per-frame signals, causally normalized to 0..1
    energy: float
    brightness: float
    flux: float
    onset: float
    flatness: float              # 0..1 noisiness (tonal <-> noisy)
    bands: list[float]           # per-band energy (0..1), bass..treble
    chroma: list[float]          # 12-dim pitch classes, L1-normalized
    # running (causal) semantic estimates
    tempo_bpm: float
    key: str
    mode: str                    # "major" | "minor"
    valence: float               # 0..1
    arousal: float               # 0..1
    # events fired this frame
    beat: bool = False
    section_change: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Track-level semantic tags (grouped, interpretable)
# --------------------------------------------------------------------------- #
@dataclass
class RhythmTags:
    tempo_bpm: float
    tempo_category: str          # e.g. "slow", "moderate", "fast"
    beat_strength: float         # 0..1, how pronounced the pulse is
    regularity: float            # 0..1, how steady the tempo is
    time_feel: str               # e.g. "straight", "swung", "loose"


@dataclass
class TonalTags:
    key: str                     # e.g. "C", "F#"
    mode: str                    # "major" | "minor"
    key_confidence: float        # 0..1
    chroma_centroid: float       # 0..11, dominant pitch class (circular mean)


@dataclass
class MoodTags:
    valence: float               # 0..1 (negative -> positive affect)
    arousal: float               # 0..1 (calm -> excited)
    label: str                   # quadrant label, e.g. "energetic", "serene"
    descriptors: list[str] = field(default_factory=list)


@dataclass
class EnergyTags:
    level: float                 # 0..1 overall loudness/energy
    dynamic_range: float         # 0..1, contrast between quiet and loud
    danceability: float          # 0..1, pulse regularity x energy
    label: str                   # e.g. "calm", "driving", "intense"


@dataclass
class TimbreTags:
    brightness: float            # 0..1 (dark -> bright), from spectral centroid
    warmth: float                # 0..1, low-band energy emphasis
    roughness: float             # 0..1, spectral flatness / noisiness
    descriptors: list[str] = field(default_factory=list)


@dataclass
class GenreHint:
    label: str
    confidence: float            # 0..1 — heuristic, treat as a hint only


@dataclass
class TrackTags:
    rhythm: RhythmTags
    tonal: TonalTags
    mood: MoodTags
    energy: EnergyTags
    timbre: TimbreTags
    genre_hints: list[GenreHint] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Time-varying signals (for morphing geometry)
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    start: float                 # seconds
    end: float                   # seconds
    label: str                   # e.g. "section_0"
    novelty: float               # 0..1, strength of the boundary into this segment


@dataclass
class Timelines:
    """Regularly-sampled signals share the ``times`` grid (seconds).

    Each signal is normalized to 0..1 unless noted, so a renderer can bind it
    to a geometry parameter without rescaling. ``chroma`` is a list of 12-dim
    vectors (one per frame) for pitch-class-driven color/shape mapping.
    """

    hop_seconds: float
    times: list[float]
    energy: list[float]              # 0..1 RMS envelope
    brightness: list[float]          # 0..1 spectral centroid over time
    flux: list[float]                # 0..1 spectral flux (change / motion)
    onset_strength: list[float]      # 0..1 onset envelope (attacks)
    chroma: list[list[float]]        # per-frame 12-dim pitch class energies
    bands: list[list[float]]         # per-frame log-spaced band energies (0..1)
    beats: list[float] = field(default_factory=list)      # beat times (s)
    segments: list[Segment] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Top-level profile
# --------------------------------------------------------------------------- #
@dataclass
class AudioMeta:
    source: str
    duration: float              # seconds
    sample_rate: int
    channels: int


@dataclass
class SemanticProfile:
    schema_version: str
    meta: AudioMeta
    tags: TrackTags
    timelines: Timelines

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str, *, indent: int | None = 2) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json(indent=indent))

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SemanticProfile":
        meta = AudioMeta(**d["meta"])
        t = d["tags"]
        tags = TrackTags(
            rhythm=RhythmTags(**t["rhythm"]),
            tonal=TonalTags(**t["tonal"]),
            mood=MoodTags(**t["mood"]),
            energy=EnergyTags(**t["energy"]),
            timbre=TimbreTags(**t["timbre"]),
            genre_hints=[GenreHint(**g) for g in t.get("genre_hints", [])],
        )
        tl = d["timelines"]
        timelines = Timelines(
            hop_seconds=tl["hop_seconds"],
            times=tl["times"],
            energy=tl["energy"],
            brightness=tl["brightness"],
            flux=tl["flux"],
            onset_strength=tl["onset_strength"],
            chroma=tl["chroma"],
            bands=tl.get("bands", []),
            beats=tl.get("beats", []),
            segments=[Segment(**s) for s in tl.get("segments", [])],
        )
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            meta=meta,
            tags=tags,
            timelines=timelines,
        )

    @classmethod
    def load(cls, path: str) -> "SemanticProfile":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
