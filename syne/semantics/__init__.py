"""Semantic data contract shared by the tagger and future renderers."""

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

__all__ = [
    "SCHEMA_VERSION",
    "AudioMeta",
    "EnergyTags",
    "GenreHint",
    "MoodTags",
    "RhythmTags",
    "Segment",
    "SemanticFrame",
    "SemanticProfile",
    "Timelines",
    "TimbreTags",
    "TonalTags",
    "TrackTags",
]
