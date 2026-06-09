"""Pluggable tagger interface.

A tagger turns a :class:`~syne.features.FeatureBundle` into the track-level
:class:`~syne.semantics.schema.TrackTags`. The heuristic DSP tagger is the
default implementation; a future deep-learning tagger (e.g. a model trained on
MTG-Jamendo tags) only has to satisfy this same interface to drop in, exactly
like the model-agnostic design in SAMAT.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from syne.features import FeatureBundle
from syne.semantics.schema import TrackTags


@runtime_checkable
class Tagger(Protocol):
    name: str

    def tag(self, features: FeatureBundle) -> TrackTags:
        """Produce track-level semantic tags from extracted features."""
        ...
