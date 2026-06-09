"""Tagging layer: pluggable interface plus the default heuristic tagger."""

from syne.tagging.base import Tagger
from syne.tagging.heuristic import HeuristicTagger

__all__ = ["Tagger", "HeuristicTagger"]
