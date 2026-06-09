"""Tagging layer.

The tagger is the swap point in ``interface -> tagger -> renderer``: the
heuristic streaming mock here is meant to be replaced by a data-driven object
implementing the same :class:`~syne.stream.StreamingTagger` protocol.
``vocab`` holds the interpretable label mappings used when aggregating a frame
stream into a track-level summary.
"""

from syne.stream import HeuristicStreamingTagger, StreamingTagger

__all__ = ["StreamingTagger", "HeuristicStreamingTagger"]
