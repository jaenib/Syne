"""Syne — semantic music tagging pipeline.

Turns an audio file into a structured :class:`SemanticProfile`: interpretable,
semantically-grouped track tags plus time-varying timelines, designed as a
stable contract for a future morphing-geometry renderer.
"""

from syne.pipeline import analyze_clip, analyze_file, frames_to_profile
from syne.semantics.schema import SemanticFrame, SemanticProfile
from syne.stream import HeuristicStreamingTagger, StreamingTagger
from syne.morph.drivers import derive_drivers

__version__ = "0.1.0"

__all__ = [
    "analyze_file", "analyze_clip", "frames_to_profile",
    "SemanticProfile", "SemanticFrame",
    "StreamingTagger", "HeuristicStreamingTagger",
    "derive_drivers",
]
