"""Syne — semantic music tagging pipeline.

Turns an audio file into a structured :class:`SemanticProfile`: interpretable,
semantically-grouped track tags plus time-varying timelines, designed as a
stable contract for a future morphing-geometry renderer.
"""

from syne.pipeline import analyze_clip, analyze_file
from syne.semantics.schema import SemanticProfile
from syne.morph.drivers import derive_drivers

__version__ = "0.1.0"

__all__ = ["analyze_file", "analyze_clip", "SemanticProfile", "derive_drivers"]
