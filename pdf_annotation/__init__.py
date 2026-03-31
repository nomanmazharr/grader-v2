"""PDF annotation package.

This package contains the implementation used by the legacy `dummy_comments_scenario.py`
entrypoint, which remains as a thin compatibility wrapper.
"""

from .annotator import annotate_pdf

__all__ = ["annotate_pdf"]
