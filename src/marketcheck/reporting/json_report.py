"""
JSON report renderer
Takes a DatasetSummary and produces a human-readable report in JSON format
"""

from __future__ import annotations

import json

from marketcheck.models.result import DatasetSummary


def render_json(summary: DatasetSummary) -> str:
    """Render a DatasetSummary as a JSON string.

    Args:
        summary: The aggregated validation summary.

    Returns:
        A pretty-printed JSON string.
    """
    data = summary.model_dump(mode="json")
    return json.dumps(data, indent=2, default=str)
