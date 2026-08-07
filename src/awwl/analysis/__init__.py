"""Post-hoc analysis: results ledger and significance testing."""

from __future__ import annotations

from awwl.analysis.results import append_result, load_results, result_row
from awwl.analysis.stats import (
    ComparisonResult,
    GroupSummary,
    compare_to_baseline,
    summarize_groups,
)

__all__ = [
    "ComparisonResult",
    "GroupSummary",
    "append_result",
    "compare_to_baseline",
    "load_results",
    "result_row",
    "summarize_groups",
]
