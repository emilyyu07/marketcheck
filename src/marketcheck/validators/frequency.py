"""Bar-frequency inference shared by validators.

Lives in its own module rather than inside a rule (or inside ingestion) for two
reasons:

1. **Architecture.** Adding a validation rule should not require touching
   ingestion. `CanonicalDataset.inferred_frequency` exists but is never
   populated by `to_canonical()`, and populating it would make every rule that
   needs a frequency depend on ingestion having done work first.
2. **Type fidelity.** `inferred_frequency` is a `str` (e.g. `"1min"`), but
   inference naturally produces a `timedelta`, and gap arithmetic needs the
   `timedelta`. Round-tripping through a string would be lossy and would
   require a format parser that nothing else needs.

Precedent: `validators/signatures.py`, created so two rules could share one
definition of a stock split. Ingestion may later call `format_frequency()` for
display without the rule depending on it.

Inference uses the **mode** of consecutive deltas. Rejected alternatives:
- `min`: a single anomalous row sets the frequency for the whole dataset.
- `mean`: skewed by exactly the gaps we are trying to find.
- `median`: breaks down once more than half the expected bars are absent.
The mode is the natural choice because the normal spacing is, by definition,
the most frequently occurring one; gaps are the minority in any dataset worth
validating.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl


def infer_frequency(
    deltas: pl.Series, dominance: float
) -> tuple[timedelta | None, float]:
    """Infer the dominant bar spacing from a series of consecutive deltas.

    Args:
        deltas: Duration series of consecutive timestamp differences (excluding overnight gaps)
        dominance: The winning value must account for more than this
            fraction of `deltas` (a strict majority at the 0.5 default) -> without
            this guard, on irregular data with no fixed grid, nearly every
            interval would be reported as a gap

    Returns:
        `(frequency, confidence)` where `frequency` is the modal delta and
        `confidence` is the fraction of deltas it accounts for. `frequency` is
        `None` when it cannot be inferred credibly: no deltas at all, a
        non-positive modal delta, or confidence below `dominance`. `confidence`
        is still returned in the failure case so callers can report why the
        inference was rejected.
    """
    clean = deltas.drop_nulls()
    if clean.len() == 0:
        return None, 0.0

    # `Series.mode()` is unsuitable here: it can return several values on a tie,
    # in a non-deterministic order, which would make the rule produce different
    # answers for identical input. `value_counts` plus an explicit sort makes the
    # winner deterministic: highest count first, then smallest delta as the
    # tie-break so the result never depends on row order.
    counts = (
        clean.value_counts()
        .rename({clean.name or "": "delta"})
        .sort(["count", "delta"], descending=[True, False])
    )

    winner = counts.row(0, named=True)
    frequency: timedelta = winner["delta"]
    confidence = winner["count"] / clean.len()

    # A non-positive delta means duplicate or unsorted timestamps reached us
    # despite the caller's preprocessing; treat as uninferable rather than
    # dividing by zero downstream.
    if frequency <= timedelta(0):
        return None, confidence

    # Strict comparison: the modal delta must be a true majority. This is what
    # makes a perfect two-way tie (each candidate at exactly 0.5) fail the guard
    # instead of being resolved arbitrarily by the tie-break above.
    if confidence <= dominance:
        return None, confidence

    return frequency, confidence


def format_frequency(freq: timedelta) -> str:
    """Render a bar spacing as a compact human-readable string.

    Produces the vocabulary used in market-data filenames (`"1min"`, `"5min"`,
    `"1h"`, `"1d"`) so a report echoes back something a user recognises. Falls
    back to a composite form for spacings that are not a clean multiple.

    Durations are formatted rather than emitted as `timedelta` objects because a
    raw `timedelta` serialises to ISO-8601 (`"PT4M"`) in JSON output, which is
    valid but not readable in a report.
    """
    total = int(freq.total_seconds())
    if total <= 0:
        return "0s"

    if total % 86400 == 0:
        return f"{total // 86400}d"
    if total % 3600 == 0:
        return f"{total // 3600}h"
    if total % 60 == 0:
        return f"{total // 60}min"
    if total < 60:
        return f"{total}s"

    minutes, seconds = divmod(total, 60)
    return f"{minutes}min{seconds}s"
