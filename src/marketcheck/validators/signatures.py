"""
Shared stock-split signature detection.

This module exists so that `numerical.suspicious_price_jump` and
`financial.corporate_action_discontinuity` agree on exactly one definition of
"this price ratio looks like a stock split". The two rules detect the *same
signal* -- a large close-to-close move -- and divide it by *cause*:

  - ratio matches a split signature  -> CorporateActionDiscontinuity (INFO):
    the data is not corrupt, it simply appears unadjusted.
  - ratio does not match             -> SuspiciousPriceJump (WARNING):
    the move has no benign explanation and may be a bad tick.

The division is mutually exclusive, so no row is reported by both rules and no
large move goes unowned. Defining the ratio set in one place is what keeps that
guarantee true: duplicating it in both rules would eventually drift and
reintroduce double-reporting.

**Scope is SPLITS ONLY, deliberately and by name.** Dividends are not detectable
this way: a dividend drop is typically 0.1-3% of price, which is
indistinguishable from ordinary daily movement. No ratio test can separate
"fell 1.5% because of a dividend" from "fell 1.5% because it fell". The helpers
here are therefore named for splits rather than for corporate actions
generally, so callers cannot mistake their coverage.

**Splits only take effect between sessions**, never mid-session. Callers must
apply these signatures only to transitions where the calendar date changes; an
intraday move that happens to land near 0.5 is a bad tick, not a 2:1 split.
"""

from __future__ import annotations

import polars as pl

# (price ratio, human label). A forward split divides the price: a 2:1 split
# turns one share into two, so the price ratio is 1/2. A reverse split
# multiplies it: a 1:2 reverse split turns two shares into one, ratio 2.0.
SPLIT_SIGNATURES: tuple[tuple[float, str], ...] = (
    (1 / 20, "20:1 split"),
    (1 / 10, "10:1 split"),
    (1 / 7, "7:1 split"),
    (1 / 5, "5:1 split"),
    (1 / 4, "4:1 split"),
    (1 / 3, "3:1 split"),
    (1 / 2, "2:1 split"),
    (2 / 3, "3:2 split"),
    (3 / 4, "4:3 split"),
    (3 / 2, "2:3 reverse split"),
    (2 / 1, "1:2 reverse split"),
    (3 / 1, "1:3 reverse split"),
    (5 / 1, "1:5 reverse split"),
    (10 / 1, "1:10 reverse split"),
)


def matches_split_ratio_expr(ratio: pl.Expr, tolerance: float) -> pl.Expr:
    """Return a boolean Polars expression: does *ratio* look like a stock split?

    Vectorized on purpose -- this is used inside rule filters over full columns,
    so it must not fall back to per-row Python.

    Args:
        ratio: Expression yielding ``close[i] / close[i-1]``.
        tolerance: Relative tolerance, e.g. 0.02 for 2%.

    Returns:
        Expression that is True where the ratio is within *tolerance* of any
        known split ratio.
    """
    return pl.any_horizontal(
        [((ratio / target) - 1).abs() <= tolerance for target, _ in SPLIT_SIGNATURES]
    )


def label_split_ratio(ratio: float | None, tolerance: float) -> str | None:
    """Return the human label for the split *ratio* matches, or None.

    Scalar counterpart to :func:`matches_split_ratio_expr`, for labelling the
    (already capped) rows a rule reports in ``details``. Returns the closest
    match in relative terms when several are within tolerance.
    """
    if ratio is None or ratio <= 0:
        return None
    best: tuple[float, str] | None = None
    for target, label in SPLIT_SIGNATURES:
        error = abs((ratio / target) - 1)
        if error <= tolerance and (best is None or error < best[0]):
            best = (error, label)
    return None if best is None else best[1]
