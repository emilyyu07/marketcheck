"""Financial validation rules """

import polars as pl

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register
from marketcheck.validators.signatures import (
    label_split_ratio,
    matches_split_ratio_expr,
)

'''
Rule: Corporate Action Discontinuity
Flags session-boundary price discontinuities whose ratio matches a known stock
split, indicating the data appears unadjusted for that action.
'''
@register
class CorporateActionDiscontinuity(ValidationRule):
    rule_id = "financial.corporate_action_discontinuity"
    rule_name = "Corporate Action Discontinuity"
    category = Category.FINANCIAL
    default_severity = Severity.INFO

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        def _pass(message: str) -> ValidationResult:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message=message,
            )

        # MissingColumns owns absent columns. `close` is the signal and
        # `timestamp` is required to identify session boundaries -- a split only
        # ever takes effect between sessions, so without dates this rule has no
        # way to distinguish a split from an intraday bad tick and must not guess.
        if "close" not in df.columns:
            return _pass("No close column present; skipped.")
        if "timestamp" not in df.columns:
            return _pass("No timestamp column present; cannot identify sessions; skipped.")

        # Need at least one consecutive pair to form a ratio.
        if df.height < 2:
            return _pass("No corporate-action discontinuities detected.")

        tolerance = context.config.split_ratio_tolerance
        has_volume = "volume" in df.columns

        # A session boundary is simply a change of calendar date between
        # consecutive rows. For daily data every pair qualifies, which is what
        # keeps the rule frequency-agnostic.
        date_col = pl.col("timestamp").dt.date()
        is_boundary = date_col != date_col.shift(1)

        prev_close = pl.col("close").shift(1)
        ratio = pl.col("close") / prev_close

        selected = [
            pl.int_range(0, df.height).alias("index"),
            pl.col("timestamp"),
            prev_close.alias("previous_close"),
            pl.col("close"),
            ratio.alias("ratio"),
            is_boundary.alias("is_session_boundary"),
        ]
        if has_volume:
            # Corroborating evidence only -- never part of the filter.
            selected.append(
                (pl.col("volume") / pl.col("volume").shift(1)).alias("volume_ratio")
            )

        base = df.select(selected).with_columns(
            ((pl.col("ratio") - 1) * 100).alias("pct_change"),
        )

        candidates = base.filter(
            # Non-positive prices would make the ratio meaningless or infinite;
            # ImpossibleValues owns those rows. Nulls fall out automatically
            # because comparisons against null yield null, not True.
            (pl.col("previous_close") > 0)
            & (pl.col("close") > 0)
            # Boundary-only, and the signature match is the sole magnitude filter.
            & pl.col("is_session_boundary")
            & matches_split_ratio_expr(pl.col("ratio"), tolerance)
        )

        affected_rows = candidates.height
        if affected_rows == 0:
            return _pass("No corporate-action discontinuities detected.")

        # Tally inferred actions over the FULL candidate set so the summary stays
        # accurate even when `details` is truncated below.
        inferred_action_counts: dict[str, int] = {}
        for row_ratio in candidates.get_column("ratio").to_list():
            label = label_split_ratio(row_ratio, tolerance)
            if label is not None:
                inferred_action_counts[label] = inferred_action_counts.get(label, 0) + 1

        capped = candidates.head(context.config.max_rows_in_details)
        violations = []
        for row in capped.iter_rows(named=True):
            violation = {
                "index": row["index"],
                "timestamp": row["timestamp"],
                "previous_close": row["previous_close"],
                "close": row["close"],
                "ratio": round(row["ratio"], 6),
                "pct_change": round(row["pct_change"], 2),
                "inferred_action": label_split_ratio(row["ratio"], tolerance),
            }
            if has_volume and row["volume_ratio"] is not None:
                violation["volume_ratio"] = round(row["volume_ratio"], 2)
            violations.append(violation)

        first = violations[0]
        # "consistent with" / "appears unadjusted", never "a split occurred":
        # the inference is unverifiable without a corporate-actions feed.
        message = (
            f"{affected_rows} price discontinuit{'y' if affected_rows == 1 else 'ies'} "
            f"consistent with stock split(s); data appears unadjusted "
            f"(first at index {first['index']}: "
            f"{first['previous_close']:g} -> {first['close']:g}, "
            f"{first['pct_change']:+g}%, consistent with {first['inferred_action']})."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={
                "violations": violations,
                "inferred_action_counts": inferred_action_counts,
                "split_ratio_tolerance": tolerance,
            },
            affected_rows=affected_rows,
        )
