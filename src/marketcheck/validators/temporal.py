"""Temporal validation rules (4 rules)."""

from datetime import time
from typing import cast

import polars as pl

from marketcheck.ingestion.schema import TZ_MIXED
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register
from marketcheck.validators.frequency import format_frequency, infer_frequency

'''
Rule: Missing Trading Sessions
Checks whether any NYSE trading session inside the dataset's own date range has
no rows at all.

Mechanics: expected sessions from context.calendar.valid_sessions(start, end)
MINUS the set of dates actually present in `timestamp` -> anything left over is
a session the exchange was open for but the data doesn't cover.
'''
@register
class MissingSessions(ValidationRule):
    rule_id = "temporal.missing_sessions"
    rule_name = "Missing Trading Sessions"
    category = Category.TEMPORAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        # MissingColumns owns absent columns; skip rather than crash.
        if "timestamp" not in df.columns:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No timestamp column present; skipped.",
            )

        # NullValues owns nulls; drop them before deriving the date range so a
        # null can't collapse the range or masquerade as a covered session.
        dates_present = set(
            df.select(pl.col("timestamp").dt.date().alias("d"))
            .drop_nulls()
            .to_series()
            .unique()
            .to_list()
        )

        # No usable timestamps -> no date range to diff against the calendar, so
        # nothing was verified. SKIP rather than claim no sessions are missing.
        if not dates_present:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No usable timestamps, so no date range to check; skipped.",
            )

        start = min(dates_present)
        end = max(dates_present)

        # Single schedule build for the whole range (already chronological).
        expected_sessions = context.calendar.valid_sessions(start, end)
        missing = [d for d in expected_sessions if d not in dates_present]

        if not missing:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No trading sessions missing.",
            )

        capped = missing[: context.config.max_rows_in_details]
        message = (
            f"{len(missing)} trading session(s) missing between {start} and {end} "
            f"(first: {missing[0]})."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={
                "missing_sessions": [str(d) for d in capped],
                "missing_count": len(missing),
                "expected_session_count": len(expected_sessions),
                "present_session_count": len(dates_present),
                "range_start": str(start),
                "range_end": str(end),
            },
            affected_rows=0,
        )


'''
Rule: Gaps Within Session
Checks whether bars are missing within a trading session (e.g. 1-minute data
that jumps from 10:15 straight to 10:23)

Key notes:
- expected spacing is inferred from data itself (most common spacing is the
  inferred grid size)
- delta-based, not grid-based: detection compares consecutive bars rather than
  diffing against a reconstructed timestamp grid from session hours
- no calendar dependency
- session boundaries excluded, so overnight gaps are never reported here
  (MissingSessions owns absent days)

Mechanics: sort a copy -> drop duplicate timestamps -> take consecutive deltas ->
discard deltas that span a session boundary -> infer the modal spacing -> any
surviving delta larger than one bar is a gap, hiding `delta/frequency - 1` bars.

'''
@register
class GapsWithinSession(ValidationRule):
    rule_id = "temporal.gaps_within_session"
    rule_name = "Gaps Within Session"
    category = Category.TEMPORAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        def _result(
            status: Status, message: str, details: dict[str, object] | None = None
        ) -> ValidationResult:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=status,
                message=message,
                details=details or {},
            )

        def _pass(message: str, details: dict[str, object] | None = None) -> ValidationResult:
            """Rule evaluated the data and found no gaps."""
            return _result(Status.PASS, message, details)

        def _skip(message: str, details: dict[str, object] | None = None) -> ValidationResult:
            """Rule could not evaluate the data -- no timestamps, no intra-session
            intervals, or no credible bar frequency. Never reported as a pass."""
            return _result(Status.SKIP, message, details)

        # MissingColumns owns absent columns; skip rather than crash.
        if "timestamp" not in df.columns:
            return _skip("No timestamp column present; skipped.")

        # Need at least two rows to form a single delta.
        if df.height < 2:
            return _skip("Fewer than 2 rows, so no interval exists; skipped.")

        min_missing = context.config.gap_min_missing_bars
        dominance = context.config.gap_frequency_dominance

        # Work on a sorted, de-duplicated COPY. Rules run independently and cannot
        # assume sorted input -- the same reasoning that drove DuplicateTimestamps
        # to group-by rather than shift. Duplicates are dropped because a zero
        # delta is not a gap and would corrupt the modal spacing;
        # UnsortedTimestamps and DuplicateTimestamps own those defects.
        ordered = (
            df.select("timestamp")
            .drop_nulls()
            .unique(subset="timestamp")
            .sort("timestamp")
        )
        if ordered.height < 2:
            return _skip(
                "Fewer than 2 distinct non-null timestamps, so no interval exists; skipped."
            )

        # A session boundary is a change of calendar date. Deltas that span one
        # are excluded from BOTH inference and detection: an overnight gap is not
        # a within-session gap, and MissingSessions owns absent days.
        #
        # This is also what makes the rule frequency-agnostic without configuration:
        # for DAILY data every delta spans a boundary, so no within-session deltas
        # remain, inference fails, and the rule skips -- which is correct, because
        # "gap within a session" is meaningless when each session holds one bar.
        date_col = pl.col("timestamp").dt.date()
        analysed = ordered.with_columns(
            pl.int_range(0, ordered.height).alias("index"),
            pl.col("timestamp").diff().alias("delta"),
            (date_col != date_col.shift(1)).alias("is_session_boundary"),
        ).filter(~pl.col("is_session_boundary").fill_null(True))

        if analysed.height == 0:
            return _skip(
                "No intra-session intervals to analyse (one bar per session); skipped."
            )

        frequency, confidence = infer_frequency(analysed.get_column("delta"), dominance)
        if frequency is None:
            # Not a regular grid (tick/event data, or an ambiguous tie). Reporting
            # gaps here would mean flagging almost every interval, so the rule
            # declines to guess and says why.
            return _skip(
                "Could not infer a consistent bar frequency "
                f"(most common spacing accounts for {confidence:.0%} of intervals, "
                f"below the {dominance:.0%} required); skipped.",
                {"frequency_confidence": round(confidence, 4)},
            )

        # A gap exists where the observed delta exceeds one bar. missing_bars
        # subtracts 1 because the delta spans from the last surviving bar TO the
        # next surviving bar: a 4-minute delta on a 1-minute grid hides 3 bars,
        # not 4.
        freq_us = int(frequency.total_seconds() * 1_000_000)
        gaps = analysed.with_columns(
            ((pl.col("delta").dt.total_microseconds() // freq_us) - 1).alias("missing_bars")
        ).filter(pl.col("missing_bars") >= min_missing)

        gap_count = gaps.height
        frequency_label = format_frequency(frequency)
        summary = {
            "inferred_frequency": frequency_label,
            "frequency_confidence": round(confidence, 4),
        }

        if gap_count == 0:
            return _pass(
                f"No gaps within sessions detected ({frequency_label} bars).", summary
            )

        # Totals are computed over the FULL gap set, before details are capped, so
        # the summary stays accurate on heavily-holed data.
        missing_series = gaps.get_column("missing_bars")
        # sum()/max() are typed as a broad scalar union; the column is integer by
        # construction, so cast explicitly rather than leaving it implicit.
        total_missing = int(cast("int", missing_series.sum()))
        largest_gap = int(cast("int", missing_series.max()))

        capped = gaps.head(context.config.max_rows_in_details)
        violations = [
            {
                # Position in SORTED order, since detection runs on a sorted copy.
                # Timestamps are the primary locator here for that reason.
                "index": row["index"],
                "gap_start": row["timestamp"] - row["delta"],
                "gap_end": row["timestamp"],
                "gap_duration": format_frequency(row["delta"]),
                "missing_bars": int(row["missing_bars"]),
            }
            for row in capped.iter_rows(named=True)
        ]

        first = violations[0]
        # Worded for review, not corruption: a genuine trading halt produces a
        # legitimate gap and this rule cannot distinguish one from a truncated
        # feed. Same ambiguity that keeps VolumeAnomaly at WARNING.
        message = (
            f"{gap_count} gap(s) within sessions, {total_missing} missing "
            f"{frequency_label} bar(s) in total "
            f"(largest {largest_gap}; first after {first['gap_start']}, "
            f"{first['missing_bars']} bar(s) absent)."
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
                "gap_count": gap_count,
                "total_missing_bars": total_missing,
                "largest_gap_bars": largest_gap,
                **summary,
            },
            # Distinct EXISTING rows that follow a gap. Missing bars are not rows,
            # so counting them would break the convention and could exceed row_count.
            affected_rows=gap_count,
        )


'''
Rule: Outside Trading Hours
Checks whether any row's timestamp falls outside NYSE regular trading hours
([09:30:00, 16:00:00) ET, half-open interval), by wall-clock time-of-day only.
'''
@register
class OutsideTradingHours(ValidationRule):
    rule_id = "temporal.outside_trading_hours"
    rule_name = "Data Outside Trading Hours"
    category = Category.TEMPORAL
    default_severity = Severity.INFO

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        # MissingColumns owns absent columns; skip rather than crash or double-report.
        if "timestamp" not in df.columns:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No timestamp column present; skipped.",
            )

        # Nothing to check with zero rows. Unlike the pairwise structural
        # rules, this rule inspects each row independently, so even a single
        # row is meaningfully checkable -- no "< 2 rows" guard needed here.
        if df.height == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No rows to inspect; skipped.",
            )

        regular_open = context.calendar.regular_open()
        regular_close = context.calendar.regular_close()

        # Vectorized pass: extract wall-clock time-of-day and keep only rows
        # outside the half-open [regular_open, regular_close) interval. Null
        # timestamps produce a null time_of_day, and comparisons against
        # null evaluate to null (not True) in the filter, so they're
        # excluded automatically (NullValues check owns reporting them).
        violations_df = (
            df.select(
                pl.int_range(0, df.height).alias("index"),
                pl.col("timestamp"),
                pl.col("timestamp").dt.time().alias("time_of_day"),
            )
            .filter(
                (pl.col("time_of_day") < regular_open)
                | (pl.col("time_of_day") >= regular_close)
            )
        )

        affected_rows = violations_df.height

        if affected_rows == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No data outside regular trading hours.",
            )

        capped = violations_df.head(context.config.max_rows_in_details)
        violations = [
            {
                "index": row["index"],
                "timestamp": row["timestamp"],
                "time_of_day": row["time_of_day"],
            }
            for row in capped.iter_rows(named=True)
        ]

        first = violations[0]
        message = (
            f"{affected_rows} row(s) outside regular trading hours "
            f"({regular_open}-{regular_close} ET) "
            f"(first at index {first['index']}: {first['time_of_day']})."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={"violations": violations},
            affected_rows=affected_rows,
        )


'''
Rule: Timezone Inconsistency
Checks whether the timestamps can be interpreted as a single, consistent
timezone, specifically as the ET wall-clock that every other temporal rule
assumes.

Two independent detections, covering disjoint failure modes:
1. Metadata - what the source actually declared, recorded during ingestion
2. Heuristic - naive timestamps that do not look like ET at all

Key notes:
- consistent timezone declarations are not flagged
- shifts searched in 30-minute increments
- reported shift is difference from ET, not source's UTC offset (e.g. India +05:30 appears as +10:30)
- OutsideTradingHours will usually also fire when the heuristic does. Both are
  allowed to report: this rule names the cause, other details the symptoms

'''
# --- Heuristic calibration (constants) -------------------------------------------------------------
# Minimum distinct times-of-day before a shift is even considered. Daily data has
# exactly one, which is what makes the daily-data trap avoidable.
_MIN_DISTINCT_TIMES = 10
# A shift must place at least this fraction of bars inside the session to be
# credible evidence of a timezone difference.
_MIN_SHIFTED_ALIGNMENT = 0.90
# ...and the data must currently align no better than this, otherwise there is
# nothing wrong to explain.
_MAX_CURRENT_ALIGNMENT = 0.50
# Candidate shifts, in minutes: every 30 minutes across the range of real world
# offsets relative to ET.
_SHIFT_STEP_MINUTES = 30
_SHIFT_RANGE_MINUTES = (-12 * 60, 14 * 60)


@register
class TimezoneInconsistency(ValidationRule):
    rule_id = "temporal.timezone_inconsistency"
    rule_name = "Timezone Inconsistency"
    category = Category.TEMPORAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        def _result(
            status: Status, message: str, details: dict[str, object] | None = None
        ) -> ValidationResult:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=status,
                message=message,
                details=details or {},
                affected_rows=df.height if status == Status.FAIL else 0,
            )

        # MissingColumns owns absent columns.
        if "timestamp" not in df.columns:
            return _result(Status.SKIP, "No timestamp column present; skipped.")

        # ---- Detection 1: what the source declared -------------------------
        # A column mixing offset-bearing and offset-free values cannot be parsed
        # as one type, so those rows became nulls during coercion. That is a true
        # inconsistency: the rows do not share a frame of reference.
        if dataset.source_timezone == TZ_MIXED:
            return _result(
                Status.FAIL,
                "Timestamps mix timezone-aware and timezone-naive values, so they "
                "do not share a single frame of reference; the unparseable rows "
                "became null during loading.",
                {"source_timezone": TZ_MIXED},
            )

        timestamps = df.get_column("timestamp").drop_nulls()
        if timestamps.len() == 0:
            return _result(
                Status.SKIP, "No usable timestamps to inspect; skipped."
            )

        # ---- Detection 2: heuristic on the values themselves ---------------
        # Guard: without several distinct times-of-day a uniform shift cannot be
        # told apart from the data simply being daily (bars stamped 00:00).
        minutes_of_day = (
            timestamps.dt.hour().cast(pl.Int64) * 60 + timestamps.dt.minute().cast(pl.Int64)
        )
        distinct_times = minutes_of_day.n_unique()
        if distinct_times < _MIN_DISTINCT_TIMES:
            return _result(
                Status.SKIP,
                f"Only {distinct_times} distinct time(s) of day, too few to "
                "distinguish a timezone shift from daily data; skipped.",
                {"distinct_times_of_day": distinct_times},
            )

        open_minutes = _minutes(context.calendar.regular_open())
        close_minutes = _minutes(context.calendar.regular_close())
        total = minutes_of_day.len()

        def alignment(shift: int) -> float:
            """Fraction of bars landing inside the session after shifting by
            `shift` minutes. Wraps modulo 24h so a shift across midnight behaves
            the same as any other."""
            shifted = (minutes_of_day + shift) % (24 * 60)
            inside = ((shifted >= open_minutes) & (shifted < close_minutes)).sum()
            return int(inside) / total

        current = alignment(0)
        # Nothing to explain if the data already sits in the session.
        if current > _MAX_CURRENT_ALIGNMENT:
            return _result(
                Status.PASS,
                "Timestamps are consistent with ET wall-clock.",
                _calibration({"current_alignment": round(current, 4)}),
            )

        candidates = range(
            _SHIFT_RANGE_MINUTES[0], _SHIFT_RANGE_MINUTES[1] + 1, _SHIFT_STEP_MINUTES
        )
        best_shift, best_alignment = 0, current
        for shift in candidates:
            if shift == 0:
                continue
            score = alignment(shift)
            # Ties resolve to the smaller absolute shift: the least surprising
            # explanation, and it keeps the result deterministic.
            if score > best_alignment or (
                score == best_alignment and abs(shift) < abs(best_shift)
            ):
                best_shift, best_alignment = shift, score

        if best_shift == 0 or best_alignment < _MIN_SHIFTED_ALIGNMENT:
            # Poorly aligned, but no uniform shift explains it -- so the cause is
            # not a timezone. OutsideTradingHours owns reporting the bars.
            return _result(
                Status.PASS,
                "No uniform timezone shift explains the timestamps; "
                "not a timezone problem.",
                _calibration(
                    {
                        "current_alignment": round(current, 4),
                        "best_alignment": round(best_alignment, 4),
                    }
                ),
            )

        offset_label = _format_offset(best_shift)
        return _result(
            Status.FAIL,
            f"Timestamps do not look like ET wall-clock: only {current:.0%} of bars "
            f"fall inside the trading session, but {best_alignment:.0%} would if "
            f"shifted by {offset_label}. Every temporal check assumes ET, so all of "
            "them are unreliable until this is resolved.",
            _calibration(
                {
                    "source_timezone": dataset.source_timezone,
                    "current_alignment": round(current, 4),
                    "suggested_shift": offset_label,
                    "suggested_shift_minutes": best_shift,
                    "alignment_after_shift": round(best_alignment, 4),
                    "distinct_times_of_day": distinct_times,
                }
            ),
        )


def _minutes(value: time) -> int:
    """Minutes since midnight for a `time`."""
    return value.hour * 60 + value.minute


def _format_offset(minutes: int) -> str:
    """Render a shift in minutes as a signed offset, e.g. `-5:00`, `+10:30`."""
    sign = "+" if minutes >= 0 else "-"
    hours, mins = divmod(abs(minutes), 60)
    return f"{sign}{hours}:{mins:02d}"


def _calibration(details: dict[str, object]) -> dict[str, object]:
    """Attach the calibration constants so the inference stays auditable.

    The thresholds are not configurable, but hiding them would make a CRITICAL
    finding unverifiable -- the same reason VolumeAnomaly echoes its multiplier
    and window.
    """
    return {
        **details,
        "min_distinct_times_of_day": _MIN_DISTINCT_TIMES,
        "min_alignment_after_shift": _MIN_SHIFTED_ALIGNMENT,
        "max_current_alignment": _MAX_CURRENT_ALIGNMENT,
    }
