"""Validation configuration model"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ValidationConfig(BaseModel):
    """Runtime configuration for a validation run.

    Unknown keys are rejected rather than ignored. Pydantic's default is to drop
    them silently, which would mean a typo like `volume_anomly_multiplier` left
    the user believing they had configured something they had not -- the same
    class of quiet dishonesty as reporting an unrun check as a pass.

    `severity_overrides` was deliberately removed rather than implemented. Each
    rule declares one class-level severity AND hardcodes the status it returns
    (VolumeAnomaly returns WARN; CorporateActionDiscontinuity maps INFO -> WARN).
    Overriding severity alone would produce incoherent results such as
    `severity=critical, status=warn`, and making it coherent would mean deriving
    status from severity, destroying those deliberate pairings. The real needs it
    implied are already served: `strict` escalates warnings for CI, and
    `disabled_rules` silences a rule entirely.
    """

    model_config = ConfigDict(extra="forbid")

    strict: bool = False
    """If True, treat WARNINGs as FAILs."""

    enabled_categories: list[str] = []
    """If non-empty, only run rules in these categories. Empty = run all."""

    disabled_rules: list[str] = []
    """Rule IDs to skip."""

    max_rows_in_details: int = 50
    """Cap on how many affected rows to include in detailed output."""

    volume_anomaly_multiplier: float = 10.0
    """VolumeAnomaly: flag a bar when volume exceeds this multiple of its rolling median.

    Default 10.0 sits deliberately between two observed bands: genuine market
    events (earnings, news, index rebalances) typically produce 2-5x normal
    volume, while the data errors this targets (share/round-lot units confusion,
    decimal shifts, a daily aggregate written onto an intraday bar) produce 100x+.
    """

    volume_anomaly_window: int = 20
    """VolumeAnomaly: centered rolling-window size, in bars, for the median baseline.

    Frequency-agnostic by design (CanonicalDataset.inferred_frequency is never
    populated): ~1 month of daily bars, or 20 minutes of intraday bars — a
    sensible local baseline at either scale.
    """

    price_jump_intraday_threshold: float = 0.20
    """SuspiciousPriceJump: fractional close-to-close move flagged WITHIN a session.

    Default 0.20 (20%). Exchange circuit breakers halt trading on 5-10% moves over
    five minutes, so a 20% move between consecutive intraday bars is well outside
    normal market behaviour.
    """

    price_jump_overnight_threshold: float = 0.50
    """SuspiciousPriceJump: fractional close-to-close move flagged ACROSS a session boundary.

    Default 0.50 (50%), deliberately looser than the intraday threshold because
    overnight gaps are structurally larger (earnings, news). Still tight enough to
    catch order-of-magnitude errors such as a decimal shift (+900%).

    Note this threshold also governs DAILY data, where every consecutive pair is a
    session boundary — which is what keeps the rule frequency-agnostic.
    """

    gap_min_missing_bars: int = 1
    """GapsWithinSession: minimum missing bars for a hole to be reported.

    Default 1 reports every gap. Single missing bars are common in real data —
    many vendors emit no bar for a minute with no trades — but they still matter
    for a backtest that assumes every bar exists, so the honest default is to
    report them and let `largest_gap_bars` convey magnitude. Raise this to
    suppress noise on illiquid names.
    """

    gap_frequency_dominance: float = 0.5
    """GapsWithinSession: fraction of deltas the modal bar spacing must EXCEED.

    Default 0.5 requires a strict majority. Guards against irregular data (tick
    or event data), where no fixed grid exists, the modal delta is arbitrary, and
    almost every interval would otherwise be reported as a gap. The strict
    comparison also rejects a perfect two-way tie as ambiguous rather than
    resolving it arbitrarily.
    """

    split_ratio_tolerance: float = 0.02
    """Relative tolerance when matching a price ratio to a known stock-split ratio.

    Default 0.02 (2%). Tolerance is required because the price also moves on genuine
    trading across the same interval, so a 2:1 split rarely produces exactly 0.5.
    """

    output_format: str = "text"
    """Default output format: 'text' or 'json'."""

    output_path: Path | None = None
    """If set, write the report to this path instead of stdout."""
