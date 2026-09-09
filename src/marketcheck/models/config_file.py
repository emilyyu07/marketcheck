"""Loading a ValidationConfig from a TOML file.


The file is flat, mapping one-to-one onto `ValidationConfig` fields:

    # marketcheck.toml
    volume_anomaly_multiplier = 15.0
    gap_min_missing_bars = 2
    disabled_rules = ["temporal.outside_trading_hours"]

Nested tables would read more nicely but require a mapping layer that can drift
out of step with the model. Flat means the file is exactly the documented field
list, and unknown keys are caught by the model itself.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from marketcheck.models.config import ValidationConfig

# Settings that describe how the command was invoked rather than how validation
# should behave. Accepting these from a file would let a committed config silently
# redirect another user's output, so they are rejected like any unknown key.
_CLI_ONLY_KEYS = frozenset({"output_format", "output_path", "strict"})


class ConfigError(Exception):
    """Raised when a config file cannot be read or is not valid.

    A dedicated exception so the CLI can report a configuration mistake as a tool
    error with a readable message, rather than surfacing a raw parse traceback.
    """


def load_config(path: Path) -> ValidationConfig:
    """Load and validate a `ValidationConfig` from a TOML file.

    Args:
        path: Path to the TOML config file.

    Returns:
        The parsed configuration.

    Raises:
        ConfigError: If the file is missing, is not valid TOML, contains unknown
            or invocation-only keys, or holds a value the model rejects.
    """
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        with path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    # Reported before model validation so the message names the specific problem
    # ("set on the command line instead") rather than a generic unknown-key error.
    invocation_keys = sorted(_CLI_ONLY_KEYS & raw.keys())
    if invocation_keys:
        raise ConfigError(
            f"{path}: {', '.join(invocation_keys)} cannot be set in a config file; "
            "the file configures validation behaviour only. Set these on the "
            "command line instead."
        )

    try:
        return ValidationConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    """Turn a pydantic ValidationError into an actionable message.

    Unknown keys are singled out and listed against the valid field names, because
    the overwhelmingly likely cause is a typo and the fix is then obvious.
    """
    unknown: list[str] = []
    other: list[str] = []

    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        if error["type"] == "extra_forbidden":
            unknown.append(location)
        else:
            other.append(f"{location}: {error['msg']}")

    lines = [f"invalid config in {path}:"]
    if unknown:
        # Exclude invocation-only keys: listing them as valid would contradict the
        # dedicated error raised for them above.
        valid = ", ".join(
            sorted(set(ValidationConfig.model_fields) - _CLI_ONLY_KEYS)
        )
        lines.append(f"  unknown setting(s): {', '.join(sorted(unknown))}")
        lines.append(f"  valid settings are: {valid}")
    lines.extend(f"  {message}" for message in other)
    return "\n".join(lines)
