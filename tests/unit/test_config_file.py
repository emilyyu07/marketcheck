"""Tests for TOML config-file loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from marketcheck.models.config import ValidationConfig
from marketcheck.models.config_file import ConfigError, load_config


def _write(tmp_path: Path, body: str, name: str = "marketcheck.toml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestLoadConfig:
    def test_empty_file_yields_defaults(self, tmp_path: Path) -> None:
        config = load_config(_write(tmp_path, ""))

        assert config == ValidationConfig()

    def test_scalar_settings_applied(self, tmp_path: Path) -> None:
        config = load_config(
            _write(
                tmp_path,
                "volume_anomaly_multiplier = 15.0\n"
                "gap_min_missing_bars = 2\n"
                "split_ratio_tolerance = 0.05\n",
            )
        )

        assert config.volume_anomaly_multiplier == 15.0
        assert config.gap_min_missing_bars == 2
        assert config.split_ratio_tolerance == 0.05

    def test_list_settings_applied(self, tmp_path: Path) -> None:
        config = load_config(
            _write(
                tmp_path,
                'disabled_rules = ["temporal.outside_trading_hours"]\n'
                'enabled_categories = ["structural", "numerical"]\n',
            )
        )

        assert config.disabled_rules == ["temporal.outside_trading_hours"]
        assert config.enabled_categories == ["structural", "numerical"]

    def test_comments_are_supported(self, tmp_path: Path) -> None:
        """A tuning file needs comments: why a threshold was changed matters as
        much as the value. This is why TOML was chosen over JSON."""
        config = load_config(
            _write(
                tmp_path,
                "# Raised because this vendor reports volume in round lots\n"
                "volume_anomaly_multiplier = 25.0\n",
            )
        )

        assert config.volume_anomaly_multiplier == 25.0

    def test_unspecified_settings_keep_defaults(self, tmp_path: Path) -> None:
        config = load_config(_write(tmp_path, "gap_min_missing_bars = 3\n"))

        assert config.gap_min_missing_bars == 3
        assert config.volume_anomaly_multiplier == ValidationConfig().volume_anomaly_multiplier


class TestLoadConfigErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.toml")

    def test_malformed_toml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not valid TOML"):
            load_config(_write(tmp_path, "this is = = not toml\n"))

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        """A silently-ignored typo would leave the user believing they had
        configured something they had not — the same quiet dishonesty as
        reporting an unrun check as a pass."""
        with pytest.raises(ConfigError, match="unknown setting"):
            load_config(_write(tmp_path, "volume_anomly_multiplier = 15.0\n"))

    def test_unknown_key_error_lists_valid_settings(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as excinfo:
            load_config(_write(tmp_path, "notathing = 1\n"))

        message = str(excinfo.value)
        assert "volume_anomaly_multiplier" in message

    def test_unknown_key_error_does_not_advertise_cli_only_keys(
        self, tmp_path: Path
    ) -> None:
        """Listing output_format as valid would contradict the dedicated error
        raised for it."""
        with pytest.raises(ConfigError) as excinfo:
            load_config(_write(tmp_path, "notathing = 1\n"))

        valid_line = next(
            line
            for line in str(excinfo.value).splitlines()
            if "valid settings" in line
        )
        assert "output_format" not in valid_line
        assert "output_path" not in valid_line
        assert "strict" not in valid_line

    def test_wrong_type_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="invalid config"):
            load_config(_write(tmp_path, 'gap_min_missing_bars = "many"\n'))

    @pytest.mark.parametrize("key", ["output_format", "output_path", "strict"])
    def test_invocation_only_keys_are_rejected(self, tmp_path: Path, key: str) -> None:
        """These describe how the command was invoked, not how validation should
        behave. Accepting them from a file would let a committed config silently
        redirect another user's output."""
        value = "true" if key == "strict" else '"somewhere"'
        with pytest.raises(ConfigError, match="cannot be set in a config file"):
            load_config(_write(tmp_path, f"{key} = {value}\n"))
