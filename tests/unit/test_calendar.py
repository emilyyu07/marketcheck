"""Tests for the calendar module."""

from marketcheck.calendar.sessions import MarketCalendar


class TestMarketCalendar:
    def test_instantiation(self) -> None:
        cal = MarketCalendar()
        assert cal is not None

    def test_regular_hours(self) -> None:
        assert MarketCalendar.regular_open().hour == 9
        assert MarketCalendar.regular_open().minute == 30
        assert MarketCalendar.regular_close().hour == 16
        assert MarketCalendar.regular_close().minute == 0

    def test_valid_sessions_returns_trading_days(self) -> None:
        cal = MarketCalendar()
        sessions = cal.valid_sessions("2024-01-02", "2024-01-05")
        # Jan 2-5 2024: Tue-Fri, all trading days
        assert len(sessions) == 4

    def test_is_trading_day(self) -> None:
        cal = MarketCalendar()
        # Jan 1 2024 is New Year's Day (holiday)
        assert cal.is_trading_day("2024-01-01") is False
        # Jan 2 2024 is a regular trading day
        assert cal.is_trading_day("2024-01-02") is True
