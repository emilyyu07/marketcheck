"""
Thin wrapper around pandas_market_calendars for NYSE session info.
Provides NYSE market awareness (i.e. valid trading days, open/close times) to validators.

"""

from __future__ import annotations

from datetime import date, time

import pandas_market_calendars as mcal


class MarketCalendar:
    """Provides NYSE trading session information.

    Wraps ``pandas_market_calendars`` to expose the specific queries
    needed by MarketCheck validators.
    """

    def __init__(self, exchange: str = "NYSE") -> None:
        self._calendar = mcal.get_calendar(exchange)

    def valid_sessions(self, start: date | str, end: date | str) -> list[date]:
        """Return all valid trading dates between *start* and *end* (inclusive).

        Args:
            start: First date of the range.
            end: Last date of the range.

        Returns:
            List of ``date`` objects for each trading session.
        """
        schedule = self._calendar.schedule(start_date=str(start), end_date=str(end))
        return [d.date() for d in schedule.index]

    def is_trading_day(self, d: date | str) -> bool:
        """Check whether a date is a valid trading session."""
        schedule = self._calendar.schedule(start_date=str(d), end_date=str(d))
        return not schedule.empty

    @staticmethod
    def regular_open() -> time:
        """NYSE regular trading hours open (ET)."""
        return time(9, 30)

    @staticmethod
    def regular_close() -> time:
        """NYSE regular trading hours close (ET)."""
        return time(16, 0)
