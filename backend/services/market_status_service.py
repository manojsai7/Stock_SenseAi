import datetime
from typing import Optional, Set
import pytz

class MarketStatusService:
    """
    Dedicated service for Indian Equity Market (NSE/BSE) trading session status.
    Uses Asia/Kolkata timezone and standard NSE operating rules:
    - Monday - Friday
    - Pre-Market: 09:00 - 09:15 IST
    - Regular Trading Session (OPEN): 09:15 - 15:30 IST
    - Post-Market / Closing: 15:30 - 16:00 IST
    - Closed: < 09:00 or >= 16:00 IST
    - Weekends: Saturday / Sunday
    - Holidays: NSE Official Holiday Calendar
    """

    # Official NSE Trading Holidays (Format: YYYY-MM-DD)
    NSE_HOLIDAYS_2026: Set[str] = {
        "2026-01-26",  # Republic Day
        "2026-03-03",  # Holi
        "2026-03-20",  # Id-Ul-Fitr
        "2026-04-03",  # Good Friday
        "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
        "2026-05-01",  # Maharashtra Day
        "2026-05-27",  # Bakri Id / Eid ul-Adha
        "2026-08-15",  # Independence Day
        "2026-10-02",  # Mahatma Gandhi Jayanti
        "2026-10-20",  # Dussehra
        "2026-11-08",  # Diwali Laxmi Pujan
        "2026-11-10",  # Diwali Balipratipada
        "2026-11-24",  # Gurunanak Jayanti
        "2026-12-25",  # Christmas
    }

    def __init__(self, tz_name: str = "Asia/Kolkata"):
        self.tz = pytz.timezone(tz_name)

    def get_current_time(self) -> datetime.datetime:
        return datetime.datetime.now(self.tz)

    def is_holiday(self, d: datetime.date) -> bool:
        return d.strftime("%Y-%m-%d") in self.NSE_HOLIDAYS_2026

    def is_trading_day(self, d: datetime.date) -> bool:
        # Weekend: 5=Saturday, 6=Sunday
        if d.weekday() >= 5:
            return False
        if self.is_holiday(d):
            return False
        return True

    def get_market_status(self, dt: Optional[datetime.datetime] = None) -> str:
        """
        Determines current market state:
        PRE_MARKET | OPEN | POST_MARKET | CLOSED | WEEKEND | HOLIDAY
        """
        if dt is None:
            dt = self.get_current_time()
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)
        else:
            dt = dt.astimezone(self.tz)

        current_date = dt.date()

        # Check weekend
        if current_date.weekday() >= 5:
            return "WEEKEND"

        # Check official holiday
        if self.is_holiday(current_date):
            return "HOLIDAY"

        current_time = dt.time()
        pre_market_start = datetime.time(9, 0)
        market_open_time = datetime.time(9, 15)
        market_close_time = datetime.time(15, 30)
        post_market_end = datetime.time(16, 0)

        if pre_market_start <= current_time < market_open_time:
            return "PRE_MARKET"
        elif market_open_time <= current_time < market_close_time:
            return "OPEN"
        elif market_close_time <= current_time < post_market_end:
            return "POST_MARKET"
        else:
            return "CLOSED"

    def is_market_open(self, dt: Optional[datetime.datetime] = None) -> bool:
        return self.get_market_status(dt) == "OPEN"

    def get_latest_completed_trading_day(self, dt: Optional[datetime.datetime] = None) -> datetime.date:
        """
        Returns the date of the most recent COMPLETED trading session.
        If today is a trading day and current time >= 15:30 IST, today's candle is complete.
        Otherwise (market is OPEN, PRE_MARKET, WEEKEND, HOLIDAY, or earlier in the morning),
        today's candle is not completed, so return the previous trading day.
        """
        if dt is None:
            dt = self.get_current_time()
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)
        else:
            dt = dt.astimezone(self.tz)

        current_date = dt.date()
        current_time = dt.time()

        # If today is a trading day and time is past 15:30 IST, today is completed
        if self.is_trading_day(current_date) and current_time >= datetime.time(15, 30):
            return current_date

        # Otherwise look back day by day until we find the latest completed trading day
        candidate = current_date - datetime.timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate -= datetime.timedelta(days=1)

        return candidate

    def get_next_trading_day(self, from_date: Optional[datetime.date] = None) -> datetime.date:
        """
        Calculates the next official NSE trading day after from_date.
        """
        if from_date is None:
            from_date = self.get_current_time().date()

        candidate = from_date + datetime.timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += datetime.timedelta(days=1)

        return candidate

market_status_service = MarketStatusService()
