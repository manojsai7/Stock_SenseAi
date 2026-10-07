import datetime
import pytz
import pytest
from backend.services.market_status_service import MarketStatusService

tz = pytz.timezone("Asia/Kolkata")

def test_market_status_weekend():
    service = MarketStatusService()
    # A known Saturday: 2026-10-10 at 11:00 AM IST
    saturday_dt = tz.localize(datetime.datetime(2026, 10, 10, 11, 0, 0))
    status = service.get_market_status(saturday_dt)
    assert status == "WEEKEND"
    assert not service.is_market_open(saturday_dt)

def test_market_status_holiday():
    service = MarketStatusService()
    # Republic Day: 2026-01-26 (Monday) at 10:30 AM IST
    holiday_dt = tz.localize(datetime.datetime(2026, 1, 26, 10, 30, 0))
    status = service.get_market_status(holiday_dt)
    assert status == "HOLIDAY"
    assert not service.is_market_open(holiday_dt)

def test_market_status_pre_market():
    service = MarketStatusService()
    # Wednesday 2026-10-07 at 09:05 AM IST
    dt = tz.localize(datetime.datetime(2026, 10, 7, 9, 5, 0))
    assert service.get_market_status(dt) == "PRE_MARKET"
    assert not service.is_market_open(dt)

def test_market_status_open():
    service = MarketStatusService()
    # Wednesday 2026-10-07 at 11:30 AM IST
    dt = tz.localize(datetime.datetime(2026, 10, 7, 11, 30, 0))
    assert service.get_market_status(dt) == "OPEN"
    assert service.is_market_open(dt)

def test_market_status_post_market():
    service = MarketStatusService()
    # Wednesday 2026-10-07 at 15:45 IST
    dt = tz.localize(datetime.datetime(2026, 10, 7, 15, 45, 0))
    assert service.get_market_status(dt) == "POST_MARKET"
    assert not service.is_market_open(dt)

def test_market_status_closed_night():
    service = MarketStatusService()
    # Wednesday 2026-10-07 at 20:00 IST
    dt = tz.localize(datetime.datetime(2026, 10, 7, 20, 0, 0))
    assert service.get_market_status(dt) == "CLOSED"
    assert not service.is_market_open(dt)

def test_latest_completed_trading_day_during_market_hours():
    service = MarketStatusService()
    # Wednesday 2026-10-07 at 12:00 PM IST (Market is OPEN)
    # Today's daily candle is NOT completed yet, so latest completed trading day is Tuesday (2026-10-06)
    dt = tz.localize(datetime.datetime(2026, 10, 7, 12, 0, 0))
    completed_date = service.get_latest_completed_trading_day(dt)
    assert completed_date == datetime.date(2026, 10, 6)

def test_latest_completed_trading_day_after_market_close():
    service = MarketStatusService()
    # Wednesday 2026-10-07 at 16:30 IST (Market is CLOSED after regular session)
    # Today's daily candle is COMPLETED, so latest completed day is 2026-10-07
    dt = tz.localize(datetime.datetime(2026, 10, 7, 16, 30, 0))
    completed_date = service.get_latest_completed_trading_day(dt)
    assert completed_date == datetime.date(2026, 10, 7)

def test_latest_completed_trading_day_on_weekend():
    service = MarketStatusService()
    # Sunday 2026-10-11 at 14:00 IST
    # Latest completed day was Friday 2026-10-09
    dt = tz.localize(datetime.datetime(2026, 10, 11, 14, 0, 0))
    completed_date = service.get_latest_completed_trading_day(dt)
    assert completed_date == datetime.date(2026, 10, 9)

def test_next_trading_day_calculation():
    service = MarketStatusService()
    # Friday 2026-10-09 -> next trading day is Monday 2026-10-12
    friday = datetime.date(2026, 10, 9)
    next_day = service.get_next_trading_day(friday)
    assert next_day == datetime.date(2026, 10, 12)
