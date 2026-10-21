"""Comprehensive unit tests for date, duration, and budget parsing (20+ test cases)."""

from datetime import datetime, timedelta
import pytest
from yatra_agent.utils.date_parser import (
    get_current_ist_datetime,
    parse_budget,
    parse_duration,
    parse_travel_date,
)


def test_relative_date_parsing_en_today():
    now = get_current_ist_datetime()
    assert parse_travel_date("today") == now.strftime("%d-%m-%Y")


def test_relative_date_parsing_en_tomorrow():
    now = get_current_ist_datetime()
    assert parse_travel_date("tomorrow") == (now + timedelta(days=1)).strftime("%d-%m-%Y")


def test_relative_date_parsing_en_day_after():
    now = get_current_ist_datetime()
    assert parse_travel_date("day after tomorrow") == (now + timedelta(days=2)).strftime("%d-%m-%Y")


def test_relative_date_parsing_hi_aaj():
    now = get_current_ist_datetime()
    assert parse_travel_date("aaj") == now.strftime("%d-%m-%Y")


def test_relative_date_parsing_hi_kal():
    now = get_current_ist_datetime()
    assert parse_travel_date("kal") == (now + timedelta(days=1)).strftime("%d-%m-%Y")


def test_relative_date_parsing_hi_parso():
    now = get_current_ist_datetime()
    assert parse_travel_date("parso") == (now + timedelta(days=2)).strftime("%d-%m-%Y")


def test_explicit_hyphen_format():
    assert parse_travel_date("15-12-2026") == "15-12-2026"


def test_explicit_slash_format():
    assert parse_travel_date("15/12/2026") == "15-12-2026"


def test_explicit_reverse_iso_format():
    res = parse_travel_date("2026-12-15")
    assert res in ("15-12-2026", "2026-12-15")


def test_named_month_format():
    res = parse_travel_date("Dec 15")
    assert res is not None
    assert "-12-" in res or "15" in res


def test_next_week_parsing():
    res = parse_travel_date("next week")
    assert res is not None


def test_this_weekend_parsing():
    res = parse_travel_date("this weekend")
    assert res is not None


def test_duration_parsing_days():
    assert parse_duration("3 days") == 3


def test_duration_parsing_din():
    assert parse_duration("4 din") == 4


def test_duration_parsing_nights():
    assert parse_duration("5 nights") == 6


def test_duration_parsing_week():
    assert parse_duration("a week") == 7
    assert parse_duration("1 week") == 7


def test_duration_parsing_weekend():
    assert parse_duration("weekend") == 2


def test_duration_parsing_invalid():
    assert parse_duration("no numbers here") is None


def test_budget_symbol_inr():
    assert parse_budget("₹8000") == 8000.0


def test_budget_thousand_suffix():
    assert parse_budget("8k") == 8000.0
    assert parse_budget("50k") == 50000.0


def test_budget_comma_separated():
    assert parse_budget("₹15,000") == 15000.0


def test_budget_lakh_suffix():
    assert parse_budget("1.5 lakh") == 150000.0
    assert parse_budget("2 lakhs") == 200000.0


def test_budget_rs_prefix():
    assert parse_budget("Rs 12000") == 12000.0
    assert parse_budget("INR 25000") == 25000.0


def test_budget_empty_and_invalid():
    assert parse_budget("flexible budget without numbers") is None


def test_current_ist_datetime_timezone():
    dt = get_current_ist_datetime()
    assert dt is not None
    assert dt.year >= 2026