"""Unit tests for date, duration, and budget parsing."""

from datetime import datetime, timedelta
import pytest
from yatra_agent.utils.date_parser import (
    get_current_ist_datetime,
    parse_budget,
    parse_duration,
    parse_travel_date,
)


def test_relative_date_parsing_en():
    """Verify relative English expressions."""
    now = get_current_ist_datetime()
    today_str = now.strftime("%d-%m-%Y")
    tomorrow_str = (now + timedelta(days=1)).strftime("%d-%m-%Y")
    day_after_str = (now + timedelta(days=2)).strftime("%d-%m-%Y")

    assert parse_travel_date("today") == today_str
    assert parse_travel_date("tomorrow") == tomorrow_str
    assert parse_travel_date("day after tomorrow") == day_after_str


def test_relative_date_parsing_hi():
    """Verify Hindi / Hinglish relative expressions."""
    now = get_current_ist_datetime()
    today_str = now.strftime("%d-%m-%Y")
    tomorrow_str = (now + timedelta(days=1)).strftime("%d-%m-%Y")
    day_after_str = (now + timedelta(days=2)).strftime("%d-%m-%Y")

    assert parse_travel_date("aaj") == today_str
    assert parse_travel_date("kal") == tomorrow_str
    assert parse_travel_date("parso") == day_after_str


def test_explicit_date_parsing():
    """Verify explicit formats."""
    assert parse_travel_date("15-12-2026") == "15-12-2026"
    assert parse_travel_date("15/12/2026") == "15-12-2026"


def test_duration_parsing():
    """Verify trip duration parsing."""
    assert parse_duration("3 days") == 3
    assert parse_duration("3 din") == 3
    assert parse_duration("5 nights") == 6
    assert parse_duration("a week") == 7
    assert parse_duration("weekend") == 2
    assert parse_duration("random text without numbers") is None


def test_budget_parsing():
    """Verify budget extraction from text."""
    assert parse_budget("₹8000") == 8000.0
    assert parse_budget("8k") == 8000.0
    assert parse_budget("₹15,000") == 15000.0
    assert parse_budget("1.5 lakh") == 150000.0
    assert parse_budget("budget: 25000") == 25000.0
