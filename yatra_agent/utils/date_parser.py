"""Relative and colloquial date, duration, and budget resolution for Indian Standard Time (IST)."""

from datetime import datetime, timedelta
import re
import zoneinfo
from typing import Optional, Tuple
from dateutil import parser as dateutil_parser

IST_TIMEZONE = "Asia/Kolkata"

WEEKDAYS_EN = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}

WEEKDAYS_HI = {
    "somwar": 0, "mangalwar": 1, "budhwar": 2, "guruwar": 3,
    "veervar": 3, "shukrawar": 4, "shaniwar": 5, "raviwar": 6, "itwar": 6
}


def get_current_ist_datetime() -> datetime:
    """Return the current time in IST timezone (Asia/Kolkata)."""
    try:
        tz = zoneinfo.ZoneInfo(IST_TIMEZONE)
        return datetime.now(tz)
    except Exception:
        # Fallback to local machine time
        return datetime.now()


def _get_next_weekday(start_date: datetime, target_weekday: int) -> datetime:
    """Get the next occurrence of target_weekday (0=Mon, ..., 6=Sun) strictly after start_date."""
    days_ahead = target_weekday - start_date.weekday()
    if days_ahead <= 0:  # Target day already happened this week or is today
        days_ahead += 7
    return start_date + timedelta(days=days_ahead)


def parse_travel_date(raw_date_str: str) -> Optional[str]:
    """Parse relative expressions or explicit dates into DD-MM-YYYY format.
    
    Handles:
        - Relative English: "today", "tomorrow", "day after tomorrow", "in 3 days", "this weekend", "next friday"
        - Relative Hindi: "aaj", "kal", "parso", "agle hafte", "agle somwar"
        - Explicit: "15-12-2026", "15/12/2026", "Dec 15", "15 December 2026"
    
    Returns:
        String in "DD-MM-YYYY" format, or None if unparseable.
    """
    if not raw_date_str:
        return None

    raw = raw_date_str.strip()
    raw_lower = raw.lower()
    now = get_current_ist_datetime()

    # 1. Immediate relatives (English + Hindi)
    if re.search(r"\b(day after tomorrow|parso)\b", raw_lower):
        return (now + timedelta(days=2)).strftime("%d-%m-%Y")
    elif re.search(r"\b(tomorrow|kal)\b", raw_lower):
        return (now + timedelta(days=1)).strftime("%d-%m-%Y")
    elif re.search(r"\b(today|aaj)\b", raw_lower):
        return now.strftime("%d-%m-%Y")
    elif re.search(r"\b(in 3 days|3 din baad|narso)\b", raw_lower):
        return (now + timedelta(days=3)).strftime("%d-%m-%Y")

    # 2. Week / Weekend
    if "this weekend" in raw_lower:
        # Target coming Saturday
        target = _get_next_weekday(now, 5) if now.weekday() != 5 else now
        return target.strftime("%d-%m-%Y")
    if "next weekend" in raw_lower:
        # Target Saturday next week
        saturday = _get_next_weekday(now, 5)
        return (saturday + timedelta(days=7)).strftime("%d-%m-%Y")
    if "next week" in raw_lower or "agle hafte" in raw_lower:
        return (now + timedelta(days=7)).strftime("%d-%m-%Y")
    if "next month" in raw_lower or "agle mahine" in raw_lower:
        return (now + timedelta(days=30)).strftime("%d-%m-%Y")

    # 3. Relative weekday (e.g., "next Friday", "agle somwar")
    for name, day_num in {**WEEKDAYS_EN, **WEEKDAYS_HI}.items():
        if f"next {name}" in raw_lower or f"agle {name}" in raw_lower or f"coming {name}" in raw_lower:
            target = _get_next_weekday(now, day_num)
            return target.strftime("%d-%m-%Y")
        elif re.search(rf"\b{name}\b", raw_lower):
            target = _get_next_weekday(now, day_num)
            return target.strftime("%d-%m-%Y")

    # 4. "in X days" regex
    match_in_days = re.search(r"in\s+(\d+)\s+days?", raw_lower)
    if match_in_days:
        days = int(match_in_days.group(1))
        return (now + timedelta(days=days)).strftime("%d-%m-%Y")

    # 5. Clean common Indian suffixes like "ko", "tarikh ko"
    cleaned = re.sub(r"\b(ko|taareekh|tarikh|date)\b", "", raw_lower).strip()

    # 6. Explicit date parsing via regex + dateutil
    date_match = re.search(r"\b(\d{1,2}[-/.](?:0[1-9]|1[0-2]|\d{1,2})[-/.]\d{2,4})\b", raw_lower)
    if not date_match:
        date_match = re.search(r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{2,4})?)\b", raw_lower)

    target_to_parse = date_match.group(1) if date_match else cleaned
    try:
        # Use dayfirst=True for Indian DD-MM-YYYY standard
        dt = dateutil_parser.parse(target_to_parse, dayfirst=True, default=now)
        # If parsed date is in the past, roll forward to current or next year
        if dt.date() < now.date() and dt.year == now.year:
            dt = dt.replace(year=now.year + 1)
        return dt.strftime("%d-%m-%Y")
    except Exception:
        pass

    return None


def parse_duration(text: str) -> Optional[int]:
    """Extract number of trip days from natural language expressions.
    
    Examples:
        "3 days" -> 3
        "5 nights" -> 6 (nights + 1)
        "a week" -> 7
        "weekend" -> 2
        "2 din" -> 2
    """
    if not text:
        return None

    t = text.lower().strip()

    if "weekend" in t:
        return 2
    if "a week" in t or "one week" in t or "1 week" in t or "ek hafta" in t:
        return 7

    # Check nights
    match_nights = re.search(r"(\d+)\s*(?:nights?|raat)", t)
    if match_nights:
        return int(match_nights.group(1)) + 1

    # Check days
    match_days = re.search(r"(\d+)\s*(?:days?|din|divas)", t)
    if match_days:
        return int(match_days.group(1))

    return None


def parse_budget(text: str) -> Optional[float]:
    """Extract budget amount in INR from natural language expressions.
    
    Examples:
        "₹8000" -> 8000.0
        "8k" -> 8000.0
        "1.5 lakh" -> 150000.0
        "10,000 rupees" -> 10000.0
    """
    if not text:
        return None

    t = text.lower().strip()

    # 1. Lakh check
    match_lakh = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lac|lacs?)", t)
    if match_lakh:
        return float(match_lakh.group(1)) * 100000.0

    # 2. 'k' suffix check (e.g., 8k, 12k)
    match_k = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*k\b", t)
    if match_k:
        return float(match_k.group(1)) * 1000.0

    # 3. Numeric check with optional commas and currency symbols
    match_num = re.search(r"(?:₹|rs\.?|inr|budget:?)\s*([0-9,]+(?:\.\d+)?)", t)
    if match_num:
        raw_num = match_num.group(1).replace(",", "")
        try:
            val = float(raw_num)
            if val >= 50:
                return val
        except ValueError:
            pass

    # 4. Standalone numbers of 3 to 7 digits (e.g., "budget 8000")
    for token in re.findall(r"\b\d{3,7}\b", t):
        try:
            val = float(token)
            if val >= 500:
                return val
        except ValueError:
            pass

    return None
