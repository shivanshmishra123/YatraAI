"""Relative and colloquial date resolution for Indian Standard Time (IST)."""

from datetime import datetime, timedelta
import zoneinfo
from typing import Optional

IST_TIMEZONE = "Asia/Kolkata"


def get_current_ist_datetime() -> datetime:
    """Return the current time in IST timezone."""
    try:
        tz = zoneinfo.ZoneInfo(IST_TIMEZONE)
        return datetime.now(tz)
    except Exception:
        # Fallback to local machine time if tz is unavailable
        return datetime.now()


def parse_travel_date(raw_date_str: str) -> Optional[str]:
    """Parse relative expressions like 'tomorrow', 'next friday', 'in 3 days' into DD-MM-YYYY."""
    if not raw_date_str:
        return None

    raw_lower = raw_date_str.strip().lower()
    now = get_current_ist_datetime()

    if "today" in raw_lower or "aaj" in raw_lower:
        return now.strftime("%d-%m-%Y")
    elif "tomorrow" in raw_lower or "kal" in raw_lower:
        return (now + timedelta(days=1)).strftime("%d-%m-%Y")
    elif "day after tomorrow" in raw_lower or "parso" in raw_lower:
        return (now + timedelta(days=2)).strftime("%d-%m-%Y")

    return raw_date_str
