"""Indian city name and alias resolver."""

from typing import Dict, Optional

# Canonical city mapping
CITY_ALIASES: Dict[str, str] = {
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "madras": "Chennai",
    "chennai": "Chennai",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "poona": "Pune",
    "pune": "Pune",
    "benares": "Varanasi",
    "banaras": "Varanasi",
    "kashi": "Varanasi",
    "varanasi": "Varanasi",
    "cochin": "Kochi",
    "kochi": "Kochi",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    "shimla": "Shimla",
    "simla": "Shimla",
    "ooty": "Udhagamandalam",
    "udhagamandalam": "Udhagamandalam",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "dilli": "Delhi",
    "jaipur": "Jaipur",
    "udaipur": "Udaipur",
    "agra": "Agra",
    "goa": "Goa",
    "manali": "Manali",
    "leh": "Leh",
    "ladakh": "Leh",
    "amritsar": "Amritsar",
    "rishikesh": "Rishikesh",
    "haridwar": "Haridwar",
    "hyderabad": "Hyderabad",
}


def resolve_city_name(raw_name: str) -> str:
    """Resolve colloquial, historical, or alternate city names to canonical Indian names."""
    if not raw_name:
        return ""
    cleaned = raw_name.strip().lower()
    return CITY_ALIASES.get(cleaned, raw_name.strip().title())
