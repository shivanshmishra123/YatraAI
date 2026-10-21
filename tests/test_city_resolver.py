"""Unit tests for city alias resolution and fuzzy matching."""

import pytest
from yatra_agent.utils.city_resolver import resolve_city_name


def test_standard_city_aliases():
    """Verify colonial and historical city names resolve to canonical Indian names."""
    assert resolve_city_name("bombay") == "Mumbai"
    assert resolve_city_name("Bombay") == "Mumbai"
    assert resolve_city_name("mumbai") == "Mumbai"
    assert resolve_city_name("bangalore") == "Bengaluru"
    assert resolve_city_name("Bengaluru") == "Bengaluru"
    assert resolve_city_name("calcutta") == "Kolkata"
    assert resolve_city_name("madras") == "Chennai"
    assert resolve_city_name("poona") == "Pune"
    assert resolve_city_name("trivandrum") == "Thiruvananthapuram"


def test_spiritual_and_heritage_aliases():
    """Verify sacred cities, pilgrimage aliases, and nicknames."""
    assert resolve_city_name("benares") == "Varanasi"
    assert resolve_city_name("banaras") == "Varanasi"
    assert resolve_city_name("kashi") == "Varanasi"
    assert resolve_city_name("allahabad") == "Prayagraj"
    assert resolve_city_name("pink city") == "Jaipur"
    assert resolve_city_name("cochin") == "Kochi"
    assert resolve_city_name("pondy") == "Puducherry"
    assert resolve_city_name("ooty") == "Udhagamandalam"
    assert resolve_city_name("gurgaon") == "Gurugram"
    assert resolve_city_name("baroda") == "Vadodara"
    assert resolve_city_name("vizag") == "Visakhapatnam"


def test_fuzzy_matching_typos():
    """Verify fuzzy matching corrects slight user typos."""
    assert resolve_city_name("bengaluruu") == "Bengaluru"
    assert resolve_city_name("mumbaii") == "Mumbai"
    assert resolve_city_name("varanasii") == "Varanasi"


def test_unknown_city_fallback():
    """Verify unknown city names return cleanly in title case."""
    assert resolve_city_name("dhanbad") == "Dhanbad"
    assert resolve_city_name("") == ""
