"""Unit tests for city alias resolution, fuzzy matching, and LRU caching (30+ aliases)."""

import pytest
from yatra_agent.utils.city_resolver import resolve_city_name


def test_standard_colonial_metro_aliases():
    """Verify colonial and historical city names resolve to canonical Indian names."""
    assert resolve_city_name("bombay") == "Mumbai"
    assert resolve_city_name("Bombay") == "Mumbai"
    assert resolve_city_name("mumbai") == "Mumbai"
    assert resolve_city_name("navi mumbai") == "Mumbai"
    assert resolve_city_name("bangalore") == "Bengaluru"
    assert resolve_city_name("Bengaluru") == "Bengaluru"
    assert resolve_city_name("calcutta") == "Kolkata"
    assert resolve_city_name("madras") == "Chennai"
    assert resolve_city_name("poona") == "Pune"
    assert resolve_city_name("trivandrum") == "Thiruvananthapuram"
    assert resolve_city_name("calicut") == "Kozhikode"
    assert resolve_city_name("trichy") == "Tiruchirappalli"
    assert resolve_city_name("tanjore") == "Thanjavur"
    assert resolve_city_name("cochin") == "Kochi"
    assert resolve_city_name("cannanore") == "Kannur"
    assert resolve_city_name("alleppey") == "Alappuzha"
    assert resolve_city_name("quilon") == "Kollam"


def test_spiritual_and_heritage_aliases():
    """Verify sacred cities, pilgrimage aliases, and nicknames."""
    assert resolve_city_name("benares") == "Varanasi"
    assert resolve_city_name("banaras") == "Varanasi"
    assert resolve_city_name("kashi") == "Varanasi"
    assert resolve_city_name("allahabad") == "Prayagraj"
    assert resolve_city_name("prayag") == "Prayagraj"
    assert resolve_city_name("pink city") == "Jaipur"
    assert resolve_city_name("city of lakes") == "Udaipur"
    assert resolve_city_name("blue city") == "Jodhpur"
    assert resolve_city_name("golden city") == "Jaisalmer"
    assert resolve_city_name("golden temple") == "Amritsar"
    assert resolve_city_name("haridwar") == "Haridwar"
    assert resolve_city_name("hardwar") == "Haridwar"
    assert resolve_city_name("hrishikesh") == "Rishikesh"
    assert resolve_city_name("rishikesh") == "Rishikesh"
    assert resolve_city_name("ayodhya") == "Ayodhya"
    assert resolve_city_name("mathura") == "Mathura"
    assert resolve_city_name("vrindavan") == "Vrindavan"
    assert resolve_city_name("shirdi") == "Shirdi"
    assert resolve_city_name("tirupati") == "Tirupati"
    assert resolve_city_name("tirumala") == "Tirupati"
    assert resolve_city_name("bodh gaya") == "Bodh Gaya"
    assert resolve_city_name("gaya") == "Bodh Gaya"
    assert resolve_city_name("rameswaram") == "Rameswaram"


def test_hill_station_and_regional_aliases():
    """Verify hill station, regional, and modernized naming aliases."""
    assert resolve_city_name("simla") == "Shimla"
    assert resolve_city_name("pondy") == "Puducherry"
    assert resolve_city_name("pondicherry") == "Puducherry"
    assert resolve_city_name("ooty") == "Udhagamandalam"
    assert resolve_city_name("gurgaon") == "Gurugram"
    assert resolve_city_name("baroda") == "Vadodara"
    assert resolve_city_name("vizag") == "Visakhapatnam"
    assert resolve_city_name("waltair") == "Visakhapatnam"
    assert resolve_city_name("gauhati") == "Guwahati"
    assert resolve_city_name("andaman") == "Port Blair"


def test_fuzzy_matching_typos():
    """Verify fuzzy matching corrects slight user typos."""
    assert resolve_city_name("bengaluruu") == "Bengaluru"
    assert resolve_city_name("mumbaii") == "Mumbai"
    assert resolve_city_name("varanasii") == "Varanasi"
    assert resolve_city_name("jodhpurr") == "Jodhpur"
    assert resolve_city_name("udaipuur") == "Udaipur"


def test_unknown_city_fallback_and_cache():
    """Verify unknown city names return cleanly in title case and LRU cache works."""
    assert resolve_city_name("dhanbad") == "Dhanbad"
    assert resolve_city_name("") == ""
    assert resolve_city_name("raipur") == "Raipur"

    # Test LRU cache info exists
    info = resolve_city_name.cache_info()
    assert info.hits >= 1 or info.misses >= 1
