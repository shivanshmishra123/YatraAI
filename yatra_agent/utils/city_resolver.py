"""Indian city name and alias resolver with fuzzy typo tolerance."""

import difflib
from typing import Dict, List, Optional

# Canonical city mapping (50+ Indian travel destinations and historical/colonial aliases)
CITY_ALIASES: Dict[str, str] = {
    # Metropolitan & Major Cities
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "navi mumbai": "Mumbai",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "madras": "Chennai",
    "chennai": "Chennai",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "old delhi": "Delhi",
    "dilli": "Delhi",
    "ncr": "Delhi",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "noida": "Noida",
    "hyderabad": "Hyderabad",
    "secunderabad": "Hyderabad",
    "poona": "Pune",
    "pune": "Pune",
    "ahmedabad": "Ahmedabad",
    "amdavad": "Ahmedabad",
    "baroda": "Vadodara",
    "vadodara": "Vadodara",
    "surat": "Surat",

    # Spiritual & Pilgrimage Hubs
    "benares": "Varanasi",
    "banaras": "Varanasi",
    "kashi": "Varanasi",
    "varanasi": "Varanasi",
    "allahabad": "Prayagraj",
    "prayag": "Prayagraj",
    "prayagraj": "Prayagraj",
    "haridwar": "Haridwar",
    "hardwar": "Haridwar",
    "rishikesh": "Rishikesh",
    "hrishikesh": "Rishikesh",
    "amritsar": "Amritsar",
    "golden temple": "Amritsar",
    "ayodhya": "Ayodhya",
    "mathura": "Mathura",
    "vrindavan": "Vrindavan",
    "puri": "Puri",
    "jagannath puri": "Puri",
    "tirupati": "Tirupati",
    "tirumala": "Tirupati",
    "shirdi": "Shirdi",
    "bodh gaya": "Bodh Gaya",
    "gaya": "Bodh Gaya",
    "rameswaram": "Rameswaram",
    "rameshwaram": "Rameswaram",

    # Rajasthan Heritage
    "jaipur": "Jaipur",
    "pink city": "Jaipur",
    "udaipur": "Udaipur",
    "lake city": "Udaipur",
    "jodhpur": "Jodhpur",
    "blue city": "Jodhpur",
    "jaisalmer": "Jaisalmer",
    "pushkar": "Pushkar",
    "bikaner": "Bikaner",
    "ajmer": "Ajmer",
    "mount abu": "Mount Abu",

    # North & Himalayan Hill Stations
    "agra": "Agra",
    "shimla": "Shimla",
    "simla": "Shimla",
    "manali": "Manali",
    "kullu": "Kullu",
    "dharamshala": "Dharamshala",
    "mcleodganj": "McLeod Ganj",
    "kasol": "Kasol",
    "leh": "Leh",
    "ladakh": "Leh",
    "spiti": "Spiti Valley",
    "kaza": "Spiti Valley",
    "srinagar": "Srinagar",
    "gulmarg": "Gulmarg",
    "pahalgam": "Pahalgam",
    "nainital": "Nainital",
    "mussoorie": "Mussoorie",
    "dehradun": "Dehradun",
    "chandigarh": "Chandigarh",

    # South India Travel & Coasts
    "goa": "Goa",
    "panaji": "Goa",
    "panjim": "Goa",
    "cochin": "Kochi",
    "kochi": "Kochi",
    "ernakulam": "Kochi",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "alleppey": "Alappuzha",
    "alappuzha": "Alappuzha",
    "munnar": "Munnar",
    "wayanad": "Wayanad",
    "varkala": "Varkala",
    "kovalam": "Kovalam",
    "calicut": "Kozhikode",
    "kozhikode": "Kozhikode",
    "cannanore": "Kannur",
    "kannur": "Kannur",
    "quilon": "Kollam",
    "kollam": "Kollam",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
    "hampi": "Hampi",
    "hosapete": "Hampi",
    "ooty": "Udhagamandalam",
    "udhagamandalam": "Udhagamandalam",
    "kodaikanal": "Kodaikanal",
    "kodai": "Kodaikanal",
    "coorg": "Madikeri",
    "madikeri": "Madikeri",
    "gokarna": "Gokarna",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    "pondy": "Puducherry",
    "madurai": "Madurai",
    "trichy": "Tiruchirappalli",
    "tiruchirappalli": "Tiruchirappalli",
    "tanjore": "Thanjavur",
    "thanjavur": "Thanjavur",
    "mahabalipuram": "Mamallapuram",
    "mamallapuram": "Mamallapuram",
    "kanyakumari": "Kanyakumari",
    "vizag": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam",
    "waltair": "Visakhapatnam",
    "vijayawada": "Vijayawada",
    "belgaum": "Belagavi",
    "belagavi": "Belagavi",
    "mangalore": "Mangaluru",
    "mangaluru": "Mangaluru",

    # East & North East
    "darjeeling": "Darjeeling",
    "gangtok": "Gangtok",
    "shillong": "Shillong",
    "guwahati": "Guwahati",
    "kaziranga": "Kaziranga",
    "tawang": "Tawang",
    "bhubaneswar": "Bhubaneswar",
    "cuttack": "Cuttack",
    "patna": "Patna",
    "ranchi": "Ranchi",

    # Islands
    "andaman": "Port Blair",
    "port blair": "Port Blair",
    "havelock": "Havelock Island",
}


def resolve_city_name(raw_name: str, fuzzy_threshold: float = 0.8) -> str:
    """Resolve colloquial, historical, or alternate city names to canonical Indian names.
    
    Includes exact match lookup and fuzzy matching for typos.
    
    Examples:
        resolve_city_name("bombay") -> "Mumbai"
        resolve_city_name("bengaluruu") -> "Bengaluru"
        resolve_city_name("pondy") -> "Puducherry"
    """
    if not raw_name:
        return ""

    cleaned = raw_name.strip().lower()

    # 1. Exact match
    if cleaned in CITY_ALIASES:
        return CITY_ALIASES[cleaned]

    # 2. Check if clean name matches any canonical title case
    for alias, canonical in CITY_ALIASES.items():
        if cleaned == canonical.lower():
            return canonical

    # 3. Fuzzy matching for slight typos
    matches = difflib.get_close_matches(cleaned, CITY_ALIASES.keys(), n=1, cutoff=fuzzy_threshold)
    if matches:
        return CITY_ALIASES[matches[0]]

    # 4. Fallback: title case of input
    return raw_name.strip().title()
