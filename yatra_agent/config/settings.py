"""Global configuration and constants for Yatra AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
TRAVEL_DOCS_DIR = os.getenv("TRAVEL_DOCS_DIR", str(DATA_DIR / "travel_docs"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "travel_vectordb"))

# Model Configuration
PRIMARY_MODEL = "gpt-4o-mini"
PRIMARY_MODEL_TEMPERATURE = 0.7
PRIMARY_MODEL_MAX_TOKENS = 4096

REASONING_MODEL = "gemini-1.5-pro"
REASONING_MODEL_TEMPERATURE = 0.3
REASONING_MODEL_MAX_TOKENS = 8192

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# RAG Configuration
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 5

# MCP Configuration
MCP_CONFIG_PATH = os.getenv(
    "MCP_CONFIG_PATH",
    str(Path(__file__).resolve().parent / "servers_config.json")
)
IRCTC_TIMEOUT = 120  # seconds
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 2
RETRY_BACKOFF = [1, 3]  # seconds

# Agent Configuration
MAX_AGENT_ITERATIONS = 15
MAX_VALIDATION_CYCLES = 2

# Indian Context Constants
DEFAULT_CURRENCY = "INR"
CURRENCY_SYMBOL = "₹"
TIMEZONE = "Asia/Kolkata"
DATE_FORMAT = "%d-%m-%Y"
PETROL_PRICE_PER_LITER = 100  # INR
DIESEL_PRICE_PER_LITER = 95   # INR
AVG_CAR_MILEAGE_KM_PER_LITER = 15
NON_HIGHWAY_TIME_MULTIPLIER = 1.3

# Budget Thresholds (per night in INR)
LUXURY_BUDGET_THRESHOLD = 3000
MID_BUDGET_THRESHOLD = 1500
ECONOMY_BUDGET_THRESHOLD = 800

# Complexity Triggers
TIGHT_BUDGET_PER_PERSON_PER_DAY = 500  # INR
MULTI_CITY_THRESHOLD = 3
LONG_DISTANCE_FLIGHT_THRESHOLD = 500  # km
LONG_TRAIN_DURATION_THRESHOLD = 12  # hours
