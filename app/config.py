import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "").strip() or GEMINI_MODEL
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")

DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "app.db"))

# domain -> display name (only domains listed in STORE_DOMAINS are searched)
KNOWN_STORES = {
    "amazon.in": "Amazon",
    "nykaa.com": "Nykaa",
    "myntra.com": "Myntra",
    "flipkart.com": "Flipkart",
    "ajio.com": "Ajio",
    "tatacliq.com": "Tata CLiQ",
}
_domains = [d.strip().lower() for d in os.getenv(
    "STORE_DOMAINS", "amazon.in,nykaa.com,myntra.com,flipkart.com,ajio.com").split(",") if d.strip()]
STORE_NAMES = {d: KNOWN_STORES.get(d, d) for d in _domains}

PER_STORE_SEARCH = os.getenv("PER_STORE_SEARCH", "false").lower() == "true"
MAX_QUERIES_PER_REQUEST = _int("MAX_QUERIES_PER_REQUEST", 2)
MAX_SNIPPETS = _int("MAX_SNIPPETS", 12)  # search results sent to the LLM; fewer = fewer tokens
RESULTS_PER_QUERY = _int("RESULTS_PER_QUERY", 8)
CACHE_TTL_SECONDS = _int("CACHE_TTL_SECONDS", 6 * 3600)
DAILY_SEARCH_LIMIT = _int("DAILY_SEARCH_LIMIT", 20)
MAX_IMAGE_BYTES = 5 * 1024 * 1024
HALF_LIFE_DAYS = 30

# --- accounts ---
SESSION_DAYS = _int("SESSION_DAYS", 30)
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # set true when served over HTTPS
MAX_LOGIN_FAILURES = 5
LOGIN_LOCK_MINUTES = 15
