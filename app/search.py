"""Tavily search restricted to shopping sites, with SQLite caching."""
import hashlib
import json
from urllib.parse import urlparse

from tavily import TavilyClient

from . import config, db

_client = None


def _tavily() -> TavilyClient:
    global _client
    if _client is None:
        if not config.TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")
        _client = TavilyClient(api_key=config.TAVILY_API_KEY)
    return _client


def store_for(url: str):
    host = urlparse(url).netloc.lower()
    for domain, name in config.STORE_NAMES.items():
        if host == domain or host.endswith("." + domain):
            return name
    return None


def _search_one(query: str, domains: list) -> list:
    key = hashlib.sha256(json.dumps([query.lower().strip(), sorted(domains)]).encode()).hexdigest()
    cached = db.cache_get(key, config.CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    resp = _tavily().search(
        query=query,
        search_depth="basic",
        max_results=config.RESULTS_PER_QUERY,
        include_domains=domains,
    )
    results = [
        {"title": r.get("title") or "", "url": r.get("url") or "", "content": r.get("content") or ""}
        for r in resp.get("results", []) if r.get("url")
    ]
    db.cache_set(key, results)
    return results


def search_stores(queries: list) -> list:
    """Returns de-duplicated results [{title, url, content, store}] from the configured stores."""
    all_domains = list(config.STORE_NAMES)
    seen, out = set(), []
    for i, q in enumerate(queries[: config.MAX_QUERIES_PER_REQUEST]):
        # per-store mode: only the first (primary) query is split by store, to save credits
        groups = [[d] for d in all_domains] if (config.PER_STORE_SEARCH and i == 0) else [all_domains]
        for domains in groups:
            for r in _search_one(q, domains):
                store = store_for(r["url"])
                if store and r["url"].startswith("https://") and r["url"] not in seen:
                    seen.add(r["url"])
                    out.append({**r, "store": store})
    return out
