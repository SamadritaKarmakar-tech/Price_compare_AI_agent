"""Optional Mem0 hosted memory for free-form preferences. Fully disabled unless MEM0_API_KEY is set.
All calls fail soft: if Mem0 is down or the SDK changed, the app keeps working with SQLite alone."""
import logging

from . import config

log = logging.getLogger(__name__)
_client = None
_disabled = False


def _get():
    global _client, _disabled
    if not config.MEM0_API_KEY or _disabled:
        return None
    if _client is None:
        try:
            from mem0 import MemoryClient
            _client = MemoryClient(api_key=config.MEM0_API_KEY)
        except Exception as e:  # missing package, bad key, ...
            _disabled = True  # warn once, not on every search
            log.warning("Mem0 is switched off (%s). Run `pip install mem0ai`, or leave MEM0_API_KEY empty in .env "
                        "to silence this. The app works fine without it.", e)
            return None
    return _client


def recall(user_id: str, query: str, limit: int = 5) -> list:
    c = _get()
    if not c:
        return []
    try:
        try:
            res = c.search(query, filters={"user_id": user_id})
        except TypeError:
            res = c.search(query, user_id=user_id)
        items = res.get("results", []) if isinstance(res, dict) else res
        return [m.get("memory", "") for m in items[:limit] if m.get("memory")]
    except Exception as e:
        log.warning("Mem0 recall failed: %s", e)
        return []


def remember(user_id: str, text: str) -> None:
    c = _get()
    if c and text:
        try:
            c.add([{"role": "user", "content": text}], user_id=user_id)
        except Exception as e:
            log.warning("Mem0 add failed: %s", e)


def forget(user_id: str) -> None:
    c = _get()
    if c:
        try:
            c.delete_all(user_id=user_id)
        except Exception as e:
            log.warning("Mem0 delete failed: %s", e)
