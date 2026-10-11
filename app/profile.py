"""Builds a user profile from tracked behaviour, with recency decay."""
import statistics
import time
from collections import defaultdict

from . import config, db

ACTION_WEIGHTS = {"search": 1, "click": 3, "save": 5, "bought": 8, "dismiss": -3}
# Only product-relevant, non-sensitive facts are stored. No health conditions, no free text.
FACT_LABELS = {"skin_type": "Skin type", "hair_type": "Hair type", "shoe_size": "Shoe size",
               "clothing_size": "Clothing size", "preferred_style": "Style you like", "diet_preference": "Diet preference"}
ALLOWED_FACTS = {"skin_type", "hair_type", "shoe_size", "clothing_size", "preferred_style", "diet_preference"}


def _decay(ts: float, now: float) -> float:
    return 0.5 ** (max(0.0, now - ts) / 86400 / config.HALF_LIFE_DAYS)


def _norm(v):
    v = (v or "").strip().lower()
    return v or None


def _top(scores: dict, n: int) -> list:
    return [(k, round(v, 2)) for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v > 0][:n]


def get_profile(user_id: str, now: float = None) -> dict:
    if not db.has_consent(user_id):  # no behaviour tracking, but explicit settings the user typed still apply
        return {"categories": [], "brands": [], "stores": [], "typical_budget": None, "facts": db.get_facts(user_id)}
    now = now or time.time()
    cats, brands, stores = defaultdict(float), defaultdict(float), defaultdict(float)
    budgets = []
    for s in db.fetch_searches(user_id):
        w = ACTION_WEIGHTS["search"] * _decay(s["created_at"], now)
        if _norm(s["category"]):
            cats[_norm(s["category"])] += w
        if s["budget"]:
            budgets.append(s["budget"])
    for i in db.fetch_interactions(user_id):
        w = ACTION_WEIGHTS.get(i["action"], 0) * _decay(i["created_at"], now)
        for table, field in ((cats, "category"), (brands, "brand"), (stores, "store")):
            if _norm(i[field]):
                table[_norm(i[field])] += w
    return {
        "categories": _top(cats, 5),
        "brands": _top(brands, 5),
        "stores": _top(stores, 3),
        "typical_budget": round(statistics.median(budgets[:10])) if budgets else None,  # newest first
        "facts": db.get_facts(user_id),
    }


def for_prompt(profile: dict) -> dict:
    """Compact version safe to put in an LLM prompt."""
    return {
        "favourite_categories": [c for c, _ in profile["categories"][:3]],
        "favourite_brands": [b for b, _ in profile["brands"][:3]],
        "typical_budget_inr": profile["typical_budget"],
        "stated_facts": profile["facts"],
    }


def save_stated_facts(user_id: str, facts) -> None:
    if not isinstance(facts, dict):
        return
    for k, v in facts.items():
        if k in ALLOWED_FACTS and isinstance(v, str) and 0 < len(v.strip()) <= 40:
            db.set_fact(user_id, k, v.strip().lower())


def update_facts(user_id: str, facts) -> dict:
    """User-edited preferences from the settings panel. Empty value removes the preference."""
    if not isinstance(facts, dict):
        return db.get_facts(user_id)
    for k, v in facts.items():
        if k not in ALLOWED_FACTS or not isinstance(v, str):
            continue
        v = v.strip().lower()
        if not v:
            db.delete_fact(user_id, k)
        elif len(v) <= 40:
            db.set_fact(user_id, k, v)
    return db.get_facts(user_id)
