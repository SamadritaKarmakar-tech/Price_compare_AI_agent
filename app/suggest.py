"""Rule-based suggestions from the profile (no ML needed to start)."""
from . import db, profile

SKIN_WORDS = ("skin", "sunscreen", "serum", "moistur", "face", "cleanser", "cream", "lotion")


def get_suggestions(user_id: str) -> dict:
    if not db.has_consent(user_id):
        return {"ideas": [], "saved": []}
    p = profile.get_profile(user_id)
    budget = p["typical_budget"]
    facts = p["facts"]
    ideas = []
    for cat, _ in p["categories"][:3]:
        q = f"best {cat}"
        if "skin_type" in facts and any(w in cat for w in SKIN_WORDS):
            q += f" for {facts['skin_type']} skin"
        if budget:
            q += f" under ₹{int(budget)}"
        ideas.append({"label": f"More {cat}", "query": q, "reason": f"You often look for {cat}"})
    if p["brands"] and p["categories"]:
        brand, cat = p["brands"][0][0], p["categories"][0][0]
        ideas.append({"label": f"{brand.title()} {cat}", "query": f"{brand} {cat}",
                      "reason": f"You engage with {brand.title()}"})
    return {"ideas": ideas[:4], "saved": db.fetch_saved(user_id)}


# ---------- LLM-written suggestions based on the saved profile ----------
import hashlib
import json

from . import llm

IDEA_SYSTEM = (
    "You suggest what a shopper in India might want to search next. Use ONLY the profile given. "
    "Do not infer or mention health conditions, religion, finances or other sensitive traits. Return ONLY JSON."
)
IDEA_PROMPT = """Profile: {profile}

Return a JSON array of 3-4 objects:
{{"label": "max 4 words", "query": "a short shopping search the user could run", "reason": "max 12 words, referring to their stated or observed preference"}}"""


def _ideas_key(user_id: str, prompt_profile: dict) -> str:
    h = hashlib.sha256(json.dumps(prompt_profile, sort_keys=True).encode()).hexdigest()[:16]
    return f"ideas:{user_id}:{h}"


def cached_ideas(user_id: str):
    """Returns [] if there is nothing to base ideas on, None if a (paid) LLM call is needed."""
    pp = profile.for_prompt(profile.get_profile(user_id))
    if not (pp["favourite_categories"] or pp["favourite_brands"] or pp["stated_facts"]):
        return []
    return db.cache_get(_ideas_key(user_id, pp), 24 * 3600)


def generate_ideas(user_id: str) -> list:
    pp = profile.for_prompt(profile.get_profile(user_id))
    raw = llm.chat_json([{"role": "system", "content": IDEA_SYSTEM},
                         {"role": "user", "content": IDEA_PROMPT.format(profile=json.dumps(pp))}], max_tokens=450)
    ideas = []
    for i in raw if isinstance(raw, list) else []:
        if isinstance(i, dict) and isinstance(i.get("query"), str) and i["query"].strip():
            ideas.append({"label": str(i.get("label") or i["query"])[:40], "query": i["query"].strip()[:120],
                          "reason": str(i.get("reason") or "")[:100]})
    ideas = ideas[:4]
    db.cache_set(_ideas_key(user_id, pp), ideas)
    return ideas
