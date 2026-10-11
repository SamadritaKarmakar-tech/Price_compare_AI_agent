"""Orchestrates: (photo ->) intent -> search -> extract -> rank -> explain -> track."""
import json
import time

from . import config, db, extract, llm, memory, profile, rank, search, vision


class LimitReached(Exception):
    pass


INTENT_SYSTEM = (
    "You turn shopping requests into JSON for a price-comparison agent in India (prices in INR). Return ONLY JSON.\n"
    "Rules: the user's explicit request always beats the profile. Use the profile only to fill gaps "
    "(e.g. budget, skin type). Never invent facts about the user."
)
INTENT_PROMPT = """User request: {text}
{image_block}
User profile (may be empty): {profile}
Remembered notes (may be empty): {memories}

Return JSON:
{{"category": "short lowercase category or null",
  "budget_inr": number|null,
  "brand": str|null,
  "must_have": [short requirements],
  "priority": "cheapest"|"best_rated"|"balanced",
  "search_queries": [1-3 short shopping queries; do NOT include store names],
  "stated_facts": {{only if the user explicitly stated them: skin_type, hair_type, shoe_size, clothing_size, preferred_style, diet_preference}},
  "clarify": "ONE short question, only if you cannot tell what product they want, else null"}}"""

EXPLAIN_SYSTEM = (
    "You write short, honest shopping recommendations. Use ONLY the data given. "
    "If price or rating is null say it was not available. Never invent specs, discounts or reviews. "
    "Reply in the same language as the user's request."
)
EXPLAIN_PROMPT = """User wants: {need}
Ranking priority: {priority}. Budget: {budget}.
Top results (already ranked by code; 'alternatives' are the same product in other stores):
{products}

Write under 110 words: the best pick and why (price, rating, fit), one alternative or cheaper store if relevant,
and any missing data. End with: "Prices come from search results and may have changed - check the link before buying." """


def _need_text(text, intent, image_info):
    bits = [text] if text else []
    if image_info:
        bits.append("(similar to the photo: " + ", ".join(
            str(v) for k, v in image_info.items()
            if k in ("category", "color", "style", "material", "brand") and v) + ")")
    if intent.get("must_have"):
        bits.append("must have: " + ", ".join(intent["must_have"]))
    if intent.get("budget_inr"):
        bits.append(f"budget: ₹{int(intent['budget_inr'])}")
    return " ".join(bits).strip()


def parse_intent(text, prof, image_info, memories):
    image_block = ""
    if image_info:
        image_block = "Photo analysis: " + json.dumps({k: v for k, v in image_info.items() if v})
    raw = llm.chat_json([
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": INTENT_PROMPT.format(
            text=text or "(no text, photo only)", image_block=image_block,
            profile=json.dumps(profile.for_prompt(prof)), memories=json.dumps(memories))},
    ], max_tokens=700)
    intent = raw if isinstance(raw, dict) else {}
    queries = [q.strip() for q in intent.get("search_queries") or [] if isinstance(q, str) and q.strip()]
    if not queries and image_info:
        queries = image_info.get("search_queries") or []
    intent["search_queries"] = queries[:3]
    if intent.get("priority") not in rank.WEIGHTS:
        intent["priority"] = "balanced"
    try:
        intent["budget_inr"] = float(intent["budget_inr"]) if intent.get("budget_inr") else None
    except (TypeError, ValueError):
        intent["budget_inr"] = None
    return intent


def explain(need, intent, top):
    slim = [{k: p.get(k) for k in ("name", "brand", "store", "price", "rating", "reviews", "score", "over_budget",
                                   "alternatives")} for p in top[:3]]
    return llm.chat([
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": EXPLAIN_PROMPT.format(
            need=need, priority=intent.get("priority"), budget=intent.get("budget_inr") or "none",
            products=json.dumps(slim, ensure_ascii=False))},
    ], max_tokens=350, temperature=0.3)


def run(user_id: str, text: str, image_bytes: bytes = None) -> dict:
    db.ensure_user(user_id)
    if not db.bump_rate(user_id, config.DAILY_SEARCH_LIMIT):
        raise LimitReached(f"Daily limit of {config.DAILY_SEARCH_LIMIT} searches reached. Try again tomorrow.")

    consent = db.has_consent(user_id)
    prof = profile.get_profile(user_id)
    image_info = vision.describe_product(image_bytes) if image_bytes else None
    memories = memory.recall(user_id, text or (image_info or {}).get("category") or "shopping preferences") if consent else []

    intent = parse_intent(text, prof, image_info, memories)
    if intent.get("clarify") and not intent["search_queries"]:
        return {"type": "clarify", "message": intent["clarify"]}
    if not intent["search_queries"]:
        return {"type": "clarify", "message": "What would you like to buy? Add a product name or a clearer photo."}

    need = _need_text(text, intent, image_info)
    results = search.search_stores(intent["search_queries"])
    if not results:
        return {"type": "empty", "message": "No results found on the configured stores. Try different words."}

    offers = extract.extract_offers(need, results)
    top = rank.rank_products(offers, intent, prof)
    if not top:
        return {"type": "empty", "message": "Found pages, but none had usable product details. Try rephrasing."}

    explanation = explain(need, intent, top)

    search_id = None
    if consent:
        search_id = db.log_search(user_id, need[:300], "image" if image_bytes else "text",
                                  intent.get("category"), intent.get("budget_inr"))
        shown = []
        for p in top:
            shown.append(p)
            shown += [{**a, "name": p["name"], "brand": p.get("brand"), "rank": p["rank"]} for a in p["alternatives"]]
        db.log_shown(search_id, shown)
        profile.save_stated_facts(user_id, intent.get("stated_facts"))
        memory.remember(user_id, text)

    return {
        "type": "results",
        "search_id": search_id,
        "intent": {k: intent.get(k) for k in ("category", "budget_inr", "priority", "must_have")},
        "image": image_info,
        "products": top,
        "explanation": explanation,
        "checked_at": time.strftime("%d %b %Y, %H:%M UTC", time.gmtime()),
        "tracking": consent,
    }
