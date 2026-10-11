"""Deterministic ranking in plain code: auditable and consistent (the LLM never does the maths)."""
import re

WEIGHTS = {
    "cheapest":   {"price": 0.50, "rating": 0.15, "fit": 0.30, "data": 0.05},
    "best_rated": {"price": 0.15, "rating": 0.45, "fit": 0.35, "data": 0.05},
    "balanced":   {"price": 0.30, "rating": 0.30, "fit": 0.35, "data": 0.05},
}
PRIOR_MEAN, PRIOR_WEIGHT = 3.8, 50  # Bayesian average: few reviews -> pull toward the mean
STOP = {"the", "for", "with", "and", "of", "a", "an", "buy", "online", "india", "in", "by", "at", "on",
        "to", "from", "price", "best", "offers"}


def price_score(price, budget, min_price):
    if not price:
        return 0.2
    if budget:
        if price <= budget:
            return 0.6 + 0.4 * (1 - price / budget)
        return max(0.0, 0.6 - 1.5 * (price - budget) / budget)
    return min_price / price if min_price else 0.5


def rating_score(rating, reviews):
    if rating is None:
        return 0.4
    n = reviews if reviews is not None else 10
    adj = (rating * n + PRIOR_MEAN * PRIOR_WEIGHT) / (n + PRIOR_WEIGHT)
    return min(1.0, max(0.0, (adj - 1) / 4))


def _score(o, intent, profile, min_price):
    w = WEIGHTS.get(intent.get("priority"), WEIGHTS["balanced"])
    budget = intent.get("budget_inr")
    parts = {
        "price": price_score(o.get("price"), budget, min_price),
        "rating": rating_score(o.get("rating"), o.get("reviews")),
        "fit": o.get("relevance", 0.5),
        "data": (bool(o.get("price")) + bool(o.get("rating")) + bool(o.get("is_product_page"))) / 3,
    }
    total = sum(w[k] * parts[k] for k in w)
    brands = {b for b, _ in profile.get("brands", [])[:3]}
    stores = {s for s, _ in profile.get("stores", [])[:2]}
    if o.get("brand") and o["brand"].lower() in brands:
        total += 0.05
    if o["store"].lower() in stores:
        total += 0.02
    o["over_budget"] = bool(budget and o.get("price") and o["price"] > budget)
    if o["over_budget"]:
        total *= 0.7
    if not o.get("is_product_page"):
        total *= 0.85
    return round(min(1.0, max(0.0, total)), 4), {k: round(v, 3) for k, v in parts.items()}


def _tokens(s):
    return [t for t in re.findall(r"\d+|[a-z]+", (s or "").lower()) if t not in STOP]


def same_product(a, b) -> bool:
    ba = re.sub(r"[^a-z0-9]", "", (a.get("brand") or "").lower())
    bb = re.sub(r"[^a-z0-9]", "", (b.get("brand") or "").lower())
    if ba and bb and ba != bb:
        return False
    ta, tb = set(_tokens(a["name"])), set(_tokens(b["name"]))
    if not ta or not tb:
        return False
    na, nb = {t for t in ta if t.isdigit()}, {t for t in tb if t.isdigit()}
    if na and nb and na != nb:  # different size/pack count -> different product
        return False
    return len(ta & tb) / len(ta | tb) >= 0.6


def rank_products(offers: list, intent: dict, profile: dict = None, top_n: int = 5) -> list:
    profile = profile or {}
    prices = [o["price"] for o in offers if o.get("price")]
    min_price = min(prices) if prices else None
    for o in offers:
        o["score"], o["breakdown"] = _score(o, intent, profile, min_price)

    groups = []  # same product listed in several stores
    for o in sorted(offers, key=lambda x: -x["score"]):
        for g in groups:
            if same_product(g[0], o) and all(x["store"] != o["store"] for x in g):
                g.append(o)
                break
        else:
            groups.append([o])

    ranked = []
    for g in groups:
        primary = dict(g[0])
        alts = sorted(g[1:], key=lambda x: (x.get("price") is None, x.get("price") or 0))
        primary["alternatives"] = [
            {k: a.get(k) for k in ("key", "store", "price", "rating", "url", "over_budget")} for a in alts]
        ranked.append(primary)
    ranked.sort(key=lambda x: -x["score"])
    for i, p in enumerate(ranked[:top_n]):
        p["rank"] = i + 1
    return ranked[:top_n]
