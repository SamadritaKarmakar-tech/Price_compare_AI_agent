import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from app import agent, config, db, extract, llm, profile, rank, search, suggest, vision

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 64


@pytest.fixture(autouse=True)
def tmpdb(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()


UID = "u" * 20

RESULTS = [
    {"title": "Minimalist SPF 50 Sunscreen 50g", "url": "https://www.nykaa.com/p/1", "content": "x", "store": "Nykaa"},
    {"title": "Minimalist SPF 50 Sunscreen 50 g", "url": "https://www.amazon.in/dp/2", "content": "x", "store": "Amazon"},
    {"title": "Sunscreens listing", "url": "https://www.myntra.com/sunscreen", "content": "x", "store": "Myntra"},
]
EXTRACTED = [
    {"i": 0, "name": "Minimalist SPF 50 Sunscreen 50g", "brand": "Minimalist", "price_inr": "₹449", "rating": 4.4,
     "review_count": 1200, "is_product_page": True, "relevance": 0.9},
    {"i": 1, "name": "Minimalist SPF 50 Sunscreen 50g", "brand": "Minimalist", "price_inr": 399, "rating": 4.3,
     "review_count": 5000, "is_product_page": True, "relevance": 0.9},
    {"i": 2, "name": "Sunscreens", "brand": None, "price_inr": 99999, "is_product_page": False, "relevance": 0.4},
    {"i": 99, "name": "bad index"},
]


def test_parse_json_variants():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Sure! [{"a": 1}] hope it helps') == [{"a": 1}]
    with pytest.raises(ValueError):
        llm.parse_json("no json here")


def test_extract_validates_and_uses_real_urls(monkeypatch):
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: EXTRACTED)
    offers = extract.extract_offers("sunscreen", RESULTS)
    assert len(offers) == 3  # index 99 dropped
    assert offers[0]["price"] == 449.0 and offers[0]["url"] == RESULTS[0]["url"]
    assert offers[2]["price"] is None  # listing page price discarded


def test_rank_groups_same_product_and_prefers_cheaper():
    offers = [dict(o, key=o["url"][-3:]) for o in [
        {"name": "Minimalist SPF 50 Sunscreen 50g", "brand": "Minimalist", "price": 449, "rating": 4.4, "reviews": 1200,
         "relevance": 0.9, "is_product_page": True, "url": "https://nykaa.com/1", "store": "Nykaa"},
        {"name": "Minimalist SPF 50 Sunscreen 50 g", "brand": "Minimalist", "price": 399, "rating": 4.4, "reviews": 1200,
         "relevance": 0.9, "is_product_page": True, "url": "https://amazon.in/2", "store": "Amazon"},
        {"name": "Minimalist SPF 50 Sunscreen 100g", "brand": "Minimalist", "price": 700, "rating": 4.4, "reviews": 1200,
         "relevance": 0.9, "is_product_page": True, "url": "https://amazon.in/3", "store": "Amazon"},
    ]]
    top = rank.rank_products(offers, {"budget_inr": 500, "priority": "balanced"})
    assert len(top) == 2  # 50g listed in two stores is grouped; 100g stays separate
    assert top[0]["store"] == "Amazon" and top[0]["price"] == 399
    assert top[0]["alternatives"][0]["store"] == "Nykaa"
    assert top[1]["over_budget"] is True


def test_bayesian_rating_and_budget_penalty():
    assert rank.rating_score(5.0, 2) < rank.rating_score(4.5, 5000)
    assert rank.price_score(450, 500, 400) > rank.price_score(600, 500, 400)


def test_profile_needs_consent_and_decays():
    db.ensure_user(UID)
    sid = db.log_search(UID, "q", "text", "sunscreen", 500)
    db.log_shown(sid, [{"key": "k1", "name": "N", "brand": "Minimalist", "store": "Nykaa", "url": "https://nykaa.com/1"}])
    assert profile.get_profile(UID)["categories"] == []  # no consent -> empty
    db.set_consent(UID, True)
    assert db.log_interaction(UID, sid, "k1", "save")
    p = profile.get_profile(UID)
    assert p["categories"][0][0] == "sunscreen" and p["brands"][0][0] == "minimalist" and p["typical_budget"] == 500
    old = profile.get_profile(UID, now=time.time() + 90 * 86400)
    assert old["categories"][0][1] < p["categories"][0][1] / 4  # 3 half-lives
    assert not db.log_interaction("other-user-0000000000", sid, "k1", "save")  # cannot touch others' searches


def test_search_filters_to_known_stores(monkeypatch):
    monkeypatch.setattr(search, "_search_one", lambda q, d: [
        {"title": "a", "url": "https://www.amazon.in/x", "content": "c"},
        {"title": "b", "url": "https://evil.example/x", "content": "c"}])
    out = search.search_stores(["q"])
    assert [r["store"] for r in out] == ["Amazon"]


def test_vision_rejects_bad_files():
    with pytest.raises(vision.ImageError):
        vision.describe_product(b"not an image")
    with pytest.raises(vision.ImageError):
        vision.describe_product(JPEG + b"0" * config.MAX_IMAGE_BYTES)


def _fake_llm(monkeypatch, results=None):
    def chat_json(messages, model=None, max_tokens=0):
        sys = messages[0]["content"]
        if "identify shopping products" in sys:
            return {"category": "sunscreen", "brand": None, "confidence": "high", "search_queries": ["spf 50 sunscreen"]}
        if "turn shopping requests" in sys:
            return {"category": "sunscreen", "budget_inr": 500, "priority": "balanced",
                    "search_queries": ["oil free sunscreen"], "stated_facts": {"skin_type": "Oily", "disease": "x"}}
        if "what a shopper in India might want" in sys:
            return [{"label": "Oily-skin moisturiser", "query": "gel moisturiser oily skin", "reason": "You said oily skin"}]
        return EXTRACTED
    monkeypatch.setattr(llm, "chat_json", chat_json)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "Buy Minimalist on Amazon. Prices may have changed.")
    monkeypatch.setattr(search, "search_stores", lambda q: results or RESULTS)


def client():
    from app.main import app
    return TestClient(app, follow_redirects=False)


def csrf_of(c, path):
    html = c.get(path).text
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


def signup(c, email="a@example.com", consent=True, password="correct horse"):
    r = c.post("/register", data={"csrf": csrf_of(c, "/register"), "email": email, "password": password,
                                  "name": "Asha", "consent": "on" if consent else ""})
    assert r.status_code == 303 and r.headers["location"] == "/", r.text
    return r


def post(c, path, page="/", **data):
    return c.post(path, data={"csrf": csrf_of(c, page), **data})


def run_search(c, text="sunscreen oily skin under 500", **extra):
    r = post(c, "/search", text=text, **extra)
    assert r.status_code == 303, r.text
    return r.headers["location"]


def test_login_required_and_register_login_logout():
    with client() as c:
        r = c.get("/")
        assert r.status_code == 303 and r.headers["location"] == "/login"
        signup(c)
        home = c.get("/")
        assert home.status_code == 200 and "Asha" in home.text
        with db.conn() as k:
            row = k.execute("SELECT id, password_hash FROM users WHERE email='a@example.com'").fetchone()
        assert row["password_hash"].startswith("scrypt$") and "correct horse" not in row["password_hash"]
        assert len(row["id"]) == 32
        assert post(c, "/logout").headers["location"] == "/login?msg=loggedout"
        assert c.get("/").status_code == 303
        r = post(c, "/login", page="/login", email="A@Example.com ", password="correct horse")
        assert r.status_code == 303 and c.get("/").status_code == 200


def test_auth_errors_and_no_user_enumeration():
    with client() as c:
        signup(c)
        c.cookies.delete("session")
        tok = csrf_of(c, "/register")
        assert c.post("/register", data={"csrf": tok, "email": "a@example.com", "password": "another pass"}).status_code == 409
        assert c.post("/register", data={"csrf": tok, "email": "nope", "password": "longenough"}).status_code == 400
        assert c.post("/register", data={"csrf": tok, "email": "b@example.com", "password": "short"}).status_code == 400
        wrong = post(c, "/login", page="/login", email="a@example.com", password="wrong wrong")
        unknown = post(c, "/login", page="/login", email="zz@example.com", password="wrong wrong")
        msg = lambda r: re.search(r'role="alert">([^<]+)<', r.text).group(1)
        assert wrong.status_code == unknown.status_code == 401 and msg(wrong) == msg(unknown)


def test_login_lockout():
    with client() as c:
        signup(c)
        c.cookies.delete("session")
        codes = [post(c, "/login", page="/login", email="a@example.com", password="bad bad bad").status_code for _ in range(6)]
        assert codes[:5] == [401] * 5 and codes[5] == 429
        assert post(c, "/login", page="/login", email="a@example.com", password="correct horse").status_code == 429


def test_csrf_protection():
    with client() as c:
        c.get("/login")
        assert c.post("/login", data={"email": "a@b.co", "password": "x"}).status_code == 403            # no token
        assert c.post("/login", data={"csrf": "forged", "email": "a@b.co", "password": "x"}).status_code == 403
        signup(c)
        assert c.post("/logout", data={"csrf": "forged"}).status_code == 403
        assert c.get("/").status_code == 200  # still logged in
    with client() as other:  # a form posted with no cookie at all (cross-site) is refused
        assert other.post("/login", data={"csrf": "x", "email": "a@b.co", "password": "x"}).status_code == 403


def test_pages_have_no_javascript_and_strict_headers():
    with client() as c:
        pages = [c.get("/login"), c.get("/register")]
        signup(c)
        pages += [c.get("/"), c.get("/account")]
        for r in pages:
            assert r.status_code == 200
            assert "<script" not in r.text.lower() and " onclick" not in r.text.lower()
            assert "default-src 'none'" in r.headers["content-security-policy"]
            assert "script-src" not in r.headers["content-security-policy"]
            assert r.headers["x-frame-options"] == "DENY" and r.headers["cache-control"] == "no-store"
    with client() as fresh:
        cookie = fresh.get("/login").headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=lax" in cookie


def test_session_expiry():
    with client() as c:
        signup(c)
        with db.conn() as k:
            k.execute("UPDATE sessions SET expires_at = 1")
        assert c.get("/").status_code == 303


def test_search_results_page_save_click_and_suggestions(monkeypatch):
    _fake_llm(monkeypatch)
    with client() as c:
        signup(c)
        loc = run_search(c)
        assert loc.startswith("/results/")
        page = c.get(loc)
        assert "Buy Minimalist on Amazon" in page.text and "₹399" in page.text and "Best pick" in page.text
        assert c.get(loc).status_code == 200  # refreshing never re-runs the search
        rid = loc.rsplit("/", 1)[1]
        key = re.search(r'name="key" value="([0-9a-f]{12})"', page.text).group(1)
        # save
        r = post(c, "/save", page=loc, rid=rid, key=key)
        assert r.status_code == 303
        home = c.get("/")
        assert "Saved items" in home.text and "Minimalist SPF 50" in home.text
        assert "oily" in home.text  # personalised idea chip built from the stated skin type
        # click tracking redirect (target comes from our DB)
        sid = re.search(r"/go/(\d+)/" + key, page.text).group(1)
        go_r = c.get(f"/go/{sid}/{key}")
        assert go_r.status_code == 302 and go_r.headers["location"].startswith("https://")
        assert any(i["action"] == "click" for i in db.fetch_interactions(_uid(c)))
        # remove from saved
        r = c.post("/unsave", data={"csrf": csrf_of(c, "/"), "search_id": sid, "key": key})
        assert r.status_code == 303 and "Saved items" not in c.get("/").text


def _uid(c):
    with db.conn() as k:
        return k.execute("SELECT id FROM users LIMIT 1").fetchone()["id"]


def test_untrusted_text_is_escaped(monkeypatch):
    evil = [{"title": "x", "url": "https://www.nykaa.com/p/1", "content": "x", "store": "Nykaa"}]
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "<script>alert(1)</script> hello")
    _fake_llm(monkeypatch, results=evil)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "<script>alert(1)</script> hello")
    import app.extract as ex
    monkeypatch.setattr(ex.llm, "chat_json", lambda *a, **k: [
        {"i": 0, "name": "<img src=x onerror=alert(1)>", "price_inr": 100, "is_product_page": True, "relevance": 1}]
        if "extract product offers" in a[0][0]["content"] else
        {"category": "x", "search_queries": ["q"]})
    with client() as c:
        signup(c)
        html = c.get(run_search(c)).text
        assert "<script>alert" not in html and "&lt;script&gt;" in html
        assert "<img src=x" not in html and "&lt;img" in html


def test_preferences_and_ai_ideas(monkeypatch):
    _fake_llm(monkeypatch)
    with client() as c:
        signup(c)
        r = post(c, "/account/preferences", page="/account", hair_type="Curly", skin_type="Oily", shoe_size="", clothing_size="",
                 preferred_style="", diet_preference="")
        assert r.status_code == 303
        acct = c.get("/account").text
        assert 'value="curly"' in acct and 'value="oily"' in acct
        post(c, "/account/preferences", page="/account", skin_type="", hair_type="Curly")  # empty removes
        assert 'value="oily"' not in c.get("/account").text
        calls = []
        orig = suggest.generate_ideas
        monkeypatch.setattr(suggest, "generate_ideas", lambda u: calls.append(1) or orig(u))
        assert post(c, "/suggestions/ai").headers["location"] == "/?msg=ai"
        post(c, "/suggestions/ai")
        assert len(calls) == 1  # second request served from cache
        assert "Oily-skin moisturiser" in c.get("/").text


def test_image_upload_and_empty_file_field(monkeypatch):
    _fake_llm(monkeypatch)
    with client() as c:
        signup(c)
        tok = csrf_of(c, "/")
        r = c.post("/search", data={"csrf": tok, "text": ""}, files={"image": ("a.jpg", JPEG, "image/jpeg")})
        assert r.status_code == 303
        res = c.get(r.headers["location"]).text
        assert "Best pick" in res
        bad = c.post("/search", data={"csrf": tok, "text": ""}, files={"image": ("a.jpg", b"junk", "image/jpeg")})
        assert bad.status_code == 400 and "JPEG, PNG or WebP" in bad.text
        empty = c.post("/search", data={"csrf": tok, "text": "sunscreen"}, files={"image": ("", b"", "application/octet-stream")})
        assert empty.status_code == 303  # browsers send an empty file part when no photo is chosen
        assert c.post("/search", data={"csrf": tok, "text": ""}).status_code == 400


def test_users_are_isolated(monkeypatch):
    _fake_llm(monkeypatch)
    with client() as a, client() as b:
        signup(a, "a@example.com")
        signup(b, "b@example.com")
        loc = run_search(a)
        page = a.get(loc).text
        key = re.search(r'name="key" value="([0-9a-f]{12})"', page).group(1)
        sid = re.search(r"/go/(\d+)/" + key, page).group(1)
        assert b.get(loc).status_code == 404                       # cannot open A's results
        assert b.get(f"/go/{sid}/{key}").status_code == 404        # cannot use A's tracked links
        assert post(b, "/save", rid=loc.rsplit("/", 1)[1], key=key).status_code == 404
        assert post(b, "/unsave", search_id=sid, key=key).status_code == 404
        assert b.get("/account/export").json()["searches"] == []


def test_no_tracking_without_consent(monkeypatch):
    _fake_llm(monkeypatch)
    with client() as c:
        signup(c, consent=False)
        loc = run_search(c)
        page = c.get(loc).text
        assert "Best pick" in page and "/go/" not in page and 'action="/save"' not in page
        assert db.fetch_searches(_uid(c)) == []
        key = re.search(r'id="p-([0-9a-f]{12})"', page).group(1)
        assert post(c, "/save", rid=loc.rsplit("/", 1)[1], key=key).status_code == 403
        post(c, "/account/consent", page="/account", choice="on")
        assert "Personalisation" in c.get("/account").text and c.get("/account").text.count("Turn off") == 1


def test_export_clear_and_delete_account(monkeypatch):
    _fake_llm(monkeypatch)
    with client() as c:
        signup(c)
        run_search(c)
        exp = c.get("/account/export")
        assert "attachment" in exp.headers["content-disposition"]
        data = exp.json()
        assert data["account"][0]["email"] == "a@example.com" and "password_hash" not in json.dumps(data) and len(data["searches"]) == 1
        # clear needs the tick box; keeps account
        assert post(c, "/account/clear", page="/account").status_code == 400
        assert post(c, "/account/clear", page="/account", confirm="yes").status_code == 303
        assert c.get("/account/export").json()["searches"] == [] and c.get("/").status_code == 200
        # delete needs tick + correct password
        assert post(c, "/account/delete", page="/account", password="correct horse").status_code == 400
        assert post(c, "/account/delete", page="/account", password="wrong wrong", confirm="yes").status_code == 401
        assert post(c, "/account/delete", page="/account", password="correct horse", confirm="yes").status_code == 303
        with db.conn() as k:
            for t in ("users", "sessions", "searches", "shown_products", "facts", "rate", "results", "interactions"):
                assert k.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"] == 0, t
        assert c.get("/").status_code == 303


def test_daily_limit_and_unknown_page(monkeypatch):
    _fake_llm(monkeypatch)
    monkeypatch.setattr(config, "DAILY_SEARCH_LIMIT", 2)
    with client() as c:
        signup(c)
        codes = [post(c, "/search", text="sunscreen").status_code for _ in range(3)]
        assert codes == [303, 303, 429]
        assert c.get("/nope").status_code == 404


# ---------------- Gemini errors, model listing, Mem0 ----------------
import httpx
import openai


def _status_error(cls, code):
    resp = httpx.Response(code, request=httpx.Request(
        "POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"))
    return cls("boom", response=resp, body=None)


class _FakeClient:
    def __init__(self, exc):
        self.chat = self
        self.completions = self
        self._exc = exc

    def create(self, **kw):
        raise self._exc


@pytest.mark.parametrize("exc, expected", [
    (_status_error(openai.NotFoundError, 404), "GEMINI_MODEL"),
    (_status_error(openai.AuthenticationError, 401), "GEMINI_API_KEY"),
    (_status_error(openai.RateLimitError, 429), "rate limit"),
])
def test_gemini_errors_become_clear_messages(monkeypatch, exc, expected):
    monkeypatch.setattr(config, "GEMINI_MODEL", "some-unavailable-model")
    monkeypatch.setattr(llm, "_get", lambda: _FakeClient(exc))
    with pytest.raises(llm.LLMError) as e:
        llm.chat([{"role": "user", "content": "hi"}])
    assert expected in str(e.value)
    assert "Traceback" not in str(e.value) and "boom" not in str(e.value)


def test_missing_model_gives_setup_instructions(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MODEL", "")
    with pytest.raises(llm.LLMError) as e:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "GEMINI_MODEL is not set" in str(e.value)


def test_unavailable_model_shows_message_on_page_not_a_crash(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-unavailable")
    monkeypatch.setattr(llm, "_get", lambda: _FakeClient(_status_error(openai.NotFoundError, 404)))
    with client() as c:
        signup(c)
        r = post(c, "/search", text="sunscreen under 500")
        assert r.status_code == 502
        assert "gemini-unavailable" in r.text and "GEMINI_MODEL" in r.text
        assert "Traceback" not in r.text
        assert c.get("/").status_code == 200  # the app keeps working


def test_gemini_client_uses_google_api_and_key(monkeypatch):
    called = {}

    def fake_openai(**kwargs):
        called.update(kwargs)
        return object()

    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-google-key")
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "OpenAI", fake_openai)
    llm._get()
    assert called["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert called["api_key"] == "test-google-key"


def test_list_models_filters_generation_models(monkeypatch, capsys):
    import list_models
    data = [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-embedding", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
    ]
    assert list_models.select(data) == ["gemini-2.0-flash", "gemini-2.5-flash"]
    assert list_models.select(data, term="2.5") == ["gemini-2.5-flash"]
    monkeypatch.setattr(list_models, "fetch", lambda: data)
    assert list_models.main([]) == 0
    out = capsys.readouterr().out
    assert "gemini-2.5-flash" in out and "GEMINI_MODEL" in out
    monkeypatch.setattr(list_models, "fetch", lambda: (_ for _ in ()).throw(httpx.ConnectError("offline")))
    assert list_models.main([]) == 1 and "Could not download Gemini models" in capsys.readouterr().out


def test_mem0_missing_package_warns_once_and_search_still_works(monkeypatch, caplog):
    from app import memory
    monkeypatch.setattr(config, "MEM0_API_KEY", "m0-key")
    monkeypatch.setattr(memory, "_client", None)
    monkeypatch.setattr(memory, "_disabled", False)
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda n, *a, **k: (_ for _ in ()).throw(ImportError("No module named 'mem0'")) if n == "mem0" else real(n, *a, **k))
    with caplog.at_level("WARNING"):
        assert memory.recall("u", "q") == [] and memory.recall("u", "q") == []
        memory.remember("u", "text")
    assert sum("Mem0 is switched off" in r.message for r in caplog.records) == 1
    assert "pip install mem0ai" in caplog.text


# ---------------- Gemini chat response handling ----------------
import types


def _ok(text):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


def test_chat_passes_model_and_generation_settings(monkeypatch):
    class _Success:
        chat = completions = None

        def __init__(self):
            self.chat = self
            self.completions = self
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return _ok("answer")

    fake = _Success()
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(llm, "_get", lambda: fake)
    assert llm.chat([{"role": "user", "content": "hi"}], max_tokens=123, temperature=0.4) == "answer"
    assert fake.kwargs["model"] == "gemini-2.5-flash"
    assert fake.kwargs["max_tokens"] == 123
    assert fake.kwargs["temperature"] == 0.4


def test_truncated_json_array_keeps_finished_objects():
    cut = '[{"i": 0, "name": "A", "price_inr": 10}, {"i": 1, "name": "B", "price_inr": 20}, {"i": 2, "name": "C", "pri'
    assert [o["name"] for o in llm.parse_json(cut)] == ["A", "B"]
    assert llm.parse_json("[]") == []
    with pytest.raises(ValueError):
        llm.parse_json('{"a": 1, "b":')  # truncated objects are not guessed at


def test_extraction_budget_scales_and_stores_are_balanced(monkeypatch):
    results = [{"title": f"a{i}", "url": f"https://amazon.in/{i}", "content": "c", "store": "Amazon"} for i in range(10)] + \
              [{"title": "n", "url": "https://nykaa.com/1", "content": "c", "store": "Nykaa"},
               {"title": "m", "url": "https://myntra.com/1", "content": "c", "store": "Myntra"}]
    seen = {}

    def fake(messages, model=None, max_tokens=0):
        seen["max_tokens"], seen["prompt"] = max_tokens, messages[1]["content"]
        return []

    monkeypatch.setattr(llm, "chat_json", fake)
    monkeypatch.setattr(config, "MAX_SNIPPETS", 6)
    extract.extract_offers("x", results)
    assert seen["max_tokens"] == 250 + 120 * 6 and seen["max_tokens"] < 1500
    assert "Nykaa" in seen["prompt"] and "Myntra" in seen["prompt"]  # not crowded out by 10 Amazon results
