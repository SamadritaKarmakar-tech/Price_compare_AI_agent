"""Server-rendered web app: FastAPI + Jinja2 + plain HTML forms. No JavaScript anywhere."""
import hmac
import logging
import re
import secrets
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.datastructures import UploadFile as BaseUploadFile  # FastAPI may pass this base class
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import agent, auth, config, db, llm, memory, profile, suggest, vision

log = logging.getLogger("price-agent")
SESSION_COOKIE, CSRF_COOKIE = "session", "csrf"

FLASH = {  # fixed messages only: a query-string code can never inject text into the page
    "saved": "Saved to your list.", "removed": "Removed from your list.", "prefs": "Preferences saved.",
    "consent_on": "Personalisation is on.", "consent_off": "Personalisation is off. We no longer track new activity.",
    "cleared": "Your history and preferences were cleared.", "ai": "Ideas updated.",
    "noideas": "Search a few things or add preferences first, then ask again.", "loggedout": "You are logged out.",
}


def _inr(v):
    if v is None:
        return "Price not shown"
    s = str(int(round(v)))
    if len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    return "₹" + s


templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))  # autoescape is on for .html
templates.env.filters["inr"] = _inr


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    yield


app = FastAPI(title="Price Compare Agent", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = (
        "default-src 'none'; style-src 'self'; img-src 'self' data:; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'none'")
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if not request.url.path.startswith("/static"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------- helpers ----------------
class LoginRequired(Exception):
    pass


def get_user(request: Request):
    if not hasattr(request.state, "user"):
        request.state.user = auth.user_from_token(request.cookies.get(SESSION_COOKIE, ""))
    return request.state.user


def current_user(request: Request) -> dict:
    user = get_user(request)
    if not user:
        raise LoginRequired()
    return user


def _cookie(resp, name, value, max_age):
    resp.set_cookie(name, value, httponly=True, samesite="lax", secure=config.COOKIE_SECURE, max_age=max_age, path="/")


def render(request: Request, template: str, status_code: int = 200, **ctx):
    token = request.cookies.get(CSRF_COOKIE) or secrets.token_urlsafe(32)
    ctx.update(user=get_user(request), flash=FLASH.get(request.query_params.get("msg")),
               csrf_field=Markup('<input type="hidden" name="csrf" value="%s">' % escape(token)))
    resp = templates.TemplateResponse(request, template, ctx, status_code=status_code)
    if CSRF_COOKIE not in request.cookies:
        _cookie(resp, CSRF_COOKIE, token, 30 * 86400)
    return resp


def check_csrf(request: Request, token: str):
    """Double-submit token: the form field must equal the HttpOnly cookie. Other sites can do neither."""
    cookie = request.cookies.get(CSRF_COOKIE, "")
    if not cookie or not token or not hmac.compare_digest(cookie, token):
        raise HTTPException(403, "This form expired. Go back, reload the page and try again.")


def go(path: str, code: str = None):
    return RedirectResponse(path + (f"?msg={code}" if code else ""), status_code=303)


@app.exception_handler(LoginRequired)
async def _login_required(request, exc):
    return go("/login")


@app.exception_handler(StarletteHTTPException)
async def _http_error(request, exc):
    msg = exc.detail if isinstance(exc.detail, str) and exc.status_code != 404 else "Page not found."
    return render(request, "error.html", status_code=exc.status_code, message=msg)


@app.exception_handler(RequestValidationError)
async def _bad_form(request, exc):
    return render(request, "error.html", status_code=400, message="That form was incomplete. Go back and try again.")


def _home_ctx(uid: str) -> dict:
    s = suggest.get_suggestions(uid)
    return {"ideas": s["ideas"], "saved": s["saved"], "ai_ideas": suggest.cached_ideas(uid) or []}


# ---------------- accounts ----------------
@app.get("/login")
def login_page(request: Request):
    return go("/") if get_user(request) else render(request, "auth.html", mode="login")


@app.post("/login")
def login(request: Request, csrf: str = Form(""), email: str = Form(""), password: str = Form("")):
    check_csrf(request, csrf)
    try:
        _, token = auth.login(email, password)
    except auth.AuthError as e:
        return render(request, "auth.html", status_code=e.status, mode="login", error=e.message, email=email)
    resp = go("/")
    _cookie(resp, SESSION_COOKIE, token, config.SESSION_DAYS * 86400)
    return resp


@app.get("/register")
def register_page(request: Request):
    return go("/") if get_user(request) else render(request, "auth.html", mode="register")


@app.post("/register")
def register(request: Request, csrf: str = Form(""), email: str = Form(""), password: str = Form(""),
             name: str = Form(""), consent: str = Form("")):
    check_csrf(request, csrf)
    try:
        _, token = auth.register(email, password, name, consent == "on")
    except auth.AuthError as e:
        return render(request, "auth.html", status_code=e.status, mode="register", error=e.message,
                      email=email, name=name)
    resp = go("/")
    _cookie(resp, SESSION_COOKIE, token, config.SESSION_DAYS * 86400)
    return resp


@app.post("/logout")
def logout(request: Request, csrf: str = Form("")):
    check_csrf(request, csrf)
    auth.end_session(request.cookies.get(SESSION_COOKIE, ""))
    resp = go("/login", "loggedout")
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


# ---------------- search ----------------
@app.get("/")
def home(request: Request, q: str = "", user: dict = Depends(current_user)):
    return render(request, "home.html", q=q[:500], **_home_ctx(user["id"]))


@app.post("/search")
def search(request: Request, csrf: str = Form(""), text: str = Form(""),
           image: Union[UploadFile, str, None] = File(None), user: dict = Depends(current_user)):
    check_csrf(request, csrf)
    text = text.strip()[:500]
    data = None
    if isinstance(image, BaseUploadFile) and image.filename:
        data = image.file.read(config.MAX_IMAGE_BYTES + 1)

    def fail(status, msg):
        return render(request, "home.html", status_code=status, error=msg, q=text, **_home_ctx(user["id"]))

    if not text and not data:
        return fail(400, "Type what you want or choose a photo.")
    try:
        result = agent.run(user["id"], text, data)
    except vision.ImageError as e:
        return fail(400, str(e))
    except agent.LimitReached as e:
        return fail(429, str(e))
    except llm.LLMError as e:  # wrong/retired model, bad key, no credits ...: tell the user what to fix
        log.warning("LLM error: %s", e)
        return fail(502, str(e))
    except RuntimeError as e:  # missing API keys
        return fail(500, str(e))
    except Exception:
        log.exception("search failed")
        return fail(502, "The AI or search service failed. Check your API keys and try again.")
    if result["type"] != "results":
        return render(request, "home.html", notice=result["message"], q=text, **_home_ctx(user["id"]))
    return go("/results/" + db.save_result(user["id"], result))  # redirect so a refresh never re-runs the search


@app.get("/results/{rid}")
def results(rid: str, request: Request, user: dict = Depends(current_user)):
    res = db.get_result(user["id"], rid)
    if not res:
        raise HTTPException(404, "That result has expired. Run the search again.")
    return render(request, "results.html", res=res, rid=rid)


def _key_in(res: dict, key: str) -> bool:
    return any(p["key"] == key or any(a["key"] == key for a in p["alternatives"]) for p in res["products"])


@app.post("/save")
def save(request: Request, csrf: str = Form(""), rid: str = Form(""), key: str = Form(""),
         user: dict = Depends(current_user)):
    check_csrf(request, csrf)
    res = db.get_result(user["id"], rid)
    if not res or not _key_in(res, key):
        raise HTTPException(404, "That result has expired.")
    if not res.get("search_id") or not db.has_consent(user["id"]):
        raise HTTPException(403, "Turn on personalisation in My account to save items.")
    db.log_interaction(user["id"], res["search_id"], key, "save")
    return RedirectResponse(f"/results/{rid}?msg=saved#p-{key}", status_code=303)


@app.post("/unsave")
def unsave(request: Request, csrf: str = Form(""), search_id: int = Form(...), key: str = Form(""),
           user: dict = Depends(current_user)):
    check_csrf(request, csrf)
    if not db.log_interaction(user["id"], search_id, key, "dismiss"):
        raise HTTPException(404, "Item not found.")
    return go("/", "removed")


@app.get("/go/{search_id}/{key}")
def open_store(search_id: int, key: str, user: dict = Depends(current_user)):
    """Logs the click, then sends the user to the store. The target is read from our own database,
    never from the request, so this cannot be used as an open redirect."""
    url = db.get_shown_url(user["id"], search_id, key)
    if not url or not url.startswith("https://"):
        raise HTTPException(404, "Link not found.")
    if db.has_consent(user["id"]):
        db.log_interaction(user["id"], search_id, key, "click")
    return RedirectResponse(url, status_code=302)


# ---------------- suggestions ----------------
@app.post("/suggestions/ai")
def ai_suggestions(request: Request, csrf: str = Form(""), user: dict = Depends(current_user)):
    check_csrf(request, csrf)
    ideas = suggest.cached_ideas(user["id"])
    if ideas is None:  # needs a paid LLM call, so it counts toward the daily limit
        def fail(status, msg):
            return render(request, "home.html", status_code=status, error=msg, q="", **_home_ctx(user["id"]))
        if not db.bump_rate(user["id"], config.DAILY_SEARCH_LIMIT):
            return fail(429, "Daily limit reached. Try again tomorrow.")
        try:
            ideas = suggest.generate_ideas(user["id"])
        except llm.LLMError as e:
            return fail(502, str(e))
        except RuntimeError as e:
            return fail(500, str(e))
        except Exception:
            log.exception("ai suggestions failed")
            return fail(502, "Could not generate suggestions right now.")
    return go("/", "ai" if ideas else "noideas")


# ---------------- account, preferences, privacy ----------------
@app.get("/account")
def account(request: Request, user: dict = Depends(current_user)):
    return render(request, "account.html", profile=profile.get_profile(user["id"]), labels=profile.FACT_LABELS)


@app.post("/account/preferences")
def save_preferences(request: Request, csrf: str = Form(""), skin_type: str = Form(""), hair_type: str = Form(""),
                     shoe_size: str = Form(""), clothing_size: str = Form(""), preferred_style: str = Form(""),
                     diet_preference: str = Form(""), user: dict = Depends(current_user)):
    check_csrf(request, csrf)
    profile.update_facts(user["id"], {
        "skin_type": skin_type, "hair_type": hair_type, "shoe_size": shoe_size, "clothing_size": clothing_size,
        "preferred_style": preferred_style, "diet_preference": diet_preference})
    return go("/account", "prefs")


@app.post("/account/consent")
def set_consent(request: Request, csrf: str = Form(""), choice: str = Form(""), user: dict = Depends(current_user)):
    check_csrf(request, csrf)
    on = choice == "on"
    db.set_consent(user["id"], on)
    return go("/account", "consent_on" if on else "consent_off")


@app.get("/account/export")
def export(user: dict = Depends(current_user)):
    return JSONResponse(db.export_user_data(user["id"]),
                        headers={"Content-Disposition": 'attachment; filename="my-data.json"'})


@app.post("/account/clear")
def clear_history(request: Request, csrf: str = Form(""), confirm: str = Form(""), user: dict = Depends(current_user)):
    check_csrf(request, csrf)
    if confirm != "yes":
        raise HTTPException(400, "Please tick the confirmation box.")
    db.delete_user_data(user["id"])
    memory.forget(user["id"])
    return go("/account", "cleared")


@app.post("/account/delete")
def delete_account(request: Request, csrf: str = Form(""), confirm: str = Form(""), password: str = Form(""),
                   user: dict = Depends(current_user)):
    check_csrf(request, csrf)

    def fail(status, msg):
        return render(request, "account.html", status_code=status, delete_error=msg,
                      profile=profile.get_profile(user["id"]), labels=profile.FACT_LABELS)

    if confirm != "yes":
        return fail(400, "Please tick the confirmation box.")
    try:
        auth.confirm_password(user["email"], password)
    except auth.AuthError as e:
        return fail(e.status, e.message)
    db.delete_account(user["id"])
    memory.forget(user["id"])
    resp = go("/register")
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp
