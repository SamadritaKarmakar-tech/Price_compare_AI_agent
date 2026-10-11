# Price Compare Agent

Describe a product, or upload a photo, and the agent compares prices across Indian shopping sites
(Amazon, Nykaa, Myntra, Flipkart, Ajio) and recommends the best option. It learns your preferences
(with your consent) and suggests what to look for next.

**Stack:** Python + FastAPI with Jinja2 templates and plain HTML forms (**no JavaScript anywhere**),
Tavily (search), Google Gemini (LLM + vision), SQLite, optional Mem0.

**Pages:** `/login`, `/register`, `/` (search + ideas + saved items), `/results/<id>`, `/account`
(preferences, personalisation, export, clear history, delete account). Everything is a normal page or form post.

Because there is no JavaScript: the browser simply waits while a search runs (usually several seconds),
there is no photo preview, and "ideas" fill the search box via a link instead of searching instantly.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then add your Gemini and Tavily keys
uvicorn app.main:app --reload
```
Open http://localhost:8000

Get a Gemini API key at https://aistudio.google.com/apikey and a Tavily key at https://tavily.com.

The default model is `gemini-3.8-flash`, which supports text and images. To use another Gemini model,
set `GEMINI_MODEL` (and optionally `GEMINI_VISION_MODEL` for photo searches) in `.env`.

You can list the models enabled for your Gemini API key with:
```bash
python list_models.py
```

## Troubleshooting

| You see | Meaning | Fix |
|---|---|---|
| *GEMINI_API_KEY is not set* | `.env` has no Gemini API key | Add a key from Google AI Studio to `GEMINI_API_KEY` |
| *Google Gemini rejected your API key* | The key is invalid or unavailable | Check `GEMINI_API_KEY` and its access in Google AI Studio |
| *Gemini does not provide the model ...* | The model is unavailable for the key | Run `python list_models.py` and set an available model in `GEMINI_MODEL` |
| Rate limit or quota error | The Gemini key has reached its request limit | Check the key's Gemini API quota and try again later |
| Photo search fails but text works | The configured model may not accept images | Use a Gemini model that supports image input for `GEMINI_VISION_MODEL` |
| `Mem0 is switched off (No module named 'mem0')` | `MEM0_API_KEY` is set but the package is not installed | Leave `MEM0_API_KEY` empty (recommended) or `pip install mem0ai`. This is only a warning |
| *TAVILY_API_KEY is not set* | Missing Tavily key | Add it to `.env` |

## Model token limits

A search makes 3-4 model calls. Their maximum answer lengths are: understanding the request 700, extracting
prices 250 + 120 per result (about 1,700 for the default 12 results; cut with `MAX_SNIPPETS`), explanation 350,
photo reading 600, "suggest ideas" 450. If a response is truncated, reduce `MAX_SNIPPETS` to lower the amount
of search text sent to the model.

## How it works

```
text and/or photo
  -> (photo) vision model -> product attributes            vision.py
  -> intent JSON (+ your profile)                          agent.py
  -> Tavily search on store domains (cached)               search.py
  -> LLM extracts price/rating from snippets               extract.py
  -> Python ranks + groups same product across stores      rank.py
  -> LLM explains the top picks                            agent.py
  -> log search/clicks/saves (only with consent)           db.py, profile.py
  -> suggestions from your profile                         suggest.py
```

Design rules: the LLM handles language (intent, extraction, explanation); plain Python handles numbers
(ranking), so results are consistent and auditable. Price, rating and URL always come from the real search
result. A missing value stays `null` and is shown as "not available" instead of being guessed.

## Ranking

Score = weighted mix of price fit, Bayesian-adjusted rating (few reviews are pulled toward the average),
fit to your request, and data completeness. Weights change with `priority` (`cheapest`, `best_rated`,
`balanced`). Over-budget items are penalised; listing pages (not a single product) are penalised.
Edit `WEIGHTS` in `app/rank.py` to tune.

## Accounts and login

- Users **create an account** (email + password) or **log in**. On sign-up the server creates a unique
  random user id; every search, click, saved item and preference is stored under that id.
- Passwords are hashed with scrypt (never stored in plain text). Login uses a random session token stored
  as an **HttpOnly, SameSite=Lax** cookie; only a hash of the token is kept in the database. Sessions last
  `SESSION_DAYS` (default 30) and end on logout.
- 5 wrong passwords for an email locks login for 15 minutes. Login errors never reveal whether an email exists.
- Every form carries a hidden CSRF token that must match an HttpOnly cookie, so other websites cannot make
  a logged-in user's browser submit forms to this app.
- Pages are sent with a strict Content-Security-Policy (`default-src 'none'`, no scripts allowed at all),
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` and `Cache-Control: no-store`. All text from the web
  is HTML-escaped by Jinja2.
- "Open on store" links go through `/go/...`, which logs the click and then redirects to a URL read from our own
  database (never from the request), so it cannot be abused as an open redirect.
- A search redirects to `/results/<id>`, so refreshing the page never re-runs a paid search. Result pages are kept
  for 6 hours and are deleted with your history or account.
- Behind HTTPS, set `COOKIE_SECURE=true`. **Do not run this on the public internet over plain HTTP.**
- Not included yet: email verification, password reset ("forgot password"), two-factor login, social login.
  These need an email service; add them before a public launch.
- Upgrading from the earlier no-login version: the database upgrades itself; old anonymous data is not linked to accounts.

## Personalisation

- Behaviour tracking needs consent (a checkbox at sign-up, changeable any time in My account).
  Without consent nothing is logged. Preferences the user types in *My preferences* still apply.
- **My preferences** panel lets users edit skin type, hair type, shoe size, clothing size, style and diet.
- **Suggest ideas for me** (button on the home page) asks the LLM for 3-4 searches based on the saved profile. Results are cached for
  24 hours per profile, and a new generation counts toward the daily limit.
- Weighted actions with 30-day half-life: search 1, click 3, save 5, bought 8, not-interested -3.
- Only these facts are kept when you state them: skin type, hair type, shoe size, clothing size, style, diet.
- Users can download their data (`GET /api/export`), clear history (`DELETE /api/data`) or delete the
  whole account (`DELETE /api/auth/account`) from the page.
- Optional Mem0: set `MEM0_API_KEY` and `pip install mem0ai`. **Read Mem0's current terms first.** At the time
  of writing, their free hosted plan grants Mem0 a licence to your content, which matters for user data.
  Self-hosting Mem0 is the safer route; this project only calls the hosted client, so adapt `memory.py` for that.

## Costs and limits

- Each request uses roughly 1-3 Tavily searches (`MAX_QUERIES_PER_REQUEST`) and 3-4 LLM calls
  (+1 for a photo). Results are cached for 6 hours. `DAILY_SEARCH_LIMIT` caps each user per day.
- `PER_STORE_SEARCH=true` searches each store separately for better coverage but uses more credits.

## Known limitations (be honest with your users)

- Prices come from search snippets and can be stale or missing. The UI always links to the live page.
- Stores that block crawlers may return thin results. For reliable prices at scale, move to affiliate
  feeds or official APIs.
- Same-product matching is by brand + name tokens. Different shades/colours with similar names can merge.
- Photo search finds *similar* products, not guaranteed exact ones.
- Login is basic: no email verification, password reset or 2FA yet (see Accounts above).
- Search snippets are untrusted text. Prompts tell the model to ignore instructions inside them, but no
  prompt defence is perfect: don't give this agent tools that can spend money or send data.

## Legal

Emails and behaviour are personal data. If you launch publicly in India, review the Digital Personal Data Protection Act, 2023 (consent notice,
deletion, security), each store's terms of service, and affiliate-disclosure rules. This is not legal advice.

## Tests

```bash
pip install pytest httpx && pytest   # 35 tests
```
APIs are mocked, so tests need no keys. Covered: parsing, extraction validation, ranking and grouping,
sign-up/login/logout, hashed passwords, login lockout, CSRF tokens, no `<script>` on any page, escaping of
untrusted text, security headers, session expiry, redirect endpoint, empty photo field, per-user isolation,
consent gating, preference editing, cached AI suggestions, account deletion, daily limit, image upload.

## Ideas for v2

Affiliate feeds, price history charts and drop alerts for saved items, Google Lens-style visual search
(needs a public image URL and a paid or limited provider), embeddings for "similar to what you liked",
Telegram/WhatsApp front end, email verification and password reset, two-factor login.
