"""SQLite storage: users, searches, interactions, facts, cache, rate limits."""
import json
import sqlite3
import time
import secrets
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id TEXT PRIMARY KEY, created_at REAL NOT NULL, consent INTEGER NOT NULL DEFAULT 0,
  email TEXT, name TEXT, password_hash TEXT);
CREATE TABLE IF NOT EXISTS sessions(
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS login_failures(email TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_login_failures ON login_failures(email, ts);
CREATE TABLE IF NOT EXISTS searches(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, query TEXT, input_type TEXT,
  category TEXT, budget REAL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_searches_user ON searches(user_id, created_at);
CREATE TABLE IF NOT EXISTS shown_products(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  search_id INTEGER NOT NULL REFERENCES searches(id) ON DELETE CASCADE,
  product_key TEXT, name TEXT, brand TEXT, store TEXT, price REAL, rating REAL, url TEXT, rank INTEGER);
CREATE INDEX IF NOT EXISTS idx_shown ON shown_products(search_id, product_key);
CREATE TABLE IF NOT EXISTS interactions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, search_id INTEGER, product_key TEXT,
  name TEXT, brand TEXT, category TEXT, store TEXT, price REAL, url TEXT,
  action TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_inter_user ON interactions(user_id, created_at);
CREATE TABLE IF NOT EXISTS facts(
  user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT, updated_at REAL, PRIMARY KEY(user_id, key));
CREATE TABLE IF NOT EXISTS results(
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_results_user ON results(user_id);
CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS rate(
  user_id TEXT NOT NULL, day TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(user_id, day));
"""


@contextmanager
def conn():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with conn() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}  # upgrade older databases
        for col in ("email", "name", "password_hash"):
            if col not in cols:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")


# ---------- accounts & sessions ----------
class EmailTaken(Exception):
    pass


def create_user(email: str, name: str, password_hash: str, consent: bool) -> str:
    uid = uuid.uuid4().hex  # the unique id every preference and search is stored under
    try:
        with conn() as c:
            c.execute("INSERT INTO users(id, created_at, consent, email, name, password_hash) VALUES(?,?,?,?,?,?)",
                      (uid, time.time(), 1 if consent else 0, email, name, password_hash))
    except sqlite3.IntegrityError:
        raise EmailTaken()
    return uid


def get_user_by_email(email: str):
    with conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def create_session(uid: str, token_hash: str, ttl_seconds: int):
    now = time.time()
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        c.execute("INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES(?,?,?,?)",
                  (token_hash, uid, now, now + ttl_seconds))


def user_for_session(token_hash: str):
    with conn() as c:
        row = c.execute(
            "SELECT u.id, u.email, u.name, u.consent FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash=? AND s.expires_at > ?", (token_hash, time.time())).fetchone()
    return dict(row) if row else None


def delete_session(token_hash: str):
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))


def recent_failures(email: str, window_seconds: int) -> int:
    with conn() as c:
        return c.execute("SELECT COUNT(*) AS n FROM login_failures WHERE email=? AND ts > ?",
                         (email, time.time() - window_seconds)).fetchone()["n"]


def record_failure(email: str):
    with conn() as c:
        c.execute("DELETE FROM login_failures WHERE ts < ?", (time.time() - 86400,))
        c.execute("INSERT INTO login_failures(email, ts) VALUES(?,?)", (email, time.time()))


def clear_failures(email: str):
    with conn() as c:
        c.execute("DELETE FROM login_failures WHERE email=?", (email,))


# ---------- users & consent ----------
def ensure_user(uid: str):
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO users(id, created_at) VALUES(?,?)", (uid, time.time()))


def set_consent(uid: str, value: bool):
    ensure_user(uid)
    with conn() as c:
        c.execute("UPDATE users SET consent=? WHERE id=?", (1 if value else 0, uid))


def has_consent(uid: str) -> bool:
    with conn() as c:
        row = c.execute("SELECT consent FROM users WHERE id=?", (uid,)).fetchone()
    return bool(row and row["consent"])


def bump_rate(uid: str, limit: int) -> bool:
    """Count one search for today. Returns False if the daily limit is reached."""
    day = time.strftime("%Y-%m-%d", time.gmtime())
    with conn() as c:
        c.execute("DELETE FROM rate WHERE day < ?", (day,))
        row = c.execute("SELECT count FROM rate WHERE user_id=? AND day=?", (uid, day)).fetchone()
        if row and row["count"] >= limit:
            return False
        c.execute(
            "INSERT INTO rate(user_id, day, count) VALUES(?,?,1) "
            "ON CONFLICT(user_id, day) DO UPDATE SET count=count+1", (uid, day))
    return True


# ---------- tracking ----------
def log_search(uid, query, input_type, category, budget) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO searches(user_id, query, input_type, category, budget, created_at) VALUES(?,?,?,?,?,?)",
            (uid, query, input_type, category, budget, time.time()))
        return cur.lastrowid


def log_shown(search_id: int, offers: list):
    with conn() as c:
        c.executemany(
            "INSERT INTO shown_products(search_id, product_key, name, brand, store, price, rating, url, rank) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            [(search_id, o["key"], o["name"], o.get("brand"), o["store"], o.get("price"),
              o.get("rating"), o["url"], o.get("rank", 0)) for o in offers])


def log_interaction(uid: str, search_id: int, product_key: str, action: str) -> bool:
    with conn() as c:
        row = c.execute(
            "SELECT sp.*, s.category FROM shown_products sp JOIN searches s ON s.id = sp.search_id "
            "WHERE sp.search_id=? AND sp.product_key=? AND s.user_id=? LIMIT 1",
            (search_id, product_key, uid)).fetchone()
        if not row:
            return False
        c.execute(
            "INSERT INTO interactions(user_id, search_id, product_key, name, brand, category, store, price, url, action, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (uid, search_id, product_key, row["name"], row["brand"], row["category"], row["store"],
             row["price"], row["url"], action, time.time()))
    return True


def get_shown_url(uid: str, search_id: int, product_key: str):
    with conn() as c:
        row = c.execute(
            "SELECT sp.url FROM shown_products sp JOIN searches s ON s.id = sp.search_id "
            "WHERE sp.search_id=? AND sp.product_key=? AND s.user_id=? LIMIT 1",
            (search_id, product_key, uid)).fetchone()
    return row["url"] if row else None


# ---------- short-lived result pages (so a page refresh never re-runs a paid search) ----------
RESULT_TTL = 6 * 3600


def save_result(uid: str, payload: dict) -> str:
    rid = secrets.token_urlsafe(16)
    with conn() as c:
        c.execute("DELETE FROM results WHERE created_at < ?", (time.time() - RESULT_TTL,))
        c.execute("INSERT INTO results(id, user_id, payload, created_at) VALUES(?,?,?,?)",
                  (rid, uid, json.dumps(payload), time.time()))
    return rid


def get_result(uid: str, rid: str):
    with conn() as c:
        row = c.execute("SELECT payload FROM results WHERE id=? AND user_id=? AND created_at > ?",
                        (rid, uid, time.time() - RESULT_TTL)).fetchone()
    return json.loads(row["payload"]) if row else None


def fetch_searches(uid: str, limit: int = 200) -> list:
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM searches WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (uid, limit))]


def fetch_interactions(uid: str, limit: int = 500) -> list:
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM interactions WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (uid, limit))]


def fetch_saved(uid: str, limit: int = 5) -> list:
    with conn() as c:
        return [dict(r) for r in c.execute(
            """SELECT i.search_id, i.product_key, i.name, i.store, i.price, i.url, MAX(i.created_at) AS ts
               FROM interactions i
               WHERE i.user_id=? AND i.action='save'
                 AND NOT EXISTS (SELECT 1 FROM interactions d WHERE d.user_id=i.user_id
                                 AND d.product_key=i.product_key AND d.action='dismiss'
                                 AND d.created_at > i.created_at)
               GROUP BY i.product_key ORDER BY ts DESC LIMIT ?""", (uid, limit))]


# ---------- explicit facts ----------
def set_fact(uid: str, key: str, value: str):
    with conn() as c:
        c.execute(
            "INSERT INTO facts(user_id, key, value, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (uid, key, value, time.time()))


def delete_fact(uid: str, key: str):
    with conn() as c:
        c.execute("DELETE FROM facts WHERE user_id=? AND key=?", (uid, key))


def get_facts(uid: str) -> dict:
    with conn() as c:
        return {r["key"]: r["value"] for r in c.execute(
            "SELECT key, value FROM facts WHERE user_id=?", (uid,))}


# ---------- cache ----------
def cache_get(key: str, ttl: int):
    with conn() as c:
        row = c.execute("SELECT value, created_at FROM cache WHERE key=?", (key,)).fetchone()
    if not row or time.time() - row["created_at"] > ttl:
        return None
    return json.loads(row["value"])


def cache_set(key: str, value):
    with conn() as c:
        c.execute("DELETE FROM cache WHERE created_at < ?", (time.time() - 7 * 86400,))
        c.execute("INSERT OR REPLACE INTO cache(key, value, created_at) VALUES(?,?,?)",
                  (key, json.dumps(value), time.time()))


# ---------- privacy: export & delete ----------
def export_user_data(uid: str) -> dict:
    with conn() as c:
        q = lambda sql: [dict(r) for r in c.execute(sql, (uid,))]
        return {
            "account": q("SELECT id, email, name, created_at, consent FROM users WHERE id=?"),
            "searches": q("SELECT * FROM searches WHERE user_id=?"),
            "shown_products": q("SELECT sp.* FROM shown_products sp JOIN searches s ON s.id=sp.search_id WHERE s.user_id=?"),
            "interactions": q("SELECT * FROM interactions WHERE user_id=?"),
            "facts": q("SELECT key, value FROM facts WHERE user_id=?"),
        }


def delete_user_data(uid: str):
    """Wipe searches, interactions, preferences and AI suggestions. Keeps the account."""
    with conn() as c:
        c.execute("DELETE FROM shown_products WHERE search_id IN (SELECT id FROM searches WHERE user_id=?)", (uid,))
        for table in ("interactions", "searches", "facts", "results"):
            c.execute(f"DELETE FROM {table} WHERE user_id=?", (uid,))
        c.execute("DELETE FROM cache WHERE key LIKE ?", (f"ideas:{uid}:%",))


def delete_account(uid: str):
    delete_user_data(uid)
    with conn() as c:
        for table in ("sessions", "rate"):
            c.execute(f"DELETE FROM {table} WHERE user_id=?", (uid,))
        c.execute("DELETE FROM users WHERE id=?", (uid,))
