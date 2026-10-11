"""Accounts: password hashing (scrypt, stdlib), server-side sessions, login throttling."""
import hashlib
import hmac
import re
import secrets

from . import config, db

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_N, _R, _P = 2 ** 15, 8, 1


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, maxmem=64 * 1024 * 1024)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                              maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(calc, bytes.fromhex(h))
    except (ValueError, TypeError):
        return False


_DUMMY = hash_password("not-a-real-password")  # makes unknown-email logins take as long as real ones


def clean_email(email: str):
    e = (email or "").strip().lower()
    return e if len(e) <= 254 and EMAIL_RE.match(e) else None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _start_session(uid: str) -> str:
    token = secrets.token_urlsafe(32)
    db.create_session(uid, _token_hash(token), config.SESSION_DAYS * 86400)
    return token  # only the hash is stored; the raw token lives in the browser cookie


def public(user: dict) -> dict:
    return {"email": user["email"], "name": user.get("name") or "", "consent": bool(user["consent"])}


def register(email: str, password: str, name: str, consent: bool):
    e = clean_email(email)
    if not e:
        raise AuthError(400, "Enter a valid email address.")
    if not (8 <= len(password or "") <= 128):
        raise AuthError(400, "Password must be 8 to 128 characters.")
    if (password or "").lower() == e:
        raise AuthError(400, "Password must not be your email address.")
    try:
        uid = db.create_user(e, (name or "").strip()[:60], hash_password(password), consent)
    except db.EmailTaken:
        raise AuthError(409, "An account with this email already exists. Log in instead.")
    return {"email": e, "name": (name or "").strip()[:60], "consent": consent}, _start_session(uid)


def _authenticate(email: str, password: str) -> dict:
    e = clean_email(email) or "invalid"
    if db.recent_failures(e, config.LOGIN_LOCK_MINUTES * 60) >= config.MAX_LOGIN_FAILURES:
        raise AuthError(429, f"Too many failed attempts. Try again in {config.LOGIN_LOCK_MINUTES} minutes.")
    user = db.get_user_by_email(e)
    ok = verify_password(password or "", user["password_hash"] if user and user["password_hash"] else _DUMMY)
    if not (user and user["password_hash"] and ok):
        db.record_failure(e)
        raise AuthError(401, "Incorrect email or password.")
    db.clear_failures(e)
    return user


def login(email: str, password: str):
    user = _authenticate(email, password)
    return public(user), _start_session(user["id"])


def confirm_password(email: str, password: str) -> None:
    """Re-check the password before a destructive action (shares the failed-attempt lockout)."""
    _authenticate(email, password)


def user_from_token(token: str):
    return db.user_for_session(_token_hash(token)) if token else None


def end_session(token: str):
    if token:
        db.delete_session(_token_hash(token))
