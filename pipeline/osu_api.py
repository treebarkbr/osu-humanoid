"""osu!api v2 OAuth2 (authorization-code grant) client + helpers.

Credentials live in a gitignored ``.env`` (``OSU_CLIENT_ID``,
``OSU_CLIENT_SECRET``, optional ``OSU_REDIRECT_URI``). Tokens are cached in a
gitignored ``.osu_token.json`` and refreshed automatically.

Docs: https://osu.ppy.sh/docs/index.html
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
TOKEN_PATH = BASE_DIR / ".osu_token.json"

AUTHORIZE_URL = "https://osu.ppy.sh/oauth/authorize"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"
API_BASE = "https://osu.ppy.sh/api/v2"

DEFAULT_REDIRECT = "http://localhost:8765/callback"
DEFAULT_SCOPE = "public identify"
UA = "osu-humanoid/0.1 (research project; local ML pipeline) by nocone"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def client_credentials() -> tuple[str, str]:
    load_env()
    cid = os.environ.get("OSU_CLIENT_ID", "").strip()
    secret = os.environ.get("OSU_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise SystemExit(
            "Missing OSU_CLIENT_ID / OSU_CLIENT_SECRET.\n"
            "Register an OAuth app at https://osu.ppy.sh/home/account/edit#oauth\n"
            f"with callback URL exactly: {redirect_uri()}\n"
            "then put the id/secret in " + str(ENV_PATH)
        )
    return cid, secret


def redirect_uri() -> str:
    load_env()
    return os.environ.get("OSU_REDIRECT_URI", DEFAULT_REDIRECT).strip()


def scope() -> str:
    load_env()
    return os.environ.get("OSU_SCOPE", DEFAULT_SCOPE).strip()


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------
def authorize_url(state: str) -> str:
    cid, _ = client_credentials()
    params = {
        "client_id": cid,
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": scope(),
        "state": state,
    }
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)


def exchange_code(code: str) -> dict:
    cid, secret = client_credentials()
    r = SESSION.post(
        TOKEN_URL,
        data={
            "client_id": cid,
            "client_secret": secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri(),
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _refresh(refresh_token: str) -> dict:
    cid, secret = client_credentials()
    r = SESSION.post(
        TOKEN_URL,
        data={
            "client_id": cid,
            "client_secret": secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": scope(),
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# token cache
# ---------------------------------------------------------------------------
def save_token(data: dict) -> None:
    data = dict(data)
    data["expires_at"] = time.time() + float(data.get("expires_in", 0))
    TOKEN_PATH.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(TOKEN_PATH, 0o600)
    except OSError:
        pass


def load_token() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        return json.loads(TOKEN_PATH.read_text())
    except (ValueError, OSError):
        return None


def get_valid_token() -> str:
    tok = load_token()
    if tok is None:
        raise SystemExit("No cached token. Run scripts/get_osu_token.py first.")
    if tok.get("expires_at", 0) - time.time() > 60:
        return tok["access_token"]
    rt = tok.get("refresh_token")
    if not rt:
        raise SystemExit("Token expired and no refresh_token. Re-run get_osu_token.py.")
    fresh = _refresh(rt)
    if "refresh_token" not in fresh:
        fresh["refresh_token"] = rt
    save_token(fresh)
    return fresh["access_token"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def api_get(path: str, params: dict | None = None, *, raw: bool = False, retries: int = 4):
    """GET an API v2 path (e.g. ``/beatmaps/1/scores``). Auto-refresh / backoff."""
    url = API_BASE + path if path.startswith("/") else path
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        token = get_valid_token()
        try:
            r = SESSION.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=60,
            )
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1.5 * (attempt + 1))
            continue

        if r.status_code == 401 and attempt == 0:
            tok = load_token() or {}
            if tok.get("refresh_token"):
                save_token(_refresh(tok["refresh_token"]) | {"refresh_token": tok["refresh_token"]})
                continue
        if r.status_code == 429:
            time.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if r.status_code >= 500:
            last_exc = requests.HTTPError(f"{r.status_code} {r.reason}")
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.content if raw else r.json()
    if last_exc:
        raise last_exc
    raise requests.HTTPError(f"api_get failed: {url}")
