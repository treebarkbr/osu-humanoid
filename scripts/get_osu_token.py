#!/usr/bin/env python3
"""Interactive osu!api v2 OAuth2 authorization (authorization-code grant).

Prereqs: register an OAuth app at
  https://osu.ppy.sh/home/account/edit#oauth
and set its "Application Callback URL" to exactly the redirect used here
(default ``http://localhost:8765/callback``).

Then put in a gitignored ``.env``:
  OSU_CLIENT_ID=12345
  OSU_CLIENT_SECRET=abcdef...
  OSU_REDIRECT_URI=http://localhost:8765/callback   # optional

Usage:
  python scripts/get_osu_token.py            # opens browser, waits for callback
  python scripts/get_osu_token.py --print-url
  python scripts/get_osu_token.py --code <code>   # headless: paste code manually
"""
from __future__ import annotations

import argparse
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import osu_api  # noqa: E402

OK_PAGE = b"<html><body><h2>osu-humanoid: authorization complete.</h2>You can close this tab.</body></html>"
FAIL_PAGE = b"<html><body><h2>Authorization failed.</h2>Check the terminal.</body></html>"


def _serve_once(host: str, port: int, path: str, state: str, timeout: float = 300.0):
    result: dict = {"code": None, "error": None, "state": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_GET(self):
            q = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(q.query)
            result["state"] = params.get("state", [None])[0]
            if "code" in params:
                result["code"] = params["code"][0]
                body = OK_PAGE
            else:
                result["error"] = params.get("error_description", params.get("error", ["unknown"]))[0]
                body = FAIL_PAGE
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    httpd = HTTPServer((host, port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    t.join(timeout)
    httpd.server_close()
    if result["code"] and result["state"] != state:
        raise SystemExit("State mismatch — possible CSRF, aborting.")
    return result["code"], result["error"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-url", action="store_true", help="just print the authorize URL")
    ap.add_argument("--code", help="exchange a code you obtained manually")
    args = ap.parse_args()

    if args.code:
        osu_api.save_token(osu_api.exchange_code(args.code))
        print(f"token saved -> {osu_api.TOKEN_PATH}")
        return 0

    state = secrets.token_urlsafe(16)
    url = osu_api.authorize_url(state)

    if args.print_url:
        print(url)
        return 0

    parsed = urllib.parse.urlparse(osu_api.redirect_uri())
    host = parsed.hostname or "localhost"
    port = parsed.port or 8765

    print("Open this URL and authorize the app:\n")
    print("  " + url + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print(f"Waiting for callback on {host}:{port} ...")

    code, error = _serve_once(host, port, parsed.path or "/callback", state)
    if error or not code:
        print(f"Authorization failed: {error}", file=sys.stderr)
        return 1

    osu_api.save_token(osu_api.exchange_code(code))
    print(f"Success. Token saved -> {osu_api.TOKEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
