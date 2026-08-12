#!/usr/bin/env python3
"""
One-time WHOOP authorization — gets WHOOP_REFRESH_TOKEN into .env.

    python3 tools/whoop_auth.py

Prints an authorization URL, waits for WHOOP's redirect on localhost:8765
(register http://localhost:8765/callback as the app's redirect URI), exchanges
the code, writes the refresh token into .env, and verifies it by fetching
today's recovery.

Run it in the VS Code terminal and the redirect just works: the browser lands
on YOUR machine's localhost:8765 and VS Code forwards the port to this script.
If the redirect page cannot connect (no forwarding), paste the code=... value
from the browser's address bar into this terminal instead — same result.
"""

import http.server
import json
import os
import secrets
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from integrations import load_env  # noqa: E402

PORT = 8765
REDIRECT = f"http://localhost:{PORT}/callback"
AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
# offline = get a refresh token; cycles = day strain; workout = workouts
SCOPE = "read:recovery read:sleep read:cycles read:workout offline"

load_env()
CID = os.getenv("WHOOP_CLIENT_ID")
SECRET = os.getenv("WHOOP_CLIENT_SECRET")
if not (CID and SECRET):
    sys.exit("WHOOP_CLIENT_ID / WHOOP_CLIENT_SECRET missing from .env")

state = secrets.token_urlsafe(12)             # WHOOP requires state, 8+ chars
got = {}
ready = threading.Event()


class Callback(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        ok = q.get("state") == [state] and "code" in q
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Roomi: authorized - you can close this tab.\n"
                         if ok else b"state mismatch or no code\n")
        if ok:
            got["code"] = q["code"][0]
            ready.set()

    def log_message(self, *_):                # keep the terminal readable
        pass


def stdin_fallback():
    """Accept a pasted code (or the whole redirect URL) if forwarding fails."""
    for line in sys.stdin:
        line = line.strip()
        if line:
            got["code"] = line.split("code=")[-1].split("&")[0]
            ready.set()
            return


url = AUTH_URL + "?" + urllib.parse.urlencode({
    "client_id": CID, "redirect_uri": REDIRECT,
    "response_type": "code", "scope": SCOPE, "state": state})
print(f"\n1. Open this in your browser and approve:\n\n{url}\n")
print(f"2. Waiting for the redirect on {REDIRECT} ...")
print("   (or paste the code=... value here if that page cannot connect)\n")

server = http.server.HTTPServer(("127.0.0.1", PORT), Callback)
threading.Thread(target=server.serve_forever, daemon=True).start()
threading.Thread(target=stdin_fallback, daemon=True).start()
ready.wait()
server.shutdown()

# ---- exchange the (single-use, short-lived) code ---------------------------
# The User-Agent is load-bearing: WHOOP is behind Cloudflare, which 403s
# Python's default UA ("error code: 1010") before the request reaches WHOOP.
UA = {"User-Agent": "roomi/0.4 (Jetson Orin Nano)"}
body = urllib.parse.urlencode({
    "grant_type": "authorization_code", "code": got["code"],
    "client_id": CID, "client_secret": SECRET, "redirect_uri": REDIRECT,
}).encode()
try:
    req = urllib.request.Request(TOKEN_URL, data=body, headers=UA)
    with urllib.request.urlopen(req) as r:
        tokens = json.load(r)
except urllib.error.HTTPError as e:
    # Show WHOOP's actual complaint (invalid_grant = code expired/used, etc.)
    # instead of a bare traceback.
    sys.exit(f"token exchange failed: HTTP {e.code}: "
             f"{e.read().decode(errors='replace')[:300]}")
refresh = tokens["refresh_token"]

# ---- write .env, replacing the placeholder or stale line -------------------
env_path = ROOT / ".env"
lines, done = [], False
for ln in env_path.read_text().splitlines():
    if ln.lstrip("# ").startswith("WHOOP_REFRESH_TOKEN="):
        ln, done = f"WHOOP_REFRESH_TOKEN={refresh}", True
    lines.append(ln)
if not done:
    lines.append(f"WHOOP_REFRESH_TOKEN={refresh}")
env_path.write_text("\n".join(lines) + "\n")

# A previous seed leaves rotated state behind; with a NEW authorization that
# state is stale and would shadow the fresh token. Start clean.
(ROOT / ".whoop_tokens.json").unlink(missing_ok=True)
print("3. Refresh token written to .env (.whoop_tokens.json cleared)")

# ---- prove the whole chain works before the voice loop relies on it --------
req = urllib.request.Request(
    "https://api.prod.whoop.com/developer/v2/recovery?limit=1",
    headers={"Authorization": f"Bearer {tokens['access_token']}", **UA})
with urllib.request.urlopen(req) as r:
    recs = json.load(r).get("records") or []
score = (recs[0].get("score") or {}) if recs else {}
if score:
    print(f"4. Verified: recovery {score.get('recovery_score', 0):.0f}%, "
          f"HRV {score.get('hrv_rmssd_milli', 0):.0f} ms — WHOOP is live.")
else:
    print("4. Token works, but no recovery record yet today — that's fine.")
