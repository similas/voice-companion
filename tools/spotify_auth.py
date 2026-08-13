#!/usr/bin/env python3
"""
One-time Spotify authorization — gets SPOTIFY_REFRESH_TOKEN into .env.

    python3 tools/spotify_auth.py

Prerequisites, in the app at developer.spotify.com/dashboard:
  - Redirect URI registered as EXACTLY:  http://127.0.0.1:8766/callback
    (Spotify no longer accepts http://localhost — loopback must be the
    literal IP. Must match this script character for character.)
  - SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET already in .env.
  - The account is Premium — playback control does not work without it.

Prints an authorization URL, catches the redirect (VS Code forwards the
port; or paste the code=... value into the terminal), exchanges the code,
writes the refresh token into .env, and verifies by naming the account and
its available playback devices. Spotify refresh tokens are long-lived and
NOT rotated, so unlike WHOOP there is no state file to manage.
"""

import base64
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

PORT = 8766
REDIRECT = f"http://127.0.0.1:{PORT}/callback"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
# modify = play/pause/skip; read = now-playing + devices; playlist-read lets
# "play my chill playlist" find private playlists too.
SCOPE = ("user-modify-playback-state user-read-playback-state "
         "user-read-currently-playing playlist-read-private")

load_env()
CID = os.getenv("SPOTIFY_CLIENT_ID")
SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
if not (CID and SECRET):
    sys.exit("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET missing from .env — "
             "create the app at developer.spotify.com/dashboard first")

state = secrets.token_urlsafe(12)
got = {}
ready = threading.Event()


class Callback(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        ok = q.get("state") == [state] and "code" in q
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Roomi: Spotify authorized - you can close this tab.\n"
                         if ok else b"state mismatch or no code\n")
        if ok:
            got["code"] = q["code"][0]
            ready.set()

    def log_message(self, *_):
        pass


def stdin_fallback():
    for line in sys.stdin:
        line = line.strip()
        if line:
            got["code"] = line.split("code=")[-1].split("&")[0]
            ready.set()
            return


url = AUTH_URL + "?" + urllib.parse.urlencode({
    "client_id": CID, "redirect_uri": REDIRECT, "response_type": "code",
    "scope": SCOPE, "state": state})
print(f"\n1. Open this in your browser and approve:\n\n{url}\n")
print(f"2. Waiting for the redirect on {REDIRECT} ...")
print("   (or paste the code=... value here if that page cannot connect)\n")

server = http.server.HTTPServer(("127.0.0.1", PORT), Callback)
threading.Thread(target=server.serve_forever, daemon=True).start()
threading.Thread(target=stdin_fallback, daemon=True).start()
ready.wait()
server.shutdown()

basic = base64.b64encode(f"{CID}:{SECRET}".encode()).decode()
body = urllib.parse.urlencode({
    "grant_type": "authorization_code", "code": got["code"],
    "redirect_uri": REDIRECT,
}).encode()
try:
    req = urllib.request.Request(
        TOKEN_URL, data=body,
        headers={"Authorization": f"Basic {basic}",
                 "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req) as r:
        tokens = json.load(r)
except urllib.error.HTTPError as e:
    sys.exit(f"token exchange failed: HTTP {e.code}: "
             f"{e.read().decode(errors='replace')[:300]}")
refresh = tokens["refresh_token"]

env_path = ROOT / ".env"
lines, done = [], False
for ln in env_path.read_text().splitlines():
    if ln.lstrip("# ").startswith("SPOTIFY_REFRESH_TOKEN="):
        ln, done = f"SPOTIFY_REFRESH_TOKEN={refresh}", True
    lines.append(ln)
if not done:
    lines.append(f"SPOTIFY_REFRESH_TOKEN={refresh}")
env_path.write_text("\n".join(lines) + "\n")
print("3. Refresh token written to .env")

# ---- prove the whole chain works -------------------------------------------
def get(path):
    req = urllib.request.Request(
        f"https://api.spotify.com/v1{path}",
        headers={"Authorization": f"Bearer {tokens['access_token']}"})
    with urllib.request.urlopen(req) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else {}

me = get("/me")
devs = (get("/me/player/devices")).get("devices") or []
print(f"4. Verified: account {me.get('display_name')!r} "
      f"({me.get('product', 'unknown')} plan)")
if devs:
    for d in devs:
        print(f"   device: {d['name']}{'  [active]' if d.get('is_active') else ''}")
    print("   Tip: set SPOTIFY_DEVICE in .env to one of these names so Roomi")
    print("   can start music when nothing is playing.")
else:
    print("   no playback devices right now — open Spotify on your phone or")
    print("   computer once; Roomi targets whatever device Spotify knows.")
if me.get("product") != "premium":
    print("   WARNING: playback control requires Premium — commands will 403.")
