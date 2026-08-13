"""
Spotify playback control via the Web API (Premium required).

Commands land on whatever Spotify Connect device is active. When nothing is
active (the 404 NO_ACTIVE_DEVICE case), the transport retries against the
device named by SPOTIFY_DEVICE — e.g. a librespot/spotifyd instance running on
this Jetson — or the first device Spotify knows about.

Unlike WHOOP, Spotify refresh tokens are long-lived and not rotated, so the
one in .env keeps working.
"""

import base64
import os
import time
import urllib.error
import urllib.parse

from integrations import api, tool

_cache = {"access": None, "until": 0.0}


def enabled() -> bool:
    return all(os.getenv(k) for k in
               ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET",
                "SPOTIFY_REFRESH_TOKEN"))


async def _token() -> str:
    if _cache["access"] and time.time() < _cache["until"]:
        return _cache["access"]
    basic = base64.b64encode(
        f"{os.environ['SPOTIFY_CLIENT_ID']}:"
        f"{os.environ['SPOTIFY_CLIENT_SECRET']}".encode()).decode()
    t = await api("https://accounts.spotify.com/api/token", method="POST",
                  headers={"Authorization": f"Basic {basic}"},
                  form={"grant_type": "refresh_token",
                        "refresh_token": os.environ["SPOTIFY_REFRESH_TOKEN"]})
    _cache.update(access=t["access_token"],
                  until=time.time() + t.get("expires_in", 3600) - 60)
    return _cache["access"]


async def _sp(method: str, path: str, data: dict = None):
    return await api(f"https://api.spotify.com/v1{path}", method=method,
                     headers={"Authorization": f"Bearer {await _token()}"},
                     data=data)


async def _device_id() -> str:
    devs = (await _sp("GET", "/me/player/devices")).get("devices") or []
    want = (os.getenv("SPOTIFY_DEVICE") or "").lower()
    d = (next((d for d in devs if want and want in d["name"].lower()), None)
         or (devs[0] if devs else None))
    if not d:
        raise RuntimeError("no Spotify device is available")
    return d["id"]


async def _transport(method: str, path: str, data: dict = None):
    """Player command, retried once against a named device if none is active."""
    try:
        await _sp(method, path, data)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        await _sp(method, f"{path}?device_id={await _device_id()}", data)


@tool("music", "Control Spotify: play something by name, pause, resume, "
               "skip songs, or identify the current song.",
      {"type": "object", "properties": {
          # The enum descriptions map spoken phrasings onto actions — "skip"
          # and "what song is this" did not land without them (measured).
          "action": {"type": "string", "enum": [
              "play", "pause", "resume", "next", "previous", "now_playing"],
              "description": "next = skip this song; resume = continue the "
                             "music; now_playing = what song is this"},
          "query": {"type": "string",
                    "description": "what to play; only with action=play"},
          "kind": {"type": "string",
                   "enum": ["track", "playlist", "album", "artist"],
                   "description": "what the query names; default track"},
      }, "required": ["action"]})
async def music(action: str, query: str = None, kind: str = "track") -> str:
    if action == "now_playing":
        item = (await _sp("GET", "/me/player/currently-playing") or {}).get("item")
        if not item:
            return "nothing is playing"
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        return f"playing {item['name']}" + (f" by {artists}" if artists else "")

    if action == "play" and query:
        kind = kind if kind in ("track", "playlist", "album", "artist") else "track"
        q = urllib.parse.urlencode({"q": query, "type": kind, "limit": 1})
        # Spotify pads search results with nulls sometimes — filter them.
        items = [i for i in ((await _sp("GET", f"/search?{q}"))
                             .get(f"{kind}s", {}).get("items") or []) if i]
        if not items:
            return f"found nothing on Spotify for {query}"
        it = items[0]
        body = ({"uris": [it["uri"]]} if kind == "track"
                else {"context_uri": it["uri"]})
        await _transport("PUT", "/me/player/play", body)
        return f"playing {it['name']}"

    verbs = {"pause": ("PUT", "/me/player/pause"),
             "resume": ("PUT", "/me/player/play"),
             "play": ("PUT", "/me/player/play"),      # bare "play" = resume
             "next": ("POST", "/me/player/next"),
             "previous": ("POST", "/me/player/previous")}
    if action not in verbs:
        return f"unknown action {action}"
    await _transport(*verbs[action])
    return f"music: {action}"


TOOLS = [music]
