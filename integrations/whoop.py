"""
WHOOP — recovery and sleep, read-only.

AUTH: WHOOP rotates refresh tokens — every refresh invalidates the token that
was used and issues a new one. A static value in .env therefore works exactly
once; after that the CURRENT token lives in .whoop_tokens.json (gitignored),
rewritten on every refresh. WHOOP_REFRESH_TOKEN is only the seed for first run.
"""

import json
import os
import time

from integrations import ROOT, api, tool
from integrations import govee

_API = "https://api.prod.whoop.com"
_STATE = ROOT / ".whoop_tokens.json"
_cache = {"access": None, "until": 0.0}


def enabled() -> bool:
    return all(os.getenv(k) for k in
               ("WHOOP_CLIENT_ID", "WHOOP_CLIENT_SECRET", "WHOOP_REFRESH_TOKEN"))


async def _token() -> str:
    if _cache["access"] and time.time() < _cache["until"]:
        return _cache["access"]
    refresh = (json.loads(_STATE.read_text())["refresh_token"]
               if _STATE.exists() else os.environ["WHOOP_REFRESH_TOKEN"])
    t = await api(f"{_API}/oauth/oauth2/token", method="POST", form={
        "grant_type": "refresh_token", "refresh_token": refresh,
        "client_id": os.environ["WHOOP_CLIENT_ID"],
        "client_secret": os.environ["WHOOP_CLIENT_SECRET"],
        "scope": "offline",
    })
    # Persist the rotated token BEFORE using the access token: if we crash in
    # between, the old refresh token is already dead and this file is the only
    # copy of the new one.
    _STATE.write_text(json.dumps({"refresh_token": t["refresh_token"]}))
    _cache.update(access=t["access_token"],
                  until=time.time() + t.get("expires_in", 3600) - 60)
    return _cache["access"]


async def _latest(path: str) -> dict:
    """Newest record, whole — callers pick score/times out of it."""
    r = await api(f"{_API}{path}",
                  headers={"Authorization": f"Bearer {await _token()}"})
    recs = r.get("records") or []
    return recs[0] if recs else {}


async def _score(path: str) -> dict:
    return (await _latest(path)).get("score") or {}


@tool("whoop_recovery", "Today's WHOOP recovery: score, HRV, resting heart rate.")
async def recovery() -> str:
    s = await _score("/developer/v2/recovery?limit=1")
    if not s:
        return "no recovery data yet today"
    return (f"recovery {s['recovery_score']:.0f}%, "
            f"HRV {s['hrv_rmssd_milli']:.0f} ms, "
            f"resting heart rate {s['resting_heart_rate']:.0f} bpm")


@tool("whoop_sleep", "Last night's WHOOP sleep: hours slept and quality.")
async def sleep() -> str:
    s = await _score("/developer/v2/activity/sleep?limit=1")
    if not s:
        return "no sleep data yet"
    st = s.get("stage_summary") or {}
    hours = (st.get("total_light_sleep_time_milli", 0)
             + st.get("total_slow_wave_sleep_time_milli", 0)
             + st.get("total_rem_sleep_time_milli", 0)) / 3.6e6
    return (f"slept {hours:.1f} hours, "
            f"sleep performance {s.get('sleep_performance_percentage', 0):.0f}%")


@tool("whoop_strain", "Today's WHOOP strain, calories burned, heart rate.")
async def strain() -> str:
    s = await _score("/developer/v2/cycle?limit=1")
    if not s:
        return "no strain data yet today"
    return (f"day strain {s.get('strain', 0):.1f} of 21, "
            f"{s.get('kilojoule', 0) / 4.184:.0f} calories, "
            f"average heart rate {s.get('average_heart_rate', 0):.0f}, "
            f"max {s.get('max_heart_rate', 0):.0f} bpm")


@tool("whoop_workout", "The most recent workout: sport, duration, strain.")
async def workout() -> str:
    rec = await _latest("/developer/v2/activity/workout?limit=1")
    if not rec:
        return "no workouts recorded"
    s = rec.get("score") or {}
    mins = ""
    try:
        from datetime import datetime
        t0 = datetime.fromisoformat(rec["start"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(rec["end"].replace("Z", "+00:00"))
        mins = f"{(t1 - t0).total_seconds() / 60:.0f} minutes, "
    except Exception:
        pass                                    # duration is nice-to-have
    sport = str(rec.get("sport_name") or "workout").replace("_", " ")
    return (f"last workout: {sport}, {mins}"
            f"strain {s.get('strain', 0):.1f}, "
            f"{s.get('kilojoule', 0) / 4.184:.0f} calories")


# WHOOP's own zones: green >= 67, yellow 34-66, red < 34. Deterministic here,
# not delegated to the model — a wrong color on a health signal reads as broken.
_ZONES = [(67, "green"), (34, "amber"), (0, "red")]


@tool("recovery_light", "Show today's WHOOP recovery on the lamp "
                        "(green good, amber medium, red poor).")
async def recovery_light() -> str:
    s = await _score("/developer/v2/recovery?limit=1")
    if not s:
        return "no recovery data to show"
    score = s["recovery_score"]
    color = next(c for lo, c in _ZONES if score >= lo)
    await govee.lamp.run(state="on", color=color, brightness=45)
    return f"lamp is {color} — recovery {score:.0f}%"


# recovery_light is appended by integrations.load() only when govee is also
# enabled; a lamp tool that cannot reach a lamp should not exist.
TOOLS = [recovery, sleep, strain, workout]
