"""
Govee lamp. Two transports, best first:

  LAN    GOVEE_LAMP_IP set — one UDP datagram straight to the lamp, ~20 ms,
         no cloud, no key. Enable "LAN Control" for the device in the Govee
         Home app and give the lamp a fixed IP in your router.
  cloud  GOVEE_API_KEY + GOVEE_DEVICE + GOVEE_MODEL — the developer REST API.
         Works for models without LAN support; adds internet latency.
"""

import json
import os
import socket

from integrations import api, tool

# Small on purpose: the model passes #rrggbb for anything unusual, names are
# only a convenience for the common asks. Matched by substring, so "warm red"
# lands on red and "warm white" on warm.
_COLORS = {
    "red": (255, 30, 10), "green": (0, 200, 60), "blue": (20, 60, 255),
    "amber": (255, 120, 0), "orange": (255, 90, 0), "yellow": (255, 200, 0),
    "purple": (160, 30, 255), "pink": (255, 60, 120), "cyan": (0, 200, 255),
    "white": (255, 255, 255), "warm": (255, 160, 70),
}


def enabled() -> bool:
    return bool(os.getenv("GOVEE_LAMP_IP")) or all(
        os.getenv(k) for k in ("GOVEE_API_KEY", "GOVEE_DEVICE", "GOVEE_MODEL"))


def _rgb(color: str) -> tuple:
    c = color.strip().lower().lstrip("#")
    if len(c) == 6 and all(ch in "0123456789abcdef" for ch in c):
        return int(c[:2], 16), int(c[2:4], 16), int(c[4:], 16)
    return next((v for k, v in _COLORS.items() if k in c), _COLORS["warm"])


async def _lan(cmd: str, value) -> None:
    msg = json.dumps({"msg": {"cmd": cmd, "data": value}}).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(msg, (os.environ["GOVEE_LAMP_IP"], 4001))


async def _cloud(name: str, value) -> None:
    await api("https://developer-api.govee.com/v1/devices/control", method="PUT",
              headers={"Govee-API-Key": os.environ["GOVEE_API_KEY"]},
              data={"device": os.environ["GOVEE_DEVICE"],
                    "model": os.environ["GOVEE_MODEL"],
                    "cmd": {"name": name, "value": value}})


@tool("lamp", "Control the room lamp: power, color, brightness.",
      {"type": "object", "properties": {
          "state": {"type": "string", "enum": ["on", "off"]},
          "color": {"type": "string", "description": "name or #rrggbb"},
          "brightness": {"type": "integer", "description": "1-100"},
      }})
async def lamp(state: str = None, color: str = None, brightness=None) -> str:
    lan = bool(os.getenv("GOVEE_LAMP_IP"))
    done = []
    if state in ("on", "off"):
        await (_lan("turn", {"value": int(state == "on")}) if lan
               else _cloud("turn", state))
        done.append(state)
    if brightness is not None:
        b = max(1, min(100, int(brightness)))
        await (_lan("brightness", {"value": b}) if lan else _cloud("brightness", b))
        done.append(f"brightness {b}%")
    if color:
        r, g, b = _rgb(str(color))
        # LAN colorwc wants colorTemInKelvin=0 to mean "use the RGB".
        await (_lan("colorwc", {"color": {"r": r, "g": g, "b": b},
                                "colorTemInKelvin": 0}) if lan
               else _cloud("color", {"r": r, "g": g, "b": b}))
        done.append(str(color))
    return f"lamp: {', '.join(done)}" if done else "lamp: nothing to change"


TOOLS = [lamp]
