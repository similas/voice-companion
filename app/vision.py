"""
Camera handling and the decision of WHEN to look.

Sending an image to a VLM is expensive: Gemma 3 has to run its vision encoder over
the frame before it can emit a single token, which on this board roughly triples
time-to-first-token. So the camera is used only when the transcript implies it.
"""

import re
import threading
import time
from typing import Optional, Tuple

import cv2

from loguru import logger


class Camera:
    """
    Keeps the webcam open and continuously drains it in a background thread.

    Why drain? V4L2 hands you the oldest frame in its buffer. If we opened the
    camera on demand, or let it sit idle between turns, "what do you see" would
    return a picture from several seconds ago — which is worse than useless,
    because it looks like it worked. Draining at a low rate keeps the newest frame
    genuinely current while costing a few percent of one core.
    """

    def __init__(self, device: int, width: int, height: int, poll_fps: float = 5.0):
        self.device = device
        self.width = width
        self.height = height
        self.interval = 1.0 / max(0.5, poll_fps)

        self._frame = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        # CAP_V4L2 is required, not optional: with the default backend the first
        # read() on this hardware returns False and the camera never produces a
        # frame. (Verified on the EMEET C960.)
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            logger.error(f"camera: cannot open /dev/video{self.device}")
            return False
        # MJPEG matters: in raw YUYV, USB bandwidth caps this camera at a few fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # NOTE: do NOT set CAP_PROP_BUFFERSIZE=1 — on this v4l2 backend it halves
        # the achievable frame rate. Freshness is handled by draining instead.

        ok, frame = cap.read()
        if not ok:
            logger.error("camera: opened but first read failed")
            cap.release()
            return False

        self._cap = cap
        with self._lock:
            self._frame = frame
            self._stamp = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        h, w = frame.shape[:2]
        logger.info(f"camera: /dev/video{self.device} {w}x{h} ready")
        return True

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._stamp = time.time()
            else:
                time.sleep(0.1)
            time.sleep(self.interval)

    def grab_jpeg(self, max_edge: int = 896, quality: int = 85
                  ) -> Optional[Tuple[bytes, Tuple[int, int], float]]:
        """Newest frame as JPEG bytes. Returns (jpeg, (w,h), age_seconds)."""
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            age = time.time() - self._stamp
        if frame is None:
            return None

        h, w = frame.shape[:2]
        # Downscale before encoding. Gemma 3 resizes to 896x896 internally, so
        # sending more pixels costs upload+encode time and buys nothing.
        longest = max(w, h)
        if longest > max_edge:
            s = max_edge / longest
            frame = cv2.resize(frame, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return buf.tobytes(), (w, h), age

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()


class VisionTrigger:
    """
    Decides whether an utterance is asking the bot to look.

    Two mechanisms, because a phrase list alone proved far too narrow. In a real
    conversation these all needed the camera and none matched the original list:

        "What is the number that I'm showing with my hands?"
        "What is the color of my shirt?"
        "What is written on this marker?"
        "What am I doing right now?"
        "How about now?"

    So on top of the configured phrases:

    1. A generic visual-question test — a question word combined with anything
       about the speaker's body, clothing, surroundings, or an object being held
       or shown.
    2. Sticky follow-ups — after a vision turn, short deictic questions like "how
       about now?" or "and this?" continue to mean "look again". Without this, the
       model answers them from whatever it last saw, which is worse than useless
       because it sounds authoritative.
    """

    # Something being asked ABOUT the visible world.
    _SUBJECT = re.compile(
        r"\b("
        r"see|seeing|look|looking|watch|show|showing|holding|hold|wearing|wear|"
        r"my\s+(hand|hands|finger|fingers|shirt|face|shoes|glasses|hair|clothes|"
        r"desk|room|screen)|"
        r"this|that|these|those|"
        r"i'?m\s+doing|i\s+am\s+doing|i'?m\s+holding|i'?m\s+wearing|"
        # "What am I doing/holding/wearing/showing right now?" — inverted word
        # order, which the i'm-prefixed patterns above miss entirely.
        r"am\s+i\s+(doing|holding|wearing|showing|pointing|touching)|"
        r"written|writing|says|say|read|colour|color|brand|label|logo|"
        r"in\s+front\s+of|on\s+the\s+table|in\s+my\s+hand|camera"
        r")\b", re.I)
    # Phrased as a question / request.
    _ASKING = re.compile(
        r"\b(what|which|who|how\s+many|how\s+much|where|is\s+this|is\s+that|"
        r"can\s+you|do\s+you|describe|tell\s+me|count|identify|any|"
        # Imperatives, not just questions: "List all the things you are seeing",
        # "Name the objects on the table", "Read this label".
        r"list|name|read|show\s+me)\b", re.I)
    # Short deictic follow-ups: "how about now?", "and now?", "this one?"
    _FOLLOWUP = re.compile(
        r"^(and\s+)?(how\s+about\s+)?(now|this|that|this\s+one|that\s+one|again)"
        r"[\s\?\.!]*$", re.I)

    def __init__(self, triggers, enabled: bool = True, smart: bool = True,
                 sticky: bool = True):
        self.enabled = enabled
        self.smart = smart
        self.sticky = sticky
        self.last_used_vision = False
        self.phrases = [t.lower().strip() for t in (triggers or []) if t.strip()]
        # Word-boundary-anchored so "look at" doesn't fire on "outlook attachment".
        self._res = [re.compile(r"\b" + re.escape(p).replace(r"\ ", r"\s+"))
                     for p in self.phrases]

    def wants_vision(self, text: str) -> bool:
        if not self.enabled or not text:
            return False
        t = text.strip()

        # 1. explicit configured phrases
        if any(r.search(t.lower()) for r in self._res):
            return True

        # 2. sticky follow-up to a vision turn
        if self.sticky and self.last_used_vision and self._FOLLOWUP.match(t):
            return True
        if (self.sticky and self.last_used_vision and len(t.split()) <= 4
                and self._SUBJECT.search(t)):
            return True

        # 3. generic "question about the visible world"
        if self.smart and self._ASKING.search(t) and self._SUBJECT.search(t):
            return True

        return False

    def note_result(self, used: bool):
        """Remember whether this turn looked, so follow-ups can inherit it."""
        self.last_used_vision = used
