#!/usr/bin/env python3
"""
Synthesize Roomi's wake chime — the sound played the moment "hey Roomi" lands.

Design goals: SHORT (you will hear it dozens of times a day, anything long gets
annoying by Tuesday), soft attack (it interrupts silence, not a rock concert),
rising (a question mark, "I'm listening", not a completion sound), and clean on
a small mono speaker at 70% volume.

The shape: two pure tones a perfect fifth apart (A5 -> E6), each a soft pluck
with exponential decay, the second starting 90 ms after the first, plus a faint
octave shimmer on the tail. A fifth is the most consonant non-octave interval —
it reads as friendly-neutral rather than alarm-like (minor) or saccharine
(major third). Total 420 ms.

Regenerate with:  python tools/make_chime.py     (writes assets/roomi_chime.wav)
"""

import wave
from pathlib import Path

import numpy as np

SR = 48000          # the sink's native rate — no resampling on playback
OUT = Path(__file__).resolve().parent.parent / "assets" / "roomi_chime.wav"


def pluck(freq, start, dur, amp, sr=SR, total=0.42):
    """A sine with a 5 ms attack and exponential release, placed at `start`."""
    n = int(total * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * freq * t) * amp
    # very slight detuned partner gives it width without chorus-mush
    tone += np.sin(2 * np.pi * freq * 1.003 * t) * amp * 0.35
    env = np.zeros(n)
    i0, i1 = int(start * sr), min(n, int((start + dur) * sr))
    seg = i1 - i0
    if seg <= 0:
        return np.zeros(n)
    attack = np.minimum(np.arange(seg) / (0.005 * sr), 1.0)
    release = np.exp(-np.arange(seg) / (dur * sr / 5.5))
    env[i0:i1] = attack * release
    return tone * env


def write(path, s):
    n = len(s)
    s[-int(0.03 * SR):] *= np.linspace(1, 0, int(0.03 * SR))
    # normalise to -16 dBFS peak: audible, never startling at 70% sink volume
    s = s / np.max(np.abs(s)) * (10 ** (-16 / 20))
    pcm = (s * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"  wrote {path}  ({n/SR*1000:.0f} ms, peak -16 dBFS)")


def main():
    a5, e6, a6 = 880.0, 1318.51, 1760.0
    # WAKE: rising fifth — a question mark, "I'm listening"
    write(OUT, pluck(a5, 0.000, 0.30, 0.55)
             + pluck(e6, 0.090, 0.33, 0.42)
             + pluck(a6, 0.180, 0.24, 0.10))
    # SLEEP: the same two notes DESCENDING — a full stop, "good night". The
    # mirror image is deliberate: the pair reads as open/close without anyone
    # being told what the sounds mean.
    write(OUT.parent / "roomi_sleep.wav",
          pluck(e6, 0.000, 0.28, 0.50)
        + pluck(a5, 0.090, 0.36, 0.45))


if __name__ == "__main__":
    main()
