#!/usr/bin/env python3
"""
Render the TTFA chart from benchmark.json + the live session CSV.

Emits a self-contained SVG with no external fonts or scripts, using colours that
stay legible on both light and dark GitHub themes (no pure white or pure black,
transparent background).

    python bench/make_chart.py --tag v0.1
"""

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Stage colours, in pipeline order.
STAGES = [
    ("stt_ms", "STT", "#4c9f70"),
    ("camera_capture_ms", "camera", "#c9a227"),
    ("llm_first_token_ms", "LLM first token", "#3f6fb5"),
    ("tts_first_audio_ms", "TTS first audio", "#a4599b"),
]
INK = "#8b949e"     # axis/label grey that reads on both themes
TEXT = "#adbac7"

# GitHub renders SVGs as separate documents, so CSS inside the file applies —
# including prefers-color-scheme. The fill= attributes stay as a fallback for
# renderers that ignore the stylesheet, and both fallback tones are mid-greys
# that remain legible on white as well as on dark.
STYLE = """<style>
  .ttl { fill: #1f2328; font-weight: 600 }
  .lbl { fill: #32383f }
  .sub, .ax { fill: #6e7781 }
  @media (prefers-color-scheme: dark) {
    .ttl { fill: #e6edf3 }
    .lbl { fill: #c9d1d9 }
    .sub, .ax { fill: #8b949e }
  }
</style>"""


def live_medians(csv_path):
    """Median measured TTFA from the real conversation, split by vision use."""
    text, vision = [], []
    if not csv_path.exists():
        return None, None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            tot = r.get("total_latency")
            if not tot:
                continue
            try:
                v = float(tot)
            except ValueError:
                continue
            (vision if r.get("vision_used") == "true" else text).append(v)
    return (round(st.median(text), 1) if text else None,
            round(st.median(vision), 1) if vision else None)


def bar(x0, y, w, h, fill, rx=2):
    return (f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(w,0.6):.1f}" '
            f'height="{h}" fill="{fill}" rx="{rx}"/>')


def build(bench, live_text, live_vision, tag="v0.1"):
    comp = bench["composed_ttfa"]
    rows = []
    v2v = comp.get("voice_to_voice")
    vi2v = comp.get("voice_image_to_voice")
    if v2v:
        rows.append(("voice → voice", v2v, live_text))
    if vi2v:
        rows.append(("voice + image → voice", vi2v, live_vision))

    # Layout
    W, LEFT, RIGHT = 900, 210, 90
    row_h, gap, bar_h = 78, 26, 26
    top = 84
    plot_w = W - LEFT - RIGHT
    peak = max([r[1]["ttfa_ms"] for r in rows]
               + [v for _, _, v in rows if v] + [1])
    # Round the axis up to a tidy 1000 ms step.
    axis_max = (int(peak / 1000) + 1) * 1000
    scale = plot_w / axis_max
    H = top + len(rows) * (row_h + gap) + 74

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,'
         f'Segoe UI,Roboto,sans-serif">', STYLE]
    o.append(f'<text x="0" y="24" font-size="17" class="ttl" fill="{TEXT}">'
             f'Time to first audio &#8212; {tag}</text>')
    o.append(f'<text x="0" y="46" font-size="12.5" class="sub" fill="{INK}">'
             f'Jetson Orin Nano Super 8GB &#183; fully local &#183; '
             f'bars = component benchmark, diamond = median of a real conversation'
             f'</text>')

    # gridlines
    for ms in range(0, axis_max + 1, 1000):
        x = LEFT + ms * scale
        o.append(f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{H-52}" '
                 f'stroke="{INK}" stroke-opacity="0.22" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{H-32}" font-size="11" class="ax" '
                 f'fill="{INK}" text-anchor="middle">{ms//1000}s</text>')

    y = top
    for label, stage, live in rows:
        o.append(f'<text x="0" y="{y+bar_h*0.72:.1f}" font-size="13.5" '
                 f'font-weight="600" class="lbl" fill="{TEXT}">{label}</text>')
        x = LEFT
        for key, name, colour in STAGES:
            val = stage.get(key)
            if not val:
                continue
            w = val * scale
            o.append(bar(x, y, w, bar_h, colour))
            if w > 46:
                o.append(f'<text x="{x + w/2:.1f}" y="{y+bar_h*0.68:.1f}" '
                         f'font-size="11" fill="#0d1117" fill-opacity="0.82" '
                         f'text-anchor="middle">{val:.0f}</text>')
            x += w
        total = stage["ttfa_ms"]
        o.append(f'<text x="{x+10:.1f}" y="{y+bar_h*0.72:.1f}" font-size="13" '
                 f'font-weight="600" class="lbl" fill="{TEXT}">'
                 f'{total/1000:.2f}s</text>')

        # live measured marker + a faint bar to the same scale
        if live:
            ly = y + bar_h + 12
            lw = live * scale
            o.append(f'<rect x="{LEFT}" y="{ly}" width="{lw:.1f}" height="10" '
                     f'fill="{INK}" fill-opacity="0.3" rx="2"/>')
            cx, cy = LEFT + lw, ly + 5
            o.append(f'<polygon points="{cx-6:.1f},{cy} {cx},{cy-6} '
                     f'{cx+6:.1f},{cy} {cx},{cy+6}" class="lbl" fill="{TEXT}"/>')
            o.append(f'<text x="{cx+12:.1f}" y="{cy+4:.1f}" font-size="11.5" '
                     f'class="ax" fill="{INK}">{live/1000:.2f}s live</text>')
        y += row_h + gap

    # legend
    lx = LEFT
    ly = H - 12
    for key, name, colour in STAGES:
        if not any(r[1].get(key) for r in rows):
            continue
        o.append(f'<rect x="{lx}" y="{ly-9}" width="11" height="11" '
                 f'fill="{colour}" rx="2"/>')
        o.append(f'<text x="{lx+16}" y="{ly}" font-size="11.5" class="ax" '
                 f'fill="{INK}">{name}</text>')
        lx += 20 + len(name) * 7.0
    o.append("</svg>")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0.1")
    args = ap.parse_args()

    d = ROOT / "bench" / "results" / args.tag
    bench = json.loads((d / "benchmark.json").read_text())
    lt, lv = live_medians(d / "live-session-2026-08-05" / "live_turns.csv")

    svg = build(bench, lt, lv, args.tag)
    out = d / "ttfa.svg"
    out.write_text(svg)
    print(f"  wrote {out.relative_to(ROOT)}")
    print(f"  composed  voice->voice        "
          f"{bench['composed_ttfa']['voice_to_voice']['ttfa_ms']:.0f} ms")
    if bench["composed_ttfa"].get("voice_image_to_voice"):
        print(f"  composed  voice+image->voice  "
              f"{bench['composed_ttfa']['voice_image_to_voice']['ttfa_ms']:.0f} ms")
    print(f"  live      voice->voice        {lt} ms")
    print(f"  live      voice+image->voice  {lv} ms")


if __name__ == "__main__":
    main()
