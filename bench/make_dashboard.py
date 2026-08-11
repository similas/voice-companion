#!/usr/bin/env python3
"""
Generate an interactive latency dashboard from benchmark results.

Why a separate page: GitHub sanitizes HTML in READMEs, so an embedded chart can
only ever be a static image there. A self-contained HTML file served by GitHub
Pages (or opened locally) can be fully interactive. The README keeps the SVG as a
fallback and links here.

The page is deliberately dependency-free — no CDN, no build step, all data inlined
as JSON — so it works offline and cannot rot when a CDN changes.

    python bench/make_dashboard.py --tag v0.1     -> docs/index.html
"""

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_live(csv_path):
    """Per-turn records from a real conversation."""
    turns = []
    if not csv_path.exists():
        return turns
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if not r.get("total_latency"):
                continue

            def num(k):
                try:
                    return float(r[k]) if r.get(k) else None
                except ValueError:
                    return None

            stt = num("t_stt_done")
            vis = num("t_vision_done")
            ttft = num("t_llm_first_token")
            tts = num("t_tts_first_audio")
            play = num("t_audio_playback_start")
            if None in (stt, ttft, tts, play):
                continue
            # Convert cumulative stamps into per-stage durations.
            turns.append({
                "turn": int(r["turn"]),
                "vision": r.get("vision_used") == "true",
                "transcript": r.get("transcript", "").strip(),
                "reply": r.get("reply", "").strip(),
                "stages": {
                    "stt": round(stt, 1),
                    "camera": round(vis - stt, 1) if vis else 0.0,
                    "llm": round(ttft - (vis or stt), 1),
                    "tts": round(tts - ttft, 1),
                    "play": round(max(play - tts, 0), 1),
                },
                "total": round(play, 1),
            })
    return turns


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>voice-companion &middot; latency __TAG__</title>
<style>
  :root{
    --bg:#ffffff; --panel:#f6f8fa; --line:#d0d7de; --ink:#1f2328;
    --dim:#656d76; --accent:#0969da;
    --stt:#3f8f63; --camera:#b8901f; --llm:#3f6fb5; --tts:#9c4f93; --play:#7d8590;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --bg:#0d1117; --panel:#161b22; --line:#30363d; --ink:#e6edf3;
      --dim:#8b949e; --accent:#4493f8;
      --stt:#57ab7a; --camera:#d4a72c; --llm:#539bf5; --tts:#c46fb8; --play:#768390;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding:28px 20px 60px}
  .wrap{max-width:1040px;margin:0 auto}
  h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
  .sub{color:var(--dim);font-size:13.5px;margin:0 0 22px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    padding:18px 20px;margin-bottom:18px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);
    margin:0 0 14px;font-weight:600}
  .tabs{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
  .tab{font:inherit;font-size:13px;padding:6px 13px;border-radius:7px;cursor:pointer;
    background:transparent;border:1px solid var(--line);color:var(--dim)}
  .tab[aria-selected=true]{background:var(--accent);border-color:var(--accent);
    color:#fff;font-weight:500}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:14px;font-size:12.5px}
  .lg{display:flex;align-items:center;gap:6px;cursor:pointer;color:var(--dim);
    user-select:none}
  .lg.off{opacity:.35;text-decoration:line-through}
  .sw{width:11px;height:11px;border-radius:3px;flex:0 0 auto}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .seg{cursor:pointer;transition:opacity .12s}
  .seg:hover{opacity:.78}
  .axis{fill:var(--dim);font-size:11px}
  .glab{fill:var(--ink);font-size:13px;font-weight:600}
  .gtot{fill:var(--ink);font-size:12.5px;font-weight:600}
  .grid{stroke:var(--line);stroke-width:1}
  #tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
    background:var(--ink);color:var(--bg);padding:7px 10px;border-radius:7px;
    font-size:12.5px;max-width:330px;z-index:9;line-height:1.45}
  #tip b{font-weight:600}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
  th{color:var(--dim);font-weight:600;font-size:11.5px;text-transform:uppercase;
    letter-spacing:.05em}
  td.n{text-align:right;font-variant-numeric:tabular-nums}
  .note{color:var(--dim);font-size:12.5px;margin-top:12px}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .kpi{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
  .kpi .v{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums}
  .kpi .k{color:var(--dim);font-size:11.5px;text-transform:uppercase;
    letter-spacing:.05em;margin-top:2px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Time to first audio &mdash; <span id="tag"></span></h1>
  <p class="sub" id="sysline"></p>

  <div class="card">
    <h2>End to end</h2>
    <div class="tabs" role="tablist" id="modeTabs"></div>
    <svg id="mainChart" role="img" aria-label="TTFA stage breakdown"></svg>
    <div class="legend" id="legend"></div>
    <p class="note" id="modeNote"></p>
  </div>

  <div class="card">
    <h2>Every turn of the live conversation</h2>
    <svg id="turnChart" role="img" aria-label="Per-turn latency"></svg>
    <p class="note">Hover a bar for what was said and what came back. Taller
      bars with a gold segment are turns that used the camera.</p>
  </div>

  <div class="card">
    <h2>Component detail (benchmark)</h2>
    <div class="kpis" id="kpis"></div>
    <table id="detail"></table>
  </div>
</div>
<div id="tip"></div>

<script>
const DATA = __DATA__;

const STAGES = [
  {k:"stt",    label:"STT",             css:"--stt"},
  {k:"camera", label:"camera capture",  css:"--camera"},
  {k:"llm",    label:"LLM first token", css:"--llm"},
  {k:"tts",    label:"TTS first audio", css:"--tts"},
  {k:"play",   label:"playback start",  css:"--play"},
];
const colour = k => getComputedStyle(document.documentElement)
  .getPropertyValue(STAGES.find(s=>s.k===k).css).trim();

const off = new Set();
let mode = "benchmark";
const tip = document.getElementById("tip");

function showTip(html, e){
  tip.innerHTML = html;
  tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8)  x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + "px";
  tip.style.top  = y + "px";
}
const hideTip = () => { tip.style.opacity = 0; };

function el(tag, attrs, kids){
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in (attrs||{})) n.setAttribute(k, attrs[k]);
  (kids||[]).forEach(c => n.appendChild(c));
  return n;
}
const txt = (s) => document.createTextNode(s);

// ---- main stacked chart -------------------------------------------------
function drawMain(){
  const svg = document.getElementById("mainChart");
  if (!svg) return;
  svg.textContent = "";
  const rows = DATA.modes[mode].rows;
  const W = 1000, L = 190, R = 108, barH = 30, rowH = 62, top = 26;
  const plot = W - L - R;
  const shown = r => STAGES.filter(s=>!off.has(s.k))
                           .reduce((a,s)=>a+(r.stages[s.k]||0),0);
  const peak = Math.max(...rows.map(shown), 1);
  const axisMax = Math.ceil(peak/1000)*1000 || 1000;
  const sc = plot/axisMax;
  const H = top + rows.length*rowH + 44;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  for (let ms=0; ms<=axisMax; ms+=1000){
    const x = L + ms*sc;
    svg.appendChild(el("line",{x1:x,y1:top-8,x2:x,y2:H-34,class:"grid",
      "stroke-opacity":ms?0.35:0.7}));
    svg.appendChild(el("text",{x:x,y:H-14,class:"axis","text-anchor":"middle"},
      [txt((ms/1000)+"s")]));
  }

  rows.forEach((r,i)=>{
    const y = top + i*rowH;
    svg.appendChild(el("text",{x:0,y:y+barH*0.7,class:"glab"},[txt(r.label)]));
    let x = L;
    STAGES.forEach(s=>{
      if (off.has(s.k)) return;
      const v = r.stages[s.k]||0;
      if (!v) return;
      const w = v*sc;
      const rect = el("rect",{x:x,y:y,width:Math.max(w,1),height:barH,
        fill:colour(s.k),rx:3,class:"seg"});
      const denom = shown(r) || 1;
      const share = (v/denom*100).toFixed(0);
      rect.addEventListener("mousemove", e=>showTip(
        `<b>${s.label}</b><br>${v.toFixed(0)} ms &middot; ${share}% of this path`, e));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
      if (w > 52) svg.appendChild(el("text",{x:x+w/2,y:y+barH*0.66,
        "text-anchor":"middle","font-size":"11.5",fill:"#0d1117",
        "fill-opacity":"0.85"},[txt(v.toFixed(0))]));
      x += w;
    });
    svg.appendChild(el("text",{x:x+10,y:y+barH*0.7,class:"gtot"},
      [txt((shown(r)/1000).toFixed(2)+"s")]));
  });
  document.getElementById("modeNote").textContent = DATA.modes[mode].note;
}

// ---- per-turn chart ----------------------------------------------------
function drawTurns(){
  const svg = document.getElementById("turnChart");
  if (!svg) return;
  svg.textContent = "";
  const turns = DATA.live;
  if (!svg) return;
  if (!turns.length){ svg.style.display = "none"; return; }
  const W = 1000, L = 40, B = 46, top = 16, H = 300;
  const plot = W - L - 16, ph = H - top - B;
  const peak = Math.max(...turns.map(t=>t.total));
  const axisMax = Math.ceil(peak/2000)*2000 || 2000;
  const sc = ph/axisMax;
  const bw = Math.min(52, plot/turns.length - 8);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  for (let ms=0; ms<=axisMax; ms+=2000){
    const y = top + ph - ms*sc;
    svg.appendChild(el("line",{x1:L,y1:y,x2:W-16,y2:y,class:"grid",
      "stroke-opacity":ms?0.35:0.7}));
    svg.appendChild(el("text",{x:L-8,y:y+4,class:"axis","text-anchor":"end"},
      [txt((ms/1000)+"s")]));
  }

  turns.forEach((t,i)=>{
    const cx = L + 12 + i*(plot/turns.length);
    let yTop = top + ph;
    STAGES.forEach(s=>{
      if (off.has(s.k)) return;
      const v = t.stages[s.k]||0;
      if (!v) return;
      const h = v*sc;
      yTop -= h;
      const rect = el("rect",{x:cx,y:yTop,width:bw,height:Math.max(h,1),
        fill:colour(s.k),class:"seg",rx:2});
      rect.addEventListener("mousemove", e=>showTip(
        `<b>turn ${t.turn}</b>${t.vision?' &middot; used camera':''}<br>` +
        `<b>${s.label}: ${v.toFixed(0)} ms</b> of ${(t.total/1000).toFixed(2)}s<br>` +
        `<span style="opacity:.75">you:</span> ${esc(t.transcript)}<br>` +
        `<span style="opacity:.75">bot:</span> ${esc(t.reply)}`, e));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
    });
    svg.appendChild(el("text",{x:cx+bw/2,y:H-B+18,class:"axis",
      "text-anchor":"middle"},[txt(t.turn)]));
    if (t.vision) svg.appendChild(el("text",{x:cx+bw/2,y:H-B+32,class:"axis",
      "text-anchor":"middle","font-size":"10"},[txt("cam")]));
  });
}
const esc = s => (s||"").replace(/[<>&]/g, c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));

// ---- legend, tabs, tables ---------------------------------------------
function drawLegend(){
  const box = document.getElementById("legend");
  box.textContent = "";
  STAGES.forEach(s=>{
    const on = !off.has(s.k);
    const d = document.createElement("div");
    d.className = "lg" + (on ? "" : " off");
    d.innerHTML = `<span class="sw" style="background:${colour(s.k)}"></span>${s.label}`;
    d.title = "click to hide this stage";
    d.onclick = ()=>{ on ? off.add(s.k) : off.delete(s.k); render(); };
    box.appendChild(d);
  });
}

function drawTabs(){
  const box = document.getElementById("modeTabs");
  box.textContent = "";
  Object.keys(DATA.modes).forEach(k=>{
    const b = document.createElement("button");
    b.className = "tab"; b.textContent = DATA.modes[k].label;
    b.setAttribute("role","tab");
    b.setAttribute("aria-selected", k===mode ? "true" : "false");
    b.onclick = ()=>{ mode = k; render(); };
    box.appendChild(b);
  });
}

function drawDetail(){
  const k = document.getElementById("kpis");
  k.innerHTML = DATA.kpis.map(x=>
    `<div class="kpi"><div class="v">${x.value}</div><div class="k">${x.key}</div></div>`
  ).join("");
  const t = document.getElementById("detail");
  t.innerHTML =
    "<thead><tr><th>stage</th><th class='n'>min</th><th class='n'>median</th>" +
    "<th class='n'>p90</th><th class='n'>max</th><th class='n'>n</th></tr></thead><tbody>" +
    DATA.detail.map(r=>`<tr><td>${r.stage}</td><td class='n'>${r.min}</td>` +
      `<td class='n'><b>${r.median}</b></td><td class='n'>${r.p90}</td>` +
      `<td class='n'>${r.max}</td><td class='n'>${r.n}</td></tr>`).join("") +
    "</tbody>";
}

function render(){ drawTabs(); drawMain(); drawTurns(); drawLegend(); }

document.getElementById("tag").textContent = DATA.tag;
document.getElementById("sysline").textContent = DATA.sysline;
drawDetail();
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0.1")
    args = ap.parse_args()

    d = ROOT / "bench" / "results" / args.tag
    b = json.loads((d / "benchmark.json").read_text())
    live = load_live(next(iter(sorted(d.glob("live-session-*/live_turns.csv"))), d / "no-live-session"))

    comp = b["composed_ttfa"]
    v2v, vi2v = comp.get("voice_to_voice"), comp.get("voice_image_to_voice")

    def bench_row(label, c):
        return {"label": label, "stages": {
            "stt": c["stt_ms"],
            "camera": c.get("camera_capture_ms", 0),
            "llm": c["llm_first_token_ms"],
            "tts": c["tts_first_audio_ms"],
            "play": 0}}

    bench_rows = [r for r in [
        bench_row("voice → voice", v2v) if v2v else None,
        bench_row("voice + image → voice", vi2v) if vi2v else None] if r]

    def live_row(label, sel):
        rows = [t for t in live if sel(t)]
        if not rows:
            return None
        med = lambda k: round(st.median([t["stages"][k] for t in rows]), 1)  # noqa: E731
        return {"label": f"{label}  (n={len(rows)})", "stages": {
            "stt": med("stt"), "camera": med("camera"), "llm": med("llm"),
            "tts": med("tts"), "play": med("play")}}

    live_rows = [r for r in [
        live_row("voice → voice", lambda t: not t["vision"]),
        live_row("voice + image → voice", lambda t: t["vision"])] if r]

    def row(stage, s):
        if not s:
            return None
        return {"stage": stage, "min": s["min"], "median": s["median"],
                "p90": s["p90"], "max": s["max"], "n": s["n"]}

    detail = [r for r in [
        row("STT (whisper tiny)", b["stt"]["latency_ms"]),
        row("LLM first token — text", b["llm_text"]["ttft_ms"]),
        row("LLM first token — vision",
            (b.get("llm_vision") or {}).get("ttft_ms")),
        row("TTS synthesis (piper)", b["tts"]["synth_ms"]),
        row("camera capture + encode",
            (b.get("camera") or {}).get("capture_ms")),
        row("camera frame age", (b.get("camera") or {}).get("frame_age_ms")),
    ] if r]

    kpis = [
        {"key": "voice → voice", "value": f"{v2v['ttfa_ms']/1000:.2f}s"} if v2v else None,
        {"key": "voice+image → voice",
         "value": f"{vi2v['ttfa_ms']/1000:.2f}s"} if vi2v else None,
        {"key": "LLM throughput", "value": f"{b['llm_text']['tokens_per_s']} tok/s"},
        {"key": "STT realtime factor", "value": f"{b['stt']['realtime_factor']}×"},
        {"key": "TTS realtime factor", "value": f"{b['tts']['realtime_factor']}×"},
        {"key": "vision triggers",
         "value": f"{b['vision_trigger']['correct']}/{b['vision_trigger']['cases']}"},
    ]
    kpis = [k for k in kpis if k]

    sysinfo = b["system"]
    sysline = (f"{sysinfo['device']} · {sysinfo['cores']} cores · "
               f"{sysinfo['models']['stt']} · {sysinfo['models']['llm']} · "
               f"{sysinfo['models']['tts']} · measured "
               f"{sysinfo['timestamp_utc'][:10]}")

    payload = {
        "tag": args.tag,
        "sysline": sysline,
        "modes": {
            "benchmark": {
                "label": "Benchmark (isolated)",
                "rows": bench_rows,
                "note": "Each stage measured on its own, nothing else running. "
                        "This is the floor the hardware can do.",
            },
            "live": {
                "label": "Live conversation (median)",
                "rows": live_rows,
                "note": "Medians from a real recorded conversation, where STT, TTS, "
                        "the camera thread and the model all compete for 6 CPU "
                        "cores. The gap versus the benchmark is contention.",
            },
        },
        "live": live,
        "detail": detail,
        "kpis": kpis,
    }

    outdir = ROOT / "docs"
    outdir.mkdir(exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(payload)).replace("__TAG__", args.tag)
    out = outdir / "index.html"
    out.write_text(html)
    print(f"  wrote {out.relative_to(ROOT)}  ({len(html)//1024} KB, self-contained)")
    print(f"  modes: benchmark ({len(bench_rows)} rows), live ({len(live_rows)} rows)")
    print(f"  per-turn bars: {len(live)}")


if __name__ == "__main__":
    main()
