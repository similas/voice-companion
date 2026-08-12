#!/usr/bin/env python3
"""
Generate the interactive latency dashboard from ALL benchmark results.

One page, every version, side by side. The point is the trajectory: the same
benchmark runs on every release (bench/benchmark.py — engine-aware since
v0.4), so each version's numbers are directly comparable, and the per-version
stack table shows WHAT changed to buy each improvement.

Why a separate page: GitHub sanitizes HTML in READMEs, so an embedded chart can
only ever be a static image there. This file is deliberately dependency-free —
no CDN, no build step, all data inlined as JSON — so it works offline, straight
from disk, and cannot rot when a CDN changes.

    python bench/make_dashboard.py          -> docs/index.html (all versions)

Every result directory under bench/results/ that contains a benchmark.json is
included automatically; nothing is hardcoded per version. Labels come from each
run's own system.models — the previous generator hardcoded "whisper"/"piper"
into the detail table, which became a lie the day the engines changed.
"""

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tag_key(tag: str):
    """Sort order for result dirs: v0.2-step1-llama-server between v0.1 and
    v0.2 — an intermediate measurement precedes the release it led to."""
    m = re.match(r"v(\d+)\.(\d+)(.*)", tag)
    if not m:
        return (99, 99, 0, tag)
    major, minor, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
    return (major, minor, 0 if suffix else 1, suffix)


def load_live(csv_path):
    """Per-turn records from a real conversation (cumulative stamps -> stages)."""
    turns = []
    if not csv_path or not csv_path.exists():
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

            stt, vis = num("t_stt_done"), num("t_vision_done")
            ttft, tts, play = (num("t_llm_first_token"),
                               num("t_tts_first_audio"),
                               num("t_audio_playback_start"))
            if None in (stt, ttft, tts, play):
                continue
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


def load_version(d: Path):
    """Normalise one result dir into the dashboard's version record.

    The DIRECTORY name is the version, not the json's own "tag" field: result
    dirs were renamed at release time (v0.2's json says tag=v0.3, step1's says
    v0.2), so trusting the recorded tag mislabels every bar by one version.
    """
    b = json.loads((d / "benchmark.json").read_text())
    b["tag"] = d.name
    models = (b.get("system") or {}).get("models") or {}
    comp = (b.get("composed_ttfa") or {}).get("voice_to_voice")
    if not comp:
        return None
    spec = b.get("stt_speculative") or {}
    stt, tts, llm = b.get("stt") or {}, b.get("tts") or {}, b.get("llm_text") or {}
    trig = b.get("vision_trigger") or {}
    live = load_live(next(iter(sorted(d.glob("live-session-*/live_turns.csv"))),
                          None))

    def get(dct, *path):
        for p in path:
            dct = (dct or {}).get(p)
        return dct

    label = b["tag"]
    if "step" in label:                       # v0.2-step1-llama-server
        label = "·".join(label.split("-")[:2])

    return {
        "tag": b["tag"],
        "label": label,
        "date": (get(b, "system", "timestamp_utc") or "")[:10],
        "stack": {
            "STT": models.get("stt", "?"),
            "LLM": models.get("llm", "?"),
            "TTS": models.get("tts", "?"),
        },
        # The composed voice->voice path. stt here is the wait a person
        # experiences (speculative wait where the version had it — that is
        # what benchmark.py composes into ttfa).
        "bars": {
            "stt": comp["stt_ms"],
            "camera": comp.get("camera_capture_ms", 0) or 0,
            "llm": comp["llm_first_token_ms"],
            "tts": comp["tts_first_audio_ms"],
        },
        "ttfa": comp["ttfa_ms"],
        "metrics": {
            "composed TTFA (ms)": comp["ttfa_ms"],
            "STT wait after turn end (ms)": comp["stt_ms"],
            "STT raw decode (ms)": get(stt, "latency_ms", "median"),
            "STT word error rate (%)": round((stt.get("wer_mean") or 0) * 100, 1),
            "STT exact-match (%)": round((stt.get("exact_match_rate") or 0) * 100),
            "LLM first token (ms)": get(llm, "ttft_ms", "median"),
            "LLM throughput (tok/s)": llm.get("tokens_per_s"),
            "TTS first audio (ms)": get(tts, "synth_ms", "median"),
            "TTS realtime factor": tts.get("realtime_factor"),
            "vision trigger (correct)": (f"{trig.get('correct')}/{trig.get('cases')}"
                                         if trig else None),
        },
        "speculative": bool(spec),
        "live": live,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>voice-companion &middot; latency evolution</title>
<style>
  :root{
    --bg:#ffffff; --panel:#f6f8fa; --line:#d0d7de; --ink:#1f2328;
    --dim:#656d76; --accent:#0969da; --hl:#fff8c5; --hlline:#d4a72c;
    --stt:#3f8f63; --camera:#b8901f; --llm:#3f6fb5; --tts:#9c4f93; --play:#7d8590;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --bg:#0d1117; --panel:#161b22; --line:#30363d; --ink:#e6edf3;
      --dim:#8b949e; --accent:#4493f8; --hl:#3a3520; --hlline:#d4a72c;
      --stt:#57ab7a; --camera:#d4a72c; --llm:#539bf5; --tts:#c46fb8; --play:#768390;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding:28px 20px 60px}
  .wrap{max-width:1060px;margin:0 auto}
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
  .gsub{fill:var(--dim);font-size:10.5px}
  .gtot{fill:var(--ink);font-size:12.5px;font-weight:600}
  .gdelta{font-size:10.5px}
  .grid{stroke:var(--line);stroke-width:1}
  #tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
    background:var(--ink);color:var(--bg);padding:7px 10px;border-radius:7px;
    font-size:12.5px;max-width:340px;z-index:9;line-height:1.45}
  #tip b{font-weight:600}
  .tblwrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:13px;min-width:640px}
  th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);
    vertical-align:top}
  th{color:var(--dim);font-weight:600;font-size:11.5px;text-transform:uppercase;
    letter-spacing:.05em;white-space:nowrap}
  td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
  td.best{font-weight:700}
  td.changed{background:var(--hl);border-left:2px solid var(--hlline)}
  td.stackcell{font-size:12.5px;max-width:180px}
  .note{color:var(--dim);font-size:12.5px;margin-top:12px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Time to first audio &mdash; every version</h1>
  <p class="sub" id="sysline"></p>

  <div class="card">
    <h2>Voice &rarr; voice, composed from measured stage medians</h2>
    <svg id="evoChart" role="img" aria-label="TTFA per version"></svg>
    <div class="legend" id="legend"></div>
    <p class="note">The same benchmark (<code>bench/benchmark.py</code>) runs on
      every release: 8 spoken prompts, 3 reps, machine to itself. STT is the
      wait <em>after</em> the turn ends &mdash; speculative decoding (v0.2+)
      does the work inside the VAD hangover, which is why the green segment
      almost disappears. Hover any segment; click a legend entry to hide a
      stage.</p>
  </div>

  <div class="card">
    <h2>The stack, per version &mdash; what bought each improvement</h2>
    <div class="tblwrap"><table id="stackTbl"></table></div>
    <p class="note">Highlighted cells changed from the previous version.
      Same GGUF twice with a different serving path is a different row value on
      purpose &mdash; the serving path was worth 1.7&thinsp;s in v0.2.
      The v0.4 voice change is audible, not just measurable:
      <a href="voices.html" style="color:var(--accent)">hear the three TTS
      candidates</a> on the same sentence, with their measured numbers.</p>
  </div>

  <div class="card">
    <h2>Quality &amp; throughput, per version</h2>
    <div class="tblwrap"><table id="metricTbl"></table></div>
    <p class="note">Bold = best. WER is measured against Piper-synthesised
      prompts &mdash; a reproducible proxy, not a claim about any human accent;
      per-prompt transcripts live in each version&rsquo;s
      <code>benchmark.json</code> under <code>stt.examples</code>.</p>
  </div>

  <div class="card">
    <h2>Live conversation, per turn</h2>
    <div class="tabs" role="tablist" id="liveTabs"></div>
    <svg id="turnChart" role="img" aria-label="Per-turn latency"></svg>
    <p class="note" id="liveNote">Real recorded conversations &mdash; STT, TTS,
      the camera thread and the model competing for 6 cores. Hover a bar for
      what was said. Versions without a bar set have no recorded session yet;
      composed numbers above are the hardware floor, these are the truth in a
      room.</p>
  </div>
</div>
<div id="tip"></div>

<script>
const DATA = __DATA__;

const STAGES = [
  {k:"stt",    label:"STT wait",        css:"--stt"},
  {k:"camera", label:"camera capture",  css:"--camera"},
  {k:"llm",    label:"LLM first token", css:"--llm"},
  {k:"tts",    label:"TTS first audio", css:"--tts"},
  {k:"play",   label:"playback start",  css:"--play"},
];
const colour = k => getComputedStyle(document.documentElement)
  .getPropertyValue(STAGES.find(s=>s.k===k).css).trim();
const STAGE_ENGINE = {stt:"STT", llm:"LLM", tts:"TTS"};

const off = new Set();
const tip = document.getElementById("tip");
let liveTag = (DATA.versions.filter(v=>v.live.length).slice(-1)[0]||{}).tag;

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
const txt = s => document.createTextNode(s);
const esc = s => (s||"").replace(/[<>&]/g, c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));

// ---- evolution chart: one stacked bar per version -----------------------
function drawEvo(){
  const svg = document.getElementById("evoChart");
  svg.textContent = "";
  const rows = DATA.versions;
  const W = 1000, L = 120, R = 150, barH = 26, rowH = 56, top = 26;
  const plot = W - L - R;
  const shown = r => STAGES.filter(s=>!off.has(s.k))
                           .reduce((a,s)=>a+(r.bars[s.k]||0),0);
  const peak = Math.max(...rows.map(shown), 1);
  const step = peak > 2500 ? 1000 : 250;
  const axisMax = Math.ceil(peak/step)*step || step;
  const sc = plot/axisMax;
  const H = top + rows.length*rowH + 44;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  for (let ms=0; ms<=axisMax; ms+=step){
    const x = L + ms*sc;
    svg.appendChild(el("line",{x1:x,y1:top-8,x2:x,y2:H-34,class:"grid",
      "stroke-opacity":ms?0.35:0.7}));
    svg.appendChild(el("text",{x:x,y:H-14,class:"axis","text-anchor":"middle"},
      [txt(ms>=1000 ? (ms/1000)+"s" : ms+"ms")]));
  }

  rows.forEach((r,i)=>{
    const y = top + i*rowH;
    svg.appendChild(el("text",{x:0,y:y+barH*0.55,class:"glab"},[txt(r.label)]));
    svg.appendChild(el("text",{x:0,y:y+barH*0.55+13,class:"gsub"},[txt(r.date)]));
    let x = L;
    STAGES.forEach(s=>{
      if (off.has(s.k)) return;
      const v = r.bars[s.k]||0;
      if (!v) return;
      const w = v*sc;
      const rect = el("rect",{x:x,y:y,width:Math.max(w,1.5),height:barH,
        fill:colour(s.k),rx:3,class:"seg"});
      const denom = shown(r) || 1;
      const eng = STAGE_ENGINE[s.k] ? `<br><span style="opacity:.75">${
        esc(r.stack[STAGE_ENGINE[s.k]])}</span>` : "";
      rect.addEventListener("mousemove", e=>showTip(
        `<b>${r.label} &middot; ${s.label}</b><br>${v.toFixed(0)} ms &middot; ` +
        `${(v/denom*100).toFixed(0)}% of this bar${eng}`, e));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
      if (w > 52) svg.appendChild(el("text",{x:x+w/2,y:y+barH*0.66,
        "text-anchor":"middle","font-size":"11.5",fill:"#0d1117",
        "fill-opacity":"0.85"},[txt(v.toFixed(0))]));
      x += w;
    });
    const tot = shown(r);
    svg.appendChild(el("text",{x:x+10,y:y+barH*0.55,class:"gtot"},
      [txt((tot/1000).toFixed(2)+"s")]));
    if (i > 0){
      const prev = shown(rows[i-1]);
      const d = (tot-prev)/prev*100;
      const better = d < 0;
      svg.appendChild(el("text",{x:x+10,y:y+barH*0.55+14,class:"gdelta",
        fill: better ? "var(--stt)" : "var(--tts)"},
        [txt((better?"":"+")+d.toFixed(0)+"% vs "+rows[i-1].label)]));
    }
  });
}

// ---- stack table ---------------------------------------------------------
function drawStack(){
  const t = document.getElementById("stackTbl");
  const vs = DATA.versions;
  const roles = ["STT","LLM","TTS"];
  let html = "<thead><tr><th></th>" +
    vs.map(v=>`<th>${v.label}</th>`).join("") + "</tr></thead><tbody>";
  roles.forEach(role=>{
    html += `<tr><th>${role}</th>`;
    vs.forEach((v,i)=>{
      const cur = v.stack[role], prev = i ? vs[i-1].stack[role] : cur;
      html += `<td class="stackcell${cur!==prev?" changed":""}">${esc(cur)}</td>`;
    });
    html += "</tr>";
  });
  html += "<tr><th>composed TTFA</th>" + vs.map((v,i)=>{
    const best = Math.min(...vs.map(x=>x.ttfa));
    return `<td class="n${v.ttfa===best?" best":""}">${(v.ttfa/1000).toFixed(2)}s</td>`;
  }).join("") + "</tr></tbody>";
  t.innerHTML = html;
}

// ---- metric evolution table ----------------------------------------------
const LOWER_IS_BETTER = k => !/tok\/s|exact|trigger/.test(k);
function drawMetrics(){
  const t = document.getElementById("metricTbl");
  const vs = DATA.versions;
  const keys = Object.keys(vs[vs.length-1].metrics);
  let html = "<thead><tr><th>metric</th>" +
    vs.map(v=>`<th class="n">${v.label}</th>`).join("") + "</tr></thead><tbody>";
  keys.forEach(k=>{
    const vals = vs.map(v=>v.metrics[k]);
    const nums = vals.filter(x=>typeof x === "number");
    const best = nums.length ? (LOWER_IS_BETTER(k) ? Math.min(...nums)
                                                   : Math.max(...nums)) : null;
    html += `<tr><td>${k}</td>` + vals.map(x=>{
      if (x === null || x === undefined) return "<td class='n'>—</td>";
      const b = (typeof x === "number" && x === best) ? " best" : "";
      return `<td class="n${b}">${typeof x === "number"
        ? (Math.abs(x) >= 100 ? x.toFixed(0) : x) : x}</td>`;
    }).join("") + "</tr>";
  });
  t.innerHTML = html + "</tbody>";
}

// ---- per-turn live chart with version tabs --------------------------------
function drawLiveTabs(){
  const box = document.getElementById("liveTabs");
  box.textContent = "";
  DATA.versions.filter(v=>v.live.length).forEach(v=>{
    const b = document.createElement("button");
    b.className = "tab";
    b.textContent = `${v.label} (${v.live.length} turns)`;
    b.setAttribute("role","tab");
    b.setAttribute("aria-selected", v.tag===liveTag ? "true" : "false");
    b.onclick = ()=>{ liveTag = v.tag; render(); };
    box.appendChild(b);
  });
}

function drawTurns(){
  const svg = document.getElementById("turnChart");
  svg.textContent = "";
  const v = DATA.versions.find(x=>x.tag===liveTag);
  const turns = v ? v.live : [];
  if (!turns.length){ svg.style.display = "none"; return; }
  svg.style.display = "";
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
      const val = t.stages[s.k]||0;
      if (!val) return;
      const h = val*sc;
      yTop -= h;
      const rect = el("rect",{x:cx,y:yTop,width:bw,height:Math.max(h,1),
        fill:colour(s.k),class:"seg",rx:2});
      rect.addEventListener("mousemove", e=>showTip(
        `<b>turn ${t.turn}</b>${t.vision?' &middot; used camera':''}<br>` +
        `<b>${s.label}: ${val.toFixed(0)} ms</b> of ${(t.total/1000).toFixed(2)}s<br>` +
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

// ---- legend ---------------------------------------------------------------
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

function render(){ drawEvo(); drawLegend(); drawStack(); drawMetrics();
                   drawLiveTabs(); drawTurns(); }
document.getElementById("sysline").textContent = DATA.sysline;
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                    help="ignored (kept for muscle memory) — the dashboard "
                         "always builds every version")
    args = ap.parse_args()
    if args.tag:
        print(f"  note: --tag is obsolete; building ALL versions")

    dirs = sorted((p for p in (ROOT / "bench" / "results").iterdir()
                   if (p / "benchmark.json").exists()),
                  key=lambda p: tag_key(p.name))
    versions = [v for v in (load_version(d) for d in dirs) if v]
    if not versions:
        raise SystemExit("no benchmark.json found under bench/results/")

    # One recording, one owner. v0.1's live session file was COPIED into the
    # v0.2 and step1 result dirs (identical md5), so without this the same 16
    # turns would appear under three version tabs and read as three separate
    # recordings. Each distinct session belongs to the earliest version that
    # carries it.
    seen = set()
    for v in versions:
        key = json.dumps(v["live"], sort_keys=True)
        if v["live"] and key in seen:
            v["live"] = []
        seen.add(key)

    newest = versions[-1]
    sysline = (f"Jetson Orin Nano Super (8 GB), everything on-device · "
               f"latest: {newest['label']} — {newest['stack']['STT']} · "
               f"{newest['stack']['LLM']} · {newest['stack']['TTS']}")

    payload = {"versions": versions, "sysline": sysline}
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(payload))
    out.write_text(html)
    print(f"  wrote {out.relative_to(ROOT)}  ({len(html)//1024} KB, self-contained)")
    for v in versions:
        print(f"    {v['label']:12} ttfa {v['ttfa']:7.1f} ms   "
              f"live turns: {len(v['live'])}")


if __name__ == "__main__":
    main()
