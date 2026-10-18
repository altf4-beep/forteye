# -*- coding: utf-8 -*-
"""
FORTEYE: Streamlit UI in the design of forteye.html (cyan/pink surveillance console).

Upload a video -> POST /api/v1/analyze -> poll /api/v1/jobs/{id} -> show timestamps and base64 evidence frames.

Run (API must be up first):
    FORTEYE_API_URL=http://127.0.0.1:8000 FORTEYE_API_KEY=<secret> streamlit run app.py
"""
import html
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("FORTEYE_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("FORTEYE_API_KEY", "forteye-secret")
POLL_SEC = 1.0
UI_TIMEOUT_SEC = int(os.getenv("FORTEYE_UI_TIMEOUT", "3600"))
MIME = {".mov": "video/quicktime", ".mp4": "video/mp4", ".m4v": "video/mp4",
        ".avi": "video/x-msvideo", ".mkv": "video/x-matroska"}
SEV = {"high": ("КРИТИЧНО", 3), "mid": ("СРЕДНЕ", 2), "low": ("НИЗКО", 1)}

st.set_page_config(page_title="FORTEYE // Анализ кассовой дисциплины", page_icon="👁", layout="wide",
                   initial_sidebar_state="collapsed")
ss = st.session_state
for key, default in (("result", None), ("seek", 0), ("video", None), ("duration", 0.0), ("night", True)):
    ss.setdefault(key, default)

# --------------------------------------------------------------------------- #
# design (ported from forteye.html)
# --------------------------------------------------------------------------- #
EYE = ('<svg class="le" viewBox="0 0 100 100" aria-hidden="true"><path d="M97 32.4A50 50 0 1 0 99.9 54.4L94.1 60.7A41 37 0 0 1 '
       '13.6 50A36.4 36.4 0 0 1 34.9 17L64.9 16.7A36.4 36.4 0 0 1 82 32.4Z"/><path d="M82 32.4C74 45 62 51.2 50.1 51.2C40 51.2 29 '
       '47 22.7 38.8C22.7 58 33 64.7 50.1 64.7C72 65 90 52 97 32.4Z"/><g class="pu"><circle cx="49.9" cy="27.1" r="18.1"/>'
       '<circle class="hl" cx="44.7" cy="23.5" r="4.9"/></g></svg>')

NIGHT = "--bg:#000;--cyan:#62D3FC;--dim:rgba(98,211,252,.55);--line:rgba(98,211,252,.3);--panel:rgba(0,6,10,.78);--ink:#fff"
DAY = "--bg:#eef3f6;--cyan:#0b6f94;--dim:rgba(11,60,80,.72);--line:rgba(11,111,148,.38);--panel:rgba(255,255,255,.82);--ink:#0b1a22"

COMMON_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Russo+One&display=swap');
:root{--pink:#B31554}
html{background:var(--bg)!important}
body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"],section.main{background:transparent!important}
.stApp{color:var(--cyan)}
.stApp,.stApp p,.stApp label,.stApp button,.stApp input{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace!important;font-size:13px}
[data-testid="stHeader"]{background:transparent!important}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{max-width:1280px;padding:6px 18px 40px!important}
#forteye-grid{position:fixed;inset:0;width:100vw;height:100vh;z-index:-1;pointer-events:none}
/* panels */
[data-testid="stColumn"],[data-testid="column"]{position:relative;background:var(--panel);border:1px solid var(--line);padding:16px!important;backdrop-filter:blur(3px)}
[data-testid="stColumn"]:before,[data-testid="column"]:before,[data-testid="stColumn"]:after,[data-testid="column"]:after{content:"";position:absolute;width:16px;height:16px;border:2px solid var(--cyan);pointer-events:none}
[data-testid="stColumn"]:before,[data-testid="column"]:before{top:-1px;left:-1px;border-right:0;border-bottom:0}
[data-testid="stColumn"]:after,[data-testid="column"]:after{bottom:-1px;right:-1px;border-left:0;border-top:0}
.ph{display:flex;justify-content:space-between;font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);margin-bottom:12px}
/* video stage */
[data-testid="stVideo"],.stVideo video,video{border:1px solid var(--line);background:#02070a;filter:grayscale(1) contrast(1.2) brightness(.95) sepia(1) hue-rotate(145deg) saturate(2.4)}
.stage{position:relative;aspect-ratio:16/9;background:#02070a;border:1px solid var(--line);display:grid;place-items:center;text-align:center;color:var(--dim);letter-spacing:.3em;font-size:12px}
.stage b{display:block;font:400 22px "Russo One",Impact,sans-serif;color:var(--pink);letter-spacing:.25em;margin-bottom:6px}
.hudline{display:flex;justify-content:space-between;gap:10px;font-size:11px;letter-spacing:.12em;color:var(--dim);margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hudline .rec{color:#ff7aa8}
.tl-bar{position:relative;height:14px;margin:12px 0 4px;border:1px solid var(--line);background:rgba(98,211,252,.05)}
.mk{position:absolute;top:-4px;width:3px;height:20px;background:var(--pink);box-shadow:0 0 8px var(--pink);transform:translateX(-1px)}
/* buttons / uploader */
.stButton>button,[data-testid^="stBaseButton"],[data-testid="stFileUploader"] button{border:1px solid var(--cyan)!important;border-radius:0!important;background:rgba(98,211,252,.08)!important;color:var(--cyan)!important;font-weight:700!important;letter-spacing:.16em;text-transform:uppercase;transition:.2s}
.stButton>button:hover:not(:disabled),[data-testid^="stBaseButton"]:hover:not(:disabled){background:var(--cyan)!important;color:#000!important;box-shadow:0 0 18px var(--cyan)}
.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"]{border-color:var(--pink)!important;background:rgba(179,21,84,.18)!important;color:#ff7aa8!important}
.stButton>button[kind="primary"]:hover:not(:disabled),[data-testid="stBaseButton-primary"]:hover:not(:disabled){background:var(--pink)!important;color:#fff!important;box-shadow:0 0 18px var(--pink)}
.stButton>button:disabled,[data-testid^="stBaseButton"]:disabled{opacity:.35}
[data-testid="stFileUploaderDropzone"]{background:rgba(98,211,252,.05)!important;border:1px dashed var(--line)!important;border-radius:0!important}
[data-testid="stFileUploaderDropzone"] *{color:var(--cyan)!important}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:0!important;background:transparent!important}
[data-testid="stExpander"] summary{color:var(--dim)}
[data-testid="stTextInput"] input{background:rgba(98,211,252,.05)!important;border-radius:0!important;color:var(--cyan)!important}
[data-testid="stToggle"] label p{letter-spacing:.14em;text-transform:uppercase;font-size:10.5px;color:var(--cyan)}
/* progress + log */
.prog{height:6px;margin-top:14px;border:1px solid var(--line)}
.prog i{display:block;height:100%;background:linear-gradient(90deg,var(--pink),var(--cyan));box-shadow:0 0 10px var(--cyan);transition:width .25s}
.log{margin-top:12px;min-height:84px;font-size:11.5px;color:var(--dim)}
.log div:before{content:"> ";color:var(--pink)}
.err{border:1px solid var(--pink);background:rgba(179,21,84,.12);color:#ff7aa8;padding:10px;margin-top:12px}
/* stats + cards */
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.stat{border:1px solid var(--line);padding:8px 10px;background:rgba(98,211,252,.05)}
.stat small{display:block;color:var(--dim);font-size:9.5px;letter-spacing:.16em;text-transform:uppercase}
.stat b{font:400 20px "Russo One",Impact,sans-serif;color:var(--ink)}
.stat b.r{color:var(--pink);text-shadow:0 0 10px var(--pink)}
.empty{color:var(--dim);text-align:center;padding:36px 8px;letter-spacing:.14em;font-size:12px}
.v{border:1px solid var(--pink);background:rgba(179,21,84,.07);padding:10px;margin-top:14px}
.shot{position:relative;border:1px solid var(--cyan);background:#02070a;aspect-ratio:16/9;overflow:hidden}
.shot img{width:100%;height:100%;object-fit:contain;display:block}
.shot .tint{position:absolute;inset:0;background:var(--cyan);mix-blend-mode:color;opacity:.12;pointer-events:none}
.shot .tc{position:absolute;left:6px;top:5px;font-size:10.5px;background:rgba(0,0,0,.65);padding:1px 6px;border:1px solid var(--pink);color:#ff7aa8}
.shot .tag{position:absolute;right:6px;bottom:5px;font-size:9.5px;letter-spacing:.14em;color:#62D3FC;background:rgba(0,0,0,.65);padding:1px 6px}
.thumbs{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:6px;margin-top:6px}
.th{position:relative;border:1px solid var(--line);aspect-ratio:16/9;overflow:hidden}
.th img{width:100%;height:100%;object-fit:cover}
.th span{position:absolute;left:3px;bottom:2px;font-size:9.5px;background:rgba(0,0,0,.65);color:#62D3FC;padding:0 4px}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 6px}
.chip{padding:2px 9px;border:1px solid var(--line);border-radius:999px;font-size:10px;letter-spacing:.12em;text-transform:uppercase}
.chip.p{border-color:var(--pink);color:#ff7aa8;background:rgba(179,21,84,.15)}
.v h3{margin:4px 0;font:400 15px "Russo One",Impact,sans-serif;color:var(--ink);letter-spacing:.04em}
.law{color:#ff7aa8;font-size:12px}
.desc{color:var(--cyan);margin:4px 0 0}
.day .chip.p,.day .law{color:var(--pink)}
"""

GRID_JS = """<script>
(function(){try{
const P=window.parent,D=P.document,DAY=__DAY__;
P.__feRun=(P.__feRun||0)+1;const id=P.__feRun;
let g=D.getElementById('forteye-grid');
if(!g){g=D.createElement('canvas');g.id='forteye-grid';g.style.cssText='position:fixed;inset:0;width:100vw;height:100vh;z-index:-1;pointer-events:none';D.body.insertBefore(g,D.body.firstChild);}
const c=g.getContext('2d');let W,H,t=0;
function rs(){W=g.width=P.innerWidth;H=g.height=P.innerHeight}
P.addEventListener('resize',rs);rs();
function draw(){
 c.clearRect(0,0,W,H);const hy=H*.4;
 const gr=c.createLinearGradient(0,hy,0,H),lc=DAY?"11,111,148":"98,211,252";
 gr.addColorStop(0,"rgba("+lc+",0)");gr.addColorStop(1,"rgba("+lc+",.42)");
 c.lineWidth=1;c.strokeStyle=gr;c.beginPath();
 for(let i=-24;i<=24;i++){c.moveTo(W/2+i*W*.012,hy);c.lineTo(W/2+i*W*.11,H)}
 c.stroke();
 for(let i=0;i<18;i++){const z=(i+t)/18,p=z*z*z,y=hy+(H-hy)*p;
  c.strokeStyle="rgba("+lc+","+(.03+p*.4)+")";c.beginPath();c.moveTo(0,y);c.lineTo(W,y);c.stroke();}
 const hz=c.createLinearGradient(0,hy-60,0,hy+30);
 hz.addColorStop(0,"rgba(179,21,84,0)");hz.addColorStop(.7,"rgba(179,21,84,.16)");hz.addColorStop(1,"rgba(0,0,0,0)");
 c.fillStyle=hz;c.fillRect(0,hy-60,W,90);
 const bg=DAY?"238,243,246":"0,0,0",tp=c.createLinearGradient(0,0,0,hy);
 tp.addColorStop(0,"rgba("+bg+",1)");tp.addColorStop(1,"rgba("+bg+",0)");c.fillStyle=tp;c.fillRect(0,0,W,hy);
}
const still=P.matchMedia("(prefers-reduced-motion: reduce)").matches;
(function loop(){if(P.__feRun!==id)return;t=(t+.006)%1;draw();if(!still)P.requestAnimationFrame(loop)})();
}catch(e){}})();
</script>"""


def header_doc(night: bool) -> str:
    v = NIGHT if night else DAY
    return f"""<style>
:root{{{v};--pink:#B31554}}
*{{box-sizing:border-box}}body{{margin:0;background:transparent;font:13px/1.55 "JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;color:var(--cyan)}}
header{{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;padding:10px 4px 12px;border-bottom:1px solid var(--line)}}
h1{{font:400 26px "Russo One",Impact,sans-serif;letter-spacing:.12em;margin:0;color:var(--ink);{'text-shadow:0 0 14px var(--cyan);' if night else ''}}}
h1 i{{font-style:normal;color:var(--pink)}}
.le{{display:inline-block;width:.8em;height:.8em;margin:0 .03em;vertical-align:-.08em;fill:currentColor;overflow:visible}}
.le .pu{{transform-box:fill-box;transform-origin:50% 0;animation:blink 4s infinite}}.le .hl{{fill:#62D3FC}}
@keyframes blink{{0%,90%,100%{{transform:scaleY(1)}}94%{{transform:scaleY(.05)}}}}
.sub{{color:var(--dim);font-size:11px;letter-spacing:.2em;text-transform:uppercase}}
.badges{{display:flex;flex-wrap:wrap;gap:8px}}
.badge{{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border:1px solid var(--line);border-radius:999px;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;background:rgba(98,211,252,.06)}}
.badge.p{{border-color:var(--pink);color:{'#ff7aa8' if night else 'var(--pink)'};background:rgba(179,21,84,.14)}}
.dot{{width:7px;height:7px;border-radius:50%;background:var(--pink);box-shadow:0 0 8px var(--pink);animation:bl 1.1s infinite}}
@keyframes bl{{50%{{opacity:.15}}}}
@media(prefers-reduced-motion:reduce){{.dot,.le .pu{{animation:none}}}}
</style>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Russo+One&display=swap" rel="stylesheet">
<header><div><h1 aria-label="FORTEYE"><i>FORT</i>{EYE}y{EYE}</h1><div class="sub">AI-аудит кассовой дисциплины по видеозаписям</div></div>
<div class="badges"><span class="badge p"><span class="dot"></span>REC</span><span class="badge">Surveillance</span><span class="badge">Online · API</span></div></header>"""


def embed_html(doc: str, height: int) -> None:
    """st.iframe on new Streamlit, components.html on old (components.html is removed in new releases)."""
    if hasattr(st, "iframe"):
        st.iframe(doc, height=height)
    else:
        components.html(doc, height=height)


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
def _check(r: requests.Response) -> requests.Response:
    if r.status_code == 401:
        raise RuntimeError("Неверный или отсутствующий X-API-Key")
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r


def api_submit(base: str, key: str, name: str, data: bytes) -> str:
    ext = os.path.splitext(name)[1].lower()
    r = requests.post(f"{base}/api/v1/analyze", headers={"X-API-Key": key},
                      files={"file": (name, data, MIME.get(ext, "application/octet-stream"))}, timeout=(10, 900))
    return _check(r).json()["job_id"]


def api_get(base: str, key: str, path: str) -> Dict[str, Any]:
    return _check(requests.get(f"{base}{path}", headers={"X-API-Key": key}, timeout=15)).json()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_rules(base: str, key: str) -> Dict[str, Dict[str, str]]:
    r = requests.get(f"{base}/api/v1/rules", headers={"X-API-Key": key}, timeout=3)
    r.raise_for_status()  # a failure raises, so it is never cached (an empty dict would stick for 5 minutes)
    return {x["type"]: x for x in r.json()}


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health(base: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"{base}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:  # API down: the main flow reports it on submit
        return {}


def health_warnings(h: Dict[str, Any]) -> List[str]:
    caps, out = h.get("capabilities") or {}, []
    if not caps.get("loaded"):  # models still warming up (or API down): nothing to judge yet
        return out
    if caps.get("vlm") in (None, "none") or caps.get("vlm_error"):
        out.append("VLM (Qwen2-VL) не загружена: анализ идёт только по YOLO, правила R01–R03, R09–R11, R16 недоступны. "
                   "Положите папку Qwen2-VL-7B-Instruct в models/ или задайте FORTEYE_VLM_PATH.")
    if not caps.get("custom_classes"):
        out.append("custom_yolo.pt не загружен: сейф, дверь и наличные по YOLO не определяются.")
    return out


def fetch_rules(base: str, key: str) -> Dict[str, Dict[str, str]]:
    try:
        return _fetch_rules(base, key)
    except Exception:  # the API may be down or lack this endpoint: the UI still works without rule metadata
        return {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def fmt(sec: float) -> str:
    sec = max(0.0, float(sec or 0))
    m = int(sec // 60)
    return f"{m:02d}:{sec - m * 60:04.1f}"


def probe_duration(data: bytes, suffix: str) -> float:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return float(n / fps) if fps > 1 and n > 0 else 0.0
    finally:
        os.unlink(path)


def meta_for(v: Dict[str, Any], rules: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    r = rules.get(v["type"], {})
    sev = SEV.get(r.get("severity", "mid"), SEV["mid"])
    return {"code": r.get("code", "—"), "sev_label": sev[0], "sev": sev[1]}


def risk_label(violations: List[Dict[str, Any]], rules: Dict[str, Dict[str, str]]) -> str:
    if not violations:
        return "НИЗКИЙ"
    avg = sum(meta_for(v, rules)["sev"] for v in violations) / len(violations)
    return "ВЫСОКИЙ" if (len(violations) >= 4 or avg > 2.2) else ("СРЕДНИЙ" if avg > 1.5 else "НИЗКИЙ")


def card_html(v: Dict[str, Any], rules: Dict[str, Dict[str, str]]) -> str:
    e, m = html.escape, meta_for(v, rules)
    desc, _, law = v["description"].partition(" Основание: ")
    frames = sorted(v.get("evidence_frames", []), key=lambda f: abs(f["timestamp_sec"] - v["timecode_sec"]))
    shot = thumbs = ""
    if frames:
        f = frames[0]
        shot = (f'<div class="shot"><img alt="" src="data:image/jpeg;base64,{f["image_base64"]}"><div class="tint"></div>'
                f'<span class="tc">T+{fmt(f["timestamp_sec"])}</span><span class="tag">СНИМОК НАРУШЕНИЯ</span></div>')
        if len(frames) > 1:
            items = "".join(f'<div class="th"><img alt="" src="data:image/jpeg;base64,{x["image_base64"]}">'
                            f'<span>{fmt(x["timestamp_sec"])}</span></div>'
                            for x in sorted(frames[1:], key=lambda x: x["timestamp_sec"]))
            thumbs = f'<div class="thumbs">{items}</div>'
    return (f'<div class="v">{shot}{thumbs}<div class="meta"><span class="chip p">⏱ {fmt(v["timecode_sec"])}</span>'
            f'<span class="chip">п. {e(m["code"])}</span><span class="chip p">{e(m["sev_label"])}</span></div>'
            f'<h3>{e(v["type"])}</h3>' + (f'<div class="law">§ {e(law.rstrip("."))}</div>' if law else "")
            + f'<p class="desc">{e(desc)}</p></div>')


def stats_html(count: str, risk: str, dur: str, red: bool) -> str:
    return (f'<div class="stats"><div class="stat"><small>Нарушений</small><b>{count}</b></div>'
            f'<div class="stat"><small>Уровень риска</small><b class="{"r" if red else ""}">{risk}</b></div>'
            f'<div class="stat"><small>Длительность</small><b>{dur}</b></div></div>')


def render_results(slot, result: Optional[Dict[str, Any]], state: str, rules: Dict[str, Dict[str, str]]) -> None:
    dur = fmt(ss.duration) if ss.duration else "—"
    with slot.container():
        status = {"idle": "ОЖИДАНИЕ", "scan": "СКАНИРОВАНИЕ"}.get(state, "")
        if result and result.get("status") == "failed":
            status = "ОШИБКА"
        elif result and result.get("status") == "done":
            status = "ОБНАРУЖЕНО" if result["violations"] else "ЧИСТО"
        st.markdown(f'<div class="ph"><span>// Журнал нарушений</span><span>{status}</span></div>', unsafe_allow_html=True)
        if not result:
            st.markdown(stats_html("0", "—", dur, False)
                        + '<div class="empty">НАРУШЕНИЙ НЕ ОБНАРУЖЕНО<br>запустите анализ</div>', unsafe_allow_html=True)
            return
        if result["status"] == "failed":
            st.markdown(stats_html("0", "—", dur, False)
                        + f'<div class="err">Анализ не удался: {html.escape(result.get("error") or "неизвестная ошибка")}</div>',
                        unsafe_allow_html=True)
            return
        vs = sorted(result["violations"], key=lambda v: v["timecode_sec"])
        st.markdown(stats_html(str(len(vs)), risk_label(vs, rules) if vs else "НИЗКИЙ", dur, bool(vs)), unsafe_allow_html=True)
        if not vs:
            st.markdown('<div class="empty">НАРУШЕНИЙ НЕ ОБНАРУЖЕНО</div>', unsafe_allow_html=True)
        for i, v in enumerate(vs):
            st.markdown(card_html(v, rules), unsafe_allow_html=True)
            if st.button(f"▶ Перейти к {fmt(v['timecode_sec'])}", key=f"seek_{i}_{v['timecode_sec']}"):
                ss.seek = int(v["timecode_sec"])
                st.rerun()
        st.download_button("Скачать JSON", json.dumps(result, ensure_ascii=False), file_name=f"{result['job_id']}.json",
                           mime="application/json")


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
night = bool(ss.night)
st.markdown(f"<style>:root{{{NIGHT if night else DAY}}}{COMMON_CSS}</style>", unsafe_allow_html=True)
try:
    embed_html(GRID_JS.replace("__DAY__", "false" if night else "true"), height=1)
except Exception:  # animated grid is cosmetic
    pass
embed_html(header_doc(night), height=104)
st.toggle("Night Vision", key="night")

left, right = st.columns([1.15, 0.85], gap="medium")
with right:
    right_slot = st.empty()
with left:
    st.markdown('<div class="ph"><span>// Источник видеосигнала</span><span>CAM-01</span></div>', unsafe_allow_html=True)
    up = st.file_uploader("Загрузить видео", type=["mov", "mp4", "m4v", "avi", "mkv"], label_visibility="collapsed")
    if up is not None and (ss.video is None or ss.video["key"] != (up.name, up.size)):
        data = up.getvalue()
        ss.video = {"key": (up.name, up.size), "name": up.name, "data": data}
        ss.duration, ss.seek, ss.result = probe_duration(data, os.path.splitext(up.name)[1]), 0, None
    video = ss.video if up is not None else None
    if video:
        ext = os.path.splitext(video["name"])[1].lower()
        st.video(video["data"], format=MIME.get(ext, "video/mp4"), start_time=int(ss.seek))
        marks = ""
        if ss.result and ss.result.get("status") == "done" and ss.duration:
            marks = "".join(f'<div class="mk" style="left:{min(99.5, v["timecode_sec"] / ss.duration * 100):.2f}%" '
                            f'title="{fmt(v["timecode_sec"])}"></div>' for v in ss.result["violations"])
        st.markdown(f'<div class="hudline"><span class="rec">● REC CAM-01</span><span>{html.escape(video["name"])}</span>'
                    f'<span>{fmt(ss.duration) if ss.duration else "—"}</span></div><div class="tl-bar">{marks}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="stage"><div><b>NO SIGNAL</b>загрузите видеозапись кассовой зоны</div></div>',
                    unsafe_allow_html=True)
    run = st.button("Запустить анализ", type="primary", disabled=video is None)
    prog_slot, log_slot, health_slot = st.empty(), st.empty(), st.empty()
    with st.expander("// Подключение к API"):
        base = st.text_input("API URL", API_URL).rstrip("/")
        key = st.text_input("X-API-Key", API_KEY, type="password")

rules = fetch_rules(base, key)
for w in health_warnings(fetch_health(base)):
    health_slot.markdown(f'<div class="err">⚠ {html.escape(w)}</div>', unsafe_allow_html=True)

if run and video:
    ss.result = None
    render_results(right_slot, None, "scan", rules)
    log: List[str] = []
    try:
        job_id = api_submit(base, key, video["name"], video["data"])
        log.append(f"ЗАДАЧА ПРИНЯТА: {job_id[:8]}")
        t0 = time.time()
        while True:
            job = api_get(base, key, f"/api/v1/jobs/{job_id}")
            prog = api_get(base, key, f"/api/v1/jobs/{job_id}/progress")
            stage = prog.get("stage", "")
            if stage.startswith("Кадр") and log and log[-1].startswith("Кадр"):
                log[-1] = stage
            elif not log or log[-1] != stage:
                log.append(stage)
            pct = int(100 * float(prog.get("progress", 0)))
            prog_slot.markdown(f'<div class="prog"><i style="width:{pct}%"></i></div>', unsafe_allow_html=True)
            log_slot.markdown('<div class="log">' + "".join(f"<div>{html.escape(x)}</div>" for x in log[-5:]) + "</div>",
                              unsafe_allow_html=True)
            if job["status"] in ("done", "failed"):
                ss.result = job
                break
            if time.time() - t0 > UI_TIMEOUT_SEC:
                raise TimeoutError("Превышено время ожидания анализа")
            time.sleep(POLL_SEC)
        if ss.result["status"] == "done":
            log.append(f"АНАЛИЗ ЗАВЕРШЁН. НАРУШЕНИЙ: {len(ss.result['violations'])}")
            log_slot.markdown('<div class="log">' + "".join(f"<div>{html.escape(x)}</div>" for x in log[-5:]) + "</div>",
                              unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        ss.result = {"job_id": "error", "status": "failed", "violations": [], "error": str(exc)}
    if ss.result and ss.result.get("status") == "done":
        st.rerun()  # redraw the page so the violation markers appear on the video timeline

render_results(right_slot, ss.result, "idle", rules)
st.markdown('<div class="ph" style="justify-content:center;margin-top:22px">FORTEYE · ВСЕ ДАННЫЕ ОБРАБАТЫВАЮТСЯ ЛОКАЛЬНО</div>',
            unsafe_allow_html=True)
