"""Streamlit front end for the multi-agent research assistant."""

import os
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

from src.pipeline.pipeline import STAGES, run_research_pipeline_stream

load_dotenv()

st.set_page_config(
    page_title="Mohan · Research Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES = [
    "New LLM models in 2026",
    "Agentic AI in production",
    "Small language models on edge",
]


# ══════════════════════════════════════════════════════════════════════
#  Design system
# ══════════════════════════════════════════════════════════════════════

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:        #07080D;
  --surface-1: #0E1017;
  --surface-2: #141825;
  --surface-3: #1A1F30;
  --line:      rgba(255,255,255,.07);
  --line-soft: rgba(255,255,255,.04);

  --text:   #ECEDF3;
  --muted:  #8B91A8;
  --faint:  #565C70;

  --accent:   #6366F1;
  --accent-2: #8B5CF6;
  --cyan:     #22D3EE;
  --ok:       #34D399;
  --warn:     #FBBF24;
  --danger:   #F87171;

  --grad: linear-gradient(135deg, #6366F1 0%, #8B5CF6 55%, #A78BFA 100%);
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px; --r-xl: 22px;
}

/* ── base ─────────────────────────────────────────────── */
html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 520px at 12% -8%,  rgba(99,102,241,.13), transparent 62%),
    radial-gradient(900px 460px at 92% 0%,   rgba(139,92,246,.10), transparent 60%),
    radial-gradient(700px 420px at 50% 108%, rgba(34,211,238,.055), transparent 65%),
    var(--bg);
}
header[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer { display: none; }
.block-container { padding: 1.4rem 2.2rem 4rem; max-width: 1280px; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #242A3C; border-radius: 9px; }
::-webkit-scrollbar-thumb:hover { background: #313852; }

h1,h2,h3,h4 { font-family: 'Inter', sans-serif; letter-spacing: -.022em; }
code, kbd { font-family: 'JetBrains Mono', monospace !important; }

@keyframes rise { from { opacity:0; transform: translateY(12px);} to {opacity:1; transform:none;} }
.rise   { animation: rise .5s cubic-bezier(.2,.7,.3,1) both; }
.rise-1 { animation-delay: .05s; } .rise-2 { animation-delay: .12s; }
.rise-3 { animation-delay: .19s; } .rise-4 { animation-delay: .26s; }

/* ── top bar ──────────────────────────────────────────── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 .15rem 1.1rem; margin-bottom: .2rem;
}
.brand { display: flex; align-items: center; gap: .65rem; }
.mark {
  width: 30px; height: 30px; border-radius: 9px; background: var(--grad);
  display: grid; place-items: center; color: #fff; font-size: 14px; font-weight: 800;
  box-shadow: 0 5px 18px rgba(99,102,241,.4);
}
.brand .wm { font-size: .97rem; font-weight: 700; letter-spacing: -.01em; }
.brand .wm span { color: var(--faint); font-weight: 500; }
.chips { display: flex; gap: .45rem; }
.chip {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .3rem .68rem; border-radius: 999px;
  border: 1px solid var(--line); background: rgba(20,24,37,.7);
  font-size: .715rem; font-weight: 550; color: var(--muted);
  font-family: 'JetBrains Mono', monospace;
}
.chip i { width: 6px; height: 6px; border-radius: 50%; background: var(--faint); }
.chip.on i  { background: var(--ok);     box-shadow: 0 0 7px rgba(52,211,153,.9); }
.chip.off i { background: var(--danger); box-shadow: 0 0 7px rgba(248,113,113,.9); }

/* ── hero ─────────────────────────────────────────────── */
.hero { padding: 1.5rem 0 1.1rem; }
.eyebrow {
  display: inline-flex; align-items: center; gap: .5rem;
  padding: .3rem .8rem; border-radius: 999px; margin-bottom: 1rem;
  border: 1px solid rgba(99,102,241,.3); background: rgba(99,102,241,.1);
  color: #B9BDFF; font-size: .68rem; font-weight: 650;
  letter-spacing: .13em; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace;
}
.eyebrow b { width: 5px; height: 5px; border-radius: 50%; background: var(--accent);
             animation: blink 2.2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.25;} }

.hero h1 {
  font-size: clamp(2.1rem, 4.4vw, 3.15rem); font-weight: 800;
  line-height: 1.04; margin: 0 0 .7rem; letter-spacing: -.035em;
}
.hero h1 .g {
  background: var(--grad); -webkit-background-clip: text;
  background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { margin: 0; color: var(--muted); font-size: 1.01rem; max-width: 620px; line-height: 1.6; }

/* ── search field ─────────────────────────────────────── */
.st-key-searchbar { margin: 1.5rem 0 .5rem; }
.st-key-searchbar [data-testid="stTextInputRootElement"],
.st-key-searchbar div[data-baseweb="input"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-md) !important;
  transition: border-color .2s ease, box-shadow .2s ease;
}
.st-key-searchbar div[data-baseweb="input"]:focus-within {
  border-color: rgba(99,102,241,.75) !important;
  box-shadow: 0 0 0 4px rgba(99,102,241,.14) !important;
}
.st-key-searchbar input {
  background: transparent !important; color: var(--text) !important;
  font-size: 1rem !important; font-weight: 500 !important;
  padding: .92rem 1rem !important;
}
.st-key-searchbar input::placeholder { color: var(--faint) !important; font-weight: 400; }

/* primary button */
.st-key-searchbar .stButton > button {
  background: var(--grad); color: #fff; border: 0;
  border-radius: var(--r-md); padding: .92rem 1rem;
  font-weight: 650; font-size: .95rem; letter-spacing: -.005em;
  box-shadow: 0 7px 22px rgba(99,102,241,.32);
  transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}
.st-key-searchbar .stButton > button:hover {
  transform: translateY(-1.5px); filter: brightness(1.08);
  box-shadow: 0 11px 30px rgba(99,102,241,.45);
}
.st-key-searchbar .stButton > button:active { transform: translateY(0); }

/* example pills */
.st-key-pills .stButton > button {
  background: rgba(20,24,37,.6); color: var(--muted);
  border: 1px solid var(--line); border-radius: 999px;
  padding: .42rem .9rem; font-size: .79rem; font-weight: 520;
  transition: all .18s ease;
}
.st-key-pills .stButton > button:hover {
  border-color: rgba(99,102,241,.5); color: var(--text);
  background: rgba(99,102,241,.1);
}
.pillrow-label {
  font-size: .715rem; color: var(--faint); font-weight: 550;
  font-family: 'JetBrains Mono', monospace; letter-spacing: .06em;
  text-transform: uppercase; padding-top: .62rem;
}

/* ── glass card shell ─────────────────────────────────── */
.card {
  border: 1px solid var(--line); border-radius: var(--r-lg);
  background: linear-gradient(180deg, rgba(20,24,37,.82), rgba(14,16,23,.82));
  backdrop-filter: blur(10px);
  box-shadow: 0 1px 0 rgba(255,255,255,.04) inset, 0 18px 42px rgba(0,0,0,.34);
}
.card-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: .8rem 1.1rem; border-bottom: 1px solid var(--line-soft);
}
.card-head .t {
  font-size: .72rem; font-weight: 650; color: var(--muted);
  letter-spacing: .1em; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace;
}
.card-head .s { font-size: .72rem; color: var(--faint); font-family: 'JetBrains Mono', monospace; }

/* ── pipeline flow graph ──────────────────────────────── */
.st-key-flowcard {
  border: 1px solid var(--line); border-radius: var(--r-lg);
  background: linear-gradient(180deg, rgba(20,24,37,.8), rgba(14,16,23,.8));
  backdrop-filter: blur(10px); margin-top: 1.5rem; padding: .55rem .7rem .2rem;
  box-shadow: 0 1px 0 rgba(255,255,255,.04) inset, 0 18px 42px rgba(0,0,0,.34);
  animation: rise .5s cubic-bezier(.2,.7,.3,1) both; animation-delay: .12s;
}
.flowwrap { padding: .35rem .2rem 0; overflow-x: auto; }
.flow { display: block; width: 100%; min-width: 900px; height: auto; }

.fl-box   { fill: var(--surface-2); stroke: var(--line); stroke-width: 1.2; transition: all .32s ease; }
.fl-title { fill: var(--muted); font-size: 13px; font-weight: 650; font-family: 'Inter', sans-serif; }
.fl-code  { fill: #5B6079; font-size: 10.5px; font-family: 'JetBrains Mono', monospace; }
.fl-sub   { fill: #565C70; font-size: 9.5px; font-family: 'Inter', sans-serif; }
.fl-tag   { fill: #4A5064; font-size: 9.5px; font-family: 'JetBrains Mono', monospace; }
.fl-data  { fill: #767C93; font-size: 10px; font-family: 'JetBrains Mono', monospace; transition: fill .3s ease; }
.fl-line  { stroke: #262C3E; stroke-width: 1.6; fill: none; transition: stroke .3s ease; }
.fl-tick  { stroke: #1E2332; stroke-width: 1; fill: none; }

.fl-g.active .fl-box   { stroke: var(--accent); stroke-width: 1.8; fill: #171B33;
                         animation: glow 1.7s ease-in-out infinite; }
.fl-g.active .fl-title { fill: #C7CAFF; }
.fl-g.active .fl-code, .fl-g.active .fl-sub { fill: #8E93C4; }
.fl-line.active { stroke: var(--accent); stroke-width: 2; stroke-dasharray: 7 5;
                  animation: dashflow .55s linear infinite; }
.fl-data.active { fill: #A5B4FC; }

.fl-g.done .fl-box   { stroke: rgba(52,211,153,.45); fill: #0F2120; }
.fl-g.done .fl-title { fill: #DDE3E8; }
.fl-g.done .fl-code  { fill: #6B7588; }
.fl-line.done { stroke: rgba(52,211,153,.42); }
.fl-data.done { fill: #7FAE9D; }

@keyframes dashflow { to { stroke-dashoffset: -24; } }
@keyframes glow {
  0%,100% { filter: drop-shadow(0 0 3px rgba(99,102,241,.3)); }
  50%     { filter: drop-shadow(0 0 13px rgba(99,102,241,.8)); }
}
.flowcap {
  display:flex; align-items:center; justify-content:center; gap:.5rem;
  padding: .7rem 0 .85rem; color: var(--faint); font-size: .76rem;
  font-family: 'JetBrains Mono', monospace;
}
.flowcap b { color: #B9BDFF; font-weight: 600; }
.flowcap .sp {
  width: 11px; height: 11px; border-radius: 50%;
  border: 1.8px solid rgba(99,102,241,.25); border-top-color: var(--accent);
  animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── score + stats ────────────────────────────────────── */
.statgrid {
  display: grid; grid-template-columns: 200px repeat(4, 1fr);
  gap: .75rem; margin: 1.3rem 0 .4rem;
}
.scorecard {
  display: flex; align-items: center; gap: .95rem; padding: 1.05rem 1.15rem;
  border: 1px solid var(--line); border-radius: var(--r-lg);
  background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(20,24,37,.85));
}
.scorecard .meta .k {
  font-size: .66rem; color: var(--faint); letter-spacing: .12em;
  text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin-bottom: .2rem;
}
.scorecard .meta .v { font-size: 1.05rem; font-weight: 700; color: var(--text); }
.scorecard .meta .d { font-size: .72rem; color: var(--muted); margin-top: .1rem; }

.stat {
  padding: 1.05rem 1.15rem; border: 1px solid var(--line);
  border-radius: var(--r-lg); background: rgba(20,24,37,.62);
  transition: border-color .22s ease, transform .22s ease;
}
.stat:hover { border-color: rgba(99,102,241,.4); transform: translateY(-2px); }
.stat .k {
  font-size: .66rem; color: var(--faint); letter-spacing: .12em;
  text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
}
.stat .v { font-size: 1.72rem; font-weight: 750; margin: .3rem 0 .05rem; letter-spacing: -.03em; }
.stat .u { font-size: .95rem; color: var(--muted); font-weight: 600; }
.stat .d { font-size: .72rem; color: var(--faint); }

/* ── tabs ─────────────────────────────────────────────── */
.stTabs { margin-top: 1.5rem; }
.stTabs [data-baseweb="tab-list"] {
  gap: .3rem; background: rgba(14,16,23,.7); padding: .32rem;
  border: 1px solid var(--line); border-radius: var(--r-md);
}
.stTabs [data-baseweb="tab-list"] button {
  background: transparent; border-radius: var(--r-sm);
  padding: .5rem 1rem; color: var(--muted);
  font-size: .845rem; font-weight: 600; transition: all .18s ease;
}
.stTabs [data-baseweb="tab-list"] button:hover { color: var(--text); background: rgba(255,255,255,.035); }
.stTabs [aria-selected="true"] {
  background: var(--surface-3) !important; color: var(--text) !important;
  box-shadow: 0 1px 0 rgba(255,255,255,.06) inset;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ── report paper ─────────────────────────────────────── */
.st-key-paper {
  border: 1px solid var(--line); border-radius: var(--r-lg);
  background: linear-gradient(180deg, rgba(18,21,32,.9), rgba(12,14,21,.9));
  padding: 2.1rem 2.5rem; margin-top: 1rem;
}
.st-key-paper p, .st-key-paper li {
  font-size: .945rem; line-height: 1.78; color: #C9CEDD;
}
.st-key-paper h1 {
  font-size: 1.72rem; font-weight: 750; color: var(--text);
  margin: 0 0 .4rem; padding-bottom: .85rem; border-bottom: 1px solid var(--line);
}
.st-key-paper h2 {
  font-size: 1.12rem; font-weight: 700; color: var(--text);
  margin: 2rem 0 .55rem; padding-left: .7rem; border-left: 2.5px solid var(--accent);
}
.st-key-paper h3 { font-size: .98rem; font-weight: 650; color: #C7CAFF; margin: 1.3rem 0 .35rem; }
.st-key-paper a { color: #A5B4FC; text-decoration: none; border-bottom: 1px solid rgba(165,180,252,.28); }
.st-key-paper a:hover { border-bottom-color: #A5B4FC; }
.st-key-paper strong { color: var(--text); font-weight: 650; }
.st-key-paper hr { border-color: var(--line); margin: 1.8rem 0; }

.st-key-review { border: 1px solid var(--line); border-radius: var(--r-lg);
                 background: rgba(18,21,32,.86); padding: 1.6rem 1.9rem; margin-top: 1rem; }
.st-key-review p, .st-key-review li { font-size: .9rem; line-height: 1.72; color: #BFC5D4; }
.st-key-review h1, .st-key-review h2, .st-key-review h3 { font-size: 1rem; color: var(--text); }
.st-key-review code {
  background: rgba(99,102,241,.12); color: #C7CAFF;
  padding: .1rem .35rem; border-radius: 5px; font-size: .85em;
}

/* ── sources ──────────────────────────────────────────── */
.srcgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px,1fr)); gap: .65rem; margin-top: 1rem; }
.src {
  display: flex; gap: .8rem; align-items: flex-start;
  padding: .85rem 1rem; border: 1px solid var(--line);
  border-radius: var(--r-md); background: rgba(20,24,37,.6);
  transition: all .2s ease; text-decoration: none;
}
.src:hover { border-color: rgba(99,102,241,.5); background: rgba(99,102,241,.08); transform: translateY(-2px); }
.src .n {
  flex: 0 0 24px; height: 24px; border-radius: 7px;
  background: rgba(99,102,241,.16); color: #A5B4FC;
  display: grid; place-items: center;
  font-size: .68rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
}
.src .b { min-width: 0; }
.src .dom { font-size: .845rem; font-weight: 620; color: var(--text); margin-bottom: .12rem; }
.src .pth {
  font-size: .715rem; color: var(--faint); font-family: 'JetBrains Mono', monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* ── token usage ──────────────────────────────────────── */
.utotal {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: .65rem; margin-top: 1rem;
}
.utotal > div {
  padding: .9rem 1.05rem; border: 1px solid var(--line);
  border-radius: var(--r-md); background: rgba(20,24,37,.62);
}
.utotal .k {
  font-size: .64rem; color: var(--faint); letter-spacing: .12em;
  text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
}
.utotal .v {
  font-size: 1.42rem; font-weight: 750; margin-top: .25rem;
  letter-spacing: -.028em; font-family: 'JetBrains Mono', monospace;
}
.ulegend {
  display: flex; gap: 1.1rem; margin: .95rem 0 .5rem;
  font-size: .72rem; color: var(--faint); font-family: 'JetBrains Mono', monospace;
}
.ulegend span { display: inline-flex; align-items: center; gap: .4rem; }
.ulegend i { width: 9px; height: 9px; border-radius: 3px; display: inline-block; }
.ulegend i.in  { background: var(--accent); }
.ulegend i.out { background: var(--cyan); }

.utable { border: 1px solid var(--line); border-radius: var(--r-md); overflow: hidden; }
.urow {
  display: grid; grid-template-columns: 200px 1fr 110px 90px;
  gap: .9rem; align-items: center;
  padding: .85rem 1.1rem; border-bottom: 1px solid var(--line-soft);
  background: rgba(20,24,37,.4); transition: background .18s ease;
}
.urow:last-child { border-bottom: 0; }
.urow:hover { background: rgba(99,102,241,.07); }
.uname { font-size: .865rem; font-weight: 620; color: var(--text); }
.umeta {
  display: block; font-size: .68rem; color: var(--faint); font-weight: 500;
  font-family: 'JetBrains Mono', monospace; margin-top: .15rem;
}
.ubar {
  display: flex; height: 8px; border-radius: 999px; overflow: hidden;
  background: rgba(255,255,255,.05);
}
.ubar .in  { background: linear-gradient(90deg, #6366F1, #818CF8); }
.ubar .out { background: linear-gradient(90deg, #22D3EE, #67E8F9); }
.unum { text-align: right; font-family: 'JetBrains Mono', monospace; }
.unum b { display: block; font-size: .95rem; font-weight: 700; color: var(--text); }
.unum span { font-size: .67rem; color: var(--faint); }
.usplit {
  text-align: right; font-size: .68rem; color: var(--faint); line-height: 1.5;
  font-family: 'JetBrains Mono', monospace;
}

/* ── trace ────────────────────────────────────────────── */
.tracenote {
  border-left: 2px solid var(--accent); border-radius: 0 var(--r-sm) var(--r-sm) 0;
  background: rgba(99,102,241,.07); padding: .8rem 1.05rem; margin: 1rem 0 .9rem;
  font-size: .83rem; color: var(--muted); line-height: 1.6;
}
.stExpander details {
  border: 1px solid var(--line) !important; border-radius: var(--r-md) !important;
  background: rgba(20,24,37,.5) !important; margin-bottom: .45rem;
}
.stExpander summary { font-size: .83rem !important; font-weight: 600 !important; }

/* ── empty state ──────────────────────────────────────── */
.empty { text-align: center; padding: 2.4rem 1rem 1rem; }
.empty .ic { font-size: 1.5rem; opacity: .45; margin-bottom: .5rem; }
.empty .t { color: var(--muted); font-size: .9rem; font-weight: 600; }
.empty .s { color: var(--faint); font-size: .8rem; margin-top: .25rem; }

/* ── sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B0D14, #07080D);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.sb-h {
  font-size: .66rem; font-weight: 650; color: var(--faint);
  letter-spacing: .14em; text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace; margin: 1.3rem 0 .6rem;
}
.sb-step { display: flex; gap: .62rem; align-items: flex-start; padding: .36rem 0; }
.sb-step .i {
  flex: 0 0 19px; height: 19px; border-radius: 6px;
  background: rgba(99,102,241,.13); color: #A5B4FC;
  display: grid; place-items: center; font-size: .63rem; font-weight: 700;
  font-family: 'JetBrains Mono', monospace; margin-top: 1px;
}
.sb-step .n { font-size: .805rem; font-weight: 600; color: #C3C9DA; line-height: 1.3; }
.sb-step .d { font-size: .715rem; color: var(--faint); line-height: 1.35; margin-top: .05rem; }
.sb-kv {
  display: flex; justify-content: space-between; align-items: center;
  padding: .36rem 0; font-size: .765rem; border-bottom: 1px solid var(--line-soft);
}
.sb-kv .k { color: var(--faint); }
.sb-kv .v { color: #C3C9DA; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: .735rem; }

/* ── download row ─────────────────────────────────────── */
.st-key-dl .stDownloadButton > button {
  background: rgba(20,24,37,.75); color: var(--muted);
  border: 1px solid var(--line); border-radius: var(--r-md);
  padding: .62rem 1rem; font-size: .83rem; font-weight: 600;
  transition: all .18s ease; width: 100%;
}
.st-key-dl .stDownloadButton > button:hover {
  border-color: rgba(99,102,241,.55); color: var(--text); background: rgba(99,102,241,.1);
}

.foot {
  text-align: center; color: #3E4356; font-size: .72rem;
  margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--line-soft);
  font-family: 'JetBrains Mono', monospace;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════
#  Pipeline flow graph (SVG)
# ══════════════════════════════════════════════════════════════════════

ROW_Y, ROW_H = 168, 104
TOOL_Y, TOOL_H = 30, 62
MID = ROW_Y + ROW_H / 2

COL = {
    "input": (16, 120), "search": (180, 176), "reader": (398, 176),
    "write": (616, 168), "critic": (826, 168), "out": (1022, 112),
}
ARROWS = {"idle": "#262C3E", "active": "#6366F1", "done": "rgba(52,211,153,.42)"}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _box(key, state, y, h, lines, tip):
    x, w = COL[key]
    cx = x + w / 2
    body = "".join(
        f'<text x="{cx}" y="{y + dy}" text-anchor="middle" class="{cls}">{_esc(txt)}</text>'
        for txt, cls, dy in lines
    )
    return (f'<g class="fl-g {state}"><title>{_esc(tip)}</title>'
            f'<rect class="fl-box" x="{x}" y="{y}" width="{w}" height="{h}" rx="12"/>{body}</g>')


def _harrow(x1, x2, state):
    return (f'<line class="fl-line {state}" x1="{x1}" y1="{MID}" x2="{x2 - 9}" y2="{MID}" '
            f'marker-end="url(#ar-{state or "idle"})"/>')


def _tool(key, state, title, sub1, sub2):
    x, w = COL[key]
    box = _box(key, state, TOOL_Y, TOOL_H,
               [(title, "fl-code", 24), (sub1, "fl-sub", 40), (sub2, "fl-sub", 53)],
               f"{title} — invoked by the agent below; result returns as a ToolMessage")
    call_x, res_x = x + 46, x + w - 46
    tip, bot = TOOL_Y + TOOL_H, ROW_Y
    mk = f'url(#ar-{state or "idle"})'
    return (box
            + f'<line class="fl-line {state}" x1="{call_x}" y1="{bot}" x2="{call_x}" y2="{tip + 9}" marker-end="{mk}"/>'
            + f'<line class="fl-line {state}" x1="{res_x}" y1="{tip}" x2="{res_x}" y2="{bot - 9}" marker-end="{mk}"/>'
            + f'<text x="{call_x - 6}" y="{(tip + bot) / 2 + 3}" text-anchor="end" class="fl-tag">call</text>'
            + f'<text x="{res_x + 6}" y="{(tip + bot) / 2 + 3}" class="fl-tag">result</text>')


def _flow_label(x, text, state):
    return (f'<line class="fl-tick" x1="{x}" y1="{MID + 14}" x2="{x}" y2="{MID + 62}"/>'
            f'<text x="{x}" y="{MID + 78}" text-anchor="middle" class="fl-data {state}">{_esc(text)}</text>')


def render_flow(status: dict) -> str:
    """Render the pipeline as an animated SVG. status: stage key -> ''|'active'|'done'."""
    st_ = lambda k: status.get(k, "")
    gap = lambda a, b: (COL[a][0] + COL[a][1] + COL[b][0]) / 2

    edges = (
        _harrow(COL["input"][0] + COL["input"][1], COL["search"][0], st_("search"))
        + _harrow(COL["search"][0] + COL["search"][1], COL["reader"][0], st_("read"))
        + _harrow(COL["reader"][0] + COL["reader"][1], COL["write"][0], st_("write"))
        + _harrow(COL["write"][0] + COL["write"][1], COL["critic"][0], st_("critic"))
        + _harrow(COL["critic"][0] + COL["critic"][1], COL["out"][0], st_("critic"))
    )
    labels = (
        _flow_label(gap("input", "search"), "topic", st_("search"))
        + _flow_label(gap("search", "reader"), "search_results", st_("read"))
        + _flow_label(gap("reader", "write"), "scraped_content", st_("write"))
        + _flow_label(gap("write", "critic"), "report", st_("critic"))
    )
    nodes = (
        _box("input", "done" if st_("search") else "", ROW_Y + 16, ROW_H - 32,
             [("Topic", "fl-title", 26), ("from the UI", "fl-sub", 42)], "The topic entered above")
        + _box("search", st_("search"), ROW_Y, ROW_H,
               [("Search Agent", "fl-title", 30), ("tools=[web_search]", "fl-code", 50),
                ("loops until it can answer", "fl-sub", 70)],
               "create_agent — calls web_search, re-reads the result, then writes a findings brief")
        + _box("reader", st_("read"), ROW_Y, ROW_H,
               [("Reader Agent", "fl-title", 30), ("tools=[scrape_url]", "fl-code", 50),
                ("picks a URL, scrapes it", "fl-sub", 70)],
               "create_agent — chooses the best article and extracts its full substance")
        + _box("write", st_("write"), ROW_Y, ROW_H,
               [("Writer Chain", "fl-title", 30), ("prompt | llm | parser", "fl-code", 50),
                ("one model call, no tools", "fl-sub", 70)],
               "LCEL chain — drafts the cited report in a single pass")
        + _box("critic", st_("critic"), ROW_Y, ROW_H,
               [("Critic Chain", "fl-title", 30), ("prompt | llm | parser", "fl-code", 50),
                ("scores it out of 10", "fl-sub", 70)],
               "LCEL chain — grades the draft against a five-point rubric")
        + _box("out", "done" if st_("critic") == "done" else st_("critic"), ROW_Y + 16, ROW_H - 32,
               [("Results", "fl-title", 26), ("report + score", "fl-sub", 42)], "Rendered in the tabs below")
    )
    tools = (_tool("search", st_("search"), "web_search", "Tavily API search", "snippets + URLs")
             + _tool("reader", st_("read"), "scrape_url", "HTTP GET · 15s timeout", "3 extractors · 5 000 chars"))
    defs = "".join(
        f'<marker id="ar-{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
        f'markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>'
        for n, c in ARROWS.items()
    )
    return ('<div class="flowwrap">'
            '<svg class="flow" viewBox="0 0 1150 330" preserveAspectRatio="xMidYMid meet" '
            'xmlns="http://www.w3.org/2000/svg">'
            f"<defs>{defs}</defs>{edges}{labels}{tools}{nodes}</svg></div>")


# ══════════════════════════════════════════════════════════════════════
#  Components
# ══════════════════════════════════════════════════════════════════════

def score_ring(score, size=70):
    """Circular gauge for the critic score."""
    r, sw = size / 2 - 6, 5.5
    circ = 2 * 3.14159 * r
    pct = (score / 10) if score is not None else 0
    color = "#34D399" if pct >= .75 else "#FBBF24" if pct >= .45 else "#F87171"
    label = f"{score:g}" if score is not None else "—"
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="flex:0 0 auto">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="{sw}"/>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-dasharray="{circ * pct:.1f} {circ:.1f}" '
        f'transform="rotate(-90 {size/2} {size/2})" style="transition:stroke-dasharray .9s cubic-bezier(.2,.7,.3,1)"/>'
        f'<text x="50%" y="52%" text-anchor="middle" dominant-baseline="middle" '
        f'fill="{color}" font-size="19" font-weight="750" font-family="Inter,sans-serif">{label}</text>'
        f"</svg>"
    )


def stats_strip(result) -> str:
    score = result.get("score")
    report = result.get("report", "")
    words = len(report.split())
    verdict = ("Strong draft" if score and score >= 7.5
               else "Needs depth" if score and score >= 4.5
               else "Weak sourcing" if score else "Not scored")
    total = result.get("usage_total", {})
    tokens = total.get("total", 0)
    cards = [
        ("Sources", f"{len(result.get('sources', []))}", "", "distinct URLs cited"),
        ("Tokens", f"{tokens/1000:.1f}" if tokens >= 1000 else f"{tokens}",
         "k" if tokens >= 1000 else "", f"{total.get('calls', 0)} model calls"),
        ("Report", f"{words:,}", "words", f"~{max(1, round(words / 220))} min read"),
        ("Runtime", f"{result.get('duration', 0):g}", "s", "4 stages end to end"),
    ]
    tiles = "".join(
        f'<div class="stat"><div class="k">{k}</div>'
        f'<div class="v">{v}<span class="u"> {u}</span></div><div class="d">{d}</div></div>'
        for k, v, u, d in cards
    )
    return (
        '<div class="statgrid rise rise-1">'
        f'<div class="scorecard">{score_ring(score)}'
        f'<div class="meta"><div class="k">Critic score</div>'
        f'<div class="v">{verdict}</div><div class="d">out of 10</div></div></div>'
        f"{tiles}</div>"
    )


def sources_grid(urls) -> str:
    if not urls:
        return ('<div class="empty"><div class="ic">◇</div><div class="t">No sources recovered</div>'
                '<div class="s">The search returned no usable URLs for this topic.</div></div>')
    cards = []
    for i, url in enumerate(urls, 1):
        p = urlparse(url)
        dom = p.netloc.replace("www.", "") or url
        path = (p.path + ("?" + p.query if p.query else "")) or "/"
        cards.append(
            f'<a class="src" href="{_esc(url)}" target="_blank" rel="noopener">'
            f'<div class="n">{i:02d}</div><div class="b">'
            f'<div class="dom">{_esc(dom)}</div><div class="pth">{_esc(path)}</div></div></a>'
        )
    return f'<div class="srcgrid">{"".join(cards)}</div>'


def usage_panel(result) -> str:
    """Per-agent token breakdown with proportional bars."""
    usage = result.get("usage", {})
    total = result.get("usage_total", {})
    grand = max(total.get("total", 0), 1)
    tool_calls = result.get("tool_calls", {})
    labels = {key: label for key, label, _ in STAGES}

    rows = []
    for key, _, _ in STAGES:
        u = usage.get(key)
        if not u:
            continue
        pct = u["total"] / grand * 100
        inp_pct = u["input"] / max(u["total"], 1) * 100
        tools = tool_calls.get(key, 0)
        # Built separately: nesting same-type quotes inside an f-string needs 3.12+.
        meta = f'{u["calls"]} model call{"s" if u["calls"] != 1 else ""}'
        if tools:
            meta += f' · {tools} tool call{"s" if tools != 1 else ""}'
        rows.append(
            f'<div class="urow">'
            f'  <div class="uname">{labels[key]}'
            f'    <span class="umeta">{meta}</span></div>'
            f'  <div class="ubar"><span class="in" style="width:{inp_pct:.1f}%"></span>'
            f'<span class="out" style="width:{100 - inp_pct:.1f}%"></span></div>'
            f'  <div class="unum"><b>{u["total"]:,}</b><span>{pct:.0f}% of run</span></div>'
            f'  <div class="usplit">{u["input"]:,} in<br/>{u["output"]:,} out</div>'
            f"</div>"
        )

    legend = (
        '<div class="ulegend"><span><i class="in"></i>input (prompt + tool results)</span>'
        '<span><i class="out"></i>output (generated)</span></div>'
    )
    summary = (
        f'<div class="utotal">'
        f'<div><div class="k">Total tokens</div><div class="v">{total.get("total", 0):,}</div></div>'
        f'<div><div class="k">Input</div><div class="v">{total.get("input", 0):,}</div></div>'
        f'<div><div class="k">Output</div><div class="v">{total.get("output", 0):,}</div></div>'
        f'<div><div class="k">Model calls</div><div class="v">{total.get("calls", 0)}</div></div>'
        f"</div>"
    )
    note = (
        '<div class="tracenote">Counts come from the provider\'s own '
        "<code>usage_metadata</code>, aggregated per stage. Agents make several model "
        "calls inside one run — one per tool-loop turn — so an agent costs more than its "
        "single visible answer suggests.</div>"
    )
    return f'{summary}{legend}<div class="utable">{"".join(rows)}</div>{note}'


def sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand" style="margin-bottom:.3rem">'
            '<div class="mark">◈</div><div class="wm">Lumen<br/>'
            '<span style="font-size:.7rem">Research Assistant</span></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-h">Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(
                f'<div class="sb-step"><div class="i">{i}</div>'
                f'<div><div class="n">{label}</div><div class="d">{desc}</div></div></div>'
                for i, (_, label, desc) in enumerate(STAGES, 1)
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-h">Stack</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(
                f'<div class="sb-kv"><span class="k">{k}</span><span class="v">{v}</span></div>'
                for k, v in [
                    ("Model", "gemini-3.1-flash-lite"), ("Framework", "LangChain 1.4"),
                    ("Search", "Tavily"), ("Extract", "trafilatura"),
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-h">Credentials</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(
                f'<div class="sb-kv"><span class="k">{name}</span>'
                f'<span class="v" style="color:{"#34D399" if ok else "#F87171"}">'
                f'{"connected" if ok else "missing"}</span></div>'
                for name, ok in [
                    ("GEMINI_API_KEY", bool(os.getenv("GEMINI_API_KEY"))),
                    ("TAVILY_API_KEY", bool(os.getenv("TAVILY_API_KEY"))),
                ]
            ),
            unsafe_allow_html=True,
        )


def topbar():
    keys = [("gemini", bool(os.getenv("GEMINI_API_KEY"))), ("tavily", bool(os.getenv("TAVILY_API_KEY")))]
    chips = "".join(f'<div class="chip {"on" if ok else "off"}"><i></i>{n}</div>' for n, ok in keys)
    st.markdown(
        '<div class="topbar rise">'
        '<div class="brand"><div class="mark">◈</div>'
        '<div class="wm">Lumen <span>/ multi-agent research</span></div></div>'
        f'<div class="chips">{chips}</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  Run + results
# ══════════════════════════════════════════════════════════════════════

NETWORK_HINTS = (
    "RemoteDisconnected", "Connection aborted", "ConnectionError", "ConnectionReset",
    "Max retries exceeded", "Temporary failure in name resolution", "timed out",
    "ServiceUnavailable", "DeadlineExceeded", "503", "502",
)
AUTH_HINTS = ("API key", "api_key", "Unauthorized", "401", "403", "PermissionDenied")
QUOTA_HINTS = ("quota", "RateLimit", "429", "ResourceExhausted", "UsageLimit")


def describe_failure(exc: Exception) -> str:
    """Turn a raw exception into something the user can act on."""
    text = f"{type(exc).__name__}: {exc}"

    if any(h.lower() in text.lower() for h in NETWORK_HINTS):
        return (
            "**Network dropped mid-run.** The connection to an upstream API closed before "
            "it answered. This is almost always transient — press **Research** again.\n\n"
            "If it keeps happening, check your internet connection, a VPN or proxy, or "
            f"whether the provider is having an outage.\n\n`{text}`"
        )
    if any(h.lower() in text.lower() for h in QUOTA_HINTS):
        return (
            "**Rate limit or quota reached.** Wait a minute and try again, or check your "
            f"Gemini / Tavily plan limits.\n\n`{text}`"
        )
    if any(h.lower() in text.lower() for h in AUTH_HINTS):
        return (
            "**Credentials rejected.** Check `GEMINI_API_KEY` and `TAVILY_API_KEY` in your "
            f".env file, then restart the app so it reloads them.\n\n`{text}`"
        )
    return f"**Pipeline failed.**\n\n`{text}`"


def run_pipeline(topic: str, tracker, live):
    status = {key: "" for key, _, _ in STAGES}
    labels = {key: label for key, label, _ in STAGES}
    order = [key for key, _, _ in STAGES]
    result = None

    for event in run_research_pipeline_stream(topic):
        stage = event.get("stage")
        if event["type"] == "start":
            status[stage] = "active"
            tracker.markdown(render_flow(status), unsafe_allow_html=True)
            live.markdown(
                f'<div class="flowcap"><span class="sp"></span>'
                f'Step {order.index(stage) + 1} / {len(order)} &nbsp;·&nbsp; '
                f"<b>{labels[stage]}</b> running …</div>",
                unsafe_allow_html=True,
            )
        elif event["type"] == "done":
            status[stage] = "done"
            tracker.markdown(render_flow(status), unsafe_allow_html=True)
            live.markdown(
                f'<div class="flowcap"><b>{labels[stage]}</b> &nbsp;·&nbsp; '
                f'{len(event["content"]):,} chars &nbsp;·&nbsp; '
                f'{event["usage"]["total"]:,} tokens</div>',
                unsafe_allow_html=True,
            )
        elif event["type"] == "finished":
            result = event["state"]

    live.empty()
    return result


def render_results(result: dict):
    report = result.get("report", "")
    st.markdown(stats_strip(result), unsafe_allow_html=True)

    tokens = result.get("usage_total", {}).get("total", 0)
    tab_report, tab_review, tab_src, tab_usage, tab_trace = st.tabs(
        ["Report", "Critic Review", f"Sources · {len(result.get('sources', []))}",
         f"Token Usage · {tokens:,}", "Agent Trace"]
    )

    with tab_report:
        with st.container(key="paper"):
            st.markdown(report or "_No report generated._")

    with tab_review:
        with st.container(key="review"):
            st.markdown(result.get("feedback", "_No review generated._"))

    with tab_src:
        st.markdown(sources_grid(result.get("sources", [])), unsafe_allow_html=True)

    with tab_usage:
        st.markdown(usage_panel(result), unsafe_allow_html=True)

    with tab_trace:
        st.markdown(
            '<div class="tracenote"><b>Why this tab exists.</b> A LangChain agent run ends with '
            "the model's <i>summary of</i> the tool result, not the tool result itself — the raw "
            "ToolMessage sits earlier in the message list and is normally discarded. "
            "Both are shown here, so the compression is measurable.</div>",
            unsafe_allow_html=True,
        )
        for title, key in [
            ("Search · raw tool output", "search_raw"),
            ("Search · agent summary", "search_results"),
            ("Reader · raw scrape", "scraped_raw"),
            ("Reader · agent summary", "scraped_content"),
            ("Extra sources · scraped directly (no LLM)", "_extra"),
        ]:
            if key == "_extra":
                extra = result.get("extra_sources", [])
                body = "\n\n".join(
                    f"--- {item['url']} ---\n{item['content']}" for item in extra
                ) or "(none gathered)"
                title = f"{title}  —  {len(extra)} source(s)"
            else:
                body = result.get(key) or ""
            with st.expander(f"{title}   —   {len(body):,} chars"):
                st.text(body or "(empty)")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    slug = "".join(c if c.isalnum() else "-" for c in result["topic"]).strip("-").lower()[:40]
    with st.container(key="dl"):
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button(
            "↓  Report (.md)", data=f"# {result['topic']}\n\n{report}\n",
            file_name=f"report-{slug}-{stamp}.md", mime="text/markdown", use_container_width=True)
        c2.download_button(
            "↓  Full run (.md)",
            data=(f"# {result['topic']}\n\n{report}\n\n---\n\n## Critic Review\n\n"
                  f"{result.get('feedback','')}\n\n---\n\n## Sources\n\n"
                  + "\n".join(f"- {u}" for u in result.get("sources", []))),
            file_name=f"research-{slug}-{stamp}.md", mime="text/markdown", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════

def main():
    st.markdown(STYLES, unsafe_allow_html=True)
    sidebar()
    topbar()

    st.markdown(
        '<div class="hero rise rise-1">'
        '<div class="eyebrow"><b></b>4 agents · live pipeline</div>'
        '<h1>Research anything.<br/><span class="g">Sourced, scored, in a minute.</span></h1>'
        "<p>Enter a topic. A search agent sweeps the live web, a reader agent extracts the "
        "best source, a writer drafts a cited report — and a critic grades it honestly.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("topic", "")
    st.session_state.setdefault("result", None)
    if "example_pick" in st.session_state:
        st.session_state.topic = st.session_state.pop("example_pick")

    with st.container(key="searchbar"):
        field, button = st.columns([5, 1], vertical_alignment="center")
        topic = field.text_input(
            "Research topic", key="topic",
            placeholder="What should I research?", label_visibility="collapsed",
        )
        run = button.button("Research  →", type="primary", use_container_width=True)

    with st.container(key="pills"):
        cols = st.columns([0.9] + [1.5] * len(EXAMPLES) + [2.2], vertical_alignment="center")
        cols[0].markdown('<div class="pillrow-label">Try</div>', unsafe_allow_html=True)
        for col, example in zip(cols[1:], EXAMPLES):
            if col.button(example, use_container_width=True):
                st.session_state.example_pick = example
                st.rerun()

    # Keyed container -> .st-key-flowcard, so the card actually wraps the graph.
    # An inline "<div>" via st.markdown would be auto-closed by the browser instead.
    with st.container(key="flowcard"):
        tracker, live = st.empty(), st.empty()

    if run:
        if not topic.strip():
            st.warning("Enter a topic to begin.")
        elif not (os.getenv("GEMINI_API_KEY") and os.getenv("TAVILY_API_KEY")):
            st.error("Missing credentials — set GEMINI_API_KEY and TAVILY_API_KEY in your .env file.")
        else:
            st.session_state.result = None
            try:
                st.session_state.result = run_pipeline(topic.strip(), tracker, live)
            except Exception as exc:
                live.empty()
                st.error(describe_failure(exc))

    result = st.session_state.result
    if result:
        tracker.markdown(render_flow({k: "done" for k, _, _ in STAGES}), unsafe_allow_html=True)
        render_results(result)
    elif not run:
        tracker.markdown(render_flow({k: "" for k, _, _ in STAGES}), unsafe_allow_html=True)
        live.markdown(
            '<div class="flowcap">idle · hover any node for detail</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="foot">LangChain · Gemini · Tavily — '
        "the critic's score is unedited</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
