import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import os
import json
import pickle
from collections import Counter

st.set_page_config(
    page_title="Random Forest Phishing Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── ICON LIBRARY ─────────────────────────────────────────────────────────────
def icon(name: str, size: int = 15, color: str = "currentColor") -> str:
    lib = {
        "shield":         f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "cpu":            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>',
        "database":       f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        "bar-chart":      f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
        "layers":         f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "alert-triangle": f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "check-circle":   f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "mail":           f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>',
    }
    return lib.get(name, "")


# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
STYLES = """
<style>
/* ── DESIGN TOKENS — ChatGPT palette ───────────────────────────────────── */
:root {
    --bg:         #212121;
    --surface:    #2f2f2f;
    --surface-2:  #3a3a3a;
    --ink:        #ececec;
    --ink-2:      #c5c5d2;
    --ink-3:      #8e8ea0;
    --accent:     #10a37f;
    --accent-h:   #0e9270;
    --accent-lt:  rgba(16,163,127,0.15);
    --acc-txt:    #10a37f;
    --danger:     #ef4444;
    --danger-lt:  rgba(239,68,68,0.12);
    --border:     #424242;
    --sb-bg:      #171717;
    --r:          12px;
    --shadow-sm:  0 1px 3px rgba(0,0,0,0.4);
    --shadow-md:  0 4px 16px rgba(0,0,0,0.5);
}

/* ── BASE ───────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Söhne', ui-sans-serif, system-ui, -apple-system,
                 BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif !important;
}
body    { background: var(--bg) !important; color: var(--ink) !important; }
.stApp  { background: var(--bg) !important; }

#MainMenu, footer, header { visibility: hidden; }

/* ── SIDEBAR — layout/width via JS ────────────────────────────────────── */
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div * {
    max-width: 100% !important;
    box-sizing: border-box !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .block-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="headerNoPadding"],
section[data-testid="stSidebar"] button[data-testid^="baseButton"],
.st-emotion-cache-1rs6os {
    display: none !important;
    visibility: hidden !important;
}

/* ── MAIN CONTENT ───────────────────────────────────────────────────────── */
.block-container {
    padding-top: 36px !important;
    padding-bottom: 48px !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}

/* ── APP HEADER ─────────────────────────────────────────────────────────── */
.app-header { margin-bottom: 28px; }

.app-name {
    font-size: 23px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 7px;
}
.app-tagline {
    font-size: 14px;
    color: var(--ink-3);
    line-height: 1.55;
}


/* ── TEXT AREA ──────────────────────────────────────────────────────────── */
.stTextArea textarea {
    font-family: 'Söhne', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 13px 15px !important;
    color: var(--ink) !important;
    background: var(--surface-2) !important;
    resize: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(16,163,127,0.18) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--ink-3) !important; }
.stTextArea label { display: none !important; }

/* ── BUTTONS ────────────────────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1px !important;
    padding: 10px 20px !important;
    box-shadow: 0 1px 4px rgba(16,163,127,0.28) !important;
    transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    background: var(--accent-h) !important;
    box-shadow: 0 3px 10px rgba(16,163,127,0.30) !important;
    transform: translateY(-1px) !important;
}
[data-testid="baseButton-primary"]:active {
    transform: translateY(0) !important;
}

[data-testid="baseButton-secondary"],
button[kind="secondary"] {
    background: #3a3a3a !important;
    color: #c5c5d2 !important;
    border: 1.5px solid #424242 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
[data-testid="baseButton-secondary"]:hover,
button[kind="secondary"]:hover {
    border-color: #5a5a5a !important;
    background: #444444 !important;
    color: #ececec !important;
}

/* ── RESULT CARD ────────────────────────────────────────────────────────── */
.result-wrap {
    margin-top: 20px;
    animation: slideUp 0.22s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-card {
    border-radius: var(--r);
    border: 1px solid var(--border);
    overflow: hidden;
    box-shadow: var(--shadow-md);
}
.result-card.spam { border-left: 4px solid var(--danger); }
.result-card.ham  { border-left: 4px solid var(--accent); }

.result-top {
    padding: 16px 20px 15px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 9px;
}
.result-verdict {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.2px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.result-verdict.spam { color: var(--danger); }
.result-verdict.ham  { color: var(--acc-txt); }

.result-body {
    padding: 16px 20px 20px;
    background: var(--surface);
}

.result-desc {
    font-size: 14px;
    color: var(--ink-2);
    line-height: 1.68;
}

/* ── EMPTY STATE ────────────────────────────────────────────────────────── */
.empty-state {
    margin-top: 24px;
    text-align: center;
    padding: 52px 24px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    box-shadow: var(--shadow-sm);
}
.empty-icon {
    width: 48px;
    height: 48px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px;
}
.empty-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--ink-2);
    margin-bottom: 7px;
    letter-spacing: -0.2px;
}
.empty-sub {
    font-size: 14px;
    color: var(--ink-3);
    line-height: 1.65;
    max-width: 320px;
    margin: 0 auto;
}

/* ── SIDEBAR COMPONENTS ────────────────────────────────────────────────── */
.sb-header {
    padding: 18px 16px 16px;
    border-bottom: 1px solid var(--border);
}
.sb-brand-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.3px;
    line-height: 1.2;
}
.sb-brand-sub {
    font-size: 11px;
    color: var(--ink-3);
    font-weight: 400;
    margin-top: 1px;
}
.sb-section {
    padding: 13px 14px 15px;
    border-bottom: 1px solid var(--border);
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
}
.sb-section-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    font-weight: 700;
    color: var(--ink-3);
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-bottom: 12px;
    user-select: none;
}
.sb-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 3px 0;
    gap: 8px;
}
.sb-key {
    font-size: 12px;
    color: var(--ink-3);
    font-weight: 400;
    flex-shrink: 0;
}
.sb-val {
    font-size: 12px;
    color: var(--ink);
    font-weight: 600;
    text-align: right;
    max-width: 60%;
    line-height: 1.4;
}

.metric-pct { font-size: 12px; color: var(--acc-txt); font-weight: 700;
              font-variant-numeric: tabular-nums; }

/* Pipeline tags */
.pipeline-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 2px; }
.pipeline-tag {
    font-size: 11px;
    color: var(--ink-2);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 3px 7px;
    font-weight: 500;
}

/* Warning */
.stAlert { border-radius: 8px !important; font-size: 14px !important; }
div[data-testid="stAlert"],
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] * { color: #111827 !important; }

/* ── SIDEBAR — dark bg, token colors apply naturally ───────────────────── */

/* ── MOBILE ─────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container {
        padding-top: 24px !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
        padding-bottom: 32px !important;
        max-width: 100% !important;
    }
    .app-name    { font-size: 20px !important; }
    .app-tagline { font-size: 13px !important; }
    /* 16px minimum prevents iOS auto-zoom on textarea focus */
    .stTextArea textarea { font-size: 16px !important; }
    .empty-state { padding: 32px 16px !important; }
    .result-top  { padding: 14px 16px !important; }
    .result-body { padding: 14px 16px 18px !important; }
    .result-verdict { font-size: 15px !important; }
    #rf-hb { top: 10px !important; }
}
@media (max-width: 480px) {
    .block-container {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }
    .app-name { font-size: 18px !important; }
    /* Stack Analyse + Clear buttons vertically */
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 8px !important; }
    div[data-testid="column"]            { min-width: 100% !important; width: 100% !important; flex: none !important; }
    /* Hide spacer gap column */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) { display: none !important; }
    .stButton > button { width: 100% !important; min-height: 44px !important; }
    .empty-state { padding: 28px 12px !important; }
    .empty-title { font-size: 15px !important; }
    .result-card { border-left-width: 3px !important; }
}
</style>
"""


# ── HAMBURGER / SIDEBAR JS ────────────────────────────────────────────────────
HAMBURGER_HTML = """
<!DOCTYPE html><html><body style="margin:0;padding:0;overflow:hidden;">
<script>
(function () {
    var pdoc   = window.parent;
    var doc    = pdoc.document;
    var SB_W   = 278;
    var SB_C   = 50;
    var BG     = '#171717';
    var OPEN_K = '__rfSidebarOpen';
    var INIT_K = '__rfHamburgerReady';

    var styleEl = doc.getElementById('rf-sb-style');
    if (!styleEl) {
        styleEl = doc.createElement('style');
        styleEl.id = 'rf-sb-style';
        doc.head.appendChild(styleEl);
    }

    if (pdoc[OPEN_K] === undefined) pdoc[OPEN_K] = true;

    var TR = 'min-width .28s cubic-bezier(.4,0,.2,1),max-width .28s cubic-bezier(.4,0,.2,1)';

    function isMobile() { return pdoc.innerWidth <= 768; }

    function applyCSS(open) {
        var mobile = isMobile();
        var w = open ? SB_W : (mobile ? 0 : SB_C);

        styleEl.textContent = [
            'section[data-testid="stSidebar"]{',
            '  display:block!important;visibility:visible!important;',
            '  min-width:'+w+'px!important;max-width:'+w+'px!important;width:'+w+'px!important;',
            '  background-color:'+BG+'!important;',
            '  border-right:'+(w>0?'1px solid #424242':'none')+'!important;',
            '  overflow:hidden!important;transition:'+TR+'!important;',
            '}',
            'section[data-testid="stSidebar"] div,',
            'section[data-testid="stSidebar"] section{',
            '  max-width:'+w+'px!important;box-sizing:border-box!important;',
            '}',
            'section[data-testid="stSidebar"]>div{',
            '  opacity:'+(open?'1':'0')+'!important;',
            '  transition:opacity .2s ease!important;',
            '  pointer-events:'+(open?'auto':'none')+'!important;',
            '}'
        ].join('');

        var sb = doc.querySelector('[data-testid="stSidebar"]');
        if (!sb) return;

        sb.style.setProperty('display',          'block',          'important');
        sb.style.setProperty('visibility',       'visible',        'important');
        sb.style.setProperty('min-width',         w+'px',          'important');
        sb.style.setProperty('max-width',         w+'px',          'important');
        sb.style.setProperty('width',             w+'px',          'important');
        sb.style.setProperty('background-color',  BG,              'important');
        sb.style.setProperty('overflow',         'hidden',         'important');
        sb.style.setProperty('transform',        'translateX(0)',  'important');

        if (mobile && open) {
            sb.style.setProperty('position', 'fixed',  'important');
            sb.style.setProperty('top',      '0',      'important');
            sb.style.setProperty('left',     '0',      'important');
            sb.style.setProperty('height',   '100vh',  'important');
            sb.style.setProperty('z-index',  '9999',   'important');
        } else {
            sb.style.removeProperty('position');
            sb.style.removeProperty('height');
            sb.style.removeProperty('z-index');
        }

        var bd = doc.getElementById('rf-bd');
        if (mobile && open && !bd) {
            bd = doc.createElement('div');
            bd.id = 'rf-bd';
            bd.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;' +
                               'background:rgba(0,0,0,0.4);z-index:9998;';
            bd.onclick = function () { pdoc.rfToggle(); };
            doc.body.appendChild(bd);
        } else if ((!mobile || !open) && bd) {
            bd.parentNode.removeChild(bd);
        }

        var el = sb.firstElementChild;
        for (var i = 0; i < 3 && el; i++) {
            el.style.setProperty('max-width',  w+'px',       'important');
            el.style.setProperty('width',      '100%',       'important');
            el.style.setProperty('overflow',   'hidden',     'important');
            el.style.setProperty('box-sizing', 'border-box', 'important');
            if (i === 0) {
                el.style.setProperty('opacity',        open ? '1' : '0',    'important');
                el.style.setProperty('pointer-events', open ? 'auto':'none', 'important');
            }
            el = el.firstElementChild;
        }
    }

    var BTN  = 32;
    var MENU = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#c5c5d2" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>';

    function btnLeft(open) {
        if (open) return SB_W - BTN - 9;
        return isMobile() ? 12 : Math.round((SB_C - BTN) / 2);
    }

    function positionBtn(open) {
        var btn = doc.getElementById('rf-hb');
        if (btn) btn.style.left = btnLeft(open) + 'px';
    }

    pdoc.rfToggle = function () {
        pdoc[OPEN_K] = !pdoc[OPEN_K];
        applyCSS(pdoc[OPEN_K]);
        positionBtn(pdoc[OPEN_K]);
    };

    applyCSS(pdoc[OPEN_K]);
    setTimeout(function () { applyCSS(pdoc[OPEN_K]); }, 300);
    setTimeout(function () { applyCSS(pdoc[OPEN_K]); }, 800);

    if (pdoc[INIT_K]) return;
    pdoc[INIT_K] = true;

    function createBtn() {
        if (doc.getElementById('rf-hb')) return;
        var btn = doc.createElement('button');
        btn.id        = 'rf-hb';
        btn.title     = 'Toggle sidebar';
        btn.innerHTML = MENU;
        btn.style.cssText = [
            'position:fixed', 'top:10px',
            'left:' + btnLeft(pdoc[OPEN_K]) + 'px',
            'z-index:999999',
            'width:' + BTN + 'px', 'height:' + BTN + 'px',
            'border:1px solid #424242',
            'border-radius:7px',
            'background:#2f2f2f',
            'cursor:pointer',
            'display:flex', 'align-items:center', 'justify-content:center',
            'box-shadow:0 1px 3px rgba(0,0,0,0.08)',
            'padding:0', 'outline:none',
            'transition:left .28s cubic-bezier(.4,0,.2,1),box-shadow .15s,border-color .15s'
        ].join(';');
        btn.onmouseenter = function () {
            btn.style.boxShadow   = '0 2px 8px rgba(0,0,0,0.5)';
            btn.style.borderColor = '#5a5a5a';
        };
        btn.onmouseleave = function () {
            btn.style.boxShadow   = '0 1px 3px rgba(0,0,0,0.4)';
            btn.style.borderColor = '#424242';
        };
        btn.onclick = function () { pdoc.rfToggle(); };
        doc.body.appendChild(btn);
    }

    function hideNativeToggle() {
        ['[data-testid="collapsedControl"]',
         '[data-testid="stSidebarCollapsedControl"]',
         'button[data-testid="baseButton-headerNoPadding"]'
        ].forEach(function (s) {
            doc.querySelectorAll(s).forEach(function (el) {
                el.style.setProperty('display', 'none', 'important');
            });
        });
    }

    function fixSecondaryButtons() {
        doc.querySelectorAll('[data-testid="baseButton-secondary"]').forEach(function(btn) {
            btn.style.setProperty('background-color', '#3a3a3a', 'important');
            btn.style.setProperty('color', '#c5c5d2', 'important');
            btn.style.setProperty('border', '1.5px solid #424242', 'important');
            btn.style.setProperty('border-radius', '8px', 'important');
            btn.onmouseenter = function() {
                btn.style.setProperty('background-color', '#444444', 'important');
                btn.style.setProperty('border-color', '#5a5a5a', 'important');
            };
            btn.onmouseleave = function() {
                btn.style.setProperty('background-color', '#3a3a3a', 'important');
                btn.style.setProperty('border-color', '#424242', 'important');
            };
        });
    }

    var raf = null;
    new MutationObserver(function () {
        if (raf) return;
        raf = pdoc.requestAnimationFrame(function () {
            raf = null;
            applyCSS(pdoc[OPEN_K]);
            createBtn();
            positionBtn(pdoc[OPEN_K]);
            hideNativeToggle();
            fixSecondaryButtons();
        });
    }).observe(doc.body, { childList: true, subtree: true });

    pdoc.addEventListener('resize', function () {
        applyCSS(pdoc[OPEN_K]);
        positionBtn(pdoc[OPEN_K]);
    });

    if (doc.readyState === 'loading') {
        doc.addEventListener('DOMContentLoaded', createBtn);
    } else {
        setTimeout(createBtn, 100);
    }
})();
</script>
</body></html>
"""


# ── MODEL LOADING  (cached — loads pre-trained pickles, runs in ~1 second) ────
@st.cache_resource(show_spinner=False)
def load_resources():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    # Point NLTK to bundled data in repo — no network downloads at runtime
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nltk.data.path.insert(0, os.path.join(base_dir, "nltk_data"))

    with open(os.path.join(base_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(base_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    return {
        "model":          model,
        "vectorizer":     tfidf,
        "stop_words":     set(stopwords.words("english")),
        "frequent_words": set(meta["frequent_words"]),
        "rare_words":     set(meta["rare_words"]),
        "stemmer":        PorterStemmer(),
        "lemmatizer":     WordNetLemmatizer(),
        "metrics":        meta["metrics"],
        "stats":          meta["stats"],
    }


# ── PREDICT ───────────────────────────────────────────────────────────────────
def predict_email(text: str, res: dict):
    from nltk.tokenize import word_tokenize

    sw  = res["stop_words"]
    fw  = res["frequent_words"]
    rw  = res["rare_words"]
    stm = res["stemmer"]
    lem = res["lemmatizer"]

    t = str(text)
    t = re.sub(r"<[^>]+>",                   " ", t)
    t = re.sub(r"https?://\S+|www\.\S+",     " ", t)
    t = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " ", t)
    t = re.sub(r"[^\w\s]",                    " ", t)
    t = re.sub(r"\d+",                        " ", t)
    t = re.sub(r"[^a-zA-Z\s]",               " ", t)
    t = re.sub(r"\s+",                        " ", t).strip().lower()
    t = " ".join(w for w in t.split() if w not in sw)
    t = " ".join(w for w in t.split() if w not in fw)
    t = " ".join(w for w in t.split() if w not in rw)

    tokens    = word_tokenize(t)
    tokens    = [stm.stem(w)      for w in tokens]
    tokens    = [lem.lemmatize(w) for w in tokens]
    processed = " ".join(tokens)

    vec   = res["vectorizer"].transform([processed])
    pred  = res["model"].predict(vec)[0]
    proba = res["model"].predict_proba(vec)[0]
    return int(pred), proba


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar(res: dict) -> None:
    m = res["metrics"]
    s = res["stats"]

    with st.sidebar:
        st.markdown("""
        <div class="sb-header">
            <div class="sb-brand-name">RF Phishing Detector</div>
            <div class="sb-brand-sub">Random Forest · v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sb-section">
            <div class="sb-section-title">{icon("cpu",12,"#8e8ea0")} &nbsp; Model</div>
            <div class="sb-row"><span class="sb-key">Algorithm</span><span class="sb-val">Random Forest</span></div>
            <div class="sb-row"><span class="sb-key">Estimators</span><span class="sb-val">100 trees</span></div>
            <div class="sb-row"><span class="sb-key">Split</span><span class="sb-val">80% / 20%</span></div>
            <div class="sb-row"><span class="sb-key">Train samples</span><span class="sb-val">{s["train"]:,}</span></div>
            <div class="sb-row"><span class="sb-key">Test samples</span><span class="sb-val">{s["test"]:,}</span></div>
            <div class="sb-row"><span class="sb-key">Task</span><span class="sb-val">Binary Classification</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sb-section">
            <div class="sb-section-title">{icon("database",12,"#8e8ea0")} &nbsp; Dataset</div>
            <div class="sb-row"><span class="sb-key">Total emails</span><span class="sb-val">{s["total"]:,}</span></div>
            <div class="sb-row"><span class="sb-key">Phishing</span><span class="sb-val">{s["spam"]:,}</span></div>
            <div class="sb-row"><span class="sb-key">Legitimate</span><span class="sb-val">{s["ham"]:,}</span></div>
            <div class="sb-row"><span class="sb-key">Source</span><span class="sb-val">Kaggle</span></div>
        </div>
        """, unsafe_allow_html=True)

        _icon = icon("bar-chart", 12, "#8e8ea0")
        _bars = ""
        for label, key in [("Accuracy","accuracy"),("Precision","precision"),
                            ("Recall","recall"),("F1-Score","f1")]:
            v = m[key]
            _bars += (
                f'<div class="sb-row">'
                f'<span class="sb-key">{label}</span>'
                f'<span class="metric-pct">{v}%</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="sb-section" style="border-bottom:none;"><div class="sb-section-title">'
            f'{_icon} &nbsp; Performance</div>{_bars}</div>',
            unsafe_allow_html=True
        )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown(STYLES, unsafe_allow_html=True)
    components.html(HAMBURGER_HTML, height=0, scrolling=False)

    if "result"    not in st.session_state: st.session_state.result    = None
    if "input_key" not in st.session_state: st.session_state.input_key = 0

    with st.spinner("Setting up Random Forest Phishing Detector — please wait on first launch..."):
        res = load_resources()

    render_sidebar(res)

    # ── App header
    st.markdown("""
    <div class="app-header">
        <div class="app-name">Random Forest Phishing Detector</div>
        <div class="app-tagline">
            Paste any email below to instantly detect whether it is a phishing email or a legitimate email.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input
    email_input = st.text_area(
        label="Email Content",
        placeholder="Paste the email subject line or body text here...",
        height=130,
        key=f"email_input_{st.session_state.input_key}",
        label_visibility="collapsed",
    )

    # ── Action buttons
    col_a, col_gap, col_c = st.columns([3.5, 0.2, 1.3])
    with col_a:
        analyse_btn = st.button("Analyse Email", use_container_width=True, type="primary")
    with col_c:
        clear_btn = st.button("Clear", use_container_width=True, type="secondary")

    if clear_btn:
        st.session_state.result    = None
        st.session_state.input_key += 1
        st.rerun()

    if analyse_btn:
        if not email_input or not email_input.strip():
            st.warning("Please paste some email text before clicking Analyse Email.")
        else:
            with st.spinner("Analysing..."):
                pred, _ = predict_email(email_input.strip(), res)
            st.session_state.result = {"pred": pred}
            st.rerun()

    # ── Result or empty state
    if st.session_state.result is not None:
        pred = st.session_state.result["pred"]

        if pred == 1:
            card_cls  = "spam"
            icon_html = icon("alert-triangle", 17, "#C0392B")
            verdict   = f'<span style="display:flex;align-items:center;gap:8px;">{icon_html} Phishing Email Detected</span>'
            desc      = ("This email exhibits characteristics commonly associated with phishing — "
                         "suspicious links, spoofed senders, urgency cues, or requests for "
                         "personal and financial information. Do not click any links or reply.")
        else:
            card_cls  = "ham"
            icon_html = icon("check-circle", 17, "#10a37f")
            verdict   = f'<span style="display:flex;align-items:center;gap:8px;">{icon_html} Legitimate Email</span>'
            desc      = ("This email does not exhibit typical phishing patterns. Based on the "
                         "content and language features analysed by the model, it appears to be "
                         "a legitimate email.")

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-card {card_cls}">
                <div class="result-top">
                    <div class="result-verdict {card_cls}">{verdict}</div>
                </div>
                <div class="result-body">
                    <div class="result-desc">{desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-icon">{icon("mail", 22, "#8e8ea0")}</div>
            <div class="empty-title">No email analysed yet</div>
            <div class="empty-sub">
                Paste an email above and click Analyse Email to get an instant classification result.
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
