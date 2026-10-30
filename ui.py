"""Shared look for both pages: colors, CSS, and small HTML components."""
import html
import re

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

C = {
    "bg": "#0A0F1A", "surface": "#111827", "surface2": "#172033", "border": "#243049",
    "text": "#E5ECF5", "muted": "#8A97AD", "accent": "#2BD9A0",
    "home": "#4C8DFF", "draw": "#8A97AD", "away": "#F5A524", "win": "#22C55E", "loss": "#EF4444",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Cairo:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 4rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
.kicker {{ color: {C['accent']}; font-size: 12px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; }}
.hero-title {{ font-size: 38px; font-weight: 700; line-height: 1.1; margin: 6px 0 6px; color: {C['text']}; }}
.hero-sub {{ color: {C['muted']}; font-size: 15px; margin-bottom: 18px; }}
.pill {{ display:inline-block; padding: 3px 10px; border: 1px solid {C['border']}; border-radius: 999px;
        color: {C['muted']}; font-size: 12px; margin: 0 6px 6px 0; background: {C['surface']}; }}
.card {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 14px; padding: 18px 20px; }}
.card h4 {{ margin: 0 0 12px; font-size: 13px; color: {C['muted']}; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }}
.vs {{ display:flex; align-items:center; justify-content:space-between; gap: 16px; }}
.vs .team {{ font-size: 26px; font-weight: 700; }}
.vs .mid {{ color: {C['muted']}; font-size: 13px; text-align:center; }}
.vs .score {{ font-size: 34px; font-weight: 700; color: {C['text']}; }}
.bar {{ display:flex; height: 12px; border-radius: 999px; overflow: hidden; margin: 14px 0 10px; }}
.legend {{ display:flex; justify-content:space-between; font-size: 13px; color: {C['muted']}; }}
.legend b {{ color: {C['text']}; font-size: 20px; display:block; }}
.badges span {{ display:inline-flex; width: 26px; height: 26px; border-radius: 7px; align-items:center; justify-content:center;
               font-size: 12px; font-weight: 700; margin-right: 5px; color: #0A0F1A; }}
.row {{ display:flex; justify-content:space-between; padding: 7px 0; border-bottom: 1px solid {C['border']}; font-size: 14px; }}
.row:last-child {{ border-bottom: none; }}
.row .k {{ color: {C['muted']}; }}
.games div {{ font-size: 13px; padding: 5px 0; color: {C['text']}; border-bottom: 1px dashed {C['border']}; }}
.games div:last-child {{ border-bottom: none; }}
.games .d {{ color: {C['muted']}; margin-right: 8px; font-variant-numeric: tabular-nums; }}
.tile {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 12px; padding: 14px 16px; }}
.tile .l {{ color: {C['muted']}; font-size: 12px; }}
.tile .v {{ font-size: 26px; font-weight: 700; margin-top: 2px; }}
.tile .s {{ font-size: 12px; margin-top: 2px; }}
.up {{ color: {C['win']}; }} .down {{ color: {C['loss']}; }}
.check {{ display:flex; gap:10px; align-items:center; padding: 10px 14px; border-radius: 10px; font-size: 14px;
         background: rgba(43,217,160,.08); border: 1px solid rgba(43,217,160,.35); color: {C['text']}; }}
.check.bad {{ background: rgba(239,68,68,.08); border-color: rgba(239,68,68,.4); }}
.report p {{ font-size: 15px; line-height: 1.75; color: {C['text']}; margin: 0 0 10px; }}
.report .rh {{ color: {C['accent']}; font-size: 13px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; margin: 18px 0 6px; }}
.report .rh:first-child {{ margin-top: 0; }}
.report ul {{ margin: 0 0 6px; padding-inline-start: 20px; }}
.report li {{ font-size: 15px; line-height: 1.7; color: {C['text']}; margin-bottom: 4px; }}
.rtl .rh {{ letter-spacing: 0; font-size: 16px; }}
.rtl li {{ font-size: 16px; }}
.report b {{ color: {C['accent']}; font-weight: 600; }}
.rtl {{ direction: rtl; text-align: right; font-family: 'Cairo', sans-serif; }}
.rtl p {{ font-size: 16px; }}
.list-item {{ display:flex; justify-content:space-between; align-items:center; padding: 9px 0; border-bottom: 1px solid {C['border']}; }}
.list-item:last-child {{ border-bottom: none; }}
.list-item .n {{ font-weight: 600; }} .list-item .x {{ color: {C['muted']}; font-size: 13px; }}
button[data-testid="stBaseButton-primary"], div.stButton > button[kind="primary"] {{
         background: {C['accent']}; border: none; border-radius: 10px; height: 46px; }}
button[data-testid="stBaseButton-primary"] p, div.stButton > button[kind="primary"] p {{
         color: #04140E !important; font-weight: 700; font-size: 15px; }}
button[data-testid="stBaseButton-primary"]:hover {{ background: #5BE6B8; }}
[data-testid="stTabs"] button p {{ font-size: 14px; font-weight: 600; }}
[data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
</style>
"""

pio.templates["pitch"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font={"family": "Inter, sans-serif", "color": C["text"], "size": 13},
    title={"font": {"size": 15, "color": C["text"]}, "x": 0, "xanchor": "left"},
    colorway=[C["accent"], C["home"], C["away"], "#B07CFF", "#FF6B8B", "#7DD3FC", "#FACC15", "#94A3B8"],
    xaxis={"gridcolor": C["border"], "zerolinecolor": C["border"], "linecolor": C["border"]},
    yaxis={"gridcolor": C["border"], "zerolinecolor": C["border"], "linecolor": C["border"]},
    legend={"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 1, "xanchor": "right"},
    margin={"l": 10, "r": 10, "t": 60, "b": 10},
    polar={"bgcolor": "rgba(0,0,0,0)", "radialaxis": {"gridcolor": C["border"], "linecolor": C["border"]},
           "angularaxis": {"gridcolor": C["border"], "linecolor": C["border"]}},
))
pio.templates.default = "pitch"

e = html.escape


def setup(title: str, icon: str):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def hero(kicker: str, title: str, subtitle: str, pills: list[str] = ()):
    chips = "".join(f'<span class="pill">{e(p)}</span>' for p in pills)
    st.markdown(f'<div class="kicker">{e(kicker)}</div><div class="hero-title">{e(title)}</div>'
                f'<div class="hero-sub">{e(subtitle)}</div>{chips}', unsafe_allow_html=True)


def html_block(content: str):
    st.markdown(content, unsafe_allow_html=True)


def badges(letters: str) -> str:
    color = {"W": C["win"], "D": C["draw"], "L": C["loss"]}
    return '<div class="badges">' + "".join(
        f'<span style="background:{color[x]}">{x}</span>' for x in letters[::-1]) + "</div>"


def matchup(home: str, away: str, m: dict, date: str) -> str:
    ph, pd_, pa = (m["p_home_win"] * 100, m["p_draw"] * 100, m["p_away_win"] * 100)
    return f"""
<div class="card">
  <div class="vs">
    <div class="team" style="color:{C['home']}">{e(home)}</div>
    <div class="mid">Most likely score<div class="score">{e(m['most_likely_score'])}</div>{e(date)}</div>
    <div class="team" style="color:{C['away']};text-align:right">{e(away)}</div>
  </div>
  <div class="bar">
    <div style="width:{ph}%;background:{C['home']}"></div>
    <div style="width:{pd_}%;background:{C['draw']}"></div>
    <div style="width:{pa}%;background:{C['away']}"></div>
  </div>
  <div class="legend">
    <div>{e(home)} win<b>{ph:.1f}%</b></div>
    <div style="text-align:center">Draw<b>{pd_:.1f}%</b></div>
    <div style="text-align:right">{e(away)} win<b>{pa:.1f}%</b></div>
  </div>
  <div class="legend" style="margin-top:10px">
    <div>Expected goals {m['expected_goals_home']}</div><div>Expected goals {m['expected_goals_away']}</div>
  </div>
</div>"""


def team_card(team: str, color: str, table: dict | None, form: dict, record: dict, venue: str) -> str:
    rows = [("League position", table["position"] if table else "–"), ("Points", table["pts"] if table else "–"),
            ("Goals for / against", f"{table['gf']} / {table['ga']}" if table else "–"),
            (f"{venue} goals scored per game", record.get("scored") if record.get("played") else "–"),
            (f"{venue} goals conceded per game", record.get("conceded") if record.get("played") else "–")]
    rows_html = "".join(f'<div class="row"><span class="k">{k}</span><span>{v}</span></div>' for k, v in rows)
    games = "".join(f'<div><span class="d">{e(g[:10])}</span>{e(g[11:])}</div>' for g in form["games"])
    return f"""
<div class="card" style="border-top:3px solid {color}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <div style="font-size:20px;font-weight:700">{e(team)}</div>{badges(form['last_results'])}
  </div>
  {rows_html}
  <h4 style="margin-top:16px">Last 5 matches</h4><div class="games">{games}</div>
</div>"""


def tile(label: str, value: str, delta: float | None = None, good_when_up: bool = True, note: str = "vs avg") -> str:
    sub = ""
    if delta is not None:
        good = (delta >= 0) == good_when_up
        sub = f'<div class="s {"up" if good else "down"}">{"▲" if delta >= 0 else "▼"} {abs(delta):.2f} {note}</div>'
    return f'<div class="tile"><div class="l">{e(label)}</div><div class="v">{e(value)}</div>{sub}</div>'


def tiles(items: list[str]):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(item, unsafe_allow_html=True)


ISO_OPEN, ISO_CLOSE = "\u2068", "\u2069"  # bidi isolates used by the template writer
LATIN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .'’&;\-–:%/]*[A-Za-z0-9%]|[A-Za-z0-9]")


def report_html(markdown: str, rtl: bool = False) -> str:
    """Render the report's small markdown subset (### headings, - bullets, **bold**) as a styled card."""

    def inline(t: str) -> str:
        t = e(t.replace(ISO_OPEN, "").replace(ISO_CLOSE, ""), quote=False)
        if rtl:  # isolate English names, dates and scores so they don't scramble the Arabic sentence
            t = LATIN_RUN.sub(lambda m: f'<span dir="ltr" style="unicode-bidi:isolate;white-space:nowrap">{m.group(0)}</span>', t)
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)

    out, in_list = [], False
    for line in markdown.splitlines():
        line = line.strip()
        is_item = line.startswith(("- ", "* ", "• "))
        if in_list and not is_item:
            out.append("</ul>")
            in_list = False
        if not line or line.startswith("## "):
            continue
        if line.startswith("#"):
            out.append(f"<div class=\"rh\">{inline(line.lstrip('#').strip())}</div>")
        elif is_item:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
        else:
            out.append(f"<p>{inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return f'<div class="card report {"rtl" if rtl else ""}">{"".join(out)}</div>'
