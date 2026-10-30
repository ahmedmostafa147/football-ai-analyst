"""Analysis agent (LLM) + validation agent (fact checker), orchestrated with a retry loop."""
import json
import os
import re

MODEL = os.getenv("ANALYST_MODEL", "claude-sonnet-5-5")

SYSTEM = """You are a professional football match analyst writing a pre-match report.
Use ONLY the facts in the provided JSON context. Never invent players, injuries, or numbers.
Every number you write must appear in the context (probabilities may be written as percentages,
e.g. 0.456 -> 45.6%). Write two sections with these exact headings:
## English
## العربية
Each section: key facts, form, home/away records, head-to-head, tactical reading of the stats,
model prediction with probabilities, and a one-line verdict. Keep each section under 250 words."""


def _client():
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic()


class AnalysisAgent:
    def __init__(self):
        self.client = _client()

    @property
    def mode(self) -> str:
        return f"llm:{MODEL}" if self.client else "template (set ANTHROPIC_API_KEY for LLM mode)"

    def run(self, ctx: dict, feedback: list[str] | None = None) -> str:
        if not self.client:
            return template_report(ctx)
        prompt = "Context:\n" + json.dumps(ctx, ensure_ascii=False, default=str)
        if feedback:
            prompt += "\n\nYour previous draft failed fact-checking. Fix these issues:\n- " + "\n- ".join(feedback)
        msg = self.client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class ValidationAgent:
    """Deterministic fact checker: every number and score in the report must be grounded in the context."""

    NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")
    SCORE = re.compile(r"(?<![\d.])(\d+)\s*[-–]\s*(\d+)(?![\d.])")
    ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    def allowed_numbers(self, ctx: dict) -> set[str]:
        raw = json.dumps(ctx, default=str)
        allowed = set(re.findall(r"\d+(?:\.\d+)?", raw))
        for v in list(allowed):
            f = float(v)
            allowed |= {f"{f:g}", str(round(f)), f"{f:.1f}", f"{f:.2f}"}
            if f <= 1:  # probability -> percentage forms
                allowed |= {f"{f * 100:.1f}%", f"{f * 100:.0f}%", f"{f * 100:g}%"}
        allowed |= {str(i) for i in range(0, 11)}  # small counts ("last 5", "2 wins") are ok
        return allowed

    def run(self, report: str, ctx: dict) -> dict:
        text = re.sub(r"\d{4}-\d{2}-\d{2}", "", report.translate(self.ARABIC_DIGITS))
        allowed = self.allowed_numbers(ctx)
        issues = []
        for n in self.NUM.findall(text):
            if n not in allowed and n.rstrip("%") not in allowed:
                issues.append(f"Ungrounded number: {n}")
        known_scores = set(self.SCORE.findall(json.dumps(ctx, default=str)))
        for s in self.SCORE.findall(text):
            if s not in known_scores and tuple(reversed(s)) not in known_scores:
                issues.append(f"Ungrounded score: {s[0]}-{s[1]}")
        for team in (ctx["match"]["home"], ctx["match"]["away"]):
            if team not in report:
                issues.append(f"Team not mentioned: {team}")
        for heading in ("## English", "## العربية"):
            if heading not in report:
                issues.append(f"Missing section: {heading}")
        return {"passed": not issues, "issues": sorted(set(issues))}


def orchestrate(ctx: dict, max_attempts: int = 3) -> dict:
    analyst, checker = AnalysisAgent(), ValidationAgent()
    feedback = None
    for attempt in range(1, max_attempts + 1):
        report = analyst.run(ctx, feedback)
        verdict = checker.run(report, ctx)
        if verdict["passed"]:
            break
        feedback = verdict["issues"]
    return {"report": report, "validation": verdict, "attempts": attempt, "mode": analyst.mode}


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def template_report(ctx: dict) -> str:
    """Offline fallback writer so the pipeline runs end-to-end without an API key."""
    h, a = ctx["match"]["home"], ctx["match"]["away"]
    t, f, m = ctx["table"], ctx["form"], ctx["model"]
    hr, ar = ctx["home_record"], ctx["away_record"]
    iso = lambda s: f"⁨{s}⁩"  # keeps English names and scores intact inside Arabic text
    fav = max(("home", "draw", "away"), key=lambda k: m[f"p_{k}_win" if k != "draw" else "p_draw"])
    fav_en = {"home": f"{h} win", "draw": "a draw", "away": f"{a} win"}[fav]
    fav_ar = {"home": f"فوز {iso(h)}", "draw": "التعادل", "away": f"فوز {iso(a)}"}[fav]
    form_ar = lambda team: " · ".join({"W": "فوز", "D": "تعادل", "L": "خسارة"}[x] for x in f[team]["last_results"])
    hh = ctx["head_to_head"]
    recent = hh["recent"][:3]
    if hh["meetings"]:
        h2h = (f"{hh['meetings']} meetings: {h} won {hh[f'{h}_wins']}, {hh['draws']} draws, {a} won {hh[f'{a}_wins']}. "
               f"Latest: " + "; ".join(recent))
        h2h_ar = (f"{hh['meetings']} مواجهة: فاز {iso(h)} {hh[f'{h}_wins']}، تعادل {hh['draws']}، فاز {iso(a)} {hh[f'{a}_wins']}. "
                  f"الأحدث: " + "، ".join(iso(g) for g in recent))
    else:
        h2h, h2h_ar = "no previous meetings in the data", "لا توجد مواجهات سابقة في البيانات"
    pos = lambda team: t[team]["position"] if t[team] else "-"
    pts = lambda team: t[team]["pts"] if t[team] else "-"

    def record_en(r: dict) -> str:
        parts = [f"{r['scored']} scored and {r['conceded']} conceded per game"]
        if r.get("shots_on_target") is not None:
            parts.append(f"{r['shots_on_target']} shots on target")
        if r.get("xg_for") is not None:
            parts.append(f"xG {r['xg_for']} for / {r['xg_against']} against")
        return ", ".join(parts)

    def record_ar(r: dict) -> str:
        parts = [f"يسجل {r['scored']} ويستقبل {r['conceded']} في المباراة"]
        if r.get("shots_on_target") is not None:
            parts.append(f"{r['shots_on_target']} تسديدة على المرمى")
        if r.get("xg_for") is not None:
            parts.append(f"أهداف متوقعة {r['xg_for']} له و{r['xg_against']} عليه")
        return "، ".join(parts)

    en = f"""## English
**{h} vs {a}** — {ctx['match']['date']}

**Table:** {h} are {pos(h)} with {pts(h)} points; {a} are {pos(a)} with {pts(a)} points.

**Form (last 5):** {h} {f[h]['last_results']} ({f[h]['points']} pts) · {a} {f[a]['last_results']} ({f[a]['points']} pts).

**{h} at home:** {record_en(hr)}.

**{a} away:** {record_en(ar)}.

**Head-to-head:** {h2h}.

**Model:** expected goals {m['expected_goals_home']} - {m['expected_goals_away']}. {h} win {_pct(m['p_home_win'])}, draw {_pct(m['p_draw'])}, {a} win {_pct(m['p_away_win'])}. Most likely score {m['most_likely_score']}.

**Verdict:** the numbers lean towards {fav_en}.
"""
    ar_txt = f"""## العربية
**{iso(h)} ضد {iso(a)}** — {iso(ctx['match']['date'])}

**الترتيب:** {iso(h)} في المركز {pos(h)} برصيد {pts(h)} نقطة، و{iso(a)} في المركز {pos(a)} برصيد {pts(a)} نقطة.

**آخر 5 مباريات:** {iso(h)}: {form_ar(h)} ({f[h]['points']} نقطة). {iso(a)}: {form_ar(a)} ({f[a]['points']} نقطة).

**{iso(h)} على أرضه:** {record_ar(hr)}.

**{iso(a)} خارج أرضه:** {record_ar(ar)}.

**المواجهات المباشرة:** {h2h_ar}.

**توقع النموذج:** الأهداف المتوقعة {iso(f"{m['expected_goals_home']} - {m['expected_goals_away']}")}. فوز {iso(h)} {iso(_pct(m['p_home_win']))}، تعادل {iso(_pct(m['p_draw']))}، فوز {iso(a)} {iso(_pct(m['p_away_win']))}. النتيجة الأرجح {iso(m['most_likely_score'])}.

**الخلاصة:** الأرقام تميل إلى {fav_ar}.
"""
    return en + "\n" + ar_txt
