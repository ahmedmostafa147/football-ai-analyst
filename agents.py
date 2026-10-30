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
    fav = max(("home", "draw", "away"), key=lambda k: m[f"p_{k}_win" if k != "draw" else "p_draw"])
    fav_en = {"home": f"{h} win", "draw": "a draw", "away": f"{a} win"}[fav]
    fav_ar = {"home": f"فوز {h}", "draw": "التعادل", "away": f"فوز {a}"}[fav]
    hh = ctx["head_to_head"]
    if hh["meetings"]:
        h2h = (f"{hh['meetings']} meetings: {h} won {hh[f'{h}_wins']}, {hh['draws']} draws, {a} won {hh[f'{a}_wins']}. "
               f"Latest: " + "; ".join(hh["recent"]))
        h2h_ar = (f"{hh['meetings']} مواجهة: فاز {h} {hh[f'{h}_wins']}، تعادل {hh['draws']}، فاز {a} {hh[f'{a}_wins']}. "
                  f"الأحدث: " + "؛ ".join(hh["recent"]))
    else:
        h2h, h2h_ar = "no previous meetings in the data", "لا توجد مواجهات سابقة في البيانات"
    pos = lambda team: t[team]["position"] if t[team] else "-"
    pts = lambda team: t[team]["pts"] if t[team] else "-"

    en = f"""## English
**{h} vs {a}** — {ctx['match']['date']}

**Table:** {h} are {pos(h)} with {pts(h)} points; {a} are {pos(a)} with {pts(a)} points.

**Form (last 5):** {h} {f[h]['last_results']} ({f[h]['points']} pts) · {a} {f[a]['last_results']} ({f[a]['points']} pts).

**{h} at home:** {hr['scored']} scored and {hr['conceded']} conceded per game, {hr['shots_on_target']} shots on target, xG {hr['xg_for']} for / {hr['xg_against']} against.

**{a} away:** {ar['scored']} scored and {ar['conceded']} conceded per game, {ar['shots_on_target']} shots on target, xG {ar['xg_for']} for / {ar['xg_against']} against.

**Head-to-head:** {h2h}.

**Model:** expected goals {m['expected_goals_home']} - {m['expected_goals_away']}. {h} win {_pct(m['p_home_win'])}, draw {_pct(m['p_draw'])}, {a} win {_pct(m['p_away_win'])}. Most likely score {m['most_likely_score']}.

**Verdict:** the numbers lean towards {fav_en}.
"""
    ar_txt = f"""## العربية
**{h} ضد {a}** — {ctx['match']['date']}

**الترتيب:** {h} في المركز {pos(h)} برصيد {pts(h)} نقطة، و{a} في المركز {pos(a)} برصيد {pts(a)} نقطة.

**آخر 5 مباريات:** {h} {f[h]['last_results']} ({f[h]['points']} نقطة) · {a} {f[a]['last_results']} ({f[a]['points']} نقطة).

**{h} على أرضه:** يسجل {hr['scored']} ويستقبل {hr['conceded']} في المباراة، {hr['shots_on_target']} تسديدة على المرمى، xG {hr['xg_for']} له و{hr['xg_against']} عليه.

**{a} خارج أرضه:** يسجل {ar['scored']} ويستقبل {ar['conceded']} في المباراة، {ar['shots_on_target']} تسديدة على المرمى، xG {ar['xg_for']} له و{ar['xg_against']} عليه.

**المواجهات المباشرة:** {h2h_ar}.

**توقع النموذج:** الأهداف المتوقعة {m['expected_goals_home']} - {m['expected_goals_away']}. فوز {h} {_pct(m['p_home_win'])}، تعادل {_pct(m['p_draw'])}، فوز {a} {_pct(m['p_away_win'])}. النتيجة الأرجح {m['most_likely_score']}.

**الخلاصة:** الأرقام تميل إلى {fav_ar}.
"""
    return en + "\n" + ar_txt
