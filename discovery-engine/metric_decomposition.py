"""Part 2: decompose 12-month SIP continuation into product outcomes and user behaviours.

Branch sizes reuse the opportunity model in reasoning.py, run twice: on all posts and on the
first-time target cohort.

Run:  python3 metric_decomposition.py   (writes exports/metric_decomposition.svg)
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from textwrap import wrap

import pandas as pd

from reasoning import OPPORTUNITIES, opportunity_table

ENGINE_DIR = Path(__file__).resolve().parent
EXPORT_SVG = ENGINE_DIR / "exports" / "metric_decomposition.svg"
GOLDEN_CSV = ENGINE_DIR / "data" / "golden_set.csv"
TARGET_COHORT = "Target: first-time investors"

BRAND, CB_ORANGE, CB_BLUE, GREY = "#00B386", "#E69F00", "#0072B2", "#6B7280"


@dataclass(frozen=True)
class Branch:
    code: str
    name: str
    outcome: str
    behaviours: tuple[str, ...]
    leading: str
    groww_data: str
    opportunities: tuple[str, ...]


BRANCHES: list[Branch] = [
    Branch(
        "0", "Activation",
        "The first SIP is set up and the first instalment succeeds",
        ("Completes KYC and the autopay mandate", "Has balance on the first debit date",
         "Learns why a first order was rejected"),
        "% of first SIPs whose first instalment succeeds",
        "Mandate registration success; first-debit success by bank",
        ("first_sip_success",),
    ),
    Branch(
        "A", "Involuntary lapse",
        "Every instalment goes through, and a failure is recovered inside 30 days",
        ("Keeps balance on the debit date", "Updates the mandate after a bank / salary-account change",
         "Notices a failed debit"),
        "Failed-instalment recovery within 7 days",
        "Debit return reasons (account closed vs insufficient funds); lapses preceded by a failure",
        ("payment_reliability",),
    ),
    Branch(
        "B", "Pause becomes stop",
        "A pause or cash squeeze ends in a resume, a skip or a smaller amount, not a cancel",
        ("Pauses for a cash need and forgets to resume", "Cancels instead of reducing",
         "Can't tell pause from stop"),
        "Paused SIPs resumed on time; cancel intents that pick skip / reduce",
        "Pause → resume rate; cancels with no prior pause or reduce",
        ("control_clarity", "cash_flexibility"),
    ),
    Branch(
        "C", "Deliberate stop",
        "At the 'should I stop?' moment the investor continues, adjusts, or restarts without a gap",
        ("Doubts after weak returns or a fall", "Asks strangers whether to stop", "Stops one SIP to start another"),
        "Stop intents that end in continue / adjust; restart within 30 days of a switch",
        "Stop-flow funnel; cancellations in the 2 weeks after an index fall",
        ("decision_support", "switch_without_gap"),
    ),
    Branch(
        "D", "Confusion drift",
        "The investor can tell whether the SIP is on track without outside help",
        ("Checks returns often", "Can't read XIRR / NAV / units", "Disengages quietly"),
        "Returns-related support contacts; explainer views",
        "First-year stop rate by returns-screen visits",
        ("understandability",),
    ),
]

RECOVERY = (
    "Recovery: a lapsed SIP counts as active again once one instalment succeeds within 30 days",
    "Lapsed SIPs with a successful instalment within 30 days",
)

# Every opportunity reached when the investor opens the pause / stop controls.
STOP_MOMENT = ("control_clarity", "cash_flexibility", "decision_support", "switch_without_gap")
PAYMENT_CODES = ("0", "A")
# Problem 1: every SIP lost to a failed payment, whether the first instalment or a later one.
PAYMENT_TRACK = ("first_sip_success", "payment_reliability")

METRIC_MECHANICS = [
    "**One failure is enough.** A monthly SIP with a single failed or skipped instalment in the measurement "
    "month counts as inactive, even if the investor never meant to stop — continuity matters, not just intent.",
    "**Reduce is always safe; skip is safe except in the measurement month.** A smaller instalment still counts; "
    "a skipped month only hurts if it falls in the 30-day window.",
    "**Recovery is a lever.** A lapsed SIP is active again after one successful instalment, so fast recovery "
    "of a failure is worth as much as preventing it.",
    "**The denominator is first-time investors.** Causes that dominate experienced investors' posts may not "
    "be the ones that end first SIPs.",
]


def branch_sizes(
    df: pd.DataFrame, lapse_rate: float = 0.40, recovery_rate: float = 0.30,
    overrides: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Per branch: share of posts about stopping, share of the lapse, and est. uplift, for all and first-timers."""
    groups = {"all": df, "ft": df[df["cohort"] == TARGET_COHORT]}
    tables = {k: opportunity_table(g, lapse_rate, recovery_rate, overrides).set_index("code") for k, g in groups.items()}
    rows = []
    for b in BRANCHES:
        row = {"code": b.code, "Branch": f"{b.code} · {b.name}" if b.code != "0" else b.name}
        for k, g in groups.items():
            t = tables[k]
            risk = g[g["is_at_risk"]]
            hit = t.loc[list(b.opportunities)] if b.opportunities else t.iloc[0:0]
            row[f"stop_posts_{k}"] = int(hit["Evidence posts"].sum())
            row[f"risk_n_{k}"] = len(risk)
            row[f"pct_stop_{k}"] = round(float(hit["% of at-risk posts"].sum()), 1)
            row[f"lapse_share_{k}"] = round(float(hit["Lapse share %"].sum()), 1)
            row[f"lost_pp_{k}"] = round(lapse_rate * float(hit["Lapse share %"].sum()), 1)
            row[f"uplift_{k}"] = round(float(hit["Est. uplift (pp)"].sum()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def focus_comparison(
    df: pd.DataFrame, lapse_rate: float = 0.40, recovery_rate: float = 0.30,
    overrides: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Payment reliability vs the stop moment, for all investors and first-timers."""
    out: dict[str, dict[str, float]] = {}
    for k, g in {"all": df, "ft": df[df["cohort"] == TARGET_COHORT]}.items():
        t = opportunity_table(g, lapse_rate, recovery_rate, overrides).set_index("code")
        out[k] = {
            "payment": round(float(t.loc[list(PAYMENT_TRACK), "Est. uplift (pp)"].sum()), 2),
            "stop_moment": round(float(t.loc[list(STOP_MOMENT), "Est. uplift (pp)"].sum()), 2),
            "payment_pct": round(float(t.loc[list(PAYMENT_TRACK), "% of at-risk posts"].sum()), 1),
            "stop_moment_pct": round(float(t.loc[list(STOP_MOMENT), "% of at-risk posts"].sum()), 1),
            "risk_n": int(g["is_at_risk"].sum()),
            "cont_n": int(g["is_continuer"].sum()),
        }
        for key, codes in (("payment", PAYMENT_TRACK), ("stop_moment", STOP_MOMENT)):
            tags = [f"t_{r}" for o in OPPORTUNITIES if o.code in codes for r in o.reasons]
            cont = g[g["is_continuer"]]
            out[k][f"{key}_n"] = int(t.loc[list(codes), "Evidence posts"].sum())
            out[k][f"{key}_cont_pct"] = round(float(cont[tags].any(axis=1).mean() * 100), 1) if len(cont) else 0.0
    return out


def lead_phrase(stop_moment: float, payment: float) -> str:
    """Which of the two bets is larger, worded so it stays true when the data changes."""
    if abs(stop_moment - payment) < 0.5:
        return f"the two are about level (stop moment +{stop_moment:.1f} vs payment +{payment:.1f} pp)"
    if stop_moment > payment:
        return f"the stop moment leads (+{stop_moment:.1f} vs +{payment:.1f} pp for payment failure)"
    return f"payment failure leads (+{payment:.1f} vs +{stop_moment:.1f} pp for the stop moment)"


def slide_title(focus: dict[str, dict[str, float]], dec: dict | None = None) -> str:
    if dec and dec.get("verdict") == "Keep the stop-moment focus":
        return (f"Focus on the stop decision: the risky moment in {dec['stop_n']} of {dec['n']} interviews; "
                "fix failed payments in parallel")
    if dec and dec.get("verdict") == "Switch focus to payment reliability":
        return (f"Focus on failed payments: the risky moment in {dec['pay_n']} of {dec['n']} interviews; "
                "support the stop decision in parallel")
    f = focus["ft"]
    if abs(f["stop_moment"] - f["payment"]) < 0.5:
        return "First SIPs end at a failed payment and at the stop decision in similar measure"
    if f["stop_moment"] > f["payment"]:
        return "First-time investors' SIPs mostly end at the stop decision, not at a failed payment"
    return "Failed payments end the most first SIPs; the stop decision is the second lever"


def golden_signals() -> dict[str, int]:
    """Counts from the hand-labelled golden set used to phrase the interview hypotheses."""
    if not GOLDEN_CSV.exists():
        return {}
    g = pd.read_csv(GOLDEN_CSV, dtype=str).fillna("")
    p2 = g[(g["problem"] == "P2") & (g["stop_reason"] != "na")]
    p2_all = g[g["problem"] == "P2"]
    p1_all = g[g["problem"] == "P1"]
    p1 = p1_all[p1_all["payment_failure"] == "yes"]
    return {
        "p2_n": len(p2),
        "p2_fear": int(p2["stop_reason"].eq("fear_of_further_loss").sum()),
        "p2_expect_timing": int(p2["stop_reason"].isin(["expectation_gap", "market_timing"]).sum()),
        "p2_need_judge": int(p2["user_need"].isin(["decision_help", "normalcy_check", "recovery_outlook"]).sum()),
        "p2_rows": len(p2_all),
        "p2_peer": int(p2_all["seeks_peer_behaviour"].eq("yes").sum()),
        "nudge_n": int(g["nudge_fit"].isin(["yes", "partial", "no"]).sum()),
        "nudge_yes": int(g["nudge_fit"].eq("yes").sum()),
        "p1_n": len(p1),
        "p1_rows": len(p1_all),
        "p1_not_notified": int(p1["failure_notified"].eq("no").sum()),
        "p1_fixable": int(p1["user_fixable"].eq("yes").sum()),
        "p1_account_change": int(p1_all["account_change"].eq("yes").sum()),
        "p1_two_accounts": int(p1_all["two_account_setup"].eq("yes").sum()),
    }


def interview_hypotheses(gs: dict[str, int]) -> pd.DataFrame:
    """What the interviews (Part 3) must confirm or overturn, one row per branch hypothesis."""
    gs = gs or {}
    p2 = gs.get("p2_n", 0)
    p1 = gs.get("p1_n", 0)
    rows = [
        ("C", "First-timers who consider stopping lack a way to judge their own SIP (goal, horizon, what is "
              "normal) — the fall is the trigger, not the cause.",
         f"Golden set: {gs.get('p2_need_judge', 0)} of {p2} stop cases need a way to judge (decision help, "
         f"normal or not, recovery); {gs.get('p2_fear', 0)} are plain fear.",
         "Walk me through the last time you thought about stopping. What did you look at? Who did you ask?"),
        ("C", "Peer behaviour ('are others continuing?') is a proxy need; showing fund-level stats would not "
              "resolve it and reads as an endorsement.",
         f"Golden set: {gs.get('p2_peer', 0)} of {gs.get('p2_rows', 0)} ask what others do; the original nudge "
         "fully fits "
         f"{gs.get('nudge_yes', 0)} of {gs.get('nudge_n', 0)}.",
         "When you asked others, what answer would have settled it for you?"),
        ("C", "Many first-timer 'stops' are switches; the new SIP depends on a second decision that often slips.",
         "Engine: switching fund or platform is one of the main stop causes in first-time investors' posts.",
         "Have you ever stopped one SIP to start another? How long was the gap?"),
        ("B", "A cash squeeze ends in a cancel because reduce / skip is unknown or not trusted.",
         "Engine: appears in posts about pausing, but rarely in first-time investors' posts — test whether it is "
         "simply unspoken.",
         "Has money ever been tight on the debit date? What did you do with the SIP?"),
        ("A", "Failed instalments go unnoticed; a bank or salary-account change is a trigger for first-timers too.",
         f"Golden set: {gs.get('p1_not_notified', 0)} of {p1} failures were not notified; "
         f"{gs.get('p1_fixable', 0)} were user-fixable; of {gs.get('p1_rows', 0)} payment cases, "
         f"{gs.get('p1_account_change', 0)} mention an account change and {gs.get('p1_two_accounts', 0)} a "
         "two-account setup.",
         "Has an instalment ever failed? How did you find out? Have you changed jobs or salary accounts since starting?"),
        ("D", "Not understanding returns (XIRR, red numbers early on) turns normal volatility into 'it isn't working'.",
         "Engine: a minor primary cause in posts about stopping — likely a weak driver on its own.",
         "Show me what you check in the app. What does that number tell you?"),
    ]
    return pd.DataFrame(rows, columns=["Branch", "Hypothesis", "Evidence so far", "Interview probe"])


# ---------------------------------------------------------------- SVG tree

def _text(x: float, y: float, lines: list[str], size: int = 15, weight: str = "normal", fill: str = "#111827",
          anchor: str = "start", gap: float = 1.3) -> str:
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if i == 0 else size * gap}">{escape(l)}</tspan>' for i, l in enumerate(lines)
    )
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{spans}</text>')


def tree_svg(sizes: pd.DataFrame, focus: dict[str, dict[str, float]], lapse_rate: float = 0.40,
             title: str | None = None) -> str:
    W, H = 1600, 900
    top = 70 if title else 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Inter, Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
    ]
    if title:
        parts.append(_text(40, 48, [title], size=28, weight="bold"))

    # Root
    rx, rw, rh = 500, 600, 92
    parts.append(f'<rect x="{rx}" y="{top}" width="{rw}" height="{rh}" rx="10" fill="{BRAND}"/>')
    parts.append(_text(W / 2, top + 36, ["Active SIP at month 12 — first-time investors"], 22, "bold", "#FFFFFF",
                       "middle"))
    parts.append(_text(W / 2, top + 66, ["% with ≥ 1 successful instalment in the last 30 days"], 16, "normal",
                       "#FFFFFF", "middle"))
    parts.append(_text(W / 2, top + rh + 30,
                       ["= Activated × (1 − involuntary lapse) × (1 − pause becomes stop) × (1 − deliberate stop) "
                        "+ recovered within 30 days"], 15, "normal", GREY, "middle"))

    # Columns
    n, margin, gap_x = len(BRANCHES), 40, 20
    cw = (W - 2 * margin - (n - 1) * gap_x) / n
    cy, ch = top + 230, 430
    bracket_y = cy - 44
    by_code = sizes.set_index("code")
    focus_codes = {"B", "C"}
    for i, b in enumerate(BRANCHES):
        x = margin + i * (cw + gap_x)
        parts.append(f'<line x1="{W / 2}" y1="{top + rh + 44}" x2="{x + cw / 2}" y2="{bracket_y - 24}" '
                     f'stroke="#D1D5DB" stroke-width="1.5"/>')
        if b.code in focus_codes:
            stroke, dash, sw, head = CB_BLUE, "", 4, CB_BLUE
        elif b.code in PAYMENT_CODES:
            stroke, dash, sw, head = CB_ORANGE, ' stroke-dasharray="10 6"', 3, CB_ORANGE
        else:
            stroke, dash, sw, head = "#9CA3AF", "", 1.5, GREY
        parts.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="{ch}" rx="10" fill="#FFFFFF" '
                     f'stroke="{stroke}" stroke-width="{sw}"{dash}/>')
        parts.append(f'<rect x="{x}" y="{cy}" width="{cw}" height="44" rx="10" fill="{head}"/>')
        parts.append(f'<rect x="{x}" y="{cy + 30}" width="{cw}" height="14" fill="{head}"/>')
        label = b.name if b.code == "0" else f"{b.code} · {b.name}"
        parts.append(_text(x + 14, cy + 29, [label], 17, "bold", "#FFFFFF"))

        y = cy + 72
        parts.append(_text(x + 14, y, ["PRODUCT OUTCOME"], 11, "bold", GREY))
        out_lines = wrap(b.outcome, 30)
        parts.append(_text(x + 14, y + 20, out_lines, 14, "bold"))
        y += 20 + len(out_lines) * 18 + 16
        parts.append(_text(x + 14, y, ["USER BEHAVIOURS"], 11, "bold", GREY))
        y += 20
        for beh in b.behaviours:
            bl = wrap(beh, 31)
            parts.append(_text(x + 14, y, ["• " + bl[0]] + ["  " + l for l in bl[1:]], 13))
            y += len(bl) * 17 + 4
        y += 10
        parts.append(_text(x + 14, y, ["LEADING INDICATOR"], 11, "bold", GREY))
        li = wrap(b.leading, 32)
        parts.append(_text(x + 14, y + 19, li, 13))

        fy = cy + ch - 62
        parts.append(f'<line x1="{x + 14}" y1="{fy - 16}" x2="{x + cw - 14}" y2="{fy - 16}" stroke="#E5E7EB"/>')
        r = by_code.loc[b.code]
        parts.append(_text(x + 14, fy, [f"Stop posts: {r['pct_stop_all']:.0f}% all · {r['pct_stop_ft']:.0f}% "
                                        f"first-time"], 13, "bold"))
        parts.append(_text(x + 14, fy + 20, [f"Est. uplift: +{r['uplift_all']:.1f} pp all · "
                                             f"+{r['uplift_ft']:.1f} pp first-time"], 13))

    # Bracket over the stop-moment branches
    xs = [margin + i * (cw + gap_x) for i, b in enumerate(BRANCHES) if b.code in focus_codes]
    x0, x1 = xs[0], xs[-1] + cw
    parts.append(f'<path d="M{x0} {bracket_y + 18} V{bracket_y} H{x1} V{bracket_y + 18}" fill="none" '
                 f'stroke="{CB_BLUE}" stroke-width="3"/>')
    parts.append(f'<rect x="{x0 + 20}" y="{bracket_y - 16}" width="{x1 - x0 - 40}" height="30" fill="#FFFFFF"/>')
    parts.append(_text((x0 + x1) / 2, bracket_y + 5,
                       [f"THE STOP MOMENT — research & MVP focus: +{focus['all']['stop_moment']:.1f} pp all · "
                        f"+{focus['ft']['stop_moment']:.1f} pp first-time"], 15, "bold", CB_BLUE, "middle"))
    px = [margin + i * (cw + gap_x) for i, b in enumerate(BRANCHES) if b.code in PAYMENT_CODES]
    parts.append(_text((px[0] + px[-1] + cw) / 2, bracket_y + 5,
                       [f"FAILED PAYMENTS — fix in parallel: +{focus['all']['payment']:.1f} pp all · "
                        f"+{focus['ft']['payment']:.1f} pp first-time"], 13, "bold", "#92400E", "middle"))

    # Recovery bar + footnote
    ry = cy + ch + 22
    parts.append(f'<rect x="{margin}" y="{ry}" width="{W - 2 * margin}" height="42" rx="8" fill="#F3F4F6" '
                 f'stroke="#9CA3AF"/>')
    parts.append(_text(margin + 16, ry + 27, [f"{RECOVERY[0]}  ·  Leading indicator: {RECOVERY[1]}"], 15,
                       "bold"))
    a, f = focus["all"], focus["ft"]
    note = (f"Sizes from {a['risk_n']} posts about stopping (all) and {f['risk_n']} from first-time "
            f"investors (small sample — a hypothesis to test). Assumes {lapse_rate * 100:.0f}% of first SIPs lapse "
            "by month 12 and 30% of each branch is recoverable. Solid blue = focus; dashed orange = parallel track.")
    parts.append(_text(margin, ry + 72, wrap(note, 190), 12, "normal", GREY))
    parts.append("</svg>")
    return "".join(parts)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    import streamlit as st

    st.cache_data = lambda *a, **k: (lambda f: f) if not a or not callable(a[0]) else a[0]
    import app

    data = app.load_posts()
    sizes, focus = branch_sizes(data), focus_comparison(data)
    EXPORT_SVG.parent.mkdir(exist_ok=True)
    from interviews import decision, load_scorecard

    EXPORT_SVG.write_text(tree_svg(sizes, focus, title=slide_title(focus, decision(load_scorecard()))))
    print(sizes.to_string())
    print(focus)
    print(f"wrote {EXPORT_SVG}")
