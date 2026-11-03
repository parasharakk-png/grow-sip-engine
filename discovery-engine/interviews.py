"""Part 3 evidence: the anonymised interview scorecard and screener, scored against the engine's hypotheses.

Reads only the anonymised files in user-research/ (never user-research/private/).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parent.parent / "user-research"
SCORECARD = RESEARCH_DIR / "scorecard.csv"
SCREENER = RESEARCH_DIR / "screener_responses_anon.csv"

HYPOTHESES = [
    ("H1_judge_own_sip", "H1", "Can't judge their own SIP (goal, horizon, what is normal)"),
    ("H2_peers_proxy", "H2", "'Are others continuing?' is a proxy need"),
    ("H3_switch_gap", "H3", "Stops are really switches"),
    ("H4_cash_cancel", "H4", "A cash squeeze ends in a cancel"),
    ("H5_failed_instalment", "H5", "Failed instalments go unnoticed; account change triggers them"),
    ("H6_confusion", "H6", "Not understanding returns makes volatility look like failure"),
]
SCREENER_STATUS = {
    "running_no_doubt": "Running, never thought of stopping",
    "stopped": "Stopped",
    "stopped_and_switched": "Stopped and started a different SIP",
    "paused_or_reduced": "Paused or reduced",
    "failed_without_choice": "Instalment failed / stopped without choosing",
}


def load_scorecard() -> pd.DataFrame:
    if not SCORECARD.exists():
        return pd.DataFrame()
    sc = pd.read_csv(SCORECARD, dtype=str).fillna("")
    return sc[sc["status_today"] != ""].reset_index(drop=True)


def _verdict(plus: int, minus: int, new: int) -> str:
    if plus >= 3 and minus <= 1:
        return "Supported"
    if new >= 2:
        return "Reframed (new insight)"
    if minus > plus:
        return "Challenged"
    return "Weak / no signal"


def hypothesis_tally(sc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, hid, text in HYPOTHESES:
        marks = sc[col]
        plus, minus = int(marks.eq("+").sum()), int(marks.eq("−").sum())
        new, none = int(marks.eq("!").sum()), int(marks.eq("0").sum())
        rows.append({"ID": hid, "Hypothesis": text, "Confirms (+)": plus, "Challenges (−)": minus,
                     "New insight (!)": new, "No signal (0)": none, "Verdict": _verdict(plus, minus, new)})
    return pd.DataFrame(rows)


def decision(sc: pd.DataFrame) -> dict[str, Any]:
    """The decision rule written in user-research/README.md before the interviews."""
    n = len(sc)
    stop_n = int(sc["at_risk_moment"].eq("stop decision").sum())
    pay_n = int(sc["at_risk_moment"].eq("failed instalment").sum())
    h1_against = int(sc["H1_judge_own_sip"].eq("−").sum())
    if h1_against > n / 2:
        verdict = "Rethink the solution"
    elif pay_n >= 3 and pay_n > stop_n:
        verdict = "Switch focus to payment reliability"
    elif stop_n >= 3:
        verdict = "Keep the stop-moment focus"
    else:
        verdict = "Inconclusive: run more interviews"
    return {"n": n, "stop_n": stop_n, "pay_n": pay_n, "h1_against": h1_against, "verdict": verdict,
            "failed_any": int(sc["failed_instalment"].str.startswith("yes").sum()),
            "account_change": int(sc["account_change"].str.startswith("yes").sum())}


INCIDENT_FIELDS = ("When", "Trigger (what they saw first)", "Who they asked, what they were told",
                   "What they did, and how it ended", "What was missing")

# Each clause of the Problem 2 statement, with the participants whose notes support it (coded from notes/P*.md).
P2_EVIDENCE = [
    ("Judges the SIP only by red or green", None,
     "Judged by invested amount vs profit or loss. P4 stopped over a product rule, not returns."),
    ("Judges it against a one-year expectation", ("P1", "P2", "P3", "P5"),
     "Expected growth in a few months or 1 year, 10–12% or 12–14% in a year, or a one-year goal."),
    ("The doubt is in months 6–12", ("P1", "P3", "P4"),
     "Doubt peaked at months 6–8, 10 and 12. P2 stopped at about months 12–14 and P5 reduced at 14–15."),
    ("They ask whoever got them started", ("P1", "P4"),
     "A colleague and a cousin settled it. The others turned to YouTubers, online research or a friend."),
    ("Then they stop, reduce, or move the money out", ("P2", "P4", "P5"),
     "Moved to an FD or RD, a lump sum, and half to a gold scheme. P1 and P3 considered or plan to stop."),
]


def critical_incidents(sc: pd.DataFrame) -> list[dict[str, str]]:
    """The critical-incident section of each participant's anonymised notes."""
    out = []
    for pid in sc["participant"]:
        path = RESEARCH_DIR / "notes" / f"{pid}.md"
        if not path.exists():
            continue
        text = path.read_text()
        section = text.split("## Critical incident", 1)[-1].split("\n## ", 1)[0]
        fields = dict(re.findall(r"^- \*\*(.+?):\*\* (.+)$", section, re.M))
        started = re.search(r"^\| Who influenced the start \| (.+?) \|$", text, re.M)
        out.append({"participant": pid, "started_by": started.group(1) if started else "",
                    **{k: fields.get(k, "") for k in INCIDENT_FIELDS}})
    return out


def p2_evidence(sc: pd.DataFrame) -> pd.DataFrame:
    h1 = tuple(sc.loc[sc["H1_judge_own_sip"].eq("+"), "participant"])
    rows = []
    for clause, who, detail in P2_EVIDENCE:
        who = h1 if who is None else who
        rows.append({"Problem statement says": clause, "Interviews": f"{len(who)} of {len(sc)}",
                     "Who": ", ".join(who), "What they said or did": detail})
    return pd.DataFrame(rows)


def screener_summary() -> pd.DataFrame:
    if not SCREENER.exists():
        return pd.DataFrame()
    s = pd.read_csv(SCREENER, dtype=str).fillna("")
    vc = s["sip_status"].map(SCREENER_STATUS).fillna(s["sip_status"]).value_counts()
    out = vc.rename_axis("SIP status").reset_index(name="Respondents")
    out.attrs.update(
        n=len(s), groww=int(s["platform"].eq("Groww").sum()), job=int(s["changed_job"].eq("Yes").sum()),
        account=int(s["changed_salary_account"].eq("Yes").sum()),
    )
    return out
