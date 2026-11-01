"""Test the PM's two problem hypotheses against the coded corpus: evidence for, evidence against,
the specific mechanism claimed, and a data-backed reframing."""

from __future__ import annotations

from typing import Any

import pandas as pd

from reasoning import NO_FAILURE, PAYMENT_SUBCAUSES, BANK_CHANGE, _pct, _snippet, opportunity_table

TARGET = "Target: first-time investors"

# ---------------------------------------------------------------- problem 1: mandate failures

MANDATE_REPORTS: list[tuple[str, str]] = [
    ("Money debited but SIP order failed / rejected",
     r"debited but|deducted but|money (got )?deducted.{0,100}(fail|reject)|(order|status|sip).{0,40}(shows?|is) (as )?(failed|rejected)"),
    ("Autopay / mandate keeps failing",
     r"autopay.{0,40}fail|mandate.{0,40}fail|auto[- ]?debit.{0,20}(stopped|not working|fail)|sip failed|failed (sip|instal|debit)|instalment.{0,20}failed"),
    ("Mandate registration pending or rejected",
     r"registration|pending|not (approved|registered)|mandate.{0,40}(reject|declin)"),
    ("Mandate cancelled or expired without user action",
     r"cancell?ed (automatically|without)|got cancelled|expired|without (my |any )?(action|permission|consent)|never requested"),
    ("Insufficient balance / bounce charges",
     r"insufficient|bounce|ach debit return|return charges?|not enough (balance|money)"),
    ("Bank account changed, frozen or closed",
     BANK_CHANGE),
]
JOB_CHANGE = r"(chang|switch)\w* (my )?(job|company|employer)|new job|job switch|left (my )?job|resign"
# ---------------------------------------------------------------- problem 2: panic in a market fall

MARKET_FALL = (r"market (is )?(fall|crash|correction|down|dip|bleed)|crash|correction|falling|bear market|\bred\b"
               r"|negative returns?|in loss|portfolio.{0,20}(down|bleed)|xirr.{0,30}(fell|drop|went)")
PANIC = r"panic|scared|fear|worried|nervous|anxious|bleeding|sleepless|lost hope"
STOP_INTENT = r"\b(stop|pause|cancel|exit|redeem|withdraw|sell)\w*"
CONTINUE_INTENT = r"continu|keep (going|investing)|stay(ed)? invested|kept (my )?sip|didn'?t stop|not stop|hold(ing)? on"
ASKS_OTHERS = (r"are (you|people|others|u) (guys )?(all )?(still )?(continu|stopp|invest)|still continu\w* (your|their)"
               r"|what (are )?others|anyone else|everyone (else )?(is )?(stopp|continu|sell)|should i (continue|stop)")


def _mask(df: pd.DataFrame, pattern: str) -> pd.Series:
    return df["post_clean"].str.lower().str.contains(pattern, regex=True)


def _quotes(df: pd.DataFrame, k: int = 3) -> list[dict[str, str]]:
    ranked = df.assign(_l=df["link_to_lapse"].map({"high": 3, "medium": 2, "low": 1}).fillna(0))
    ranked = ranked.sort_values(["_l", "opportunity_score"], ascending=False).head(k)
    return [
        {"id": r["item_id"], "source": r["source"], "group": r["cohort"], "text": _snippet(r["post_clean"], 220),
         "url": r["url"]}
        for _, r in ranked.iterrows()
    ]


def payment_subcause_table(df: pd.DataFrame) -> pd.DataFrame:
    """Payment-tagged posts split by sub-cause: all tagged posts, the sized (at-risk) set, and first-timers in it."""
    pay = df[df["t_involuntary_system"]]
    label = pay["payment_subcause"]
    sets = {
        "Sized posts (at risk)": pay["is_at_risk"],
        "First-time, sized": pay["is_at_risk"] & (pay["cohort"] == TARGET),
        "All payment-tagged": pd.Series(True, index=pay.index),
    }
    rows = []
    for i, (group, name, mine) in enumerate([s[:3] for s in PAYMENT_SUBCAUSES] + [NO_FAILURE]):
        idx = i if i < len(PAYMENT_SUBCAUSES) else -1
        row = {"Group": group, "Sub-cause": name, "In my original list": mine}
        for col, m in sets.items():
            n = int((m & (label == idx)).sum())
            row[col] = n
            row[f"{col} %"] = _pct(n, int(m.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def _opp_row(opps: pd.DataFrame, code: str) -> dict[str, Any]:
    row = opps[opps["code"] == code].iloc[0]
    return {
        "label": row["Opportunity"],
        "rank": int(opps.index[opps["code"] == code][0]) + 1,
        "uplift": float(row["Est. uplift (pp)"]),
        "pct_risk": float(row["% of at-risk posts"]),
        "pct_cont": float(row["% among continuers"]),
        "evidence_n": int(row["Evidence posts"]),
    }


def mandate_hypothesis(df: pd.DataFrame, opps: pd.DataFrame) -> dict[str, Any]:
    risk, cont = df[df["is_at_risk"]], df[df["is_continuer"]]
    area = risk[risk["t_involuntary_system"]]
    hits = {label: _mask(area, pat) for label, pat in MANDATE_REPORTS}
    unspecified = ~pd.concat(hits.values(), axis=1).any(axis=1)
    rows = [{"What users report": label, "Posts": int(m.sum())} for label, m in hits.items()]
    reports = pd.DataFrame(rows).sort_values("Posts", ascending=False)
    reports = pd.concat([reports, pd.DataFrame(
        [{"What users report": "Payment / platform failure, cause not stated", "Posts": int(unspecified.sum())}]
    )], ignore_index=True)
    reports["% of payment-failure posts"] = reports["Posts"].map(lambda n: _pct(n, len(area)))

    mentions_mandate = df["t_involuntary_system"]
    job = _mask(df, JOB_CHANGE)
    bank = _mask(df, BANK_CHANGE)
    mechanism = area[_mask(area, JOB_CHANGE) | _mask(area, BANK_CHANGE)]
    return {
        "opp": _opp_row(opps, "payment_reliability"),
        "activation_opp": _opp_row(opps, "first_sip_success"),
        "area_n": len(area),
        "risk_n": len(risk),
        "pct_risk": _pct(len(area), len(risk)),
        "pct_cont": _pct(int(cont["t_involuntary_system"].sum()), len(cont)),
        "first_timer_n": int((area["cohort"] == TARGET).sum()),
        "reports": reports,
        "subcauses": payment_subcause_table(df),
        "job_change_with_mandate": int((mentions_mandate & job).sum()),
        "bank_change_with_mandate": int((mentions_mandate & bank).sum()),
        "job_change_any": int(job.sum()),
        "mechanism_quotes": _quotes(mechanism, 3),
        "quotes": _quotes(area[~area["item_id"].isin(mechanism["item_id"])], 3),
    }


def panic_hypothesis(df: pd.DataFrame, opps: pd.DataFrame) -> dict[str, Any]:
    fall = _mask(df, MARKET_FALL)
    panic = _mask(df, PANIC)
    stop = _mask(df, STOP_INTENT)
    cont_intent = _mask(df, CONTINUE_INTENT)
    asks = _mask(df, ASKS_OTHERS)
    groups = {
        "Posts about stopping": df["is_at_risk"],
        "Continuers": df["is_continuer"],
        "First-time": df["cohort"] == TARGET,
        "Other investors": df["cohort"] != TARGET,
    }
    by_group = pd.DataFrame(
        [{"Group": g, "Posts": int(m.sum()), "% mentioning a market fall": _pct(int((fall & m).sum()), int(m.sum()))}
         for g, m in groups.items()]
    )
    fall_posts = df[fall]
    fall_stop = df[fall & stop]
    return {
        "opp": _opp_row(opps, "decision_support"),
        "fall_n": int(fall.sum()),
        "fall_first_timer_n": int((fall & (df["cohort"] == TARGET)).sum()),
        "fall_stop_n": len(fall_stop),
        "fall_continue_n": int((fall & cont_intent).sum()),
        "panic_n": int((fall & panic).sum()),
        "asks_others_n": int((fall & asks).sum()),
        "asks_others_first_n": int((fall & asks & (df["cohort"] == TARGET)).sum()),
        "by_group": by_group,
        "pct_fall_risk": float(by_group.iloc[0, 2]),
        "pct_fall_cont": float(by_group.iloc[1, 2]),
        "quotes_for": _quotes(df[fall & (panic | asks) & ~_mask(df, r"not volatility|structural")], 3),
        "quotes_against": (
            _quotes(df[fall & cont_intent & df["is_continuer"]], 1)
            + _quotes(df[df["is_at_risk"] & df["t_poor_returns"] & stop & ~panic], 2)
        ),
        "fall_posts_n": len(fall_posts),
    }


def build_hypotheses(df: pd.DataFrame, lapse_rate: float, recovery_rate: float,
                     overrides: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    opps = opportunity_table(df, lapse_rate, recovery_rate, overrides).reset_index(drop=True)
    return {"mandate": mandate_hypothesis(df, opps), "panic": panic_hypothesis(df, opps), "opps": opps}


def strength(n: int) -> str:
    return "Strong" if n >= 30 else "Moderate" if n >= 10 else "Weak" if n >= 3 else "Not found"
