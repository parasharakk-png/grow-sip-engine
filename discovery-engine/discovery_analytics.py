"""Quantify themes and opportunity areas for 12-month SIP continuation."""

from __future__ import annotations

import re

import pandas as pd

# Maps to assignment metric decomposition
METRIC_BRANCH_LABELS = {
    "A_involuntary_lapse": "A — Involuntary lapse (payment / mandate)",
    "B_pause_becomes_stop": "B — Pause / skip becomes permanent stop",
    "C_deliberate_stop_expectations": "C — Deliberate stop / expectations & returns",
    "D_confusion_drift": "D — Confusion & engagement drift",
    "other": "Other / weak SIP signal",
}

TRIGGER_ADDRESSABILITY: dict[str, float] = {
    "failed_debit": 1.0,
    "mandate_autopay_issue": 1.0,
    "cash_crunch": 0.7,
    "red_returns": 0.6,
    "expectations_not_met": 0.6,
    "confusion_metrics": 0.9,
    "market_volatility": 0.4,
    "life_event": 0.3,
    "platform_bug": 0.85,
    "fees_charges": 0.5,
    "none_obvious": 0.2,
    "other": 0.4,
    "unknown": 0.2,
}

LAPSE_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0, "none": 0.25, "unknown": 0.5}


def assign_metric_branch(row: pd.Series) -> str:
    t = str(row.get("trigger", ""))
    life = str(row.get("lifecycle", ""))
    beh = str(row.get("behaviour", ""))

    if life == "payment_failed" or t in ("failed_debit", "mandate_autopay_issue"):
        return "A_involuntary_lapse"
    if life in ("paused",) or beh in ("paused_sip", "reduced_amount"):
        return "B_pause_becomes_stop"
    if beh == "stopped_sip" or life == "stopped" or t in (
        "red_returns",
        "expectations_not_met",
        "market_volatility",
    ):
        return "C_deliberate_stop_expectations"
    if t == "confusion_metrics" or beh in (
        "searched_youtube",
        "posted_reddit_forum",
        "asked_friend_family",
    ):
        return "D_confusion_drift"
    if t == "none_obvious" and row.get("link_to_lapse") in ("none", "low", ""):
        return "other"
    return "D_confusion_drift" if t == "confusion_metrics" else "other"


def row_opportunity_score(row: pd.Series) -> float:
    trigger = str(row.get("trigger", "unknown"))
    link = str(row.get("link_to_lapse", "unknown"))
    addr = TRIGGER_ADDRESSABILITY.get(trigger, 0.3)
    lapse = LAPSE_WEIGHT.get(link, 0.5)
    return round(lapse * addr, 3)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["metric_branch"] = out.apply(assign_metric_branch, axis=1)
    out["opportunity_score"] = out.apply(row_opportunity_score, axis=1)
    out["unmet_need"] = out["content"].apply(_extract_line, key="UNMET_NEED")
    return out


def _extract_line(text: str, key: str) -> str:
    m = re.search(rf"^{key}:[ \t]*(.*)$", text, re.M)
    return (m.group(1).strip() if m else "") or ""


def opportunity_by_metric_branch(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("metric_branch", dropna=False)
        .agg(
            posts=("item_id", "count"),
            avg_opportunity=("opportunity_score", "mean"),
        )
        .reset_index()
    )
    g["label"] = g["metric_branch"].map(METRIC_BRANCH_LABELS)
    g["posts_pct"] = (g["posts"] / len(df) * 100).round(1)
    g = g.sort_values("posts", ascending=False)
    return g
