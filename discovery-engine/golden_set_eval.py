"""Score the discovery engine against the hand-labelled golden set (data/golden_set.csv).

Run:  python3 golden_set_eval.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import hypotheses as h

GOLDEN_CSV = Path(__file__).resolve().parent / "data" / "golden_set.csv"

CODEBOOK: dict[str, tuple[str, tuple[str, ...]]] = {
    "problem": ("P2 = panic stop in a market fall (priority); P1 = payment / mandate failure", ("P1", "P2")),
    "case_type": (
        "core_positive = the problem as hypothesised; early_signal = the moment before a stop; "
        "hard_negative = looks like the problem to a keyword rule but is not; counter_evidence = challenges the solution",
        ("core_positive", "early_signal", "hard_negative", "counter_evidence"),
    ),
    "own_experience": ("The writer describes their own investing, not news or advice to others", ("yes", "no")),
    "fall_experienced": ("The writer's own portfolio or fund is falling / negative now (not a hypothetical crash)",
                         ("yes", "no", "na")),
    "emotion": ("Dominant emotion about the investment", ("anxious", "discouraged", "confused", "calm", "none", "na")),
    "action": ("What the writer did or is weighing",
               ("stopped_or_paused", "reduced", "considering_stop", "considering_switch", "continued", "none", "na")),
    "fall_drove_action": ("The fall / negative returns caused the action", ("yes", "no", "na")),
    "stop_reason": ("Why they stop or consider stopping",
                    ("fear_of_further_loss", "expectation_gap", "market_timing", "fund_underperformance", "cash_need",
                     "family_influence", "na")),
    "seeks_peer_behaviour": ("Explicitly asks what other investors are doing / did", ("yes", "no", "na")),
    "user_need": ("What they need in that moment",
                  ("peer_behaviour", "normalcy_check", "recovery_outlook", "decision_help", "fund_choice",
                   "cash_planning", "none", "na")),
    "nudge_fit": ("Would the proposed nudge (share of investors in this fund who kept their SIP + average amount "
                  "invested, shown when pausing / stopping in a fall) address this person's concern?",
                  ("yes", "partial", "no", "na")),
    "payment_failure": ("An instalment failed, was rejected or never executed", ("yes", "no", "na")),
    "failure_cause": ("Cause of the (possible) failure",
                      ("insufficient_balance", "bank_account_issue", "mandate_cancelled", "setup_incomplete",
                       "platform_technical", "unknown", "na")),
    "account_change": ("A job, bank or salary-account change is mentioned", ("yes", "no", "na")),
    "two_account_setup": ("Salary lands in one account, SIP debits from another", ("yes", "no", "na")),
    "failure_notified": ("The platform told the user about the failure in time", ("yes", "no", "unknown", "na")),
    "user_fixable": ("The user could have prevented or fixed it with the right information", ("yes", "no", "na")),
    "lapse_outcome": ("Effect on the SIP", ("stopped", "missed_instalment", "at_risk", "none", "na")),
    "label_confidence": ("Labeller's confidence in this row", ("high", "medium", "low")),
}

STOP_ACTIONS = {"stopped_or_paused", "reduced", "considering_stop"}
HUMAN_DRIVER = {
    "fear_of_further_loss": "fall_emotion", "expectation_gap": "fall_emotion", "market_timing": "market_timing",
    "fund_underperformance": "fund_specific", "cash_need": "cash_or_life", "family_influence": "other",
}
ENGINE_DRIVER = {
    "poor_returns": "fall_emotion", "decision_doubt": "fall_emotion", "market_timing": "market_timing",
    "fund_switch": "fund_specific", "fund_suspension": "fund_specific", "cash_or_goal": "cash_or_life",
}
CAUSE_CHECK_ORDER = [(5, "bank_account_issue"), (4, "insufficient_balance"), (3, "mandate_cancelled"),
                     (2, "setup_incomplete"), (0, "platform_technical"), (1, "platform_technical")]


SELECTION_LOG = GOLDEN_CSV.parent / "selection_log.csv"
METADATA = ("cohort", "source", "url", "Platform")


def load_golden(path: Path = GOLDEN_CSV) -> pd.DataFrame:
    """Golden rows; a post dropped as a duplicate is scored against the copy the corpus kept."""
    g = pd.read_csv(path, dtype=str).fillna("")
    if "human_verified" in g:
        g["human_verified"] = g["human_verified"].str.strip().str.lower()
    g["golden_item_id"] = g["item_id"]
    if SELECTION_LOG.exists():
        dup = pd.read_csv(SELECTION_LOG, dtype=str).fillna("").set_index("item_id")["duplicate_of"]
        g["item_id"] = g["item_id"].map(lambda i: dup.get(i) or i)
    return g


def selection_check(golden: pd.DataFrame) -> dict:
    """Did the selection pipeline keep the posts a human calls own experience, and drop the rest?"""
    if not SELECTION_LOG.exists():
        return {}
    log = pd.read_csv(SELECTION_LOG, dtype=str).fillna("").set_index("item_id")
    rows = []
    for _, r in golden.iterrows():
        rec = log.loc[r["golden_item_id"]] if r["golden_item_id"] in log.index else None
        decision = rec["decision"] if rec is not None else "not in selection log"
        rows.append({"item_id": r["golden_item_id"], "Human: own experience": r["own_experience"],
                     "Pipeline kept": "yes" if r["item_id"] != r["golden_item_id"] or decision.startswith("kept")
                     else "no", "Pipeline decision": decision})
    t = pd.DataFrame(rows)
    human_yes, kept = t["Human: own experience"].eq("yes"), t["Pipeline kept"].eq("yes")
    return {
        "table": t,
        "agree": int((human_yes == kept).sum()),
        "n": len(t),
        "wrongly_dropped": t[human_yes & ~kept],
        "wrongly_kept": t[~human_yes & kept],
    }


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("₹", "rs ")
    return re.sub(r"\s+", " ", text.lower()).strip()


def validate(golden: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """Problems with the golden set itself: unknown ids, invalid codes, quotes not found in the post."""
    issues = []
    posts = df.set_index("item_id")["post_text"]
    for _, row in golden.iterrows():
        rid = row["item_id"]
        if rid not in posts.index:
            issues.append(f"{rid}: not in corpus")
            continue
        for field, (_, allowed) in CODEBOOK.items():
            if row[field] not in allowed:
                issues.append(f"{rid}: {field}={row[field]!r} not in codebook")
        body = _norm(posts[rid])
        for seg in [s.strip(" .") for s in row["evidence_quote"].split("...")]:
            if seg and _norm(seg) not in body:
                issues.append(f"{rid}: quote segment not found in post: {seg[:60]!r}")
    return issues


def _engine_cause(text: pd.DataFrame) -> pd.Series:
    out = pd.Series("unknown", index=text.index)
    for idx, label in reversed(CAUSE_CHECK_ORDER):
        out[h._mask(text, h.MANDATE_REPORTS[idx][1])] = label
    return out


def engine_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """The engine's view of each golden row: auditable rules (regex) and the LLM codebook."""
    m = lambda pat: h._mask(rows, pat)  # noqa: E731
    yn = lambda s: s.map({True: "yes", False: "no"})  # noqa: E731
    return pd.DataFrame({
        "rule_fall": yn(m(h.MARKET_FALL)),
        "rule_anxious": yn(m(h.PANIC)),
        "llm_anxious": yn(rows["llm_emotion"].eq("anxious")),
        "engine_stop": yn(rows["is_at_risk"]),
        "rule_fall_stop": yn(m(h.MARKET_FALL) & m(h.STOP_INTENT)),
        "llm_fall_stop": yn(rows["trigger"].isin(["market_volatility", "red_returns"]) & rows["is_at_risk"]),
        "rule_asks_others": yn(m(h.ASKS_OTHERS)),
        "engine_driver": rows["primary_reason"].map(ENGINE_DRIVER).fillna("unclear"),
        "rule_payment_failure": yn(rows["t_involuntary_system"]),
        "llm_payment_failure": yn(rows["trigger"].isin(["mandate_autopay_issue", "failed_debit", "platform_bug"])),
        "rule_cause": _engine_cause(rows),
        "rule_account_change": yn(m(h.JOB_CHANGE) | m(h.BANK_CHANGE)),
    }, index=rows.index)


CHECKS: list[tuple[str, str, str, str]] = [
    # (problem, label shown, human column / derived key, engine column)
    ("P2", "Own portfolio is falling", "fall_experienced", "rule_fall"),
    ("P2", "Anxious / panicking", "is_anxious", "rule_anxious"),
    ("P2", "Anxious / panicking", "is_anxious", "llm_anxious"),
    ("P2", "Stopping, pausing or reducing", "is_stop", "engine_stop"),
    ("P2", "Stop caused by the fall", "fall_stop", "rule_fall_stop"),
    ("P2", "Stop caused by the fall", "fall_stop", "llm_fall_stop"),
    ("P2", "Asks what others are doing", "seeks_peer_behaviour", "rule_asks_others"),
    ("P2", "Why they stop (driver)", "driver", "engine_driver"),
    ("P1", "Instalment failed", "payment_failure", "rule_payment_failure"),
    ("P1", "Instalment failed", "payment_failure", "llm_payment_failure"),
    ("P1", "Cause of failure", "failure_cause", "rule_cause"),
    ("P1", "Job / bank / salary-account change", "account_change", "rule_account_change"),
]
METHOD = {"rule": "Text rule (regex)", "llm": "LLM codebook", "engine": "Engine (rules + LLM)"}


def evaluate(golden: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (scores per check, row-level disagreements, golden rows joined with engine labels)."""
    golden = golden.drop(columns=[c for c in METADATA if c in golden and c in df])
    g = golden.merge(df.rename(columns={"emotion": "llm_emotion"}), on="item_id", how="inner")
    g = g[g["own_experience"].eq("yes")].copy()
    g["is_anxious"] = g["emotion"].eq("anxious").map({True: "yes", False: "no"})
    g["is_stop"] = g["action"].isin(STOP_ACTIONS).map({True: "yes", False: "no"})
    g["fall_stop"] = (g["action"].isin(STOP_ACTIONS) & g["fall_drove_action"].eq("yes")).map({True: "yes", False: "no"})
    g["driver"] = g["stop_reason"].map(HUMAN_DRIVER).fillna("na")
    eng = engine_labels(g)
    g = pd.concat([g, eng], axis=1)

    scores, misses = [], []
    for problem, label, human, engine in CHECKS:
        rows = g[g["problem"].eq(problem)]
        if human == "driver":
            rows = rows[rows["is_stop"].eq("yes")]
        if human == "failure_cause":
            rows = rows[rows["payment_failure"].eq("yes")]
        agree = rows[human].eq(rows[engine])
        rec = {"Problem": problem, "What is checked": label, "Engine method": METHOD[engine.split("_")[0]],
               "Rows": len(rows), "Agreement %": round(agree.mean() * 100, 1) if len(rows) else None}
        if set(rows[human].unique()) <= {"yes", "no"}:
            tp = int(((rows[human] == "yes") & (rows[engine] == "yes")).sum())
            pred, actual = int((rows[engine] == "yes").sum()), int((rows[human] == "yes").sum())
            rec["Precision %"] = round(tp / pred * 100, 1) if pred else None
            rec["Recall %"] = round(tp / actual * 100, 1) if actual else None
        scores.append(rec)
        for _, r in rows[~agree].iterrows():
            misses.append({"item_id": r["item_id"], "Problem": problem, "What is checked": label,
                           "Engine method": rec["Engine method"], "Human label": r[human], "Engine label": r[engine],
                           "case_type": r["case_type"], "Why (human rationale)": r["rationale"], "url": r["url"]})
    return pd.DataFrame(scores), pd.DataFrame(misses), g


def summary_counts(golden: pd.DataFrame) -> dict[str, int]:
    p2 = golden[golden["problem"].eq("P2") & golden["own_experience"].eq("yes")]
    stop = p2[p2["action"].isin(STOP_ACTIONS)]
    return {
        "rows": len(golden),
        "p2": int(golden["problem"].eq("P2").sum()),
        "p1": int(golden["problem"].eq("P1").sum()),
        "p2_stop": len(stop),
        "p2_stop_fall_emotion": int(stop["stop_reason"].isin(["fear_of_further_loss", "expectation_gap"]).sum()),
        "p2_stop_timing": int(stop["stop_reason"].eq("market_timing").sum()),
        "p2_stop_fund": int(stop["stop_reason"].eq("fund_underperformance").sum()),
        "p2_stop_cash": int(stop["stop_reason"].eq("cash_need").sum()),
        "p2_nudge_yes": int(p2["nudge_fit"].eq("yes").sum()),
        "p2_nudge_partial": int(p2["nudge_fit"].eq("partial").sum()),
        "p2_nudge_no": int(p2["nudge_fit"].eq("no").sum()),
        "p2_seeks_peer": int(p2["seeks_peer_behaviour"].eq("yes").sum()),
        "p1_failures": int(golden["payment_failure"].eq("yes").sum()),
        "p1_two_account": int(golden["two_account_setup"].eq("yes").sum()),
        "p1_not_notified": int(golden["failure_notified"].eq("no").sum()),
        "p1_fixable": int(golden["user_fixable"].eq("yes").sum()),
    }


def main() -> None:
    import app  # noqa: PLC0415  (loads and enriches the corpus exactly as the dashboard does)

    df = app.load_posts()
    golden = load_golden()
    issues = validate(golden, df)
    print(f"Golden set: {len(golden)} rows | validation issues: {len(issues)}")
    for i in issues:
        print("  -", i)
    scores, misses, _ = evaluate(golden, df)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(scores.to_string(index=False))
        print(f"\n{len(misses)} disagreements")
        print(misses[["item_id", "What is checked", "Engine method", "Human label", "Engine label"]].to_string(index=False))
    print(summary_counts(golden))
    sel = selection_check(golden)
    if sel:
        print(f"\nSelection screen agrees with the human 'own experience' label on {sel['agree']} of {sel['n']}")
        print(sel["table"][sel["table"]["Human: own experience"].ne("yes") | sel["table"]["Pipeline kept"].ne("yes")]
              .to_string(index=False))


if __name__ == "__main__":
    main()
