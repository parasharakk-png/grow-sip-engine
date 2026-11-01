#!/usr/bin/env python3
"""Map screener responses onto the engine's problem-fit labels and apply the qualify rules in 01_screener.md.

Reads the Google Form export from private/ (gitignored) and writes screener_responses_anon.csv: respondent
codes R1..Rn, no timestamps, so it is safe to commit and to load in the discovery engine.

Run:  python3 user-research/screener_summary.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "screener_responses_anon.csv"

MONTHS = {"Less than 3 months ago": "lt3", "3–6 months ago": "3_6", "6–12 months ago": "6_12",
          "12–18 months ago": "12_18", "More than 18 months ago": "gt18"}
STATUS = {
    "Still running, and I've never thought of stopping it.": "running_no_doubt",
    "Still running, but I've seriously considered stopping or pausing it.": "running_considering_stop",
    "I've paused it or reduced the amount.": "paused_or_reduced",
    "I've stopped it.": "stopped",
    "I've stopped it and started a different SIP instead.": "stopped_and_switched",
    "An instalment failed or the SIP stopped without me intentionally stopping it.": "failed_without_choice",
    "I'm not sure.": "unknown",
}
AT_RISK = {"running_considering_stop", "paused_or_reduced", "stopped", "stopped_and_switched", "failed_without_choice"}
COLUMNS = {"Q1": "months_since_start", "Q2": "first_ever", "Q3": "platform", "Q4": "sip_status", "Q5": "age_band",
           "Q6": "income", "Q7": "works_in_finance", "Q8": "monthly_amount", "Q9": "changed_job",
           "Q10": "changed_salary_account"}


def load() -> pd.DataFrame:
    files = sorted((HERE / "private").glob("*.csv"))
    if not files:
        raise SystemExit("No screener export in user-research/private/")
    raw = pd.read_csv(files[-1], dtype=str).fillna("")
    df = pd.DataFrame({new: raw[next(c for c in raw.columns if c.startswith(q + "."))]
                       for q, new in COLUMNS.items()})
    df["months_since_start"] = df["months_since_start"].map(MONTHS).fillna("unknown")
    df["sip_status"] = df["sip_status"].map(STATUS).fillna("unknown")
    df["first_ever"] = df["first_ever"].str.startswith("No").map({True: "yes", False: "no"})
    df["works_in_finance"] = df["works_in_finance"].str.startswith("Yes").map({True: "yes", False: "no"})
    df.insert(0, "respondent", [f"R{i + 1}" for i in range(len(df))])
    return df


def qualify(r: pd.Series) -> str:
    """Rules from 01_screener.md; 'reserve' = fails only on timing or the 33-40 / non-salaried allowance."""
    if r["first_ever"] != "yes" or r["works_in_finance"] == "yes" or r["platform"] == "Through a financial advisor":
        return "drop"
    soft = []
    if r["months_since_start"] in ("lt3", "gt18", "unknown"):
        soft.append("timing")
    if r["age_band"] == "33-40":
        soft.append("age 33-40")
    elif r["age_band"] not in ("22-26", "27-32"):
        return "drop"
    if r["income"] != "Salaried employee":
        soft.append("not salaried")
    return "qualify" if not soft else "reserve: " + ", ".join(soft)


def main() -> None:
    df = load()
    df["at_risk"] = df["sip_status"].isin(AT_RISK).map({True: "yes", False: "no"})
    df["life_change"] = ((df["changed_job"] == "Yes") | (df["changed_salary_account"] == "Yes")).map(
        {True: "yes", False: "no"})
    df["screen"] = df.apply(qualify, axis=1)
    df.to_csv(OUT, index=False)
    print(df[["respondent", "months_since_start", "platform", "sip_status", "age_band", "income", "changed_job",
              "changed_salary_account", "screen"]].to_string(index=False))
    print("\nSIP disrupted (at risk) vs life change since starting:")
    print(pd.crosstab(df["at_risk"], df["life_change"], margins=True).to_string())
    print(f"\nWrote {OUT.name} ({len(df)} respondents)")


if __name__ == "__main__":
    main()
