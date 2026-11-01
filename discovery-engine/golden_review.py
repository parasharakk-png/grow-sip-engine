"""Write a review sheet for hand-checking the golden set (evidence-log/golden_set_review.md).

Each row shows the post, the labels already filled in, and a line to confirm or correct them. After the review,
set human_verified=yes (and fix any label) in data/golden_set.csv, then rerun golden_set_eval.py.

Run:  python3 golden_review.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from golden_set_eval import CODEBOOK, load_golden

ENGINE_DIR = Path(__file__).resolve().parent
CORPUS = ENGINE_DIR / "data" / "grow_sip_956.csv"
OUT = ENGINE_DIR.parent / "evidence-log" / "golden_set_review.md"

FIELDS = {
    "P2": ("own_experience", "fall_experienced", "emotion", "action", "fall_drove_action", "stop_reason",
           "seeks_peer_behaviour", "user_need", "nudge_fit"),
    "P1": ("own_experience", "payment_failure", "failure_cause", "account_change", "two_account_setup",
           "failure_notified", "user_fixable", "lapse_outcome"),
}


def _post(content: str) -> str:
    m = re.search(r"^POST:[ \t]*(.*?)(?:\nURL:|\Z)", content, re.M | re.S)
    text = re.sub(r"\s+", " ", m.group(1) if m else content).strip()
    return text if len(text) <= 900 else text[:900].rsplit(" ", 1)[0] + " …"


def main() -> None:
    golden = load_golden()
    corpus = pd.read_csv(CORPUS, dtype=str).fillna("").set_index("item_id")
    lines = [
        "# Golden set review",
        "",
        f"{len(golden)} posts. For each one, read the post and check every label against the definition. Mark "
        "**OK** or write the correction (e.g. `action = reduced`). Definitions are at the end.",
        "",
    ]
    for _, r in golden.iterrows():
        iid = r["item_id"]
        row = corpus.loc[iid] if iid in corpus.index else None
        lines += [f"## #{iid} · {r['problem']} · {r['case_type']}", ""]
        if row is not None:
            if row["url"]:
                lines.append(f"[Source]({row['url']})")
                lines.append("")
            lines += [f"> {_post(row['content'])}", ""]
        else:
            lines += ["*Post not in the current corpus; check against the quote below.*", ""]
        if r["evidence_quote"]:
            lines += [f"**Quote used:** \"{r['evidence_quote']}\"", ""]
        lines += ["| Label | Value |", "|---|---|"]
        lines += [f"| {f} | {r[f] or '—'} |" for f in FIELDS.get(r["problem"], ())]
        lines += ["", f"**Rationale:** {r['rationale']}", "", "**Your check:** OK ☐ · Correction: ______", ""]
    lines += ["## Label definitions", "", "| Label | Definition | Allowed values |", "|---|---|---|"]
    lines += [f"| {f} | {d} | {', '.join(v)} |" for f, (d, v) in CODEBOOK.items()]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(golden)} items, {int((~golden['item_id'].isin(corpus.index)).sum())} not in corpus)")


if __name__ == "__main__":
    main()
