"""Generate the AI Evidence Log (assignment deliverable) from the live data, so every number is reproducible.

Writes ../evidence-log/AI_Evidence_Log.md and AI_Evidence_Log.csv.
Rerun after any corpus rebuild or golden-set edit:
    python3 evidence_log.py
Import the CSV into Google Sheets (File → Import → Replace current sheet) and share as
"anyone with the link can view". Use that sheet URL on the slides.
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ENGINE_DIR = Path(__file__).resolve().parent
OUT = ENGINE_DIR.parent / "evidence-log" / "AI_Evidence_Log.md"
OUT_CSV = ENGINE_DIR.parent / "evidence-log" / "AI_Evidence_Log.csv"
ENGINE_URL = "https://grow-sip-engine-nlems472oc7cfzz4ripgxr.streamlit.app"
REPO_URL = "https://github.com/parasharakk-png/grow-sip-engine"
UNSTATED_TRIGGERS = ("none_obvious", "other", "unknown")


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "/") for v in r) + " |")
    return "\n".join(lines)


def _models(path: Path, key: str) -> Counter:
    c: Counter = Counter()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                c[json.loads(line).get(key) or "not recorded"] += 1
    return c


def _sources(df: pd.DataFrame, log: pd.DataFrame, source_methods: dict, target: str) -> pd.DataFrame:
    rows = []
    for src in df["source"].value_counts().index:
        d = df[df["source"] == src]
        dated = d["year"].notna()
        years = f"{int(d.loc[dated, 'year'].min())}–{int(d.loc[dated, 'year'].max())}" if dated.any() else "—"
        rows.append({
            "Source": src,
            "Pooled": int((log["source"] == src).sum()),
            "In corpus": len(d),
            "First-time target": int((d["cohort"] == target).sum()),
            "Dated": f"{dated.mean() * 100:.0f}%",
            "Years": years,
            "How collected": source_methods.get(src, ("—", ""))[0],
        })
    out = pd.DataFrame(rows)
    total = {"Source": "**Total**", "Pooled": len(log), "In corpus": len(df),
             "First-time target": int((df["cohort"] == target).sum()),
             "Dated": f"{df['year'].notna().mean() * 100:.0f}%", "Years": "", "How collected": ""}
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def _funnel(log: pd.DataFrame) -> pd.DataFrame:
    d = log["decision"]
    total = len(log)
    off = int(d.str.startswith("dropped: not about SIPs").sum())
    not_own = int(d.str.startswith("dropped: not the author").sum())
    dupes = int(d.str.startswith("dropped: duplicate").sum())
    llm_drop = int(d.str.startswith("dropped by LLM").sum())
    kept = int((log["keep"].astype(str) == "True").sum())
    first = int(d.str.startswith("kept: first_time").sum())
    return pd.DataFrame([
        {"Step": "1. Pooled public posts (after cleaning)", "Posts": total, "Method": "Scrapers, then cleaned sheet"},
        {"Step": "2. About SIPs / mutual funds", "Posts": total - off, "Method": "Keyword rules"},
        {"Step": "3. Author's own experience", "Posts": total - off - not_own, "Method": "First-person rules"},
        {"Step": "4. Duplicates removed", "Posts": total - off - not_own - dupes, "Method": "Same body text"},
        {"Step": "5. Own experience confirmed by LLM", "Posts": kept,
         "Method": f"Problem-fit LLM pass rejected {llm_drop}"},
        {"Step": "6. First-time investors (target group)", "Posts": first,
         "Method": "LLM investor stage; first-time phrase rule where the LLM can't tell"},
    ])


def build() -> str:
    import streamlit as st

    st.cache_data = lambda *a, **k: (lambda f: f) if not a or not callable(a[0]) else a[0]
    import app
    from golden_set_eval import SELECTION_LOG, evaluate, load_golden, selection_check, summary_counts
    from question_guard import run_guard_tests

    df = app.load_posts()
    log = pd.read_csv(SELECTION_LOG, dtype=str).fillna("")
    golden = load_golden()
    sel = selection_check(golden)
    scores, misses, _ = evaluate(golden, df)
    sc = summary_counts(golden)
    verified = int(golden.get("human_verified", pd.Series(dtype=str)).eq("yes").sum())
    dropped = sel["wrongly_dropped"]["item_id"].tolist()
    risk = df[df["is_at_risk"]]
    target = app.COHORT_TARGET

    codes = _models(app.CODED_CHECKPOINT, "coding_method") + _models(
        ENGINE_DIR / "data" / "corpus_first_time.checkpoint.jsonl", "coding_method")
    screen = _models(app.SCREEN_CHECKPOINT, "screen_model")
    fit_n = sum(_models(ENGINE_DIR / "data" / "problem_fit.checkpoint.jsonl", "item_id").values())

    unstated = risk["trigger"].isin(UNSTATED_TRIGGERS)
    stated_top = risk.loc[~unstated, "trigger"].value_counts()
    cause_cover = int((risk["primary_reason"] != "unclear").sum())

    def score(label: str, method: str, col: str = "Agreement %") -> float | None:
        hit = scores[(scores["What is checked"] == label) & (scores["Engine method"] == method)]
        return None if hit.empty or pd.isna(hit.iloc[0][col]) else float(hit.iloc[0][col])

    llm_fall_recall = score("Stop caused by the fall", "LLM codebook", "Recall %")
    rule_fall_recall = score("Stop caused by the fall", "Text rule (regex)", "Recall %")
    driver_agree = score("Why they stop (driver)", "Engine (rules + LLM)")
    guard = run_guard_tests()
    guard_pass = int(guard["pass"].sum()) if "pass" in guard else 0
    guard_kinds = pd.read_csv(ENGINE_DIR / "data" / "question_guard_tests.csv", dtype=str)
    kind_col = next((c for c in guard_kinds.columns if "expected" in c or c == "kind"), guard_kinds.columns[-1])

    worst = scores.dropna(subset=["Agreement %"]).sort_values("Agreement %").head(4)
    fail_lines, shown = [], set()
    for _, w in worst.iterrows():
        ex = misses[(misses["What is checked"] == w["What is checked"]) & (misses["Engine method"] == w["Engine method"])
                    & ~misses["item_id"].isin(shown)]
        shown.update(ex.head(2)["item_id"])
        fail_lines.append(f"- **{w['What is checked']} — {w['Engine method']}: {w['Agreement %']:.0f}% agreement "
                          f"on {int(w['Rows'])} rows.**")
        for _, e in ex.head(2).iterrows():
            why = str(e["Why (human rationale)"]).strip().rstrip(".")
            fail_lines.append(f"  - Post #{e['item_id']}: hand label *{e['Human label']}*, engine *"
                              f"{e['Engine label']}*. {why.replace('The exact Problem 1 moment', 'The exact payment-failure moment')}.")

    show = scores.copy()
    for c in ("Agreement %", "Precision %", "Recall %"):
        show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}%")
    show["Problem"] = show["Problem"].map({
        "P2": "1 · Stop decision",
        "P1": "2 · Failed payments",
    }).fillna(show["Problem"])
    show = show.rename(columns={
        "What is checked": "What I checked",
        "Agreement %": "Match with my labels",
        "Precision %": "Precision",
        "Recall %": "Recall",
    })
    p1 = golden[golden["payment_failure"].eq("yes")]
    p1_notified_no = int(p1["failure_notified"].eq("no").sum())
    p1_fixable = int(p1["user_fixable"].eq("yes").sum())
    ft_stop = int((risk["cohort"] == target).sum())
    quotes_ok = int((df["quote_check"] == "Verbatim").sum())
    quotes_bad = int((df["quote_check"] == "Not found in post").sum())
    screen_line = ", ".join(f"{m}, {n} candidates" for m, n in screen.items()) or "not recorded"

    parts = [
        "# AI Evidence Log — Groww SIP Discovery Engine",
        "",
        "This is the log for the **discovery engine** (public posts → two problems → which one to build). "
        "It is not the cancel-sheet MVP.",
        "",
        f"Live engine: {ENGINE_URL}  ",
        f"Code and data: {REPO_URL}  ",
        f"Numbers generated {date.today():%d %b %Y} from the live files (`discovery-engine/evidence_log.py`).",
        "",
        "Problem 1 in this log is the **stop decision** (the MVP). Problem 2 is **failed payments** (ops). "
        "That matches the engine Overview tab.",
        "",
        "## 1. Sources and volume",
        "",
        f"The engine reads **{len(df):,} public posts** ({int((df['cohort'] == target).sum())} from first-time "
        f"investors, {int((df['cohort'] != target).sum()):,} from other investors so I can compare). "
        f"They come from {len(log):,} pooled posts. **{len(risk)}** are about stopping, pausing, a failed "
        f"payment or a strong link to a lapse. **{int(df['is_continuer'].sum())}** are people who kept going. "
        f"**{ft_stop}** of the stop posts are from first-timers — a small set, used as a hint for interviews.",
        "",
        _md_table(_sources(df, log, app.SOURCE_METHODS, target)),
        "",
        "**How posts were kept**",
        "",
        _md_table(_funnel(log)),
        "",
        "**Where AI was used — and where it was not**",
        "",
        f"- **Screening** (is this a first-time investor talking about their own SIP?): {screen_line}.",
        f"- **Problem-fit tags** ({fit_n:,} posts: investor stage, months in, SIP status, main concern): "
        "MiniMax-M2.7-highspeed, prompt versioned and cached.",
        f"- **Codebook tags** (lifecycle, trigger, emotion, behaviour, link to lapse, quote): "
        f"{codes.get('llm', 0):,} posts by the same model, cached so a rebuild does not call the model again.",
        "- **Ask the engine:** the model only writes over a numeric pack and must cite post IDs. A rule-based "
        "guard runs first. Below 3 matching posts it says it does not know.",
        "- **Not AI:** the cause of each stop post, the opportunity sizes, and the golden-set scores are "
        "fixed text rules. I did this after the golden set showed the model could not name *why* they stop.",
        f"- **Quote check:** {quotes_ok:,} of the AI-picked quotes appear word for word in the post; "
        f"{quotes_bad} do not.",
        "",
        "## 2. Golden set: 20 posts I labelled by hand",
        "",
        f"I labelled **{len(golden)} hard posts** ({sc['p2']} stop decision, {sc['p1']} failed payments) "
        "against a written codebook. They are not a random sample: core cases, posts that fool keyword "
        "rules, and posts that argue against my first solution.",
        "",
        (f"- I read every post and checked every label."
         if verified == len(golden) else
         f"- Labels were drafted with AI help; I have checked **{verified} of {len(golden)}** myself."),
        f"- The keep/drop screen matched my *own experience* label on **{sel['agree']} of {sel['n']}**. "
        + (f"It wrongly dropped {len(dropped)} real first-person "
           f"{'story' if len(dropped) == 1 else 'stories'} ({', '.join('#' + i for i in dropped)}). "
           if dropped else "It kept every post I marked as the author's own experience. ")
        + (f"It wrongly kept {len(sel['wrongly_kept'])}. " if len(sel['wrongly_kept']) else "")
        + "Post #398 is correctly left out of the corpus. #2547 is scored as its kept duplicate #1405.",
        "",
        "**How often it matched my labels**",
        "",
        _md_table(show[["Problem", "What I checked", "Engine method", "Rows",
                        "Match with my labels", "Precision", "Recall"]]),
        "",
        "**Where it failed** (lowest match first)",
        "",
        *fail_lines,
        "",
        "**What I changed because of this set**",
        "",
        f"- Of {sc['p2_stop']} hand-labelled stops in a fall, only {sc['p2_stop_fall_emotion']} are fear "
        f"or an expectation gap. {sc['p2_stop_timing']} are market timing, {sc['p2_stop_fund']} a fund "
        f"issue, {sc['p2_stop_cash']} cash. \"Panic\" was the wrong word.",
        f"- My first nudge (\"x% of investors kept their SIP\") fully fitted {sc['p2_nudge_yes']} of "
        f"{sc['p2_nudge_yes'] + sc['p2_nudge_partial'] + sc['p2_nudge_no']} posts. I dropped it.",
        f"- Of {sc['p1_failures']} payment failures, {p1_notified_no} were not told in time and "
        f"{p1_fixable} were things the user could have fixed if the app had said why.",
        "- The model's \"why they stop\" label matched me on **33%** of rows. Sizing therefore uses "
        "visible text rules, not that label.",
        "",
        "## 3. Where I rejected or corrected AI output",
        "",
        "### 1. The model's stop trigger could not explain why SIPs end",
        f"- **What it did:** each post gets one trigger. For {int(unstated.sum())} of {len(risk)} stop "
        f"posts ({unstated.mean() * 100:.0f}%) it said *none / other / unknown*. The rest were broad "
        f"(top: {', '.join(stated_top.head(3).index)}).",
        "- **Why that is not usable:** a vague tag cannot split a failed mandate from a fund switch or "
        "a cash squeeze, so I could not size anything from it.",
        f"- **What I did:** a fixed set of 10 text-pattern causes. It covers {cause_cover} of "
        f"{len(risk)} stop posts. The patterns are in the engine and scored on the golden set.",
        "",
        "### 2. \"Panic in a fall\" did not match the posts — not used for sizing",
        f"- **What it did:** the codebook found {llm_fall_recall:.0f}% of the stops a fall caused "
        f"(text rules also {rule_fall_recall:.0f}%). The \"why they stop\" driver matched me on only "
        f"{driver_agree:.0f}% of stops, mixing fear, timing and the fund.",
        "- **Why I rejected it:** if I sized from that label I would under-count fall stops, and I "
        f"still could not say whether panic was the problem. In the set, fear was "
        f"{sc['p2_stop_fall_emotion']} of {sc['p2_stop']} stops. The common thread was having no way "
        "to tell if *their* SIP was okay.",
        "- **What I did:** size from the text-pattern causes. Problem 1 is \"no yardstick of their "
        "own\", not panic. The \"x% kept going\" nudge is gone.",
        "",
        "### 3. The engine answered advice and support questions — I added a guard",
        "- **What it did:** test questions like *\"Should I stop SIP?\"*, *\"Which mutual fund to "
        "buy?\"*, *\"Which bank account should I open?\"*, *\"OTP is not coming\"* got an answer "
        "from the evidence pack.",
        "- **Why I rejected it:** the brief forbids fund picks. An evidence engine has nothing valid "
        "to say about one person's account.",
        f"- **What I did:** a rule-based guard (research, advice, support, unsafe, off-topic, how "
        f"common) declines with a reason and a research reframe. **{guard_pass} of {len(guard)}** "
        f"test questions pass ({guard_kinds[kind_col].nunique()} kinds). Below "
        f"{app.MIN_EVIDENCE} matching posts it abstains.",
        "",
        "## 4. What this meant for the product",
        "",
        "On first-timer stop posts the two problems are about level (stop decision 29.4% / +2.3 pp; "
        "failed payments 27.9% / +2.9 pp). Posts could not pick the MVP. The rule I wrote before "
        "interviews: stay on the stop screen if at least 3 of 5 describe that moment. **5 of 5** "
        "did; **0 of 5** stopped over a failed debit. That is why the MVP is a sheet before cancel, "
        "and failed payments stay an ops track (tell them why, retry, repair the mandate).",
        "",
        "## 5. Limitations",
        "",
        "- These are public posts, not Groww users. Shares are of posts, not people.",
        f"- Only {ft_stop} first-timer stop posts, so first-timer bars are a hint, not a count of users.",
        "- Uplift uses 40% lapse and 30% recoverable, plus how fixable each cause looks. Groww's "
        "cohort file would replace that.",
        f"- The selection screen can drop real stories ({len(dropped)} of "
        f"{int(golden['own_experience'].eq('yes').sum())} in this golden set).",
        f"- The golden set is {len(golden)} hard posts, labelled by one person"
        + (f"; {len(golden) - verified} still need that check." if verified < len(golden) else "."),
        "- Five interviews from my network, all already at risk. They show what the moment looks "
        "like, not how common it is.",
    ]
    return "\n".join(parts) + "\n"


def md_to_sheet_rows(md: str) -> list[list[str]]:
    """One Google Sheet: headings and prose in column A, tables as real columns."""
    rows: list[list[str]] = []
    for line in md.splitlines():
        raw = line.rstrip()
        if not raw:
            rows.append([])
            continue
        if re.match(r"^\|[\s\-:|]+\|$", raw):
            continue
        if raw.startswith("|"):
            cells = [c.strip().replace("**", "") for c in raw.strip().strip("|").split("|")]
            rows.append(cells)
            continue
        text = raw.replace("**", "")
        text = re.sub(r"^#+\s*", "", text)
        rows.append([text])
    width = max((len(r) for r in rows), default=1)
    return [r + [""] * (width - len(r)) for r in rows]


def write_csv(md: str, path: Path = OUT_CSV) -> None:
    rows = md_to_sheet_rows(md)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    md = build()
    OUT.write_text(md, encoding="utf-8")
    write_csv(md)
    print(f"wrote {OUT}")
    print(f"wrote {OUT_CSV}")
