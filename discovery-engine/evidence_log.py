"""Generate the AI Evidence Log (assignment deliverable) from the live data, so every number is reproducible.

Writes ../evidence-log/AI_Evidence_Log.md. Rerun after any corpus rebuild or golden-set edit:
    python3 evidence_log.py
Paste the output into a Google Doc shared as "anyone with the link can view".
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ENGINE_DIR = Path(__file__).resolve().parent
OUT = ENGINE_DIR.parent / "evidence-log" / "AI_Evidence_Log.md"
ENGINE_URL = "https://groww-sip-discovery-engine-ve6n6mkspofse9xbwkzrsh.streamlit.app"
REPO_URL = "https://github.com/parasharakk-png/groww-sip-discovery-engine"
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
    from golden_set_eval import evaluate, load_golden, summary_counts, validate
    from question_guard import run_guard_tests

    df = app.load_posts()
    log = pd.read_csv(app.SELECTION_LOG, dtype=str).fillna("")
    golden = load_golden()
    issues = validate(golden, df)
    missing = [i.split(":")[0] for i in issues if i.endswith("not in corpus")]
    scores, misses, _ = evaluate(golden, df)
    sc = summary_counts(golden)
    verified = int(golden.get("human_verified", pd.Series(dtype=str)).fillna("").str.strip().ne("").sum())
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
                              f"{e['Engine label']}*. {why}.")

    show = scores.copy()
    for c in ("Agreement %", "Precision %", "Recall %"):
        show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}%")
    show = show.rename(columns={"Agreement %": "Agreement", "Precision %": "Precision", "Recall %": "Recall"})

    parts = [
        "# AI Evidence Log — Groww SIP Discovery Engine",
        f"Generated {date.today():%d %b %Y} from the live data by `discovery-engine/evidence_log.py`. "
        f"Engine: {ENGINE_URL} · Code and data: {REPO_URL}",
        "",
        "## 1. Sources and volume",
        f"The engine analyses **{len(df):,} public posts** ({int((df['cohort'] == target).sum())} from first-time "
        f"investors, {int((df['cohort'] != target).sum()):,} from other investors as a comparison group), selected "
        f"from {len(log):,} pooled posts. {len(risk)} posts describe stopping, pausing, a failed payment or a strong "
        f"link to lapse; {int(df['is_continuer'].sum())} describe continuing.",
        "",
        _md_table(_sources(df, log, app.SOURCE_METHODS, target)),
        "",
        "**How posts were selected**",
        "",
        _md_table(_funnel(log)),
        "",
        "**AI used at each step**",
        "",
        "- **First-pass screening** (is this a first-time investor's own experience?): "
        f"{', '.join(f'{m}, {n} candidates' for m, n in screen.items())}.",
        f"- **Problem-fit coding** ({fit_n:,} posts: investor stage, months since start, SIP status, core concern): "
        "MiniMax-M2.7-highspeed through an Anthropic-compatible gateway; prompt versioned and cached.",
        f"- **Codebook coding** (lifecycle, trigger, emotion, behaviour, link to lapse, quote): "
        f"{codes.get('llm', 0):,} posts coded by the LLM (latest pass MiniMax-M2.7-highspeed), cached in "
        "checkpoints so reruns reuse them.",
        "- **Answer layer** (Ask the engine): the LLM reasons over a numeric evidence pack and cites post IDs; every "
        "question is checked by a rule-based guard first.",
        "- **Not AI:** the cause taxonomy, opportunity sizing and golden-set scoring are fixed, auditable rules.",
        f"- **Quote check:** {int((df['quote_check'] == 'Verbatim').sum()):,} AI-extracted quotes were found verbatim "
        f"in their post; {int((df['quote_check'] == 'Not found in post').sum())} were not.",
        "",
        "## 2. Golden set: hand labels vs the engine",
        f"**{len(golden)} posts** ({sc['p2']} for Problem 2 — stopping in a market fall, my priority; {sc['p1']} for "
        "Problem 1 — payment / mandate failure) are labelled against a written codebook with 19 fields. Rows were "
        "chosen to be hard: core cases, hard negatives that fool keyword rules, early signals and counter-evidence "
        "to my own solution.",
        "",
        (f"- **Hand check:** all {len(golden)} labels checked by me, reading each post."
         if verified == len(golden) else
         f"- **Hand check:** labels were drafted with AI help; **{verified} of {len(golden)}** checked by me so "
         "far (column `human_verified`). Scores below are against the current labels."),
        (f"- {len(missing)} golden posts ({', '.join('#' + m for m in missing)}) left the corpus in the latest rebuild "
         "and are excluded from scoring." if missing else "- All golden posts are in the current corpus."),
        "",
        "**How often the engine agreed with me**",
        "",
        _md_table(show[["Problem", "What is checked", "Engine method", "Rows", "Agreement", "Precision", "Recall"]]),
        "",
        "**Where it failed** (lowest agreement first, with the labelling rationale)",
        "",
        *fail_lines,
        "",
        "**What the golden set changed**",
        "",
        f"- Of {sc['p2_stop']} hand-labelled stops in a fall, only {sc['p2_stop_fall_emotion']} are fear or an "
        f"expectation gap; {sc['p2_stop_timing']} are market timing, {sc['p2_stop_fund']} fund-specific and "
        f"{sc['p2_stop_cash']} cash needs. \"Panic\" was too narrow.",
        f"- My original nudge (share of investors continuing in this fund) fully fits {sc['p2_nudge_yes']} posts and "
        f"partly fits {sc['p2_nudge_partial']}; it does not fit {sc['p2_nudge_no']}.",
        f"- Of {sc['p1_failures']} payment failures, {sc['p1_not_notified']} were not notified in time and "
        f"{sc['p1_fixable']} were fixable by the user with the right information.",
        "",
        "## 3. Where I rejected or corrected AI output",
        "",
        "### 1. The LLM's stop trigger could not explain stopping — replaced with an auditable cause taxonomy",
        f"- **AI output:** the codebook LLM gives each post one trigger. For {int(unstated.sum())} of {len(risk)} "
        f"posts about stopping ({unstated.mean() * 100:.0f}%) it said *none obvious / other / unknown*, and the "
        f"stated ones were broad (top: {', '.join(stated_top.head(3).index)}).",
        "- **Why I rejected it:** a vague label per post can't separate a failed mandate from a fund switch or a "
        "cash need, so trigger counts said nothing about *why* SIPs end.",
        f"- **Correction:** a fixed, versioned text-pattern taxonomy of 10 causes (reasoning.py) assigns each post a "
        f"primary cause; it covers {cause_cover} of {len(risk)} stop posts. Patterns are visible in the app and "
        "scored against the golden set.",
        "",
        "### 2. The AI's reading of why people stop in a fall didn't match the posts — not used for sizing",
        f"- **AI output:** the LLM codebook found only {llm_fall_recall:.0f}% of the stops that a fall caused "
        f"(text rules: {rule_fall_recall:.0f}%), and the engine's stop driver matched the hand labels on only "
        f"{driver_agree:.0f}% of stops, mixing up fear, market timing and fund-specific reasons.",
        "- **Why I rejected it:** sizing from the LLM label would under-count fall-driven stops, and a driver that "
        "can't tell fear from timing can't say whether 'panic' is the problem. Reading the posts, fear was a "
        f"minority ({sc['p2_stop_fall_emotion']} of {sc['p2_stop']} stops, with expectation gaps); the common thread "
        "was having no way to judge whether the SIP was on track.",
        "- **Correction:** sizing uses the text-pattern causes, not the LLM's fall-causation label; Problem 2 is "
        "reframed from \"panic\" to \"no way to judge my own SIP at the stop "
        "moment\", and fund-level social proof was dropped from the solution.",
        "",
        "### 3. The AI answered personal-advice and support questions — added a question guard",
        "- **AI output:** when I tested it with real user questions — *\"Should I stop SIP?\"*, *\"Which mutual fund "
        "to buy?\"*, *\"Which bank account should I open?\"*, *\"OTP is not coming\"* — the answer layer answered "
        "them from the evidence pack instead of declining.",
        "- **Why I rejected it:** the brief forbids fund recommendations, and an evidence engine has nothing valid "
        "to say about one person's account.",
        f"- **Correction:** a rule-based guard classifies every question first (research, advice, support, unsafe, "
        f"off-topic, prevalence) and declines with a reason and a research reframe; {guard_pass} of {len(guard)} "
        f"test questions pass ({guard_kinds[kind_col].nunique()} kinds). The LLM also has a refusal step, and below "
        f"{app.MIN_EVIDENCE} matching posts the engine abstains.",
        "",
        "## 4. Limitations",
        "- Public posts are not a random sample of investors; shares are of posts, not people.",
        f"- Few first-time investors post about stopping ({int((risk['cohort'] == target).sum())} posts), so "
        "first-time splits are hypotheses for the interviews, not measurements.",
        "- Uplift estimates combine post shares with editable assumptions (lapse rate, recovery, severity, "
        "addressability); Groww cohort data would replace them.",
        f"- The golden set is small ({len(golden)} rows) and checked by one person"
        + (f"; {len(golden) - verified} rows still await that check." if verified < len(golden) else "."),
    ]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
