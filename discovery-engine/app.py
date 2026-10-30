"""
Groww SIP Discovery Engine — reasons about causes and sizes opportunities (not review summarization).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from discovery_analytics import (
    METRIC_BRANCH_LABELS,
    enrich_dataframe,
    opportunity_by_metric_branch,
)
from golden_set_eval import CODEBOOK, evaluate, load_golden, summary_counts, validate
from hypotheses import build_hypotheses, strength
from llm_client import make_client, stream_text
from plain_labels import BRANCH_PLAIN, LAPSE_PLAIN, opportunity_score_plain, plain_branch, plain_trigger
from question_guard import MESSAGES, GuardResult, classify_question, is_intervention_question, run_guard_tests
from reasoning import (
    LLM_INSTRUCTIONS,
    LLM_REFUSAL_PREFIX,
    MIN_EVIDENCE,
    OPPORTUNITIES,
    SMALL_SAMPLE,
    STOP_REASONS,
    build_reasoned_analysis,
    format_compact_pack,
    opportunity_table,
    tag_themes,
)
from rule_based_narrative import reasoned_narrative, reasoned_summary

ENGINE_DIR = Path(__file__).resolve().parent
DATA_CSV = ENGINE_DIR / "data" / "grow_sip_956.csv"

NORTH_STAR = (
    "**Business metric:** % of first-time MF investors with an active SIP at month 12 "
    "(≥1 successful instalment in last 30 days)."
)

FIELDS = {
    "source": "SOURCE",
    "lifecycle": "LIFECYCLE",
    "trigger": "TRIGGER",
    "emotion": "EMOTION",
    "behaviour": "BEHAVIOUR",
    "link_to_lapse": "LINK_TO_LAPSE",
    "segment": "SEGMENT",
    "workaround": "WORKAROUND",
    "quote": "QUOTE",
    "selection": "SELECTION",
}
FREE_TEXT = {"segment", "workaround", "quote", "selection"}
COHORT_TARGET = "Target: first-time / young"
COHORT_COMPARISON = "Comparison: other investors"

SELECTION_LOG = ENGINE_DIR / "data" / "selection_log.csv"
POST_DATES = ENGINE_DIR / "data" / "post_dates.csv"
CODED_CHECKPOINT = ENGINE_DIR / "coded_posts.checkpoint.jsonl"
SCREEN_CHECKPOINT = ENGINE_DIR / "data" / "screening.checkpoint.jsonl"
GOLDEN_HUMAN = "Hand-labelled golden set"

# How each source was collected (collect.py) and what that means for reuse.
SOURCE_METHODS = {
    "Play Store": ("Unofficial google-play-scraper library (Groww app reviews)",
                   "Not an official API: check Google Play's terms before reuse"),
    "App Store": ("Apple's public customer-reviews RSS feed (Groww app)", "Public feed"),
    "Reddit": ("Public subreddit search RSS feeds, keyword queries", "Public feed; keyword search over-selects "
               "stop / pause / autopay threads"),
    "TradingQ&A": ("The forum's public search endpoint, keyword queries", "Public endpoint"),
    "YouTube": ("Unofficial youtube-comment-downloader library", "Not the official API; dates are relative "
                "('3 weeks ago'), kept at month precision"),
}

# The nine discovery questions from the problem statement, in its order.
PROBLEM_QUESTIONS: list[tuple[str, str]] = [
    ("Why they start", "Why do people start their first SIP and what do they expect?"),
    ("How behaviour changes", "How does investor behaviour change over the first months?"),
    ("Pause, reduce or stop", "What leads investors to pause, reduce or stop a SIP?"),
    ("What they check", "What do investors check once they have started, and how often?"),
    ("What confuses them", "What confuses investors in the app?"),
    ("Outside the app", "What do investors do outside the app when unsure?"),
    ("Continuers vs drop-offs", "What separates investors who continue from those who drop off?"),
    ("Segment differences", "How do patterns differ for first-time and young investors?"),
    ("Unmet needs", "What unmet needs come up repeatedly?"),
]


def load_secrets() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        s = st.secrets
        out["api_key"] = str(s.get("ANTHROPIC_API_KEY", "") or "")
        out["base_url"] = str(s.get("ANTHROPIC_BASE_URL", "") or "").strip() or None
        out["model"] = str(s.get("ANTHROPIC_MODEL", "") or "claude-3-5-haiku-20241022")
    except Exception:
        pass
    if not out.get("api_key"):
        from dotenv import load_dotenv

        load_dotenv(ENGINE_DIR / ".env")
        out["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        out["base_url"] = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip() or None
        out["model"] = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022").strip()
    return out


def _extract_field(text: str, key: str, free_text: bool) -> str:
    m = re.search(rf"^{key}:[ \t]*(.*)$", text, re.M)
    val = m.group(1).strip() if m else ""
    if free_text:
        return val
    return val or "unknown"


def _extract_post(text: str) -> str:
    m = re.search(r"^POST:[ \t]*(.*?)(?:\nURL:|\Z)", text, re.M | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else text


@st.cache_data(show_spinner=False)
def load_posts() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, dtype=str).fillna("")
    df["item_id"] = df["item_id"].astype(str)
    for col, key in FIELDS.items():
        df[col] = df["content"].apply(lambda t, k=key, f=col in FREE_TEXT: _extract_field(t, k, f))
    df["post_text"] = df["content"].apply(_extract_post)
    df["post_snippet"] = df["post_text"].str.slice(0, 200)
    is_comparison = (df["selection"] == "") | df["selection"].str.startswith("comparison")
    df["cohort"] = is_comparison.map({True: COHORT_COMPARISON, False: COHORT_TARGET})
    if POST_DATES.exists():
        dates = pd.read_csv(POST_DATES, dtype=str).fillna("")
        df = df.merge(dates, on="item_id", how="left").fillna({"published": "", "precision": ""})
    else:
        df["published"], df["precision"] = "", ""
    df["year"] = pd.to_numeric(df["published"].str[:4], errors="coerce")
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731
    df["quote_check"] = [
        "No quote" if not q.strip() else "Verbatim" if norm(q) in norm(p) else "Not found in post"
        for q, p in zip(df["quote"], df["post_text"])
    ]
    golden = load_golden()
    verified = golden.get("human_verified", pd.Series("", index=golden.index)).fillna("").str.lower()
    df["in_golden"] = df["item_id"].isin(set(golden["item_id"].astype(str)))
    df["golden_verified"] = df["item_id"].isin(set(golden.loc[verified.isin(["yes", "y", "true", "1"]), "item_id"].astype(str)))
    return tag_themes(enrich_dataframe(df))


def _file_version(*paths: Path) -> str:
    h = hashlib.sha1()
    for p in paths:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


@st.cache_data(show_spinner=False)
def provenance() -> dict[str, str]:
    """Versions shown under every answer, so a reviewer can tell which data, rules and prompt produced it."""
    def top_model(path: Path, key: str) -> str:
        if not path.exists():
            return "unknown"
        counts: dict[str, int] = {}
        for line in path.read_text().splitlines():
            try:
                m = json.loads(line).get(key) or "unknown"
            except ValueError:
                continue
            counts[m] = counts.get(m, 0) + 1
        return max(counts, key=counts.get) if counts else "unknown"

    return {
        "dataset": _file_version(DATA_CSV, POST_DATES),
        "rules": _file_version(ENGINE_DIR / "reasoning.py", ENGINE_DIR / "rule_based_narrative.py",
                               ENGINE_DIR / "question_guard.py"),
        "prompt": hashlib.sha1(LLM_INSTRUCTIONS.encode()).hexdigest()[:8],
        "coding_model": top_model(CODED_CHECKPOINT, "coding_model"),
        "screen_model": top_model(SCREEN_CHECKPOINT, "screen_model"),
    }


def render_post_card(p: pd.Series, extra: str = "") -> None:
    """One evidence post with where it came from and how far its labels can be trusted."""
    text = re.sub(r"\s+", " ", p.get("post_clean") or p["post_text"]).strip()
    labels = {t.code: t.label for t in STOP_REASONS}
    with st.container(border=True):
        st.markdown(f"> {text[:360]}{'…' if len(text) > 360 else ''}")
        when = p["published"][:7] if p["precision"] == "month" else p["published"]
        st.caption(
            f"**#{p['item_id']}** · {p['source']} · {when or 'date unknown'} · {p['cohort']} · "
            f"[open source ↗]({p['url']})" + (f" · {extra}" if extra else "")
        )
        tags = [
            f"Cause: {labels.get(p['primary_reason'], 'not stated')}",
            f"Behaviour: {p['behaviour'].replace('_', ' ')}",
            f"Stage: {p['lifecycle'].replace('_', ' ')}",
            f"Emotion: {p['emotion'].replace('_', ' ')}",
        ]
        st.caption(" · ".join(tags))
        st.caption(
            f"Tags: codebook by {provenance()['coding_model']}, cause by text rules · Quote check: {p['quote_check']}"
            + _review_status(p)
        )


def _review_status(p: pd.Series) -> str:
    if p["golden_verified"]:
        return f" · **In the {GOLDEN_HUMAN.lower()}**, labels verified by a human"
    if p["in_golden"]:
        return f" · **In the {GOLDEN_HUMAN.lower()}**, labels awaiting human verification"
    return " · not individually reviewed"


def table_view(data: pd.DataFrame, label: str = "Table view") -> None:
    with st.expander(label):
        st.dataframe(data, hide_index=True, width="stretch")


def render_ai_reasoning(question: str, a: dict, cfg: dict[str, str]) -> None:
    pack = format_compact_pack(a)
    cache = st.session_state.setdefault("ai_answers", {})
    key = (question, cfg["model"], hash(pack))
    if key in cache:
        st.caption(f"Source: AI reasoning ({cfg['model']}) · cached")
        text, usage = cache[key]
        _show_ai_answer(text)
        _ai_provenance(a, cfg, usage)
        return
    st.caption(f"Source: AI reasoning ({cfg['model']}) · usually 20–60 s; the computed answer is in the next tab")
    prompt = f"{LLM_INSTRUCTIONS}\n\n=== EVIDENCE PACK ===\n{pack}"
    usage: dict = {}
    try:
        with st.spinner("Model is thinking before it writes…"):
            client = make_client(cfg["api_key"], cfg.get("base_url"))
            chunks = stream_text(client, cfg["model"], prompt, usage=usage)
            first = next(chunks)
        if first.lstrip().startswith(LLM_REFUSAL_PREFIX[:6]):
            text = (first + "".join(chunks)).strip()
            _show_ai_answer(text)
        else:
            with st.container(border=True):
                text = st.write_stream(itertools.chain([first], chunks))
        cache[key] = (text, usage)
        _ai_provenance(a, cfg, usage)
    except Exception as e:  # noqa: BLE001
        st.warning(
            f"AI reasoning unavailable ({str(e)[:200]}). The computed reasoning tab has the full answer."
        )


def _ai_provenance(a: dict, cfg: dict[str, str], usage: dict) -> None:
    v = provenance()
    tokens = (f" · {usage['input_tokens']:,} input / {usage['output_tokens']:,} output tokens"
              if usage.get("input_tokens") is not None and usage.get("output_tokens") is not None else "")
    st.caption(
        f"Model {cfg['model']} · prompt {v['prompt']} · dataset {v['dataset']} · evidence pack from "
        f"{a['pop_n']} posts, {len(a['relevant_posts'])} quoted{tokens}. The model sees only this pack; "
        "every number must come from it."
    )


def _computed_provenance(a: dict) -> None:
    v = provenance()
    st.caption(
        f"Computed by rules {v['rules']} · dataset {v['dataset']} ({a['corpus_n']} posts; tags by {v['coding_model']} "
        f"against `codebook.json`, causes by text rules) · {a['pop_n']} posts used for this question."
    )


def _show_ai_answer(text: str) -> None:
    if not text:
        st.warning("The model returned no answer. Use the computed reasoning tab.")
    elif text.startswith(LLM_REFUSAL_PREFIX):
        st.info(text)
    else:
        with st.container(border=True):
            st.markdown(text)


def _ask_instead(question: str) -> None:
    st.session_state.discovery_q_input = question
    st.session_state.discovery_question = question


def render_guard_message(question: str, g: GuardResult) -> None:
    title, body = MESSAGES[g.kind]
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body)
        st.caption(
            f"Question check: {g.reason}. Neither the computed nor the AI reasoning was run, "
            "so no answer is generated for this question."
        )
        if g.reframe:
            st.markdown(f"**A research question the engine can answer instead:** {g.reframe}")
            st.button("Ask this instead", key="guard_reframe", on_click=_ask_instead, args=(g.reframe,))
        else:
            st.markdown("Try one of the nine discovery questions above.")


def render_computed_answer(a: dict, summary: dict) -> None:
    st.caption("Source: computed reasoning (rules over coded data) · instant")
    with st.container(border=True):
        st.markdown("**Answer**")
        st.markdown(summary["answer"])
        if summary["why"]:
            st.markdown("**Why**")
            st.markdown("\n".join(f"- {w}" for w in summary["why"]))
        if summary["metric"]:
            st.markdown("**What it means for the 12-month metric**")
            st.markdown(summary["metric"])
        st.markdown("**Validate next**")
        st.markdown("\n".join(f"- {v}" for v in summary["validate"]))
    _computed_provenance(a)
    with st.expander("Full computed reasoning (mechanisms, more evidence, all linked opportunities)"):
        st.markdown(reasoned_narrative(a))


def render_guard_tests() -> None:
    res = run_guard_tests()
    passed = int(res["pass"].sum())
    with st.expander(f"How questions are checked before answering ({passed}/{len(res)} test questions routed correctly)"):
        st.markdown(
            "Every question is checked first, without AI. **Research** questions about how investors behave are "
            "answered. **Personal advice** (which fund, should I stop), **account support** (OTP, login, KYC), "
            "**fear / FOMO tactics** and **off-topic** questions are declined, and where possible a research "
            "version of the question is offered instead. The AI prompt repeats the same rules as a second layer."
        )
        st.dataframe(
            res.rename(columns={"question": "Question", "expected": "Expected", "got": "Engine", "pass": "Correct",
                                "note": "Why it is in the test set"}),
            hide_index=True,
            width="stretch",
        )


def _overrides_from_state() -> dict[str, dict[str, float]]:
    return st.session_state.get("opp_overrides", {})


P1_TERMS = re.compile(r"payment|mandate|auto-?pay|debit|retry|bank|salary|instal|fail", re.I)
P2_TERMS = re.compile(r"fall|crash|dip|correction|panic|volatil|bear|other investors|others|social proof|nudge", re.I)


def _evidence_basis(a: dict) -> tuple[int, str]:
    """The number of posts the answer actually rests on: the named theme if there is one, else the group."""
    if a["focus"]:
        f = max(a["focus"], key=lambda x: x["in_group"])
        return f["in_group"], f"posts that mention {f['label'].lower()}"
    return a["pop_n"], "posts"


def render_not_enough_evidence(df: pd.DataFrame, a: dict, n: int, label: str) -> None:
    with st.container(border=True):
        st.markdown("**Not enough evidence to answer this reliably.**")
        st.markdown(
            f"Only {n} {label} were found in this group; the engine needs at least {MIN_EVIDENCE} before it "
            "draws a conclusion. Relevant posts that use other wording may be missed."
        )
        st.caption("No computed or AI answer was generated. Try one of the nine discovery questions above.")
    if a["relevant_posts"]:
        st.markdown("**Closest posts**")
        posts = df.set_index("item_id", drop=False)
        for e in a["relevant_posts"][:3]:
            render_post_card(posts.loc[e["id"]])


def _problem_code(question: str, a: dict) -> str | None:
    codes = {f["code"] for f in a["focus"]}
    if "involuntary_system" in codes or a["intent"].code == "payment":
        return "payment_reliability"
    if codes & {"market_timing", "decision_doubt", "poor_returns"} or P2_TERMS.search(question):
        return "decision_support"
    if P1_TERMS.search(question):
        return "payment_reliability"
    return None


def _golden_check(question: str, a: dict) -> str:
    code = _problem_code(question, a)
    return _golden_note(code) if code else ""


def render_intervention_card(question: str, a: dict) -> None:
    code = _problem_code(question, a)
    if code is None:
        relevant = a["opportunities"][a["opportunities"]["Relevant to question"]]
        code = (relevant if not relevant.empty else a["opportunities"]).iloc[0]["code"]
    o = next(o for o in OPPORTUNITIES if o.code == code)
    with st.container(border=True):
        st.markdown("**This asks whether a product change would work. Public posts can't show that; only an "
                    "experiment can.**")
        st.markdown(
            "Posts show the need behind the idea and who has it, which is what the answer below describes. "
            f"Closest opportunity: **{o.label}**."
        )
        st.markdown(f"- **How to test it:** {o.experiment}\n- **Metric to watch:** {o.business_metric}")
        if code == "decision_support":
            st.markdown(
                "- **Guardrail:** keep any message neutral and factual. Fund-level 'others are continuing' numbers "
                "can read as a fund endorsement or a herd signal, which the brief rules out."
            )


def render_reasoned_results(df: pd.DataFrame, question: str, use_llm: bool) -> None:
    guard = classify_question(question)
    if not guard.allowed:
        render_guard_message(question, guard)
        return
    lapse = st.session_state.get("lapse_rate", 40) / 100
    recovery = st.session_state.get("recovery_rate", 30) / 100
    a = build_reasoned_analysis(df, question, lapse, recovery, _overrides_from_state())
    intervention = is_intervention_question(question)
    if intervention:
        a["is_hypothesis"] = False
    intent = a["intent"]
    basis_n, basis_label = _evidence_basis(a)
    if basis_n < MIN_EVIDENCE:
        render_not_enough_evidence(df, a, basis_n, basis_label)
        return

    summary = reasoned_summary(a)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Read as", intent.title)
    c2.metric("Posts analysed", f"{a['pop_n']} / {a['corpus_n']}")
    c3.metric("First-time investor posts", a["first_timer_n"])
    c4.metric("Top opportunity uplift", f"+{summary['uplift']} pp" if summary["uplift"] is not None else "—")
    st.caption(
        f"Group: {intent.population}, compared against {intent.base} (n={a['base_n']}). "
        f"A cause could be inferred for {a['explained_pct']}% of the group."
    )
    if basis_n < SMALL_SAMPLE:
        st.warning(
            f"**Small sample:** this answer rests on {basis_n} {basis_label}. Treat it as a lead for interviews, "
            "not a finding."
        )
    if intervention:
        render_intervention_card(question, a)
    golden = _golden_check(question, a)
    if golden:
        st.info(golden)

    cfg = load_secrets()
    if use_llm and cfg.get("api_key"):
        ai_tab, rules_tab = st.tabs(["AI reasoning", "Computed reasoning"])
        with rules_tab:
            render_computed_answer(a, summary)
        with ai_tab:
            render_ai_reasoning(question, a, cfg)
    else:
        render_computed_answer(a, summary)

    with st.expander("Data behind this answer (causes, kind of stop, opportunities, segments, all evidence posts)"):
        render_answer_data(df, a)


def render_answer_data(df: pd.DataFrame, a: dict) -> None:
    intent = a["intent"]
    t1, t2, t3, t4, t5 = st.tabs(
        ["Root causes", "Kind of stop", "Opportunity comparison", "Segments & pathways", "Evidence"]
    )
    with t1:
        prof = a["profile"].drop(columns=["code"]).rename(
            columns={"% comparison": f"% in {intent.base}", "Lift": "Lift (×)"}
        )
        st.dataframe(prof, hide_index=True, width="stretch")
        st.caption(
            "Lift = how many times more common in this group than in the comparison base. "
            "'% among continuers' shows whether the same theme also appears among people who kept going — "
            "if it does, the theme alone does not explain stopping."
        )
        chart = a["profile"].set_index("Theme")[["% of group", "% among continuers"]]
        st.bar_chart(chart, horizontal=True, stack=False, color=[CB_ORANGE, CB_BLUE])
        if not a["needs"].empty:
            st.markdown("**Stated unmet needs, grouped**")
            st.dataframe(a["needs"], hide_index=True, width="stretch")
    with t2:
        sc = a["stop_classes"].drop(columns=["code"])
        st.dataframe(sc, hide_index=True, width="stretch")
        st.bar_chart(sc.set_index("Kind of stop")["% of group"], horizontal=True)
        st.caption("Each post is counted once. Only 'Deliberate exit' is a decision to leave investing.")
    with t3:
        render_opportunity_table(a["opportunities"], highlight=True)
    with t4:
        st.markdown("**First-time vs other investors (same group)**")
        st.dataframe(a["segments"], hide_index=True, width="stretch")
        if not a["lifecycle"].empty:
            st.markdown("**Journey stage (where tagged)**")
            st.dataframe(a["lifecycle"], hide_index=True, width="stretch")
        st.markdown("**Most common cause → behaviour pathways**")
        st.dataframe(a["pathways"], hide_index=True, width="stretch")
    with t5:
        labels = {t.code: t.label for t in intent.lens}
        posts = df.set_index("item_id", drop=False)
        st.caption(f"Up to 3 posts per theme, most relevant to the question first. {MIN_EVIDENCE}+ matching posts "
                   f"are needed for an answer; fewer than {SMALL_SAMPLE} is labelled a small sample.")
        for code, items in a["evidence"].items():
            st.markdown(f"**{labels.get(code, code)}**")
            for e in items:
                render_post_card(posts.loc[e["id"]])


def render_opportunity_table(opps: pd.DataFrame, highlight: bool = False) -> None:
    cols = [
        "Opportunity",
        "Metric branch",
        "Evidence posts",
        "% of at-risk posts",
        "Lift vs all posts",
        "% among continuers",
        "First-timer share %",
        "Lapse share %",
        "Addressability",
        "Leverage index",
        "Est. uplift (pp)",
        "Confidence",
    ]
    if highlight and "Relevant to question" in opps:
        cols.insert(1, "Relevant to question")
    st.dataframe(
        opps[cols],
        hide_index=True,
        width="stretch",
        column_config={
            "% of at-risk posts": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Est. uplift (pp)": st.column_config.NumberColumn(format="+%.2f pp"),
            "Lift vs all posts": st.column_config.NumberColumn(format="%.1f×"),
        },
    )
    st.bar_chart(opps.set_index("Opportunity")["Est. uplift (pp)"], horizontal=True, y_label="",
                 x_label="Est. uplift (pp)", color=BRAND)
    with st.expander("Product direction per opportunity (no fund advice, no fear/urgency)"):
        for _, o in opps.iterrows():
            ids = ", ".join(f"#{i}" for i in o["evidence_ids"])
            st.markdown(f"**{o['Opportunity']}** — {o['Product direction']}  \nEvidence: {ids}")
    st.caption(
        "Lapse share = share of at-risk posts × severity (normalised). Leverage = lapse share × "
        "addressability. Est. uplift = assumed 12-month lapse rate × lapse share × addressability × "
        "recovery rate. Edit the assumptions in the Opportunity sizing tab."
    )


OPP_SHORT = {
    "payment_reliability": "Payment & mandate reliability",
    "switch_without_gap": "Switch fund without a gap",
    "decision_support": "Support at 'should I stop?'",
    "cash_flexibility": "Cash-flow flexibility",
    "control_clarity": "Pause / skip controls",
    "understandability": "'Is my SIP working?'",
}
BRAND, CB_ORANGE, CB_BLUE = "#00B386", "#E69F00", "#0072B2"
OPP_VIEWS = {
    "Estimated uplift": "Percentage points added to 12-month SIP continuation if the fix works, under the "
                        "assumptions in use. This is the ranking metric.",
    "Share of stop posts": "Share of posts about stopping whose primary cause falls in this area: the size of the "
                           "evidence behind each bet.",
    "Stoppers vs continuers": "How often each theme is mentioned by people who stopped vs people who kept going. A "
                              "theme just as common among continuers does not explain drop-off.",
    "Severity vs addressability": "The two judgement inputs. Top-right = the moment usually ends the SIP and Groww can "
                                  "fix it. Bubble size = evidence posts.",
}


def _opp_ratio(o: pd.Series) -> float | None:
    cont = float(o["% among continuers"])
    return round(float(o["Mentioned in at-risk %"]) / cont, 1) if cont else None


def render_opportunity_assumptions() -> dict[str, dict[str, float]]:
    defaults = pd.DataFrame(
        [{"code": o.code, "Opportunity": o.label, "Severity": o.severity, "Addressability": o.addressability}
         for o in OPPORTUNITIES]
    )
    with st.expander("Assumptions (open to test your own judgement)"):
        c1, c2 = st.columns(2)
        c1.slider(
            "Assumed % of first-time investors with no active SIP at month 12",
            10, 80, key="lapse_rate",
            help="Replace with Groww cohort data. Public posts can't measure this.",
        )
        c2.slider("Assumed % of addressable lapses a good fix recovers", 5, 80, key="recovery_rate")
        st.markdown(
            "**Severity**: if nothing changes, how likely this moment ends with no active SIP.  \n"
            "**Addressability**: how much Groww can fix it within the constraints (no fund advice, no fear or FOMO)."
        )
        edited = st.data_editor(
            defaults,
            hide_index=True,
            disabled=["code", "Opportunity"],
            column_config={
                "code": None,
                "Severity": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05),
                "Addressability": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05),
            },
            key="opp_editor",
        )
    overrides = {
        r["code"]: {"severity": float(r["Severity"]), "addressability": float(r["Addressability"])}
        for _, r in edited.iterrows()
    }
    st.session_state["opp_overrides"] = overrides
    changed = not edited[["Severity", "Addressability"]].equals(defaults[["Severity", "Addressability"]])
    st.caption(
        f"In use: {st.session_state['lapse_rate']}% of first-time investors lapse by month 12 · a good fix recovers "
        f"{st.session_state['recovery_rate']}% of addressable lapses · severity and addressability "
        f"{'**edited by you**' if changed else 'at default judgement'}."
    )
    return overrides


def _opp_bar_chart(opps: pd.DataFrame, value: str, title: str, fmt: str) -> None:
    d = pd.DataFrame({"name": opps["Short name"], "value": opps[value], "posts": opps["Evidence posts"],
                      "confidence": opps["Confidence"]})
    d["label"] = d["value"].map(lambda v: fmt.format(v))
    base = alt.Chart(d).encode(
        y=alt.Y("name:N", sort=alt.EncodingSortField(field="value", order="descending"), title=None,
                axis=alt.Axis(labelLimit=260, labelFontSize=13)),
        x=alt.X("value:Q", title=title, axis=alt.Axis(grid=False)),
        tooltip=[alt.Tooltip("name:N", title="Opportunity"), alt.Tooltip("value:Q", title=title, format=".2f"),
                 alt.Tooltip("posts:Q", title="Evidence posts"), alt.Tooltip("confidence:N", title="Confidence")],
    )
    chart = base.mark_bar(color=BRAND, cornerRadiusEnd=4, height=26) + base.mark_text(
        align="left", dx=5, fontSize=13, fontWeight="bold").encode(text="label:N")
    st.altair_chart(chart.properties(height=55 * len(d)), use_container_width=True)
    table_view(d.sort_values("value", ascending=False)[["name", "value", "posts", "confidence"]].rename(
        columns={"name": "Opportunity", "value": title, "posts": "Evidence posts", "confidence": "Confidence"}))


def _opp_contrast_chart(opps: pd.DataFrame) -> None:
    order = opps.sort_values("Mentioned in at-risk %", ascending=False)["Short name"].tolist()
    groups = ["Stopped or lapsed", "Kept investing"]
    long = pd.concat([
        pd.DataFrame({"name": opps["Short name"], "Group": groups[0], "pct": opps["Mentioned in at-risk %"]}),
        pd.DataFrame({"name": opps["Short name"], "Group": groups[1], "pct": opps["% among continuers"]}),
    ])
    long["label"] = long["pct"].map(lambda v: f"{v:.0f}%")
    base = alt.Chart(long).encode(
        y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=13)),
        yOffset=alt.YOffset("Group:N", sort=groups),
        x=alt.X("pct:Q", title="% of posts mentioning the theme", axis=alt.Axis(grid=False)),
        color=alt.Color("Group:N", sort=groups, scale=alt.Scale(domain=groups, range=[CB_ORANGE, CB_BLUE]),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
        tooltip=[alt.Tooltip("name:N", title="Opportunity"), "Group:N", alt.Tooltip("pct:Q", title="% of posts")],
    )
    chart = base.mark_bar(cornerRadiusEnd=3) + base.mark_text(align="left", dx=4, fontSize=12).encode(text="label:N")
    st.altair_chart(chart.properties(height=75 * len(opps)), use_container_width=True)
    table_view(long.pivot(index="name", columns="Group", values="pct").reindex(order)[groups]
               .rename_axis("Opportunity").reset_index())


def _opp_matrix_chart(opps: pd.DataFrame) -> None:
    d = pd.DataFrame({"name": opps["Short name"], "sev": opps["Severity"], "addr": opps["Addressability"],
                      "posts": opps["Evidence posts"], "uplift": opps["Est. uplift (pp)"]})
    x = alt.X("addr:Q", scale=alt.Scale(domain=[0, 1.05]), title="Addressability: can Groww fix it?")
    y = alt.Y("sev:Q", scale=alt.Scale(domain=[0, 1.05]), title="Severity: does it end the SIP?")
    tip = [alt.Tooltip("name:N", title="Opportunity"), alt.Tooltip("posts:Q", title="Evidence posts"),
           alt.Tooltip("uplift:Q", title="Est. uplift (pp)", format=".2f")]
    points = alt.Chart(d).mark_circle(color=BRAND, opacity=0.7, stroke="#004D3A", strokeWidth=1).encode(
        x=x, y=y, size=alt.Size("posts:Q", scale=alt.Scale(range=[200, 3000]), legend=None), tooltip=tip)
    labels = alt.Chart(d).mark_text(dy=-22, fontSize=12, fontWeight="bold").encode(x=x, y=y, text="name:N")
    mid = pd.DataFrame({"v": [0.5]})
    guides = (alt.Chart(mid).mark_rule(strokeDash=[4, 4], color="#9E9E9E").encode(x="v:Q")
              + alt.Chart(mid).mark_rule(strokeDash=[4, 4], color="#9E9E9E").encode(y="v:Q"))
    st.altair_chart((guides + points + labels).properties(height=440), use_container_width=True)
    table_view(d.rename(columns={"name": "Opportunity", "sev": "Severity", "addr": "Addressability",
                                 "posts": "Evidence posts", "uplift": "Est. uplift (pp)"}))


def render_opportunity_compare(opps: pd.DataFrame) -> None:
    view = st.segmented_control(
        "Compare the six opportunities by", list(OPP_VIEWS), default="Estimated uplift", key="opp_view"
    ) or "Estimated uplift"
    st.caption(OPP_VIEWS[view])
    if view == "Estimated uplift":
        _opp_bar_chart(opps, "Est. uplift (pp)", "Estimated uplift to 12-month continuation (pp)", "+{:.2f} pp")
    elif view == "Share of stop posts":
        _opp_bar_chart(opps, "% of at-risk posts", "% of posts about stopping (primary cause)", "{:.1f}%")
    elif view == "Stoppers vs continuers":
        _opp_contrast_chart(opps)
    else:
        _opp_matrix_chart(opps)

    st.markdown("**Ranked opportunities**")
    show_all = st.toggle("Show every calculated column", key="opp_all_cols")
    cols = ["Rank", "Opportunity", "Metric branch", "Evidence posts", "% of at-risk posts", "Mentioned in at-risk %",
            "% among continuers", "Est. uplift (pp)", "Confidence"]
    if show_all:
        cols += ["Lift vs all posts", "First-timer share %", "Severity", "Addressability", "Lapse share %",
                 "Leverage index", "Business metric", "Experiment"]
    st.dataframe(
        opps[cols],
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "% of at-risk posts": st.column_config.ProgressColumn(
                "Share of stop posts", min_value=0, max_value=100, format="%.1f%%",
                help="Posts about stopping whose primary cause is in this area"),
            "Mentioned in at-risk %": st.column_config.NumberColumn(
                "Mentioned by stoppers", format="%.1f%%", help="Any mention of the theme in posts about stopping"),
            "% among continuers": st.column_config.NumberColumn(
                "Mentioned by continuers", format="%.1f%%", help="Any mention of the theme by people who kept investing"),
            "Est. uplift (pp)": st.column_config.NumberColumn("Est. uplift", format="+%.2f pp"),
            "Lift vs all posts": st.column_config.NumberColumn(format="%.1f×"),
        },
    )
    st.download_button("Download the opportunity table (CSV)", opps.drop(columns=["evidence_ids"]).to_csv(index=False),
                       file_name="opportunity_sizing.csv")


def _golden_note(code: str) -> str:
    c = summary_counts(load_golden())
    if code == "payment_reliability":
        return (f"Hand-labelled golden set: of {c['p1_failures']} failed instalments, {c['p1_not_notified']} users "
                f"were not told in time, {c['p1_fixable']} could have fixed it with the right prompt, and "
                f"{c['p1_two_account']} come from salary landing in a different account from the SIP mandate.")
    if code == "decision_support":
        fit = c["p2_nudge_yes"] + c["p2_nudge_partial"] + c["p2_nudge_no"]
        return (f"Hand-labelled golden set: of {c['p2_stop']} stops during a fall, {c['p2_stop_fall_emotion']} are "
                f"emotional reactions, {c['p2_stop_timing']} deliberate timing, {c['p2_stop_fund']} fund-specific and "
                f"{c['p2_stop_cash']} cash needs. A 'how many kept their SIP' message fully fits {c['p2_nudge_yes']} "
                f"of {fit} posts.")
    return ""


GOLDEN_CHECK_FOR = {"payment_reliability": "Instalment failed", "decision_support": "Why they stop (driver)"}


def _validation_gaps(df: pd.DataFrame, o: pd.Series, code: str, ratio: float | None) -> list[str]:
    gaps = []
    n = int(o["Evidence posts"])
    if n < SMALL_SAMPLE:
        gaps.append(f"Small sample: {n} evidence posts (fewer than {SMALL_SAMPLE}).")
    sources = o["Sources"]
    if len(sources) == 1:
        gaps.append(f"Evidence comes from a single source ({next(iter(sources))}).")
    else:
        gaps.append("Sources: " + ", ".join(f"{s} {c}" for s, c in sources.items()) + ".")
    if float(o["First-timer share %"]) < 25:
        gaps.append(f"Only {o['First-timer share %']:.0f}% of evidence posts come from first-time or young investors.")
    if ratio is not None and ratio < 1.0:
        gaps.append("Mentioned at least as often by investors who kept going, so it may explain complaints more "
                    "than stopping.")
    check = GOLDEN_CHECK_FOR.get(code)
    if check:
        scores, _, _ = evaluate(load_golden(), df)
        agree = scores.loc[scores["What is checked"].eq(check), "Agreement %"].iloc[0]
        gaps.append(f"Golden set: the text rules agree with hand labels {agree:.0f}% of the time on "
                    f"'{check.lower()}'.")
    else:
        gaps.append("Not covered by the hand-labelled golden set: the cause detection for this area has not been "
                    "checked by hand.")
    ev = df[df["is_at_risk"] & df["primary_reason"].isin(next(x.reasons for x in OPPORTUNITIES if x.code == code))]
    if not ev.empty:
        gaps.append(f"{int((~ev['in_golden']).sum())} of {len(ev)} evidence posts carry AI tags only "
                    "(not individually reviewed).")
    gaps.append("Shares are of public posts, not of Groww users: validate the size with product analytics.")
    gaps.append("Severity, addressability, the metric and the experiment are judgements and have not been tested.")
    return gaps


def render_opportunity_deep_dive(df: pd.DataFrame, opps: pd.DataFrame) -> None:
    by_code = opps.set_index("code")
    code = st.selectbox(
        "Choose an opportunity", opps["code"], key="opp_focus",
        format_func=lambda c: f"#{by_code.at[c, 'Rank']}  {by_code.at[c, 'Opportunity']}",
    )
    o = by_code.loc[code]
    n_risk = int(df["is_at_risk"].sum())
    ratio = _opp_ratio(o)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rank", f"#{o['Rank']} of {len(opps)}")
    m2.metric("Evidence posts", int(o["Evidence posts"]))
    m3.metric("Share of stop posts", f"{o['% of at-risk posts']:.1f}%")
    m4.metric("Stoppers vs continuers", f"{ratio}×" if ratio else "n/a",
              help="How many times more often people who stopped mention this theme than people who kept going")
    m5.metric("Est. uplift", f"+{o['Est. uplift (pp)']:.2f} pp")

    if ratio is None or ratio >= 1.5:
        verdict = "**separates drop-offs from continuers**, so fixing it should move the metric"
    elif ratio >= 1.0:
        verdict = "is only **slightly** more common among drop-offs, so it explains part of the drop-off at best"
    else:
        verdict = ("is **at least as common among people who kept investing**, so it is a weak explanation of "
                   "drop-off on its own")
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("**Why it matters for the 12-month metric**")
            st.markdown(
                f"- **Metric branch:** {o['Metric branch']}\n"
                f"- **Evidence:** primary cause in {int(o['Evidence posts'])} of {n_risk} posts about stopping "
                f"({o['% of at-risk posts']:.1f}%); {o['First-timer share %']:.0f}% of these are from first-time or young investors.\n"
                f"- **Contrast:** mentioned by {o['Mentioned in at-risk %']:.1f}% of people who stopped vs "
                f"{o['% among continuers']:.1f}% of people who kept investing. The theme {verdict}.\n"
                f"- **Confidence: {o['Confidence']}** (High = 25+ posts from 2+ sources; Medium = 10+ posts)."
            )
        with st.container(border=True):
            st.markdown("**How the estimate is built**")
            lapse, rec = st.session_state["lapse_rate"], st.session_state["recovery_rate"]
            st.markdown(
                f"1. Share of stop posts {o['% of at-risk posts']:.1f}% × severity {o['Severity']:.2f}, normalised "
                f"across the six areas = **{o['Lapse share %']:.1f}% of lapses**\n"
                f"2. Lapse share × addressability {o['Addressability']:.2f} = **leverage {o['Leverage index']:.1f}**\n"
                f"3. {lapse}% lapse rate × {o['Lapse share %']:.1f}% × {o['Addressability']:.2f} × {rec}% recovery "
                f"= **+{o['Est. uplift (pp)']:.2f} pp** to 12-month continuation"
            )
    with right:
        with st.container(border=True):
            st.markdown("**Product direction (no fund advice, no fear or urgency)**")
            st.markdown(o["Product direction"])
        with st.container(border=True):
            st.markdown("**Qualitative judgements (not yet tested)**")
            st.markdown(f"- **Business metric to move:** {o['Business metric']}\n"
                        f"- **Suggested experiment:** {o['Experiment']}")
        note = _golden_note(code)
        if note:
            st.info(note)

    with st.container(border=True):
        st.markdown("**Validation gaps**")
        st.markdown("\n".join(f"- {g}" for g in _validation_gaps(df, o, code, ratio)))
    with st.expander("Matching rule: which posts count as evidence for this opportunity"):
        reasons = next(x.reasons for x in OPPORTUNITIES if x.code == code)
        st.markdown(
            f"A post about stopping counts when its **primary cause** is one of: "
            + ", ".join(f"*{t.label}*" for t in STOP_REASONS if t.code in reasons)
            + ". The primary cause is the first cause (in the taxonomy order) whose text pattern matches the post."
        )
        st.dataframe(
            pd.DataFrame([{"Cause": t.label, "Text pattern (regex)": t.pattern} for t in STOP_REASONS
                          if t.code in reasons]),
            hide_index=True, width="stretch",
        )
        st.caption("A 'post about stopping' is one coded as stopped / paused / reduced / switched, at the stage "
                   "stopped / paused / payment failed, or with a high link to lapse.")

    st.markdown("**What users said** (strongest evidence posts)")
    posts = df.set_index("item_id", drop=False)
    for pid in o["evidence_ids"]:
        if pid in posts.index:
            render_post_card(posts.loc[pid])


def render_opportunity_robustness(df: pd.DataFrame, opps: pd.DataFrame,
                                  overrides: dict[str, dict[str, float]]) -> None:
    lapse, rec = st.session_state["lapse_rate"] / 100, st.session_state["recovery_rate"] / 100
    top = opps.iloc[0]

    st.markdown("**1. The lapse and recovery assumptions change the size of each bet, never the order.**")
    st.caption("Both multiply every opportunity equally. The table shows the #1 bet's uplift under other assumptions.")
    grid = pd.DataFrame(
        {f"{r}% recovered": [round(l * float(top["Leverage index"]) * r / 10000, 2) for l in (20, 30, 40, 50, 60)]
         for r in (10, 20, 30, 50)},
        index=[f"{l}% lapse by month 12" for l in (20, 30, 40, 50, 60)],
    )
    st.dataframe(grid, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="+%.2f pp") for c in grid.columns})

    st.markdown(f"**2. Could any other bet overtake #1 ({top['Opportunity']})?**")
    rows = []
    for _, o in opps.iloc[1:].iterrows():
        need = float(top["Leverage index"]) / float(o["Lapse share %"]) if o["Lapse share %"] else float("inf")
        best = {**overrides, o["code"]: {"severity": 1.0, "addressability": 1.0}}
        ranked = opportunity_table(df, lapse, rec, best)
        best_rank = int(ranked.index[ranked["code"].eq(o["code"])][0]) + 1
        rows.append({
            "Opportunity": o["Opportunity"],
            "Leverage now": float(o["Leverage index"]),
            "Addressability needed to tie #1": "not reachable (needs > 1.0)" if need > 1 else f"{need:.2f}",
            "Rank at maximum severity and addressability": f"#{best_rank}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    def lowest(field: str) -> float:
        current = overrides.get(top["code"], {}).get(field.lower(), float(top[field]))
        value = current
        for v in [round(current - 0.05 * i, 2) for i in range(1, int(current / 0.05) + 1)]:
            trial = {**overrides, top["code"]: {**overrides.get(top["code"], {}), field.lower(): v}}
            if opportunity_table(df, lapse, rec, trial).iloc[0]["code"] != top["code"]:
                break
            value = v
        return value

    min_sev, min_addr = lowest("Severity"), lowest("Addressability")
    st.markdown("**3. How far can the #1 bet's own scores fall before it loses the top spot?**")
    r1, r2 = st.columns(2)
    r1.metric("Stays #1 down to severity", f"{min_sev:.2f}", f"from {float(top['Severity']):.2f}", delta_color="off")
    r2.metric("Stays #1 down to addressability", f"{min_addr:.2f}", f"from {float(top['Addressability']):.2f}",
              delta_color="off")
    st.success(
        f"**{top['Opportunity']}** stays the top bet across every lapse and recovery assumption, and keeps the top "
        f"spot even if its severity falls to {min_sev:.2f} or its addressability to {min_addr:.2f}. The ranking rests "
        "on the evidence share, not on the judgement scores."
        if min_sev <= 0.5 and min_addr <= 0.5 else
        f"**{top['Opportunity']}** is #1, but the lead depends on the judgement scores: it loses the top spot below "
        f"severity {min_sev:.2f} or addressability {min_addr:.2f}. Validate these in interviews."
    )


def render_opportunity_method(df: pd.DataFrame) -> None:
    reason_label = {t.code: t.label for t in STOP_REASONS}
    st.markdown("**Which causes feed each opportunity**")
    st.dataframe(
        pd.DataFrame([{"Opportunity": o.label, "Metric branch": o.metric_branch,
                       "Causes grouped": ", ".join(reason_label.get(r, r) for r in o.reasons),
                       "Default severity": o.severity, "Default addressability": o.addressability}
                      for o in OPPORTUNITIES]),
        hide_index=True, width="stretch",
    )
    st.markdown("**Definitions**")
    st.dataframe(
        pd.DataFrame([
            ("Posts about stopping", "Stopped, paused, payment failed, or a strong link to lapse in the codebook.",
             "Evidence"),
            ("Share of stop posts", "Posts about stopping whose primary cause is in the area ÷ all posts about stopping.",
             "Evidence"),
            ("Mentioned by stoppers / continuers", "Any mention of the area's themes among people who stopped vs "
             "people who kept investing (same rule for both groups).", "Evidence"),
            ("Lift vs all posts", "Mention rate among stoppers ÷ mention rate in all posts.", "Evidence"),
            ("Severity", "If nothing changes, how likely this moment ends with no active SIP (0–1).", "Judgement"),
            ("Addressability", "How much Groww can fix it within the constraints (0–1).", "Judgement"),
            ("Lapse share %", "Share of stop posts × severity, normalised so the six areas add to 100%.", "Derived"),
            ("Leverage index", "Lapse share % × addressability. Sets the ranking.", "Derived"),
            ("Est. uplift (pp)", "Lapse rate × lapse share × addressability × recovery rate.", "Derived"),
            ("Confidence", "High = 25+ evidence posts from 2+ sources; Medium = 10+; otherwise Low.", "Evidence"),
        ], columns=["Term", "Definition", "Type"]),
        hide_index=True, width="stretch",
    )
    scores, _, _ = evaluate(load_golden(), df)
    pick = lambda label: scores.loc[scores["What is checked"].eq(label), "Agreement %"].iloc[0]  # noqa: E731
    st.markdown("**Limitations a reviewer should weigh**")
    st.markdown(
        f"""
- **Public posts, not Groww data.** Shares describe what people write about, not how many users lapse. The lapse rate
  is an assumption until replaced with cohort data.
- **Primary causes come from text rules.** On the hand-labelled golden set, the rules detect payment failures with
  {pick('Instalment failed'):.0f}% agreement but identify *why* someone stops during a fall with only
  {pick('Why they stop (driver)'):.0f}% agreement. Treat Payment & mandate reliability as solid and the
  stop-moment areas as directional.
- **Severity and addressability are judgement.** The Robustness view shows how much the ranking depends on them.
- **Source mix:** {round((df['source'] == 'Reddit').mean() * 100)}% of posts are from Reddit, whose users are more
  engaged than a typical first-time investor.
        """
    )


def render_opportunity_tab(df: pd.DataFrame) -> None:
    st.subheader("Opportunity sizing: which fix moves 12-month SIP continuation the most?")
    st.markdown(
        "Every post about stopping or lapsing gets one **primary cause** from its text. Causes are grouped into six "
        "product opportunity areas, each tied to a branch of the metric, and sized with explicit assumptions."
    )
    overrides = render_opportunity_assumptions()
    opps = opportunity_table(df, st.session_state["lapse_rate"] / 100, st.session_state["recovery_rate"] / 100,
                             overrides)
    opps.insert(0, "Rank", range(1, len(opps) + 1))
    opps["Short name"] = opps["code"].map(OPP_SHORT)
    top, second = opps.iloc[0], opps.iloc[1]
    total = float(opps["Est. uplift (pp)"].sum())
    lead = float(top["Est. uplift (pp)"]) / float(second["Est. uplift (pp)"]) if second["Est. uplift (pp)"] else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Posts about stopping analysed", int(df["is_at_risk"].sum()))
    k2.metric("Uplift from the #1 bet", f"+{top['Est. uplift (pp)']:.2f} pp", help=top["Opportunity"])
    k3.metric("Lead over the #2 bet", f"{lead:.1f}×", help=second["Opportunity"])
    k4.metric("All six bets combined", f"+{total:.2f} pp")
    ratio = _opp_ratio(top)
    st.success(
        f"**{top['Opportunity']}** is the highest-leverage bet. It is the primary cause in "
        f"{top['% of at-risk posts']:.0f}% of posts about stopping, is mentioned "
        f"{f'{ratio}× more often' if ratio else 'far more often'} by people who stopped than by people who kept "
        f"investing ({top['Mentioned in at-risk %']:.0f}% vs {top['% among continuers']:.0f}%), and is worth about "
        f"+{top['Est. uplift (pp)']:.2f} pp: {lead:.1f}× the next bet, *{second['Opportunity']}*."
    )

    compare_t, deep_t, robust_t, branch_t, method_t = st.tabs(
        ["Compare", "Deep dive", "Robustness", "Metric branches", "Method & limits"]
    )
    with compare_t:
        render_opportunity_compare(opps)
    with deep_t:
        render_opportunity_deep_dive(df, opps)
    with robust_t:
        render_opportunity_robustness(df, opps, overrides)
    with branch_t:
        render_metric_branches_tab(df)
    with method_t:
        render_opportunity_method(df)


def render_workflow_tab() -> None:
    st.subheader("How this engine works")
    st.markdown(NORTH_STAR)
    st.markdown(
        """
| Step | What happens | Output |
|------|----------------|--------|
| **1. Collect** | Play Store, App Store, Reddit, YouTube, TradingQ&A | Raw posts + URLs |
| **2. Select** | SIP relevance → author's own experience → first-time / young (rules) → LLM screen | Target posts + comparison posts |
| **3. Code** | LLM applies a fixed codebook (lifecycle, trigger, behaviour, link to lapse…) | Structured tags per post |
| **4. Reason** | Text-based cause taxonomy → primary cause + kind of stop per post | Why, not just what |
| **5. Contrast** | Each theme compared with all posts **and** with investors who continued | Which causes actually separate drop-offs |
| **6. Size** | Causes → opportunity areas → leverage & estimated pp uplift | Ranked, comparable bets |
| **7. Answer** | Question routed to the right group; LLM reasons over a numeric evidence pack and cites posts | Reasoned answer + hypotheses to test |
| **8. Validate** | Golden-set agreement + interviews (Evidence & validation tab) | Quality checks |

**Why step 4 exists:** the codebook trigger is one label per post and is often vague, so trigger counts
cannot explain stopping. The cause taxonomy reads the post text with fixed, auditable patterns.

**Why answers differ by question:** the engine detects what the question is about — a cause (returns,
cash, payments…), an action (pause, reduce, cancel) or a yes/no hypothesis — narrows to the matching
posts, tests the theme against stopping (risk ratio, share among continuers) and reasons from the posts
that best match the question.
        """
    )
    st.markdown("**Cause taxonomy**")
    st.dataframe(
        pd.DataFrame([{"Cause": t.label, "Kind of stop": t.stop_class, "Mechanism": t.mechanism} for t in STOP_REASONS]),
        hide_index=True,
        width="stretch",
    )


def render_header(df: pd.DataFrame) -> None:
    st.title("Groww SIP Discovery Engine")
    st.markdown(
        "AI-powered discovery over public investor conversations — **why first-time investors keep or lose "
        "their SIP**, which opportunity areas matter most, and how each insight was validated."
    )
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Goal metric**")
        st.caption("% of first-time MF investors with an active SIP 12 months after their first SIP "
                   "(≥1 successful instalment in the last 30 days).")
    with c2.container(border=True):
        st.markdown("**Evidence**")
        st.caption(f"{len(df)} coded public posts from Reddit, TradingQ&A, YouTube, Play Store and App Store: "
                   f"{int((df['cohort'] == COHORT_TARGET).sum())} first-time / young investors, "
                   f"{int((df['cohort'] == COHORT_COMPARISON).sum())} other investors for comparison.")
    with c3.container(border=True):
        st.markdown("**Guardrails**")
        st.caption("No fund or stock recommendations. No fear, FOMO or urgency nudges. Not investment advice.")


def render_sidebar(df: pd.DataFrame) -> None:
    cfg = load_secrets()
    with st.sidebar:
        st.header("Corpus")
        st.metric("Coded posts", len(df))
        c1, c2 = st.columns(2)
        c1.metric("First-time / young", int((df["cohort"] == COHORT_TARGET).sum()))
        c2.metric("Comparison", int((df["cohort"] == COHORT_COMPARISON).sum()))
        c1.metric("About stopping", int(df["is_at_risk"].sum()))
        c2.metric("Continuers", int(df["is_continuer"].sum()))
        st.markdown("**Sources**")
        st.dataframe(
            df["source"].value_counts().rename_axis("Source").reset_index(name="Posts"),
            hide_index=True, width="stretch",
        )
        st.divider()
        if cfg.get("api_key"):
            st.caption(f"AI reasoning: on · `{cfg['model']}`")
        else:
            st.caption("AI reasoning: off (no API key) · computed reasoning only")


EXPLORER_PAGE_SIZE = 20


def _options(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(v for v in df[col].unique() if v and v != "unknown")


def render_browse_tab(df: pd.DataFrame) -> None:
    st.subheader("Evidence explorer")
    st.caption("Every post the engine uses, with where it came from, how it was tagged and whether a human has "
               "checked it. Open the source to read the original.")
    labels = {t.code: t.label for t in STOP_REASONS}
    labels["unclear"] = "Reason not stated"
    pretty = lambda v: v.replace("_", " ").capitalize()  # noqa: E731

    q = st.text_input("Search by keyword", key="ex_q", placeholder="e.g. autopay, salary account, crash")
    c1, c2, c3 = st.columns(3)
    sel_src = c1.multiselect("Source", _options(df, "source"), key="ex_src")
    sel_r = c2.multiselect("Primary cause", list(labels), format_func=labels.get, key="ex_cause")
    sel_b = c3.multiselect("Metric branch", sorted(df["metric_branch"].unique()),
                           format_func=lambda x: METRIC_BRANCH_LABELS.get(x, x), key="ex_branch")
    c4, c5, c6 = st.columns(3)
    sel_beh = c4.multiselect("Behaviour", _options(df, "behaviour"), format_func=pretty, key="ex_beh")
    sel_life = c5.multiselect("Journey stage", _options(df, "lifecycle"), format_func=pretty, key="ex_life")
    sel_emo = c6.multiselect("Emotion", _options(df, "emotion"), format_func=pretty, key="ex_emo")
    years = df["year"].dropna()
    c7, c8, c9 = st.columns([2, 2, 1])
    lo, hi = int(years.min()), int(years.max())
    yr = c7.slider("Publication year", lo, hi, (lo, hi), key="ex_year")
    cohort = c8.radio("Group", ["All", COHORT_TARGET, COHORT_COMPARISON], horizontal=True, key="ex_group")
    as_table = c9.toggle("Table", key="ex_table", help="Switch between cards and a table")
    f1, f2, f3 = st.columns(3)
    only_risk = f1.checkbox("Only posts about stopping", key="ex_risk")
    only_golden = f2.checkbox(f"Only the {GOLDEN_HUMAN.lower()}", key="ex_golden")
    undated = f3.checkbox("Include posts without a date", value=True, key="ex_undated")

    view = df
    if q:
        view = view[view["content"].str.contains(re.escape(q), case=False, na=False)]
    for col, sel in (("source", sel_src), ("primary_reason", sel_r), ("metric_branch", sel_b),
                     ("behaviour", sel_beh), ("lifecycle", sel_life), ("emotion", sel_emo)):
        if sel:
            view = view[view[col].isin(sel)]
    if cohort != "All":
        view = view[view["cohort"] == cohort]
    in_range = view["year"].between(*yr)
    view = view[in_range | (view["year"].isna() & undated)]
    if only_risk:
        view = view[view["is_at_risk"]]
    if only_golden:
        view = view[view["in_golden"]]
    view = view.sort_values("published", ascending=False)

    pages = max(1, -(-len(view) // EXPLORER_PAGE_SIZE))
    p1, p2 = st.columns([1, 4])
    page = int(p1.number_input("Page", 1, pages, 1, key=f"ex_page_{len(view)}"))
    start = (page - 1) * EXPLORER_PAGE_SIZE
    chunk = view.iloc[start:start + EXPLORER_PAGE_SIZE]
    p2.caption(f"{len(view)} of {len(df)} posts · newest first · showing {start + 1 if len(view) else 0}–"
               f"{start + len(chunk)} · page {page} of {pages}")
    if view.empty:
        st.info("No posts match these filters.")
        return
    if as_table:
        st.dataframe(
            chunk.assign(cause=chunk["primary_reason"].map(labels))[
                ["item_id", "published", "source", "cohort", "cause", "behaviour", "quote_check", "in_golden",
                 "post_snippet", "url"]
            ].rename(columns={"item_id": "ID", "published": "Published", "source": "Source", "cohort": "Group",
                              "cause": "Inferred cause", "behaviour": "Behaviour", "quote_check": "Quote check",
                              "in_golden": "Golden set", "post_snippet": "Post", "url": "Link"}),
            hide_index=True, width="stretch",
            column_config={"Link": st.column_config.LinkColumn(display_text="open")},
        )
    else:
        for _, p in chunk.iterrows():
            render_post_card(p)
    st.download_button("Download the filtered posts (CSV)",
                       view.drop(columns=[c for c in view.columns if c.startswith(("t_", "_"))]).to_csv(index=False),
                       file_name="evidence_filtered.csv")


def render_discovery_tab(df: pd.DataFrame) -> None:
    st.subheader("Ask the engine")
    st.caption("Pick one of the nine discovery questions from the problem statement, or type your own. "
               "The engine routes it to the right group of posts, reasons about causes, contrasts with "
               "investors who continued, and links each claim to source posts.")
    st.session_state.setdefault("discovery_question", "")
    cols = st.columns(3)
    for i, (label, q) in enumerate(PROBLEM_QUESTIONS):
        if cols[i % 3].button(f"Q{i + 1} · {label}", key=f"pq_{i}", help=q, width="stretch"):
            st.session_state.discovery_q_input = q
            st.session_state.discovery_question = q
    question = st.text_area(
        "Your question",
        placeholder="e.g. Do payment failures make first-time investors cancel?",
        height=80,
        key="discovery_q_input",
    )
    cfg = load_secrets()
    c1, c2 = st.columns([1, 3])
    if c1.button("Run discovery", type="primary", width="stretch"):
        st.session_state.discovery_question = question.strip()
    use_llm = c2.checkbox(
        "Use AI reasoning (computed reasoning is always shown alongside)",
        value=bool(cfg.get("api_key")),
        disabled=not cfg.get("api_key"),
        help=None if cfg.get("api_key") else "Set ANTHROPIC_API_KEY in .env or Streamlit secrets.",
    )
    render_guard_tests()
    if st.session_state.discovery_question:
        st.divider()
        render_reasoned_results(df, st.session_state.discovery_question, use_llm)


def _strength(n: int) -> str:
    return "High" if n >= 80 else "Medium" if n >= 30 else "Low"


@st.cache_data(show_spinner=False)
def question_coverage(_df: pd.DataFrame, n_posts: int) -> pd.DataFrame:
    rows = []
    for i, (label, q) in enumerate(PROBLEM_QUESTIONS):
        a = build_reasoned_analysis(_df, q)
        prof = a["profile"]
        top = prof.iloc[0] if len(prof) else None
        n = min(a["pop_n"], a["continuers_n"]) if a["intent"].code == "contrast" else a["pop_n"]
        rows.append({
            "#": f"Q{i + 1}",
            "Discovery question": label,
            "Posts analysed": a["pop_n"],
            "Strongest signal": f"{top['Theme']} ({top['% of group']}%)" if top is not None else "—",
            "Evidence strength": _strength(n),
        })
    return pd.DataFrame(rows)


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview — what the engine found")
    lapse = st.session_state["lapse_rate"] / 100
    recovery = st.session_state["recovery_rate"] / 100
    opps = opportunity_table(df, lapse, recovery, _overrides_from_state())
    top = opps.iloc[0]
    risk = df[df["is_at_risk"]]
    kinds = risk["stop_class"].value_counts(normalize=True).mul(100)
    not_stop = (
        pd.crosstab(df["link_to_lapse"], df["cohort"], normalize="columns").mul(100)
        .reindex(["none"]).fillna(0)
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Posts analysed", len(df), help="Coded public posts from five sources")
    m2.metric("Posts about stopping", len(risk), help="Stopped, paused, payment failed, or a strong link to lapse")
    m3.metric("Top opportunity", f"+{top['Est. uplift (pp)']} pp", help=top["Opportunity"], delta=top["Opportunity"],
              delta_color="off")
    m4.metric("All six opportunities", f"+{round(opps['Est. uplift (pp)'].sum(), 1)} pp",
              help="Estimated uplift on the 12-month active-SIP metric under the assumptions in Opportunity sizing")

    st.markdown("#### Key findings")
    f1, f2 = st.columns(2)
    with f1.container(border=True):
        st.markdown(f"**1. Most SIPs end without a decision to stop.**")
        st.markdown(
            f"{kinds.get('involuntary', 0):.0f}% of posts about stopping describe a payment, mandate or platform "
            f"failure; only {kinds.get('exit', 0):.0f}% are a deliberate exit (returns or cash need)."
        )
    with f2.container(border=True):
        st.markdown(f"**2. {top['Opportunity']} is the highest-leverage fix.**")
        st.markdown(
            f"It appears in {top['% of at-risk posts']}% of posts about stopping vs {top['% among continuers']}% "
            f"among continuers; est. +{top['Est. uplift (pp)']} pp on the goal metric."
        )
    f3, f4 = st.columns(2)
    with f3.container(border=True):
        st.markdown("**3. Many 'stops' are redirects, not exits.**")
        st.markdown(
            f"{kinds.get('redirect', 0):.0f}% stop one SIP to move money elsewhere (another fund, platform or "
            "lump sum). A stop-then-restart flow without a gap keeps them active."
        )
    with f4.container(border=True):
        st.markdown("**4. First-timers talk about starting, not stopping.**")
        st.markdown(
            f"{not_stop.at['none', COHORT_TARGET]:.0f}% of first-time / young investors' posts are not about "
            f"stopping (vs {not_stop.at['none', COHORT_COMPARISON]:.0f}% for others). Their early doubts — "
            "what to pick, how much, whether it's working — are where retention starts."
        )

    st.markdown("#### Coverage of the nine discovery questions")
    st.dataframe(
        question_coverage(df, len(df)),
        hide_index=True,
        width="stretch",
        column_config={"Evidence strength": st.column_config.TextColumn(help="High ≥ 80 posts, Medium 30–79, Low < 30")},
    )
    st.caption("Open **Ask the engine** and click a question to see the full reasoning, numbers and source posts.")

    with st.expander("Limitations to keep in mind"):
        st.markdown(
            f"""
- **Public posts are not a random sample.** {round((df['source'] == 'Reddit').mean() * 100)}% come from Reddit;
  app-store reviews and YouTube comments rarely state age or experience, so few pass the first-time filter.
- **Shares are of posts, not of investors.** Uplift estimates combine post shares with editable assumptions
  (12-month lapse rate, recovery rate, severity, addressability) — replace them with Groww cohort data.
- **Few continuers ({int(df['is_continuer'].sum())} posts)**, so contrasts with continuers are directional.
- Every claim links to source posts; the **Evidence & validation** tab shows how posts were selected and checked.
"""
        )


def _quote_list(quotes: list[dict[str, str]]) -> None:
    for q in quotes:
        group = "first-time / young" if q["group"] == COHORT_TARGET else "other investor"
        st.markdown(f"> {q['text']}  \n> — [#{q['id']}]({q['url']}) · {q['source']} · {group}")


def _problem_mandate(m: dict) -> None:
    o = m["opp"]
    with st.container(border=True):
        st.markdown("### Problem 1 · SIPs lapse when the payment mandate breaks")
        st.markdown(
            "**My hypothesis:** investors set up autopay on their salary account. When they change jobs and get a "
            "new salary account, the old mandate fails and they don't come back to Groww to update the bank account."
        )
        v1, v2 = st.columns(2)
        v1.success(f"**Problem area — strongly supported.** Payment and mandate failures are the #{o['rank']} "
                   f"cause of SIPs ending in public posts.")
        v2.warning(f"**Specific cause (job change → new salary account) — not visible in public posts.** "
                   f"{m['job_change_with_mandate']} post links a job change to a mandate; "
                   f"{m['bank_change_with_mandate']} mention a bank-account change. Validate with Groww data.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Posts about stopping that cite a payment failure", f"{m['pct_risk']}%",
                  help=f"{m['area_n']} of {m['risk_n']} posts about stopping or lapsing")
        c2.metric("Same among investors who continued", f"{m['pct_cont']}%",
                  delta=f"{round(m['pct_risk'] / max(m['pct_cont'], 0.1), 1)}× more common among drop-offs",
                  delta_color="off")
        c3.metric("Estimated uplift if fixed", f"+{o['uplift']} pp", help="On the 12-month active-SIP metric; "
                  "assumptions editable in Opportunity sizing")
        c4.metric("Of which first-time / young investors", f"{m['first_timer_n']} of {m['area_n']}")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**What users actually report when the payment fails**")
            st.bar_chart(m["reports"].set_index("What users report")["Posts"], horizontal=True, y_label="",
                         x_label="Posts", color=BRAND)
            table_view(m["reports"])
        with right:
            st.markdown("**Reading the chart**")
            unstated = m["reports"].iloc[-1]["% of payment-failure posts"]
            st.markdown(
                f"- **{unstated}%** of failure posts can't say *why* it failed — the app doesn't tell them.\n"
                "- Repeated autopay failures and mandates cancelled *without the user's action* dominate.\n"
                "- Bank-account change is rare in public posts — but job changers may simply never post; "
                "this needs Groww's own debit-return data."
            )

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Evidence for**")
            _quote_list(m["quotes"])
        with e2:
            st.markdown("**Evidence against / gaps**")
            st.markdown(
                f"- Only **{m['job_change_with_mandate']}** post connects a job change with a mandate "
                f"({m['job_change_any']} mention a job change at all).\n"
                f"- Only **{m['first_timer_n']} of {m['area_n']}** failure posts come from first-time / young "
                "investors — first-timers rarely post about it, so this rests on the comparison group.\n"
                "- Public posts over-represent people who noticed the failure; silent lapses are invisible here."
            )
            if m["mechanism_quotes"]:
                st.markdown("Closest example of the bank-account case:")
                _quote_list(m["mechanism_quotes"][:1])

        st.markdown("#### Conclusion → refined problem and solution")
        st.info(
            "**Refined problem:** a first-time investor's SIP silently lapses when an autopay instalment fails — "
            "they aren't told why, nothing retries, and repairing the mandate (including moving to a new bank "
            "account after a job change) means work outside their routine."
        )
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                "**Solution direction**\n"
                "1. Detect every failed or revoked mandate and tell the user the reason in plain language "
                "(insufficient balance, account closed, mandate cancelled by bank).\n"
                "2. One-tap recovery: *pay this instalment now* or automatic retry before the 30-day window ends.\n"
                "3. *Change bank account* and re-register the mandate inside the same flow — this covers the "
                "job-change case without depending on it.\n"
                "4. When the bank returns *account closed*, prompt to add the new account before the next date."
            )
        with s2:
            st.markdown(
                "**Validate with Groww data before building**\n"
                "- % of first-time SIP lapses preceded by a failed instalment.\n"
                "- Mix of bank return reasons (account closed vs insufficient funds vs mandate cancelled) — "
                "*account closed* is the proxy for the job-change hypothesis.\n"
                "- % of failed SIPs that never get another successful instalment.\n\n"
                "**Success metrics:** failed-instalment recovery within 7 days (leading); "
                "12-month active-SIP rate of first-time investors (goal)."
            )


def _problem_panic(p: dict) -> None:
    o = p["opp"]
    with st.container(border=True):
        st.markdown("### Problem 2 · Investors stop SIPs in panic when the market falls")
        st.markdown(
            "**My hypothesis:** after 4–5 months of regular SIPs, a market fall makes the investor panic and stop. "
            "**My original solution:** a nudge on the fund screen opening a bottom sheet with the % of users still "
            "continuing SIPs in that fund and the average amount invested."
        )
        v1, v2, v3 = st.columns(3)
        v1.success(f"**Market falls create doubt — supported.** {p['fall_n']} posts discuss a fall, "
                   f"{p['fall_first_timer_n']} of them from first-time / young investors.")
        separates = p["pct_fall_risk"] > p["pct_fall_cont"] * 1.3
        (v2.success if separates else v2.warning)(
            f"**{'Falls do separate drop-offs' if separates else 'Panic is not what separates drop-offs'}.** "
            f"A fall is mentioned in {p['pct_fall_risk']}% of posts about stopping vs {p['pct_fall_cont']}% of "
            "continuers."
        )
        v3.error("**Original solution conflicts with the brief.** Fund-level social proof reads as a fund "
                 "endorsement and can turn into a herd signal.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Posts discussing a market fall", p["fall_n"])
        c2.metric("…that consider stopping, pausing or selling", p["fall_stop_n"])
        c3.metric("…that ask others what to do", p["asks_others_n"],
                  help="e.g. 'are you guys still continuing your SIPs?' — the need behind social proof")
        c4.metric("Estimated uplift (decision support)", f"+{o['uplift']} pp", help=f"Opportunity rank #{o['rank']}")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**How often each group mentions a market fall**")
            st.bar_chart(p["by_group"].set_index("Group")["% mentioning a market fall"], horizontal=True,
                         y_label="", x_label="% of the group's posts", color=BRAND)
            table_view(p["by_group"])
        with right:
            st.markdown("**Reading the chart**")
            st.markdown(
                (f"- Continuers mention falls as often or more ({p['pct_fall_cont']}% vs {p['pct_fall_risk']}%) — "
                 "falls alone don't end SIPs.\n" if not separates else
                 f"- Falls are more common among people who stop ({p['pct_fall_risk']}% vs {p['pct_fall_cont']}%).\n")
                + "- What differs is having a way to judge whether *my* SIP is on track; without it, people ask "
                "strangers and act on the answer.\n"
                f"- The 'should I stop?' doubt opportunity covers {o['pct_risk']}% of posts about stopping."
            )

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Evidence for**")
            _quote_list(p["quotes_for"])
        with e2:
            st.markdown("**Evidence against**")
            _quote_list(p["quotes_against"])

        st.markdown("#### Checking my solution against the brief")
        st.dataframe(
            pd.DataFrame([
                {"Element": "% of users still continuing SIPs in this fund",
                 "Risk": "Reads as endorsing the fund (no fund recommendations); a low % during a crash "
                         "becomes a signal to exit",
                 "Refined": "Drop fund-level stats. If social proof is tested at all, use a platform-wide, "
                            "factual statement and A/B test it"},
                {"Element": "Average money invested in the fund",
                 "Risk": "Doesn't help the decision; invites comparison",
                 "Refined": "Show the user's own progress against their goal and horizon"},
                {"Element": "Nudge on the fund screen when markets fall",
                 "Risk": "Draws attention to the fall — an anxiety / urgency nudge",
                 "Refined": "Show only at intent: when the user taps Stop SIP"},
                {"Element": "Choice: continue or stop",
                 "Risk": "Binary — stopping is the only alternative",
                 "Refined": "Offer reduce amount, skip one month or change date (one instalment in 30 days "
                            "keeps the SIP active)"},
            ]),
            hide_index=True, width="stretch",
        )

        st.markdown("#### Conclusion → refined problem and solution")
        st.info(
            "**Refined problem:** when markets fall, first-time investors have no way to judge whether their SIP "
            "is still on track, so they ask strangers 'should I stop?' and cancel instead of adjusting."
        )
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                "**Solution direction — a neutral 'before you stop' sheet**\n"
                "1. The user's own goal, horizon and progress.\n"
                "2. A factual line on what their recent instalments bought (more units at lower prices).\n"
                "3. What stopping changes for their goal — no fear framing.\n"
                "4. Alternatives to cancelling: reduce, skip one month, change date."
            )
        with s2:
            st.markdown(
                "**Validate with Groww data before building**\n"
                "- Cancellation rate of first-time SIPs in the 2 weeks after a >5% index fall vs normal weeks.\n"
                "- Share of cancellations preceded by fund-page visits during the fall.\n"
                "- A/B test the stop-flow sheet.\n\n"
                "**Success metrics:** % of stop intents that become reduce / skip (leading); "
                "90-day and 12-month active-SIP rate (goal)."
            )


def render_problem_validation_tab(df: pd.DataFrame) -> None:
    st.subheader("Problem validation — testing my two problem hypotheses against the data")
    st.markdown(
        "I started with two problems from my own product experience with Groww. Before designing solutions I "
        f"tested each against the {len(df)} coded posts: **Is the problem real? Is my explanation of the cause "
        "supported? What does the evidence say instead? How should the solution change?**"
    )
    h = build_hypotheses(
        df, st.session_state["lapse_rate"] / 100, st.session_state["recovery_rate"] / 100, _overrides_from_state()
    )
    m, p = h["mandate"], h["panic"]
    _problem_mandate(m)
    _problem_panic(p)

    st.markdown("### Final prioritisation")
    st.dataframe(
        pd.DataFrame([
            {"Problem": "1 · Mandate / payment failure", "Evidence strength": strength(m["area_n"]),
             "Share of posts about stopping": f"{m['pct_risk']}%", "Among continuers": f"{m['pct_cont']}%",
             "Est. uplift": f"+{m['opp']['uplift']} pp", "First-timer voice": "Low (mostly other investors)",
             "My original cause": "Not supported yet — validate", "Decision": "Priority 1"},
            {"Problem": "2 · Panic stop in a market fall", "Evidence strength": strength(p["fall_n"]),
             "Share of posts about stopping": f"{p['opp']['pct_risk']}%", "Among continuers": f"{p['opp']['pct_cont']}%",
             "Est. uplift": f"+{p['opp']['uplift']} pp", "First-timer voice": "High",
             "My original cause": "Partly — doubt, not panic alone", "Decision": "Priority 2 (redesigned)"},
        ]),
        hide_index=True, width="stretch",
    )
    st.success(
        f"**Conclusion:** fix mandate failures first — they are the largest, most fixable cause of SIPs ending "
        f"({m['pct_risk']}% of posts about stopping vs {m['pct_cont']}% of continuers, est. +{m['opp']['uplift']} pp). "
        "Then address market-fall doubt for first-timers with a neutral stop-moment sheet instead of fund-level "
        "social proof. Both refined problems go into user interviews and Groww data checks before build."
    )


def main() -> None:
    st.set_page_config(page_title="Groww SIP Discovery Engine", page_icon="📈", layout="wide")
    try:
        df = load_posts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.session_state.setdefault("lapse_rate", 40)
    st.session_state.setdefault("recovery_rate", 30)

    render_header(df)
    render_sidebar(df)

    tabs = st.tabs(
        ["Overview", "Problem validation", "Ask the engine", "Opportunity sizing", "Evidence & validation",
         "Evidence explorer", "How it works"]
    )
    with tabs[0]:
        render_overview(df)
    with tabs[1]:
        render_problem_validation_tab(df)
    with tabs[2]:
        render_discovery_tab(df)
    with tabs[3]:
        render_opportunity_tab(df)
    with tabs[4]:
        render_evidence_log(df)
    with tabs[5]:
        render_browse_tab(df)
    with tabs[6]:
        render_workflow_tab()


def render_metric_branches_tab(df: pd.DataFrame) -> None:
    st.subheader("Where posts about stopping come from — 4 metric branches")
    risk = df[df["is_at_risk"]]
    no_link = df[~df["is_at_risk"] & (df["link_to_lapse"] == "none")]
    m1, m2, m3 = st.columns(3)
    m1.metric("Posts about stopping or lapsing", len(risk), help="Stopped, paused, payment failed, or a strong link to lapse")
    m2.metric("Posts with a weak link to stopping", len(df) - len(risk) - len(no_link))
    m3.metric(
        "Posts not about stopping", len(no_link),
        help="Mostly 'how do I start / which amount' posts. They are not used here, but they answer why people "
        "start, what they check and what confuses them in Discovery Q&A.",
    )
    st.caption(
        f"The branches and stop reasons below use only the {len(risk)} posts about stopping or lapsing."
    )

    branch = opportunity_by_metric_branch(risk).copy()
    branch["Reason"] = branch["metric_branch"].map(plain_branch).replace(
        {BRANCH_PLAIN["other"]: "Stop mentioned, branch unclear"}
    )
    branch["Drop-off relevance"] = branch["avg_opportunity"].map(lambda s: opportunity_score_plain(float(s)))
    st.dataframe(
        branch.rename(columns={"posts": "Number of posts", "posts_pct": "% of posts about stopping"})[
            ["Reason", "Number of posts", "% of posts about stopping", "Drop-off relevance"]
        ],
        hide_index=True,
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What triggered the stop (codebook trigger)**")
        stated = risk[~risk["trigger"].isin(["none_obvious", "other", "unknown"])]
        triggers = stated["trigger"].map(plain_trigger).value_counts()
        st.bar_chart(triggers, horizontal=True, color=BRAND)
        table_view(triggers.rename_axis("Trigger").reset_index(name="Posts"))
        st.caption(f"{len(risk) - len(stated)} of {len(risk)} posts about stopping state no trigger and are left out.")
    with c2:
        st.markdown("**How often each group posts about stopping**")
        levels = ["high", "medium", "low"]
        share = (
            pd.crosstab(df["link_to_lapse"], df["cohort"], normalize="columns")
            .reindex(levels).fillna(0).mul(100).round(1)
        )
        share.index = [LAPSE_PLAIN[l] for l in levels]
        st.bar_chart(share, horizontal=True, stack=False, color=[CB_ORANGE, CB_BLUE][: share.shape[1]])
        table_view(share.rename_axis("Link to lapse").reset_index())
        not_stop = (
            pd.crosstab(df["link_to_lapse"], df["cohort"], normalize="columns").reindex(["none"]).fillna(0).mul(100)
        )
        st.caption(
            "% of each group's posts. Not about stopping: "
            + ", ".join(f"{c} {not_stop.at['none', c]:.0f}%" for c in not_stop.columns)
            + ". First-timers mostly post about starting, so stop evidence relies more on the comparison group."
        )


@st.cache_data(show_spinner=False)
def load_selection_log() -> pd.DataFrame:
    return pd.read_csv(SELECTION_LOG, dtype=str).fillna("") if SELECTION_LOG.exists() else pd.DataFrame()


def render_selection_funnel() -> None:
    log = load_selection_log()
    if log.empty:
        st.info("Run `python3 build_corpus.py` to create the selection log.")
        return
    total = len(log)
    off_topic = log["decision"].str.startswith("dropped: not about SIPs").sum()
    not_own = log["decision"].str.startswith("dropped: not the author").sum()
    rule_kept = (log["rule_keep"] == "True").sum()
    kept = (log["keep"] == "True").sum()
    funnel = pd.DataFrame([
        {"Step": "1. Pooled public posts", "Posts": total, "How": "Scraped + cleaned sheet"},
        {"Step": "2. About SIPs / mutual funds", "Posts": total - off_topic, "How": "Keyword rules"},
        {"Step": "3. Author's own experience", "Posts": total - off_topic - not_own, "How": "First-person rules"},
        {"Step": "4. First-time or young (rules)", "Posts": rule_kept, "How": "Phrase rules, age / experience exclusions"},
        {"Step": "5. Confirmed by LLM screen", "Posts": kept, "How": "LLM checks own story, stage and age"},
    ])
    c1, c2 = st.columns([3, 2])
    c1.dataframe(
        funnel, hide_index=True, width="stretch",
        column_config={"Posts": st.column_config.ProgressColumn(min_value=0, max_value=int(total), format="%d")},
    )
    with c2:
        st.metric("Rule candidates confirmed by the LLM", f"{kept / max(rule_kept, 1) * 100:.0f}%",
                  help="The other candidates were news, guides, essays or experienced / older investors")
        st.metric("Coded by LLM against a fixed codebook", "100%", help="Lifecycle, trigger, behaviour, link to lapse…")
    dropped = log[log["decision"].str.startswith("dropped by LLM screen")]
    if len(dropped):
        with st.expander(f"Why the LLM rejected {len(dropped)} rule candidates (sample)"):
            st.dataframe(
                dropped[["item_id", "source", "llm_reason"]].head(15).rename(
                    columns={"item_id": "ID", "source": "Source", "llm_reason": "LLM reason"}
                ),
                hide_index=True, width="stretch",
            )
    st.download_button(
        "Download full selection log (every post, decision and reason)",
        log.to_csv(index=False), file_name="selection_log.csv",
    )


def render_data_quality(df: pd.DataFrame) -> None:
    log = load_selection_log()
    v = provenance()
    dated = df["year"].notna()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Posts in the corpus", len(df), help=f"Dataset version {v['dataset']}")
    m2.metric("With a publication date", f"{dated.mean() * 100:.0f}%",
              help=f"{int((df['precision'] == 'month').sum())} YouTube posts are dated to the month only")
    m3.metric("Quotes found verbatim", f"{(df['quote_check'] == 'Verbatim').mean() * 100:.0f}%",
              help="The codebook quote the AI extracted appears word-for-word in the post")
    m4.metric("Missing source link", int((df["url"].str.strip() == "").sum()))
    m5.metric(f"In the {GOLDEN_HUMAN.lower()}", int(df["in_golden"].sum()),
              help=f"{int(df['golden_verified'].sum())} verified by a human so far; the rest await verification")

    pooled = log["source"].value_counts() if not log.empty else pd.Series(dtype=int)
    rows = []
    for src, g in df.groupby("source"):
        method, terms = SOURCE_METHODS.get(src, ("Unknown", ""))
        y = g["year"].dropna()
        rows.append({
            "Source": src,
            "Posts pooled": int(pooled.get(src, 0)),
            "In corpus": len(g),
            "Share of corpus %": round(len(g) / len(df) * 100, 1),
            "First-time / young": int((g["cohort"] == COHORT_TARGET).sum()),
            "Published": f"{int(y.min())}–{int(y.max())}" if len(y) else "unknown",
            "Collection method (collect.py)": method,
            "Terms and caveats": terms,
        })
    sources = pd.DataFrame(rows).sort_values("In corpus", ascending=False)
    st.dataframe(sources, hide_index=True, width="stretch",
                 column_config={"Share of corpus %": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                                    format="%.1f%%")})
    st.caption("A source is listed only if posts from it reached the corpus. Pooled = posts entering the first-time "
               "selection pipeline, before the relevance, own-experience and first-time filters; comparison posts "
               "come from the earlier SIP corpus.")

    st.markdown("**Posts by publication year**")
    by_year = df.loc[dated, "year"].astype(int).value_counts().sort_index()
    st.bar_chart(by_year, color=BRAND, x_label="Publication year", y_label="Posts")
    table_view(by_year.rename_axis("Year").reset_index(name="Posts"))
    st.caption(f"n = {int(dated.sum())} dated posts. Reflects what was collected and selected, not how much people "
               "posted each year.")

    scores, _, _ = evaluate(load_golden(), df)
    pick = lambda label: scores.loc[scores["What is checked"].eq(label), "Agreement %"].iloc[0]  # noqa: E731
    with st.container(border=True):
        st.markdown("**How to read this dataset**")
        st.markdown(
            f"""
- **Selection bias:** public posts over-represent people with problems, and keyword searches over-select stop,
  pause and autopay threads. Nothing here measures how common a behaviour is among all investors.
- **Not a random sample:** posts were filtered for first-time / young investors and topped up with a comparison
  group, so shares describe this corpus, not Groww's users.
- **Tags are AI output:** every post was coded by `{v['coding_model']}` against `codebook.json` (screening by
  `{v['screen_model']}`); causes come from auditable text rules. On the hand-labelled golden set the rules agree
  {pick('Instalment failed'):.0f}% on payment failure but only {pick('Why they stop (driver)'):.0f}% on why people
  stop in a fall.
- **Human review:** {int(df['in_golden'].sum())} posts are hand-labelled in the golden set,
  {int(df['golden_verified'].sum())} of them verified by a human so far; the other {len(df) - int(df['in_golden'].sum())}
  carry AI tags only.
- **Small samples:** answers need at least {MIN_EVIDENCE} matching posts, and anything resting on fewer than
  {SMALL_SAMPLE} is labelled a small sample. Only {int(df['is_continuer'].sum())} posts come from investors who
  continued, so every stopper-vs-continuer contrast is directional.
            """
        )


def render_evidence_log(df: pd.DataFrame) -> None:
    st.subheader("Evidence & validation")
    st.caption("How posts were selected, how the AI's labels were checked, and how to verify any insight.")

    st.markdown("#### 1. Sources and data quality")
    render_data_quality(df)

    st.markdown("#### 2. How the corpus was selected")
    render_selection_funnel()
    st.caption(
        f"The {int((df['cohort'] == COHORT_COMPARISON).sum())} comparison posts (other investors) come from the "
        "earlier SIP corpus, so each first-timer pattern can be contrasted with the wider population."
    )

    st.markdown("#### 3. How well causes are identified")
    c1, c2, c3, c4 = st.columns(4)
    risk = df[df["is_at_risk"]]
    c1.metric("Posts about stopping", len(risk))
    c2.metric("Cause found by engine", f"{round((risk['primary_reason'] != 'unclear').mean() * 100, 1)}%",
              help="Auditable text patterns (see How it works)")
    stated = ~risk["trigger"].isin(["none_obvious", "other", "unknown"])
    c3.metric("Trigger stated by LLM codebook", f"{round(stated.mean() * 100, 1)}%")
    c4.metric("Reddit share", f"{round((df['source'] == 'Reddit').mean() * 100, 1)}%", help="Source bias to keep in mind")

    render_golden_set(df)

    st.markdown("#### 5. Validation protocol")
    st.markdown(
        """
1. **Label quality:** the golden set above is scored automatically; every miss is listed with the reason.
2. **Insight quality:** for each answer in *Ask the engine*, open 3 cited posts and check they support the claim.
3. **Challenge test:** check counter-intuitive claims (e.g. themes equally common among continuers) in interviews.
4. **Primary research:** the *Validate next* hypotheses in each answer become the interview guide.
        """
    )


def render_golden_set(df: pd.DataFrame) -> None:
    golden = load_golden()
    issues = validate(golden, df)
    scores, misses, joined = evaluate(golden, df)
    c = summary_counts(golden)
    rows_checked = int(scores["Rows"].sum())
    agree_cells = int((scores["Rows"] * scores["Agreement %"] / 100).round().sum())

    st.markdown(f"#### 4. Golden set: {c['rows']} hand-labelled posts for the two hypotheses")
    st.caption(
        f"{c['p2']} posts for Problem 2 (panic stop in a market fall, the priority) and {c['p1']} for Problem 1 "
        "(payment / mandate failure). Chosen on purpose, not at random: core cases, early signals, hard negatives "
        "that fool keyword rules, and counter-evidence against the proposed solution. Each row carries an evidence "
        "quote from the post and the labeller's reasoning."
    )
    if issues:
        st.warning("Golden set issues: " + "; ".join(issues))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Posts labelled", c["rows"])
    m2.metric("Label checks", rows_checked)
    m3.metric("Engine agreement", f"{round(agree_cells / rows_checked * 100, 1)}%")
    m4.metric("Disagreements", len(misses))

    st.dataframe(
        scores,
        hide_index=True,
        width="stretch",
        column_config={
            "Agreement %": st.column_config.ProgressColumn("Agreement %", min_value=0, max_value=100, format="%.0f%%"),
            "Precision %": st.column_config.NumberColumn(format="%.0f%%"),
            "Recall %": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    fit_n = c["p2_nudge_yes"] + c["p2_nudge_partial"] + c["p2_nudge_no"]
    fall_stop_precision = float(scores.loc[
        scores["What is checked"].eq("Stop caused by the fall") & scores["Engine method"].str.startswith("Text"),
        "Precision %"].iloc[0])
    with st.container(border=True):
        st.markdown("**What hand-labelling changed in the conclusions**")
        st.markdown(
            f"""
- **A fall plus a stop is not panic.** Of {c['p2_stop']} Problem 2 posts that stop, pause, reduce or weigh stopping,
  only **{c['p2_stop_fall_emotion']}** are an emotional reaction to a fall. {c['p2_stop_timing']} are deliberate
  market timing, {c['p2_stop_fund']} blame a specific fund and {c['p2_stop_cash']} need the cash. When the
  keyword rule flags "stop caused by a fall" it is right only {fall_stop_precision:.0f}% of the time.
- **The proposed nudge fits a narrow slice.** Showing how many investors kept their SIP fully answers the concern in
  **{c['p2_nudge_yes']} of {fit_n}** posts, partly in {c['p2_nudge_partial']}, and not at all in {c['p2_nudge_no']}
  (timers, fund-specific exits, cash needs). Only {c['p2_seeks_peer']} posts explicitly ask what others are doing.
- **Two design risks surfaced:** comparing with peers' *returns* demotivated a student (post 659), so show
  behaviour, never returns; and a user in debt (post 1093) must not be nudged to continue.
- **For Problem 1, the account matters more than the job change.** No failure post blames a job change, but
  {c['p1_two_account']} of {c['p1_failures']} failures happen because salary lands in a different account from the SIP mandate;
  {c['p1_not_notified']} users were not told in time and {c['p1_fixable']} could have fixed it with the right
  prompt. The engine has no code for this yet.
            """
        )

    with st.expander(f"Where the engine disagreed with the human labels ({len(misses)})"):
        st.dataframe(
            misses,
            hide_index=True,
            width="stretch",
            column_config={"url": st.column_config.LinkColumn("Post", display_text="open")},
        )
    with st.expander("Codebook (label definitions)"):
        st.dataframe(
            pd.DataFrame([{"Field": f, "Definition": d, "Allowed values": ", ".join(v)} for f, (d, v) in CODEBOOK.items()]),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Full golden set"):
        view = golden.merge(df[["item_id", "cohort", "source", "url"]], on="item_id", how="left")
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("Post", display_text="open")})
        st.download_button("Download golden set (CSV)", view.to_csv(index=False), file_name="golden_set.csv")
        st.download_button(
            "Download engine-vs-human comparison (CSV)",
            joined.drop(columns=[col for col in joined.columns if col.startswith("t_") or col in ("content", "_blob")])
            .to_csv(index=False),
            file_name="golden_set_engine_comparison.csv",
        )


if __name__ == "__main__":
    main()
