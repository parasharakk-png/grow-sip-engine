"""
Groww SIP Discovery Engine — reasons about causes and sizes opportunities (not review summarization).
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from discovery_analytics import (
    METRIC_BRANCH_LABELS,
    enrich_dataframe,
    opportunity_by_metric_branch,
)
from hypotheses import build_hypotheses, strength
from llm_client import make_client, stream_text
from plain_labels import BRANCH_PLAIN, LAPSE_PLAIN, opportunity_score_plain, plain_branch, plain_trigger
from reasoning import (
    LLM_INSTRUCTIONS,
    OPPORTUNITIES,
    STOP_REASONS,
    build_reasoned_analysis,
    format_compact_pack,
    opportunity_table,
    tag_themes,
)
from rule_based_narrative import reasoned_narrative

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
    return tag_themes(enrich_dataframe(df))


def render_ai_reasoning(question: str, a: dict, cfg: dict[str, str]) -> None:
    pack = format_compact_pack(a)
    cache = st.session_state.setdefault("ai_answers", {})
    key = (question, cfg["model"], hash(pack))
    if key in cache:
        st.caption(f"Source: AI reasoning ({cfg['model']}) · cached")
        st.markdown(cache[key])
        return
    st.caption(f"Source: AI reasoning ({cfg['model']}) · usually 20–60 s; the computed answer is in the next tab")
    prompt = f"{LLM_INSTRUCTIONS}\n\n=== EVIDENCE PACK ===\n{pack}"
    try:
        with st.spinner("Model is thinking before it writes…"):
            client = make_client(cfg["api_key"], cfg.get("base_url"))
            chunks = stream_text(client, cfg["model"], prompt)
            first = next(chunks)
        text = st.write_stream(itertools.chain([first], chunks))
        cache[key] = text
    except Exception as e:  # noqa: BLE001
        st.warning(
            f"AI reasoning unavailable ({str(e)[:200]}). The computed reasoning tab has the full answer."
        )


def _overrides_from_state() -> dict[str, dict[str, float]]:
    return st.session_state.get("opp_overrides", {})


def render_reasoned_results(df: pd.DataFrame, question: str, use_llm: bool) -> None:
    lapse = st.session_state.get("lapse_rate", 40) / 100
    recovery = st.session_state.get("recovery_rate", 30) / 100
    a = build_reasoned_analysis(df, question, lapse, recovery, _overrides_from_state())
    intent = a["intent"]

    st.markdown("### How the engine read your question")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Question type", intent.title)
    c2.metric("Posts analysed", f"{a['pop_n']} / {a['corpus_n']}")
    c3.metric("Cause inferred for", f"{a['explained_pct']}%")
    c4.metric("First-time investor posts", a["first_timer_n"])
    st.caption(f"Group: {intent.population}. Compared against: {intent.base} (n={a['base_n']}).")

    if a["pop_n"] == 0:
        st.info("No posts match. Try one of the nine discovery questions above.")
        return

    st.markdown("---")
    cfg = load_secrets()
    computed = reasoned_narrative(a)
    if use_llm and cfg.get("api_key"):
        ai_tab, rules_tab = st.tabs(["AI reasoning", "Computed reasoning"])
        with rules_tab:
            st.caption("Source: computed reasoning (rules over coded data) · instant")
            st.markdown(computed)
        with ai_tab:
            render_ai_reasoning(question, a, cfg)
    else:
        st.caption("Source: computed reasoning (rules over coded data)")
        st.markdown(computed)

    st.markdown("---")
    st.markdown("### The numbers behind the reasoning")
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
        st.bar_chart(chart, horizontal=True)
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
        for code, items in a["evidence"].items():
            st.markdown(f"**{labels.get(code, code)}**")
            for e in items:
                st.markdown(f"- [#{e['id']}]({e['url']}) *({e['source']})* {e['text']}")


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
    st.bar_chart(opps.set_index("Opportunity")["Est. uplift (pp)"], horizontal=True, y_label="", x_label="Est. uplift (pp)")
    with st.expander("Product direction per opportunity (no fund advice, no fear/urgency)"):
        for _, o in opps.iterrows():
            ids = ", ".join(f"#{i}" for i in o["evidence_ids"])
            st.markdown(f"**{o['Opportunity']}** — {o['Product direction']}  \nEvidence: {ids}")
    st.caption(
        "Lapse share = share of at-risk posts × severity (normalised). Leverage = lapse share × "
        "addressability. Est. uplift = assumed 12-month lapse rate × lapse share × addressability × "
        "recovery rate. Edit the assumptions in the Opportunity sizing tab."
    )


def render_opportunity_tab(df: pd.DataFrame) -> None:
    st.subheader("Opportunity sizing — which fix moves the 12-month metric most?")
    st.markdown(
        "Each at-risk post (stopped, paused, payment failed, or high link to lapse) is assigned one "
        "**primary cause** from its text. Causes are grouped into product-addressable opportunity areas "
        "and sized with explicit assumptions you can change."
    )
    c1, c2 = st.columns(2)
    c1.slider(
        "Assumed % of first-time investors with no active SIP at month 12",
        10, 80, key="lapse_rate",
        help="Replace with Groww cohort data. Public posts can't measure this.",
    )
    c2.slider(
        "Assumed % of addressable lapses a good fix recovers",
        5, 80, key="recovery_rate",
    )

    st.markdown("**Severity and addressability (edit to test your own judgement)**")
    defaults = pd.DataFrame(
        [
            {"code": o.code, "Opportunity": o.label, "Severity": o.severity, "Addressability": o.addressability}
            for o in OPPORTUNITIES
        ]
    )
    edited = st.data_editor(
        defaults,
        hide_index=True,
        disabled=["code", "Opportunity"],
        column_config={
            "code": None,
            "Severity": st.column_config.NumberColumn(
                min_value=0.0, max_value=1.0, step=0.05,
                help="If nothing changes, how likely does this moment end with no active SIP?",
            ),
            "Addressability": st.column_config.NumberColumn(
                min_value=0.0, max_value=1.0, step=0.05,
                help="How much can Groww fix this within the constraints (no fund advice, no fear/FOMO)?",
            ),
        },
        key="opp_editor",
    )
    overrides = {
        r["code"]: {"severity": float(r["Severity"]), "addressability": float(r["Addressability"])}
        for _, r in edited.iterrows()
    }
    st.session_state["opp_overrides"] = overrides

    opps = opportunity_table(
        df, st.session_state["lapse_rate"] / 100, st.session_state["recovery_rate"] / 100, overrides
    )
    render_opportunity_table(opps)
    total = opps["Est. uplift (pp)"].sum()
    top = opps.iloc[0]
    st.success(
        f"Highest leverage: **{top['Opportunity']}** (≈ +{top['Est. uplift (pp)']} pp). "
        f"All six areas together: ≈ +{round(total, 2)} pp under these assumptions."
    )


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


def render_browse_tab(df: pd.DataFrame) -> None:
    st.subheader("Browse posts")
    st.caption("Every post the engine uses, with its inferred cause. Open the link to read the original.")
    labels = {t.code: t.label for t in STOP_REASONS}
    labels["unclear"] = "Reason not stated"
    c1, c2 = st.columns(2)
    sel_r = c1.multiselect("Primary cause", list(labels), format_func=labels.get)
    sel_b = c2.multiselect(
        "Metric branch", sorted(df["metric_branch"].unique()), format_func=lambda x: METRIC_BRANCH_LABELS.get(x, x)
    )
    c3, c4, c5 = st.columns([2, 1, 2])
    cohort = c3.radio("Group", ["All", COHORT_TARGET, COHORT_COMPARISON], horizontal=True)
    only_risk = c4.checkbox("Only posts about stopping")
    q = c5.text_input("Keyword")
    view = df
    if cohort != "All":
        view = view[view["cohort"] == cohort]
    if sel_r:
        view = view[view["primary_reason"].isin(sel_r)]
    if sel_b:
        view = view[view["metric_branch"].isin(sel_b)]
    if only_risk:
        view = view[view["is_at_risk"]]
    if q:
        view = view[view["content"].str.contains(q, case=False, na=False)]
    view = view.assign(cause=view["primary_reason"].map(labels))
    st.dataframe(
        view[["item_id", "cohort", "source", "cause", "stop_class", "post_snippet", "url"]].rename(
            columns={"item_id": "ID", "cohort": "Group", "source": "Source", "cause": "Inferred cause",
                     "stop_class": "Kind of stop", "post_snippet": "Post", "url": "Link"}
        ),
        hide_index=True,
        width="stretch",
        column_config={"Link": st.column_config.LinkColumn(display_text="open")},
    )
    st.caption(f"{len(view)} of {len(df)} posts")


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
                         x_label="Posts")
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
                         y_label="", x_label="% of the group's posts")
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
         "Browse posts", "How it works"]
    )
    with tabs[0]:
        render_overview(df)
    with tabs[1]:
        render_problem_validation_tab(df)
    with tabs[2]:
        render_discovery_tab(df)
    with tabs[3]:
        render_opportunity_tab(df)
        st.divider()
        render_metric_branches_tab(df)
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
        st.bar_chart(stated["trigger"].map(plain_trigger).value_counts(), horizontal=True)
        st.caption(f"{len(risk) - len(stated)} of {len(risk)} posts about stopping state no trigger and are left out.")
    with c2:
        st.markdown("**How often each group posts about stopping**")
        levels = ["high", "medium", "low"]
        share = (
            pd.crosstab(df["link_to_lapse"], df["cohort"], normalize="columns")
            .reindex(levels).fillna(0).mul(100).round(1)
        )
        share.index = [LAPSE_PLAIN[l] for l in levels]
        st.bar_chart(share, horizontal=True, stack=False)
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


def render_evidence_log(df: pd.DataFrame) -> None:
    st.subheader("Evidence & validation")
    st.caption("How posts were selected, how the AI's labels were checked, and how to verify any insight.")

    st.markdown("#### 1. How the corpus was selected")
    render_selection_funnel()
    st.caption(
        f"The {int((df['cohort'] == COHORT_COMPARISON).sum())} comparison posts (other investors) come from the "
        "earlier SIP corpus, so each first-timer pattern can be contrasted with the wider population."
    )

    st.markdown("#### 2. How well causes are identified")
    c1, c2, c3, c4 = st.columns(4)
    risk = df[df["is_at_risk"]]
    c1.metric("Posts about stopping", len(risk))
    c2.metric("Cause found by engine", f"{round((risk['primary_reason'] != 'unclear').mean() * 100, 1)}%",
              help="Auditable text patterns (see How it works)")
    stated = ~risk["trigger"].isin(["none_obvious", "other", "unknown"])
    c3.metric("Trigger stated by LLM codebook", f"{round(stated.mean() * 100, 1)}%")
    c4.metric("Reddit share", f"{round((df['source'] == 'Reddit').mean() * 100, 1)}%", help="Source bias to keep in mind")

    st.markdown("#### 3. Validation protocol")
    st.markdown(
        """
1. **Label quality:** hand-label the 20-post sample below; record agreement with the engine's cause and 2–3 misses.
2. **Insight quality:** for each answer in *Ask the engine*, open 3 cited posts and check they support the claim.
3. **Challenge test:** check counter-intuitive claims (e.g. themes equally common among continuers) in interviews.
4. **Primary research:** the *Validate next* hypotheses in each answer become the interview guide.
        """
    )
    st.markdown("**Random sample for hand-labelling** (fixed seed, reproducible)")
    sample = risk.sample(min(20, len(risk)), random_state=42)
    st.dataframe(
        sample[["item_id", "primary_reason", "stop_class", "trigger", "url", "post_snippet"]],
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download sample as CSV",
        sample[["item_id", "primary_reason", "stop_class", "trigger", "url", "post_text"]]
        .assign(human_label="", agree="")
        .to_csv(index=False),
        file_name="evidence_log_sample.csv",
    )


if __name__ == "__main__":
    main()
