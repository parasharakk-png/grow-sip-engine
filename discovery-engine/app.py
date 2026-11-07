"""
Groww SIP Discovery Engine — reasons about causes and sizes opportunities (not review summarization).
"""

from __future__ import annotations

import hashlib
import json
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
from golden_set_eval import evaluate, load_golden, summary_counts
from hypotheses import build_hypotheses
from interviews import critical_incidents, decision, hypothesis_tally, load_scorecard, p2_evidence
from metric_decomposition import (
    BRANCHES,
    METRIC_MECHANICS,
    RECOVERY,
    lead_phrase,
    slide_title,
    branch_sizes,
    focus_comparison,
    golden_signals,
    interview_hypotheses,
    tree_svg,
)
from plain_labels import BRANCH_PLAIN, LAPSE_PLAIN, opportunity_score_plain, plain_branch, plain_trigger
from question_guard import MESSAGES, GuardResult, classify_question, is_intervention_question
from reasoning import (
    LLM_INSTRUCTIONS,
    MIN_EVIDENCE,
    OPPORTUNITIES,
    SMALL_SAMPLE,
    STOP_REASONS,
    build_reasoned_analysis,
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
    "stage": "STAGE",
    "months": "MONTHS",
    "status": "STATUS",
    "concern": "CONCERN",
    "yardstick": "YARDSTICK",
    "decides_outside": "DECIDES_OUTSIDE",
    "life_change": "LIFE_CHANGE",
    "age_band": "AGE_BAND",
    "fit_quote": "FIT_QUOTE",
    "query_branch": "QUERY_BRANCH",
}
FREE_TEXT = {"segment", "workaround", "quote", "selection", "fit_quote", "query_branch"}
COHORT_TARGET = "Target: first-time investors"
COHORT_COMPARISON = "Comparison: other investors"

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
        st.caption(f"Cause: {labels.get(p['primary_reason'], 'not stated')}")
        st.caption(
            f"Quote: {p['quote_check']}" + _review_status(p)
        )


def _review_status(p: pd.Series) -> str:
    if p["golden_verified"] or p["in_golden"]:
        return " · in the 20 I labelled"
    return ""


def table_view(data: pd.DataFrame, label: str = "Table view") -> None:
    with st.expander(label):
        st.dataframe(data, hide_index=True, width="stretch")


def _computed_provenance(a: dict) -> None:
    v = provenance()
    st.caption(
        f"Computed by rules {v['rules']} · dataset {v['dataset']} ({a['corpus_n']} posts; tags by {v['coding_model']} "
        f"against `codebook.json`, causes by text rules) · {a['pop_n']} posts used for this question."
    )


def _ask_instead(question: str) -> None:
    st.session_state.discovery_q_input = question
    st.session_state.discovery_question = question


def render_guard_message(question: str, g: GuardResult) -> None:
    title, body = MESSAGES[g.kind]
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body)
        st.caption(
            f"Question check: {g.reason}. No answer is generated for this question."
        )
        if g.reframe:
            st.markdown(f"**A research question the engine can answer instead:** {g.reframe}")
            st.button("Ask this instead", key="guard_reframe", on_click=_ask_instead, args=(g.reframe,))
        else:
            st.markdown("Try a research question about why first-time investors' SIPs fail or stop.")


def render_computed_answer(a: dict) -> None:
    with st.container(border=True):
        st.markdown(reasoned_narrative(a))
    _computed_provenance(a)


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
        st.caption("No answer was generated. Try a broader question about why SIPs fail or stop.")
    if a["relevant_posts"]:
        st.markdown("**Closest posts**")
        posts = df.set_index("item_id", drop=False)
        for e in a["relevant_posts"][:3]:
            render_post_card(posts.loc[e["id"]])


def _problem_code(question: str, a: dict) -> str | None:
    codes = {f["code"] for f in a["focus"]}
    if "activation_failure" in codes and "involuntary_system" not in codes:
        return "first_sip_success"
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


def render_reasoned_results(df: pd.DataFrame, question: str) -> None:
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
    c3.metric("First-time posts", a["first_timer_n"], help="Matching posts from first-time investors")
    c4.metric("Top uplift", f"+{summary['uplift']} pp" if summary["uplift"] is not None else "—")
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

    render_computed_answer(a)

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
        "Leverage = share of at-risk posts × addressability. Est. uplift = assumed 12-month lapse rate × "
        "share of at-risk posts × addressability × recovery rate."
    )


OPP_SHORT = {
    "first_sip_success": "First SIP goes live",
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
    "Severity vs addressability": "Two judgements per opportunity. Top-right = the moment usually ends the SIP and "
                                  "Groww can fix it. Bubble size = evidence posts. Only addressability enters the "
                                  "uplift.",
}


def _opp_ratio(o: pd.Series) -> float | None:
    cont = float(o["% among continuers"])
    return round(float(o["Mentioned in at-risk %"]) / cont, 1) if cont else None


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


def render_opportunity_compare(opps: pd.DataFrame) -> None:
    view = st.segmented_control(
        "Compare the opportunities by", list(OPP_VIEWS), default="Estimated uplift", key="opp_view"
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
                f"1. Share of stop posts = **{o['% of at-risk posts']:.1f}%**, used as its share of lapses\n"
                f"2. Share × addressability {o['Addressability']:.2f} = **leverage {o['Leverage index']:.1f}**\n"
                f"3. {lapse}% lapse rate × {o['% of at-risk posts']:.1f}% × {o['Addressability']:.2f} × {rec}% recovery "
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


def render_opportunity_tab(df: pd.DataFrame) -> None:
    st.subheader("If Groww fixed one thing, which one moves month-12 the most?")
    lapse, rec = st.session_state["lapse_rate"], st.session_state["recovery_rate"]
    st.markdown(
        f"Each post about stopping gets one cause from the text. I grouped those into six product areas. "
        f"I don't have Groww's cohort file, so I take **{lapse:.0f}% of first SIPs gone by month 12** and "
        f"**{rec:.0f}% of each cause can come back** (sidebar)."
    )
    opps = opportunity_table(df, lapse / 100, rec / 100)
    opps.insert(0, "Rank", range(1, len(opps) + 1))
    opps["Short name"] = opps["code"].map(OPP_SHORT)
    top, second = opps.iloc[0], opps.iloc[1]
    total = float(opps["Est. uplift (pp)"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stop posts", int(df["is_at_risk"].sum()))
    k2.metric("Largest area", f"+{top['Est. uplift (pp)']:.1f} pp", help=top["Opportunity"])
    k3.metric("Next", f"+{second['Est. uplift (pp)']:.1f} pp", help=second["Opportunity"])
    k4.metric("If all six were fixed", f"+{total:.1f} pp")

    ratio = _opp_ratio(top)
    how = f"{ratio}× more often" if ratio else "more often"
    st.markdown(
        f"**{top['Short name']}** is the largest in this set. It's the main cause in "
        f"{top['% of at-risk posts']:.0f}% of stop posts, shows up {how} among people who stopped than among people "
        f"who kept going, and comes to about +{top['Est. uplift (pp)']:.1f} pp. Next is **{second['Short name']}** "
        f"(+{second['Est. uplift (pp)']:.1f} pp). Posts can't tell you which of these actually ends a first SIP — "
        "that's why I used interviews to pick the MVP."
    )

    show = opps.rename(columns={
        "Short name": "Area",
        "% of at-risk posts": "% of stop posts",
        "First-timer share %": "% first-timers",
        "Est. uplift (pp)": "If this were fixed",
    })[["Rank", "Area", "% of stop posts", "% first-timers", "If this were fixed", "Confidence"]]
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "% of stop posts": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "% first-timers": st.column_config.NumberColumn(format="%.0f%%"),
            "If this were fixed": st.column_config.NumberColumn(format="+%.1f pp"),
        },
    )
    _opp_bar_chart(opps, "Est. uplift (pp)", "If this were fixed (pp on the 12-month rate)", "+{:.1f} pp")
    st.caption(
        "These are shares of public posts, not of Groww users. Change the two assumptions in the sidebar and the "
        "bars move."
    )


def render_how_it_works(df: pd.DataFrame) -> None:
    st.subheader("Where the numbers come from")
    st.markdown(
        "I started with public posts about SIPs. I kept the ones where someone is talking about their own SIP, "
        "dropped duplicates, and marked first-time investors. Each post gets a cause from the text. Those causes "
        "are sized, then checked against a small set I labelled by hand and against five interviews."
    )
    st.markdown(
        """
1. **Collect** — Play Store, App Store, Reddit, YouTube, TradingQ&A.
2. **Keep** — their own SIP, not a duplicate, first 12 months where I can tell.
3. **Cause** — one reason per stop post, from the words on the page.
4. **Size** — share of stop posts × the two assumptions in the sidebar.
5. **Check** — 20 hand-labelled posts, then 5 interviews.
"""
    )

    st.markdown("**What I collected**")
    rows = []
    for src, g in df.groupby("source"):
        rows.append({
            "Source": src,
            "Posts": len(g),
            "First-timers": int((g["cohort"] == COHORT_TARGET).sum()),
        })
    st.dataframe(pd.DataFrame(rows).sort_values("Posts", ascending=False), hide_index=True, width="stretch")

    golden = load_golden()
    scores, misses, _ = evaluate(golden, df)
    c = summary_counts(golden)
    checks = int(scores["Rows"].sum())
    agree = int((scores["Rows"] * scores["Agreement %"] / 100).round().sum())
    show = scores.copy()
    show["Problem"] = show["Problem"].map({
        "P2": "Stop decision",
        "P1": "Failed payments",
    })
    show = (show.groupby(["Problem", "What is checked"], as_index=False)["Agreement %"].mean())
    show = show.rename(columns={"What is checked": "What I checked", "Agreement %": "Match with my labels"})
    order = {"Stop decision": 0, "Failed payments": 1}
    show["_o"] = show["Problem"].map(order)
    show = show.sort_values(["_o", "Match with my labels"], ascending=[True, False]).drop(columns="_o")

    st.markdown("**The 20 posts I labelled by hand**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Posts", c["rows"], help=f"{c['p2']} stop decision, {c['p1']} failed payments")
    m2.metric("How often the engine matched", f"{agree / checks * 100:.0f}%")
    m3.metric("Places it didn't", len(misses))
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "Match with my labels": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        },
    )
    st.caption(
        "These 20 were picked to be hard, not typical. A mismatch is useful: it shows where a keyword would have "
        "fooled the engine. Full notes are in the Evidence Log."
    )


def render_header(df: pd.DataFrame) -> None:
    st.title("Groww SIP Discovery Engine")
    st.markdown(
        "Why first-time investors keep or lose their SIP — from public posts and five interviews."
    )
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**The Groww number**")
        st.caption("Share of first-time MF investors with an SIP still running 12 months later "
                   "(at least one debit in the last 30 days).")
    with c2.container(border=True):
        st.markdown("**What I used**")
        st.caption(f"{len(df)} public posts · {int((df['cohort'] == COHORT_TARGET).sum())} first-timers · "
                   f"9 screener replies · 5 interviews.")
    with c3.container(border=True):
        st.markdown("**What I will not do**")
        st.caption("No fund picks. No fear or FOMO. Not investment advice.")


def render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.header("Corpus")
        st.metric("Coded posts", len(df))
        c1, c2 = st.columns(2)
        c1.metric("First-time", int((df["cohort"] == COHORT_TARGET).sum()))
        c2.metric("Comparison", int((df["cohort"] == COHORT_COMPARISON).sum()))
        c1.metric("About stopping", int(df["is_at_risk"].sum()))
        c2.metric("Continuers", int(df["is_continuer"].sum()))
        st.markdown("**Sources**")
        st.dataframe(
            df["source"].value_counts().rename_axis("Source").reset_index(name="Posts"),
            hide_index=True, width="stretch",
        )


EXPLORER_PAGE_SIZE = 20


def _options(df: pd.DataFrame, col: str) -> list[str]:
    return sorted(v for v in df[col].unique() if v and v != "unknown")


def render_browse_tab(df: pd.DataFrame) -> None:
    st.subheader("Read the posts")
    st.caption("Open any row to the original. This is the same set the other tabs are built on.")
    labels = {t.code: t.label for t in STOP_REASONS}
    labels["unclear"] = "Reason not stated"

    q = st.text_input("Search", key="ex_q", placeholder="e.g. should I stop, autopay, red")
    c1, c2, c3 = st.columns(3)
    sel_src = c1.multiselect("Source", _options(df, "source"), key="ex_src")
    sel_r = c2.multiselect("Cause", list(labels), format_func=labels.get, key="ex_cause")
    cohort = c3.radio("Who", ["All", "First-timers", "Other investors"], horizontal=True, key="ex_group")
    f1, f2 = st.columns(2)
    only_risk = f1.checkbox("Only posts about stopping", key="ex_risk")
    only_golden = f2.checkbox("Only the 20 I labelled", key="ex_golden")

    view = df
    if q:
        view = view[view["content"].str.contains(re.escape(q), case=False, na=False)]
    if sel_src:
        view = view[view["source"].isin(sel_src)]
    if sel_r:
        view = view[view["primary_reason"].isin(sel_r)]
    if cohort == "First-timers":
        view = view[view["cohort"] == COHORT_TARGET]
    elif cohort == "Other investors":
        view = view[view["cohort"] == COHORT_COMPARISON]
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
    p2.caption(f"{len(view)} of {len(df)} posts · {start + 1 if len(view) else 0}–{start + len(chunk)}")
    if view.empty:
        st.info("Nothing matches.")
        return
    for _, p in chunk.iterrows():
        render_post_card(p)


def render_discovery_tab(df: pd.DataFrame) -> None:
    st.subheader("Ask the engine")
    st.caption("Ask why first-time investors' SIPs fail or stop. The engine routes the question to the matching "
               "posts, compares them with investors who continued, and links each claim to source posts.")
    st.session_state.setdefault("discovery_question", "")
    question = st.text_area(
        "Your question",
        placeholder="e.g. Do payment failures make first-time investors cancel?",
        height=80,
        key="discovery_q_input",
    )
    c1, _ = st.columns([1, 3])
    if c1.button("Run discovery", type="primary", width="stretch"):
        st.session_state.discovery_question = question.strip()
    if st.session_state.discovery_question:
        st.divider()
        render_reasoned_results(df, st.session_state.discovery_question)


P1_STATEMENT = (
    "First-time investors lose SIPs they meant to keep: the first SIP never goes live, or a later instalment "
    "fails, and they aren't told why or how to fix it."
)
P2_STATEMENT = (
    "In months 6–12, first-time investors judge their SIP only by red or green against a one-year expectation. "
    "With no way to tell what's normal, they ask whoever got them started, then stop, reduce, or move the money "
    "out of SIPs."
)


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview — the two problems behind lost first SIPs")
    focus = focus_comparison(df, st.session_state["lapse_rate"] / 100, st.session_state["recovery_rate"] / 100)
    sc = load_scorecard()
    dec = decision(sc) if not sc.empty else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Posts analysed", len(df), help="Coded public posts from five sources")
    m2.metric("First-time investors", int((df["cohort"] == COHORT_TARGET).sum()), help="Posts from first-time investors")
    m3.metric("Stop posts", int(df["is_at_risk"].sum()), help="Posts about stopping, lapsing or a failed SIP")
    m4.metric("Interviews", dec["n"] if dec else 0, help="Plus a 9-person screener survey")

    a, f = focus["all"], focus["ft"]
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("#### Problem 1 · The stop decision")
        st.markdown(f"*{P2_STATEMENT}*")
        k1, k2 = st.columns(2)
        k1.metric("Share of stop posts", f"{f['stop_moment_pct']:.0f}%",
                  help=f"First-time investors; {a['stop_moment_pct']:.0f}% among all investors")
        k2.metric("Est. uplift", f"+{f['stop_moment']:.1f} pp",
                  help=f"On the 12-month active-SIP rate of first-time investors; +{a['stop_moment']:.1f} pp for all")
        if dec:
            st.markdown(f"- **Interviews:** {dec['stop_n']} of {dec['n']} describe a stop decision as the moment "
                        "their SIP was at risk.")
        st.markdown("- **Decision:** research and MVP focus — neutral support for judging one's own SIP.")
    with c2.container(border=True):
        st.markdown("#### Problem 2 · Failed payments")
        st.markdown(f"*{P1_STATEMENT}*")
        k1, k2 = st.columns(2)
        k1.metric("Share of stop posts", f"{f['payment_pct']:.0f}%",
                  help=f"First-time investors; {a['payment_pct']:.0f}% among all investors")
        k2.metric("Est. uplift", f"+{f['payment']:.1f} pp",
                  help=f"On the 12-month active-SIP rate of first-time investors; +{a['payment']:.1f} pp for all")
        if dec:
            st.markdown(f"- **Interviews:** {dec['failed_any']} of {dec['n']} had a failed instalment, found only "
                        "by opening the app; none stopped because of it.")
        st.markdown("- **Decision:** parallel track — an operations fix (setup tracking, failure reasons, retries).")


def _interview_findings(tally: pd.DataFrame, dec: dict) -> list[str]:
    """One finding per row of interview_hypotheses(), in the same order as interviews.HYPOTHESES (H1–H6)."""
    n = dec["n"]
    return [
        f"{tally.at['H1', 'Confirms (+)']} of {n} only looked at red or green against what they hoped for in a year",
        f"{tally.at['H2', 'Confirms (+)']} of {n} asked someone they trust (colleague, cousin, YouTuber), not the app",
        f"{tally.at['H3', 'New insight (!)']} of {n} moved the money to FD / RD, a lump sum or gold — still investing, "
        "just not in this SIP",
        "Nobody stopped because they were short of cash. One person skipped a month after a low balance",
        f"{dec['failed_any']} of {n} had a failed instalment and only saw it in the app. Nobody stopped over it. "
        f"{dec['account_change']} changed job or salary account with no mandate issue",
        "Numbers confused a couple of people (one used ChatGPT for XIRR). One stop was an ELSS lock-in rule, not returns",
    ]


def render_decomposition_tab(df: pd.DataFrame) -> None:
    st.subheader("Where a first SIP is lost by month 12")
    lapse = st.session_state["lapse_rate"] / 100
    recovery = st.session_state["recovery_rate"] / 100
    overrides = _overrides_from_state()
    sizes = branch_sizes(df, lapse, recovery, overrides)
    focus = focus_comparison(df, lapse, recovery, overrides)
    a, f = focus["all"], focus["ft"]
    sc = load_scorecard()
    dec = decision(sc) if not sc.empty else None

    with st.container(border=True):
        st.markdown(
            "Groww's number: **share of first-time mutual fund investors who still have an SIP running 12 months "
            "after they started.** Running means at least one instalment went through in the last 30 days."
        )
        st.markdown("\n".join(f"- {m}" for m in METRIC_MECHANICS))

    st.markdown("#### 1 · Five places it can fall over")
    st.markdown(
        "The first SIP has to get set up. Then the debit has to go through every month. If they pause because "
        "cash is tight, they need a way back. If they open cancel because the number looks red, they need a way "
        "to judge. If they can't read the returns screen, they drift. A successful debit within 30 days still "
        "brings them back from any of those."
    )
    st.image(tree_svg(sizes, focus, lapse), width="stretch")
    st.download_button("Download as slide (SVG)", tree_svg(sizes, focus, lapse, title=slide_title(focus, dec)),
                       file_name="metric_decomposition.svg", mime="image/svg+xml")

    st.markdown("#### 2 · What has to happen, and what I would watch")
    st.table(
        pd.DataFrame([{
            "Branch": b.name if b.code == "0" else f"{b.code} · {b.name}",
            "What has to be true": b.outcome,
            "What people do": " · ".join(b.behaviours),
            "Number to watch": b.leading,
            "What Groww can pull": b.groww_data,
            "Where to work": ", ".join(OPP_SHORT[c] for c in b.opportunities) or "—",
        } for b in BRANCHES] + [{
            "Branch": "Recovery",
            "What has to be true": RECOVERY[0].split(": ", 1)[1].capitalize(),
            "What people do": "Notices the miss · pays it · repairs the mandate",
            "Number to watch": RECOVERY[1],
            "What Groww can pull": "Days from lapse to the next successful instalment",
            "Where to work": OPP_SHORT["payment_reliability"],
        }]).set_index("Branch"),
    )

    st.markdown("#### 3 · How the 12-month drop splits")
    st.markdown(
        f"I don't have Groww's cohort file, so I take **{lapse * 100:.0f}% of first SIPs as gone by month 12** "
        "(sidebar) and split that using the reason people named when they wrote about stopping. "
        "Orange is first-timers."
    )
    sized = sizes
    groups = [f"All investors (n={a['risk_n']})", f"First-time (n={f['risk_n']})"]
    long = pd.concat([
        pd.DataFrame({"Branch": sized["Branch"], "Group": groups[0], "pp": sized["lost_pp_all"]}),
        pd.DataFrame({"Branch": sized["Branch"], "Group": groups[1], "pp": sized["lost_pp_ft"]}),
    ])
    long["label"] = long["pp"].map(lambda v: f"{v:.1f} pp")
    base = alt.Chart(long).encode(
        y=alt.Y("Branch:N", sort=sized["Branch"].tolist(), title=None, axis=alt.Axis(labelLimit=260, labelFontSize=13)),
        yOffset=alt.YOffset("Group:N", sort=groups),
        x=alt.X("pp:Q", title=f"Percentage points of the {lapse * 100:.0f}% lapse", axis=alt.Axis(grid=False),
                scale=alt.Scale(domain=[0, float(long["pp"].max()) * 1.12])),
        color=alt.Color("Group:N", sort=groups, scale=alt.Scale(domain=groups, range=[CB_BLUE, CB_ORANGE]),
                        legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
        tooltip=["Branch:N", "Group:N", alt.Tooltip("pp:Q", title="pp of the lapse", format=".1f")],
    )
    chart = base.mark_bar(cornerRadiusEnd=3) + base.mark_text(align="left", dx=4, fontSize=12).encode(text="label:N")
    st.altair_chart(chart.properties(height=80 * len(sized)), use_container_width=True)
    table_view(sized.rename(columns={
        "pct_stop_all": "% of stop posts (all)", "pct_stop_ft": "% of stop posts (first-time)",
        "lost_pp_all": "pp of lapse (all)", "lost_pp_ft": "pp of lapse (first-time)",
        "uplift_all": "Est. uplift pp (all)", "uplift_ft": "Est. uplift pp (first-time)",
    })[["Branch", "% of stop posts (all)", "% of stop posts (first-time)", "pp of lapse (all)",
        "pp of lapse (first-time)", "Est. uplift pp (all)", "Est. uplift pp (first-time)"]])
    if f["risk_n"] < SMALL_SAMPLE:
        st.warning(
            f"Only {f['risk_n']} stop posts are from first-timers — they mostly write about starting. "
            "Treat the orange bars as a hint to take to interviews and Groww data, not a count of users."
        )

    st.markdown("#### 4 · Two problems. I built on the stop decision.")
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Problem 1 · The stop decision** (pause, skip, reduce, cancel, switch)")
        m1, m2 = st.columns(2)
        m1.metric("First-time", f"+{f['stop_moment']:.1f} pp", help="If this cause were fixed, for first-time investors")
        m2.metric("All investors", f"+{a['stop_moment']:.1f} pp", help="If this cause were fixed, for all investors")
        st.markdown(
            f"- {a['stop_moment_pct']:.0f}% of stop posts overall, {f['stop_moment_pct']:.0f}% among first-timers. "
            "They all meet on one screen: pause / cancel.\n"
            "- They need their own numbers explained, not a fund pick. That's where a short AI reply can help "
            "without becoming advice.\n"
            + (f"- **I built the MVP here.** {dec['stop_n']} of {dec['n']} interviews: the SIP was at risk "
               "because they decided to stop."
               if dec else "- Interviews still to land.")
        )
    with c2.container(border=True):
        st.markdown("**Problem 2 · Failed payments** (first debit and later ones)")
        m1, m2 = st.columns(2)
        m1.metric("First-time", f"+{f['payment']:.1f} pp", help="If this cause were fixed, for first-time investors")
        m2.metric("All investors", f"+{a['payment']:.1f} pp", help="If this cause were fixed, for all investors")
        st.markdown(
            f"- {a['payment_pct']:.0f}% of stop posts overall, {f['payment_pct']:.0f}% among first-timers. "
            "A lot of the first-timer ones never went live (KYC, mandate, first order bounced).\n"
            "- This is ops: tell them why it failed, retry, repair the mandate. AI doesn't add much.\n"
            + (f"- **Not the MVP.** {dec['pay_n']} of {dec['n']} interviews stopped over a failed debit. "
               f"{dec['failed_any']} had one, and only saw it in the app. Size it with Groww debit-return data."
               if dec else "- Interviews still to land.")
        )
    ft_small = " — small set" if f["risk_n"] < SMALL_SAMPLE else ""
    if dec:
        wrap = (f"{dec['stop_n']} of {dec['n']} interviews were a stop decision, "
                f"{dec['pay_n']} a failed instalment. That's why I built on cancel.")
    else:
        wrap = "Interviews decide which of these I build on."
    st.info(
        f"On public posts, {lead_phrase(a['stop_moment'], a['payment'])}. "
        f"In first-timer posts ({f['risk_n']} about stopping{ft_small}), "
        f"{lead_phrase(f['stop_moment'], f['payment'])}. "
        f"Posts don't tell you which moment actually ends a first SIP. {wrap}"
    )

    st.markdown("#### 5 · What I took into interviews")
    hyp = interview_hypotheses(golden_signals()).drop(columns="Interview probe")
    if dec:
        tally = hypothesis_tally(sc).set_index("ID")
        hyp["What people said"] = _interview_findings(tally, dec)
        hyp["Verdict"] = list(tally["Verdict"])
        hyp = hyp.rename(columns={"Hypothesis": "What I thought", "Evidence so far": "From posts"})
    else:
        hyp = hyp.rename(columns={"Hypothesis": "What I thought", "Evidence so far": "From posts"})
    st.table(hyp.set_index("Branch").rename_axis(None))
    st.caption(
        f"Bars assume {lapse * 100:.0f}% of first SIPs are gone by month 12 and {recovery * 100:.0f}% of each "
        "cause can come back. Change those in the sidebar."
    )


def _quote_list(quotes: list[dict[str, str]]) -> None:
    for q in quotes:
        group = "first-time" if q["group"] == COHORT_TARGET else "other investor"
        st.markdown(f"> {q['text']}  \n> — [#{q['id']}]({q['url']}) · {q['source']} · {group}")


def _payment_subcauses(t: pd.DataFrame) -> None:
    st.markdown("**Why the payment failed, as far as the text says**")
    st.caption(
        "One cause per post, first match in the text. Sized posts are the ones about stopping that go into the bars."
    )
    cols = ["First-time, sized", "Sized posts (at risk)", "All payment-tagged"]
    view = t[["Group", "Sub-cause", "In my original list"]].copy()
    for c in cols:
        view[c] = [f"{n} ({p}%)" for n, p in zip(t[c], t[f"{c} %"])]
    st.table(view.set_index("Sub-cause").rename_axis(None))

    no_fail = t[t["Group"] == "No failure described"]
    mine = t[t["In my original list"].str.startswith("Yes")]
    live = t[t["Group"] == "First SIP never went live"]
    unknown = t[t["Sub-cause"] == "Failed, no reason given"]
    ft, sized = "First-time, sized", "Sized posts (at risk)"
    st.markdown(
        f"- The three causes I started with (low balance, old account, gateway down) cover "
        f"{int(mine[sized].sum())} of {int(t[sized].sum())} sized posts, and {int(mine[ft].sum())} of "
        f"{int(t[ft].sum())} first-timer ones.\n"
        f"- A first SIP that never went live (KYC, mandate, first order bounced) is "
        f"{int(live[ft].sum())} of {int(t[ft].sum())} first-timer sized posts. That wasn't on my list. It's a setup "
        "problem, not a later miss.\n"
        f"- **{unknown[f'{sized} %'].iloc[0]}%** of sized posts say it failed and give no reason. That's bigger than "
        "any one cause I named.\n"
        f"- **{no_fail[f'{sized} %'].iloc[0]}%** don't describe a failure at all (how does autopay work?). So the "
        "payment bar is an upper bound."
    )


HYPOTHESIS_P1 = (
    "**What I thought:** a lot of SIPs die without anyone choosing to stop. The debit fails because Groww or the "
    "gateway is down, the account is short that day, or autopay is still on an old salary account after a job change."
)
HYPOTHESIS_P2 = (
    "**What I thought:** people stop on purpose for one of three reasons. They didn't understand the fund. The "
    "market keeps falling so they see a loss. Or they put extra money in during the fall, it keeps falling, and they stop."
)


def _p1_interviews(sc: pd.DataFrame, dec: dict) -> None:
    failed = sc[sc["failed_instalment"].str.startswith("yes")]
    changed = sc[sc["account_change"].str.startswith("yes")]
    st.markdown(f"#### What {dec['n']} people told me")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stopped over a failed debit", f"{dec['pay_n']} of {dec['n']}")
    c2.metric("Had a failure at some point", f"{len(failed)} of {dec['n']}")
    c3.metric("Changed job or salary account", f"{len(changed)} of {dec['n']}")
    found = " · ".join(f"{r.participant}: {r.failed_instalment[5:-1]}" for r in failed.itertuples())
    st.markdown(
        f"- They only found the failure by opening the app ({found}). Nobody remembered a notification, and nobody "
        "stopped over it.\n"
        f"- {', '.join(changed['participant'])} changed job or salary account. No mandate problem. So my job-change "
        "story isn't in this set either — I need Groww's bank-return file for that.\n"
        f"- Everyone here had been running for {sc['months_since_start'].astype(int).min()}+ months, so a first SIP "
        "that never went live can't show up. That part is posts only."
    )


def _problem_metrics(focus: dict, key: str, interviews: int | None, dec: dict | None, label: str) -> None:
    f, a = focus["ft"], focus["all"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Share of stop posts", f"{f[f'{key}_pct']}%",
              help=f"First-time investors: {f[f'{key}_n']} of {f['risk_n']} posts about stopping or lapsing have a "
                   f"{label} as their main cause. All investors: {a[f'{key}_pct']}% ({a[f'{key}_n']} of {a['risk_n']}).")
    c2.metric("Among continuers", f"{f[f'{key}_cont_pct']}%",
              help=f"First-time investors who kept their SIP ({f['cont_n']} posts) and mention it. All investors: "
                   f"{a[f'{key}_cont_pct']}%.")
    c3.metric("Est. uplift", f"+{f[key]:.1f} pp",
              help=f"Added to the 12-month active-SIP rate of first-time investors if fixed: 40% lapse × share × "
                   f"addressability × 30% recovery. All investors: +{a[key]:.1f} pp.")
    c4.metric("Interviews", f"{interviews} of {dec['n']}" if dec else "Pending",
              help=f"Participants whose risky moment was a {label}")


def _problem_mandate(m: dict, focus: dict, sc: pd.DataFrame, dec: dict | None) -> None:
    o = m["opp"]
    with st.container(border=True):
        st.markdown("### Problem 2 · Failed payments")
        st.markdown(HYPOTHESIS_P1)
        st.markdown("#### What the posts show")
        v1, v2 = st.columns(2)
        v1.success(f"**The area is real.** Failed payments are the #{o['rank']} reason SIPs end in public posts.")
        v2.warning(
            f"**The job-change story is not in the posts.** {m['job_change_with_mandate']} post links a job "
            f"change to a mandate; {m['bank_change_with_mandate']} mention a bank-account change. I need Groww's "
            "debit-return file for that."
        )
        _problem_metrics(focus, "payment", dec["pay_n"] if dec else None, dec,
                         "failed payment (first SIP never going live, or a later instalment failing)")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**What they write when a payment fails**")
            st.bar_chart(m["reports"].set_index("What users report")["Posts"], horizontal=True, y_label="",
                         x_label="Posts", color=BRAND)
            table_view(m["reports"])
        with right:
            st.markdown("**How I read this**")
            unstated = m["reports"].iloc[-1]["% of payment-failure posts"]
            st.markdown(
                f"- **{unstated}%** of failure posts can't say why it failed. The app didn't tell them.\n"
                "- The common story is autopay failing again and again, or the mandate getting cancelled without them doing it.\n"
                "- Bank-account change is rare here. People who changed jobs may just never post."
            )

        _payment_subcauses(m["subcauses"])

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**In their words**")
            _quote_list(m["quotes"])
        with e2:
            st.markdown("**What this set can't show**")
            st.markdown(
                f"- Only **{m['job_change_with_mandate']}** post connects a job change with a mandate "
                f"({m['job_change_any']} mention a job change at all).\n"
                f"- Only **{m['first_timer_n']} of {m['area_n']}** failure posts are from first-timers. They rarely "
                "write about this, so the size comes from the comparison group.\n"
                "- People who never noticed the failure don't post. Silent lapses are missing."
            )
            if m["mechanism_quotes"]:
                st.markdown("Closest example of the bank-account case:")
                _quote_list(m["mechanism_quotes"][:1])

        if dec:
            _p1_interviews(sc, dec)

        st.markdown("#### What I would build (ops, not the MVP)")
        st.info(f"**The problem:** {P1_STATEMENT}")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                "**If Groww built this**\n"
                "1. Tell them why it failed, in plain words (short of money, account closed, bank cancelled it).\n"
                "2. Let them pay this instalment now, or retry before 30 days are up.\n"
                "3. Change bank account and set up autopay in the same flow.\n"
                "4. If the bank says the account is closed, ask for the new one before the next date."
            )
        with s2:
            st.markdown(
                "**What I'd ask Groww for**\n"
                "- Of first-timer lapses, how many had a failed debit just before?\n"
                "- Why did the bank send it back (closed vs no money vs mandate cancelled)?\n"
                "- After a fail, how many never get another successful debit?\n\n"
                "**If it worked:** more failed instalments recovered in 7 days, and a higher 12-month active rate."
            )


def _deliberate_research(df: pd.DataFrame) -> None:
    """Posts collected specifically about stopping a SIP on purpose, after the same filters as the corpus."""
    path = ENGINE_DIR / "data" / "deliberate_stop_candidates.csv"
    log_path = ENGINE_DIR / "data" / "selection_log.csv"
    if not path.exists() or not log_path.exists():
        return
    from problem_fit import LABELS

    cand = pd.read_csv(path, dtype=str).fillna("")
    urls = set(cand["url"])
    base = df[df["url"].isin(urls)]
    log = pd.read_csv(log_path, dtype=str).fillna("")
    logged = log[log["url"].isin(urls)].drop_duplicates("url")
    dropped = int((~logged["decision"].str.startswith("kept")).sum())
    labels = {t.code: t.label for t in STOP_REASONS}
    status = (base["status"].map(LABELS["sip_status"]).fillna("Not stated")
              .value_counts().rename_axis("SIP status").reset_index(name="Posts"))
    cause = (base["primary_reason"].map(labels).fillna("Reason not stated")
             .value_counts().rename_axis("Cause").reset_index(name="Posts"))
    running = int(base["status"].eq("running_no_doubt").sum())
    unsure = int(base["status"].eq("running_considering_stop").sum())
    acted = int(base["status"].isin(["stopped", "stopped_and_switched", "paused_or_reduced"]).sum())
    new = int(base["query_branch"].eq("deliberate_stop").sum())
    st.markdown("#### Extra search: people who meant to stop")
    st.markdown(
        f"I also searched Reddit, TradingQ&A, YouTube, Play Store and App Store for people who **chose** to stop: "
        f"a fall, flat returns, moving the money, or averaging that went wrong. **{len(cand)} posts** (capped at 200). "
        f"Same filters as the rest — their own SIP, not a duplicate — kept **{len(base)}**. "
        f"{len(base) - new} were already in the main set; **{new} are new**. **{dropped}** were dropped."
    )
    st.markdown(
        f"Of the {len(base)} kept: **{running}** are still going with no doubt, **{unsure}** are unsure, and "
        f"**{acted}** have stopped, paused or moved the money. Most of this set is the doubt moment or a switch — "
        "not a finished exit caused by the fall alone."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What they are doing with the SIP**")
        st.dataframe(status, hide_index=True)
    with c2:
        st.markdown("**Cause read from the text**")
        st.dataframe(cause, hide_index=True)


def _problem_panic(p: dict, focus: dict, sc: pd.DataFrame, dec: dict | None, df: pd.DataFrame) -> None:
    o = p["opp"]
    with st.container(border=True):
        st.markdown("### Problem 1 · The stop decision")
        st.markdown(HYPOTHESIS_P2)
        st.info(f"**What I now think the problem is:** {P2_STATEMENT}")
        st.markdown("#### What the posts show")
        v1, v2 = st.columns(2)
        v1.success(
            f"**A fall makes people wonder.** {p['fall_n']} posts talk about a fall, "
            f"{p['fall_first_timer_n']} of them from first-timers."
        )
        separates = p["pct_fall_risk"] > p["pct_fall_cont"] * 1.3
        (v2.success if separates else v2.warning)(
            f"**{'Falls show up more in stop posts' if separates else 'Panic is not what splits stop vs continue'}.** "
            f"A fall is in {p['pct_fall_risk']}% of stop posts vs {p['pct_fall_cont']}% of people who kept going."
        )
        _problem_metrics(focus, "stop_moment", dec["stop_n"] if dec else None, dec,
                         "stop decision (pause, reduce, skip, stop or switch)")
        f = focus["ft"]
        st.caption(
            f"People who kept going mention the stop screen as often as people who stopped "
            f"({f['stop_moment_cont_pct']}% vs {f['stop_moment_pct']}%). The doubt is shared. What differs is "
            f"what they do next. {p['fall_n']} posts mention a fall; {p['fall_stop_n']} of those think about "
            f"stopping, pausing or selling; {p['asks_others_n']} ask others ('are you guys still continuing?')."
        )

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**How often each group mentions a fall**")
            st.bar_chart(p["by_group"].set_index("Group")["% mentioning a market fall"], horizontal=True,
                         y_label="", x_label="% of the group's posts", color=BRAND)
            table_view(p["by_group"])
        with right:
            st.markdown("**How I read this**")
            st.markdown(
                (f"- People who kept going mention falls as often or more ({p['pct_fall_cont']}% vs "
                 f"{p['pct_fall_risk']}%). A fall by itself doesn't end the SIP.\n" if not separates else
                 f"- Falls are more common among people who stop ({p['pct_fall_risk']}% vs {p['pct_fall_cont']}%).\n")
                + "- What they don't have is a way to tell if *their* SIP is okay. So they ask someone and do "
                "what that person says.\n"
                f"- {o['pct_risk']}% of stop posts are this should-I-stop moment."
            )

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**In their words**")
            _quote_list(p["quotes_for"])
        with e2:
            st.markdown("**The other side**")
            _quote_list(p["quotes_against"])

        _deliberate_research(df)

        if dec:
            st.markdown(f"#### What {dec['n']} people told me")
            st.markdown(
                f"**{dec['stop_n']} of {dec['n']}** — the SIP was at risk because they decided to stop. "
                "Each line of the problem, and who it came from:"
            )
            st.dataframe(
                p2_evidence(sc),
                hide_index=True,
                column_config={
                    "Problem statement says": st.column_config.TextColumn(width="medium"),
                    "Interviews": st.column_config.TextColumn(width="small"),
                    "Who": st.column_config.TextColumn(width="small"),
                    "What they said or did": st.column_config.TextColumn(width="large"),
                },
            )

        st.markdown("#### What I built")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                "**A sheet before they cancel**\n"
                "1. Their own goal, how long they've been in, and how it's doing.\n"
                "2. A plain line on what the last instalments bought (more units when the price was down).\n"
                "3. What stopping changes for that goal — no scare line.\n"
                "4. Other than cancel: lower the amount, skip a month, move the date."
            )
        with s2:
            st.markdown(
                "**What I'd still check with Groww**\n"
                "- Do first-timer cancels jump in the two weeks after a >5% index fall?\n"
                "- How many of those people were on the fund page during the fall?\n"
                "- A/B the sheet in the cancel flow.\n\n"
                "**If it worked:** more people pick skip or reduce instead of cancel, and more SIPs still running at 90 days and 12 months."
            )


def _interview_evidence(sc: pd.DataFrame, dec: dict) -> None:
    with st.container(border=True):
        st.markdown("### The five interviews")
        st.markdown(
            f"{dec['n']} first-time SIP investors, {sc['months_since_start'].astype(int).min()}–"
            f"{sc['months_since_start'].astype(int).max()} months in. I recruited them from people I know who had "
            "stopped, reduced, or thought about stopping. P1–P5."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Interviews", dec["n"])
        c2.metric("At risk because they decided to stop", f"{dec['stop_n']} of {dec['n']}")
        c3.metric("At risk because a debit failed", f"{dec['pay_n']} of {dec['n']}")
        st.markdown(
            f"**{dec['stop_n']} of {dec['n']}** were a stop decision. **{dec['pay_n']}** a failed debit. "
            "The rule I wrote before talking to them: stay on the stop screen if at least 3 of 5 describe that moment. "
            "That's what happened, so that's what I built."
        )

        st.markdown("**Who I spoke to**")
        st.table(sc.rename(columns={
            "participant": "Participant", "age_band": "Age band", "months_since_start": "Months in",
            "app": "App", "status_today": "Status today", "account_change": "Job / account change",
            "failed_instalment": "Failed instalment", "critical_moment": "The moment it was at risk",
        })[["Participant", "Age band", "Months in", "App", "Status today", "Job / account change",
            "Failed instalment", "The moment it was at risk"]].set_index("Participant").rename_axis(None))

        st.markdown("**What happened, person by person**")
        for inc in critical_incidents(sc):
            with st.expander(f"{inc['participant']} · {inc['When'].rstrip('.')}"):
                st.markdown(
                    f"- **Who got them started:** {inc['started_by']}\n"
                    f"- **What they saw first:** {inc['Trigger (what they saw first)']}\n"
                    f"- **Who they asked:** {inc['Who they asked, what they were told']}\n"
                    f"- **What they did:** {inc['What they did, and how it ended']}\n"
                    f"- **What was missing:** {inc['What was missing']}"
                )
        st.caption(
            "Five people from my network, chosen because they were at risk. This shows what the moment looks like, "
            "not how common it is. P3 is on another app. Notes: user-research/notes/P1–P5.md."
        )


def render_problem_validation_tab(df: pd.DataFrame) -> None:
    st.subheader("Did I have the right problem?")
    st.markdown(
        f"I checked each one in **{len(df)} public posts** (is it real?) and **{len(load_scorecard())} interviews** "
        "(what does the moment look like?). Problem 1 is the stop decision — that's the MVP. Problem 2 is failed "
        "payments — that's ops."
    )
    h = build_hypotheses(
        df, st.session_state["lapse_rate"] / 100, st.session_state["recovery_rate"] / 100, _overrides_from_state()
    )
    m, p = h["mandate"], h["panic"]
    focus = focus_comparison(df, st.session_state["lapse_rate"] / 100, st.session_state["recovery_rate"] / 100,
                             _overrides_from_state())
    sc = load_scorecard()
    dec = decision(sc) if not sc.empty else None
    _problem_panic(p, focus, sc, dec, df)
    _problem_mandate(m, focus, sc, dec)
    if dec:
        _interview_evidence(sc, dec)
    else:
        st.info("Interview scorecard not found (user-research/scorecard.csv).")

    keep = bool(dec) and dec["verdict"] == "Keep the stop-moment focus"
    interviews_stop = f"{dec['stop_n']} of {dec['n']}" if dec else "Pending"
    interviews_pay = (f"{dec['pay_n']} of {dec['n']}; {dec['failed_any']} had a failure, none stopped over it"
                      if dec else "Pending")
    st.markdown("### Side by side")
    ft = focus["ft"]
    st.dataframe(
        pd.DataFrame([
            {"Problem": "1 · The stop decision", "Share of stop posts": f"{ft['stop_moment_pct']}%",
             "Among people who kept going": f"{ft['stop_moment_cont_pct']}%",
             "If fixed": f"+{ft['stop_moment']:.1f} pp",
             "Interviews": interviews_stop,
             "What I got wrong": "I thought it was panic. It's that they have no way to judge their own SIP.",
             "What I did": "Built the MVP here" if keep else "Pending interviews"},
            {"Problem": "2 · Failed payments", "Share of stop posts": f"{ft['payment_pct']}%",
             "Among people who kept going": f"{ft['payment_cont_pct']}%",
             "If fixed": f"+{ft['payment']:.1f} pp",
             "Interviews": interviews_pay,
             "What I got wrong": "Low balance shows up. Job-change almost doesn't. First SIPs that never go live weren't on my list.",
             "What I did": "Ops track, not the MVP" if keep else "Pending interviews"},
        ]),
        hide_index=True, width="stretch",
    )
    st.caption(f"First-timers: {ft['risk_n']} posts about stopping.")
    wrap = (
        f"{dec['stop_n']} of {dec['n']} interviews were a stop decision, {dec['pay_n']} a failed debit. "
        "I built on cancel. Failed payments stay an ops job: tell them why, retry, repair the mandate."
        if dec else
        "Interviews decide which of these I build on."
    )
    st.info(
        f"On public posts, {lead_phrase(focus['all']['stop_moment'], focus['all']['payment'])}. "
        f"In first-timer posts, {lead_phrase(focus['ft']['stop_moment'], focus['ft']['payment'])}. {wrap}"
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
        ["Overview", "Metric decomposition", "Problem validation", "Ask the engine", "Opportunity sizing",
         "How it works & evidence", "Evidence explorer"]
    )
    with tabs[0]:
        render_overview(df)
    with tabs[1]:
        render_decomposition_tab(df)
    with tabs[2]:
        render_problem_validation_tab(df)
    with tabs[3]:
        render_discovery_tab(df)
    with tabs[4]:
        render_opportunity_tab(df)
    with tabs[5]:
        render_how_it_works(df)
    with tabs[6]:
        render_browse_tab(df)


def render_metric_branches_tab(df: pd.DataFrame) -> None:
    st.subheader("Where posts about stopping come from — 4 metric branches")
    risk = df[df["is_at_risk"]]
    no_link = df[~df["is_at_risk"] & (df["link_to_lapse"] == "none")]
    m1, m2, m3 = st.columns(3)
    m1.metric("Stop posts", len(risk), help="Stopped, paused, payment failed, or a strong link to lapse")
    m2.metric("Weak link", len(df) - len(risk) - len(no_link), help="Posts with a weak link to stopping")
    m3.metric(
        "Not about stopping", len(no_link),
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


if __name__ == "__main__":
    main()
