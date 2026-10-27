"""Deterministic reasoned answer (used when no LLM is configured or it returns nothing)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from reasoning import STOP_CLASSES, STOP_REASONS

HYPOTHESES: dict[str, tuple[str, str]] = {
    "payment_reliability": (
        "Most lapses after a failed debit happen because the investor did not notice the failure, "
        "not because they chose to stop.",
        "Wrong if interviewees noticed the failure quickly and chose not to retry.",
    ),
    "control_clarity": (
        "Investors who pause intend to resume, but nothing in the product brings them back.",
        "Wrong if paused investors say they had already decided to quit when they paused.",
    ),
    "switch_without_gap": (
        "When investors change fund or platform, the new SIP starts later than the old one "
        "stopped — or never starts.",
        "Wrong if switchers consistently restart within the same instalment cycle.",
    ),
    "decision_support": (
        "'Should I stop?' doubt comes from having no personal reference point for 'on track', "
        "not from returns alone.",
        "Wrong if doubters cite a clear return threshold they set at the start and simply missed.",
    ),
    "understandability": (
        "First-time investors cannot tell from the app whether their SIP is working.",
        "Wrong if interviewees correctly explain their own returns and feel reassured by them.",
    ),
    "cash_flexibility": (
        "Cash-squeezed investors cancel because they don't know skip / reduce / date change exist.",
        "Wrong if they knew these options and still preferred to cancel.",
    ),
}


def _lift_str(lift: Any, base_label: str) -> str:
    if lift is None or pd.isna(lift):
        return ""
    if lift >= 1.2:
        return f", **{lift}×** more common than in {base_label}"
    if lift <= 0.8:
        return f", less common than in {base_label} ({lift}×)"
    return f", about as common as in {base_label}"


def _contrast_sentence(p_group: float, p_cont: float) -> str:
    if p_cont <= 0 and p_group > 0:
        return "Almost never mentioned by investors who continued — a strong separator."
    if p_group >= 1.5 * p_cont:
        return f"Much rarer among continuers ({p_cont}%), so it separates drop-offs from stayers."
    if p_cont >= p_group:
        return (
            f"Just as common among continuers ({p_cont}%) — on its own it does **not** predict stopping."
        )
    return f"Somewhat more common than among continuers ({p_cont}%)."


def _segment_section(a: dict[str, Any]) -> list[str]:
    seg: pd.DataFrame = a["segments"]
    ft_col, other_col = seg.columns[1], seg.columns[2]
    seg = seg.assign(diff=seg[ft_col] - seg[other_col]).sort_values("diff", ascending=False)
    lines = ["### How first-time investors differ"]
    for _, r in seg.head(2).iterrows():
        if r["diff"] > 0:
            lines.append(f"- More often **{r['Theme'].lower()}**: {r[ft_col]}% vs {r[other_col]}% for others.")
    for _, r in seg.tail(2).iloc[::-1].iterrows():
        if r["diff"] < 0:
            lines.append(f"- Less often **{r['Theme'].lower()}**: {r[ft_col]}% vs {r[other_col]}% for others.")
    lines.append(
        f"- *Read with care:* only {a['first_timer_n']} at-risk posts carry a first-time-investor tag, "
        "so these gaps are hypotheses for interviews, not findings. Income stability and source of money "
        "are not tagged in public posts at all — cover them in interviews."
    )
    lines.append("")
    return lines


CO_INTERPRETATION: dict[str, str] = {
    "redirect": "the usual response is to move the money elsewhere, not to stop investing",
    "undecided": "it usually turns into an open 'should I stop?' question taken outside the app",
    "involuntary": "the stop is often caused by the system rather than by a decision",
    "friction": "people try to adjust the SIP and get stuck on the controls",
    "exit": "it combines with a deliberate decision to leave",
}

ACTION_LABEL = {"pause": "pausing or skipping", "reduce": "reducing the amount", "cancel": "stopping or cancelling"}


def _focus_answer(a: dict[str, Any], f: dict[str, Any]) -> str:
    rr, rw, rwo = f["risk_ratio"], f["risk_rate_with"], f["risk_rate_without"]
    core = (
        f"posts that mention **{f['label'].lower()}** involve stopping, pausing or lapsing "
        f"{rr}× as often as other posts ({rw}% vs {rwo}%), and it appears in {f['share_of_at_risk']}% of "
        f"at-risk posts vs {f['share_of_continuers']}% of posts by people who kept investing "
        f"(#{f['rank_among_causes']} of {f['n_causes']} causes)."
    )
    if not a["is_hypothesis"]:
        return f"Across {f['mentions']} public posts, " + core
    lead = {"Supported": "**Yes.**", "Partly supported": "**Partly.**", "Not supported": "**Not on its own.**"}
    return f"{lead[f['verdict']]} In public posts, " + core


def _focus_reasoning(a: dict[str, Any], f: dict[str, Any]) -> list[str]:
    theme = next((t for t in STOP_REASONS if t.code == f["code"]), None)
    lines = [f"**{f['label']}**"]
    if theme:
        lines.append(f"- *Mechanism:* {theme.mechanism}")
    if f["co_occurring"]:
        label, _, pct = f["co_occurring"][0]
        co_theme = next((t for t in STOP_REASONS if t.label == label), None)
        interp = CO_INTERPRETATION.get(co_theme.stop_class, "") if co_theme else ""
        others = "; ".join(f"{l.lower()} {p}%" for l, _, p in f["co_occurring"][1:])
        lines.append(
            f"- *Underlying cause:* in this group, {pct}% of these posts also talk about "
            f"**{label.lower()}** — {interp}." + (f" Also: {others}." if others else "")
        )
    if f["behaviour"]:
        did = ", ".join(f"{k} ({v})" for k, v in f["behaviour"].items())
        lines.append(f"- *What they did:* {did}.")
    if f["stop_kind"]:
        top_kind, n_kind = max(f["stop_kind"].items(), key=lambda x: x[1])
        lines.append(
            f"- *Kind of stop:* mostly **{STOP_CLASSES[top_kind][0].lower()}** ({n_kind} of {f['in_group']} posts) — "
            f"{STOP_CLASSES[top_kind][1][0].lower()}{STOP_CLASSES[top_kind][1][1:]}"
        )
    if f["needs"]:
        lines.append("- *What they wished for:* " + " · ".join(n[:90] for n in f["needs"][:2]))
    ev = a["evidence"].get(f["code"], [])[:2]
    if ev:
        lines.append("- *Evidence:* " + " ".join(f"[#{e['id']}] “{e['text'][:110]}…”" for e in ev))
    if f["verdict"] != "Supported" and f["share_of_continuers"] >= f["share_of_at_risk"]:
        lines.append(
            f"- *Counter-evidence:* investors who kept going mention it just as often "
            f"({f['share_of_continuers']}%), so it explains complaints more than it explains stopping."
        )
    lines.append("")
    return lines


def reasoned_narrative(a: dict[str, Any]) -> str:
    intent = a["intent"]
    n = a["pop_n"]
    if n == 0:
        return "No posts match this question. Try wording it around stopping, pausing, payments, checking returns or starting."

    prof: pd.DataFrame = a["profile"]
    top = prof[prof["Posts"] > 0].head(4)
    is_stop_lens = intent.lens is STOP_REASONS
    base_label = intent.base
    focus = a["focus"]
    lines: list[str] = ["### Answer"]

    if a["action_n"]:
        lines.append(f"*Looking at the {a['action_n']} posts about {ACTION_LABEL[a['action']]}.*")
    elif a["action"] and is_stop_lens:
        lines.append(
            f"Only **{a['action_count']}** public posts talk about {ACTION_LABEL[a['action']]} — too few to "
            f"analyse on their own, which is itself a finding: people rarely mention this option. "
            f"Below is the wider group of {n} at-risk posts."
        )
    needs: pd.DataFrame = a["needs"]
    needs_mode = intent.code == "needs" and not needs.empty and not focus
    if needs_mode:
        lines.append(
            f"Across {n} at-risk posts that state an unmet need, the most common needs are "
            + "; ".join(f"**{r['Need theme'].lower()}** ({r['% of stated needs']}%)" for _, r in needs.head(3).iterrows())
            + "."
        )
    elif focus:
        lines += [_focus_answer(a, f) for f in focus[:2]]
        others = [r for _, r in top.iterrows() if r["code"] not in {f["code"] for f in focus}][:2]
        if others and is_stop_lens:
            lines.append(
                "For comparison, the biggest causes in the same group are "
                + " and ".join(f"**{r['Theme'].lower()}** ({r['% of group']}%)" for r in others) + "."
            )
    elif is_stop_lens and intent.code != "contrast":
        classes = a["stop_classes"]
        named = classes[classes["code"] != "unclear"].sort_values("Posts", ascending=False)
        c1, c2 = named.iloc[0], named.iloc[1]
        exit_pct = float(classes.loc[classes["code"] == "exit", "% of group"].iloc[0])
        lead = top.iloc[0] if not top.empty else None
        lines.append(
            f"Across **{n}** public posts ({intent.population}), stopping is rarely one "
            f"'I quit' decision. The largest groups are **{c1['Kind of stop'].lower()}** "
            f"({c1['% of group']}%) and **{c2['Kind of stop'].lower()}** ({c2['% of group']}%); "
            f"only **{exit_pct}%** are deliberate exits over returns or a cash need."
        )
        if lead is not None:
            lines.append(
                f"The single biggest cause is **{lead['Theme'].lower()}** "
                f"({lead['% of group']}% of the group{_lift_str(lead['Lift'], base_label)})."
            )
    elif top.empty:
        lines.append(f"{n} posts matched, but none carry a recognisable pattern for this question.")
    else:
        t1 = top.iloc[0]
        parts = [f"Among **{n}** {intent.population}, the most common pattern is "
                 f"**{t1['Theme'].lower()}** ({t1['% of group']}%)"]
        if len(top) > 1:
            t2 = top.iloc[1]
            parts.append(f", followed by **{t2['Theme'].lower()}** ({t2['% of group']}%)")
        lines.append("".join(parts) + ".")
        lifted = top.dropna(subset=["Lift"]).sort_values("Lift", ascending=False)
        if not lifted.empty and lifted.iloc[0]["Lift"] >= 1.2:
            t = lifted.iloc[0]
            lines.append(f"Most specific to this group: **{t['Theme'].lower()}** ({t['Lift']}× vs {base_label}).")

    lines += ["", "### Reasoning"]
    if needs_mode:
        for _, r in needs.head(4).iterrows():
            theme = next((t for t in STOP_REASONS if t.label == r["Need theme"]), None)
            lines.append(f"**{r['Need theme']}** — {r['Posts']} posts ({r['% of stated needs']}% of stated needs)")
            lines.append(f"- *In their words:* {r['Examples']}")
            if theme:
                lines.append(f"- *Why the need exists:* {theme.mechanism}")
            lines.append("")
    elif focus:
        for f in focus[:2]:
            lines += _focus_reasoning(a, f)
    else:
        for i, (_, row) in enumerate(top.iterrows(), start=1):
            lines.append(
                f"**{i}. {row['Theme']}** — {row['Posts']} posts, {row['% of group']}% of the group"
                f"{_lift_str(row['Lift'], base_label)}."
            )
            lines.append(f"- *Mechanism:* {row['Mechanism']}")
            lines.append(f"- *Continuers:* {_contrast_sentence(row['% of group'], row['% among continuers'])}")
            ev = a["evidence"].get(row["code"], [])[:2]
            if ev:
                lines.append("- *Evidence:* " + " ".join(f"[#{e['id']}] “{e['text'][:110]}…”" for e in ev))
            lines.append("")

    if intent.code == "segment":
        lines += _segment_section(a)
    if intent.code == "journey" and not a["lifecycle"].empty:
        lines.append("### How it changes along the journey")
        for _, r in a["lifecycle"].iterrows():
            lines.append(
                f"- **{r['Stage']}** ({r['Posts']} posts): top cause *{r['Top cause'].lower()}* "
                f"({r['Top cause %']}%), {r['High link to lapse %']}% with a high link to lapse."
            )
        lines.append("")
    if intent.code == "contrast" or (not focus and intent.code in ("stop", "needs")):
        lines += _separators_section(a)

    opps: pd.DataFrame = a["opportunities"]
    relevant = opps[opps["Relevant to question"]]
    ranked = (relevant if not relevant.empty else opps).head(2 if focus else 3)
    lines.append("### What it means for the 12-month metric")
    for i, (_, o) in enumerate(ranked.iterrows(), start=1):
        lines.append(
            f"{i}. **{o['Opportunity']}** — leverage **{o['Leverage index']}**, ≈ **+{o['Est. uplift (pp)']} pp** "
            f"under current assumptions ({o['Evidence posts']} at-risk posts, confidence {o['Confidence']}).  \n"
            f"   *Direction:* {o['Product direction']}"
        )

    lines += ["", "### Validate next"]
    for _, o in ranked.iterrows():
        h = HYPOTHESES.get(o["code"])
        if h:
            lines.append(f"- **Hypothesis:** {h[0]} *{h[1]}*")
    reddit = a["source_mix"].get("Reddit", 0)
    weak = f" Only {a['continuers_n']} continuer posts exist, so stayer-vs-drop-off contrasts are weak." if a["continuers_n"] < 30 else ""
    lines.append(
        f"- **Data limitation:** {reddit}% of this group is Reddit (skews experienced), "
        f"{a['first_timer_n']} posts are tagged first-time investors.{weak} Treat sizes as directional."
    )
    return "\n".join(lines)


def _separators_section(a: dict[str, Any]) -> list[str]:
    sp: pd.DataFrame = a["stop_profile"]
    sp = sp[sp["Posts"] >= 3].copy()
    if sp.empty:
        return []
    sp["sep"] = sp["% of group"] - sp["% among continuers"]
    separators = sp.sort_values("sep", ascending=False).head(2)
    non_pred = sp[sp["% among continuers"] >= sp["% of group"]].sort_values("% among continuers", ascending=False).head(2)
    lines = [f"### What separates people who continue (n={a['continuers_n']} continuer posts)"]
    for _, r in separators.iterrows():
        if r["sep"] > 0:
            lines.append(f"- **{r['Theme']}**: {r['% of group']}% of this group vs {r['% among continuers']}% of continuers.")
    for _, r in non_pred.iterrows():
        lines.append(
            f"- **{r['Theme']}** is mentioned by {r['% among continuers']}% of continuers vs "
            f"{r['% of group']}% here — people complain about it and keep investing anyway."
        )
    lines.append("")
    return lines
