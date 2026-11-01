"""Reasoning layer: root causes, stayer-vs-dropoff contrast, and opportunity sizing.

Codebook triggers are one vague label per post, so causes are derived from the
post text itself with a fixed, auditable pattern taxonomy (no free-form labels).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")


@dataclass(frozen=True)
class Theme:
    code: str
    label: str
    pattern: str
    mechanism: str
    stop_class: str = ""


BANK_CHANGE = (r"chang\w* (my |the )?bank( account)?|bank account (is |got )?(frozen|closed|changed)"
               r"|account (is |got )?(frozen|closed)|new (bank|salary) account|salary account|update (my |the )?bank")

# One sub-cause per payment post: the first match wins, so explicit causes come before stages and the
# catch-alls. Most patterns stay inside one sentence ([^.]) so a word elsewhere in a long post doesn't count.
# (group, sub-cause, in my original list?, pattern)
_FAIL = r"(fail|pending|reject|error|issue|problem|unable|can'?t|cannot|not able|doesn'?t (work|happen)|not (approved|registered|available|activ|work))"
_SETUP = r"(set ?up|setup|activat|regist|generat|enabl|e-?mandate|auto[- ]?debit option)"
PAYMENT_SUBCAUSES: list[tuple[str, str, str, str]] = [
    ("First SIP never went live", "KYC not validated or pending", "No",
     r"\bkyc\b[^.]{0,40}(not|pending|valid|reject|fail|nil|stuck|still)|(not|pending|reject\w*)[^.]{0,30}\bkyc\b"),
    ("Running SIP debit failed", "Mandate limit, expiry or date mismatch", "No",
     r"(autopay|auto pay|mandate|upi)[^.]{0,15}limit|mandate (has |got )?expir|due date[^.]{0,30}mandate"
     r"|mandate start date|capping on|more than the (max|limit)"),
    ("Running SIP debit failed", "Low or insufficient balance", "Yes",
     r"insufficient|low balance|no (money|balance|funds) in|not enough (balance|money|funds)"
     r"|(debit|mandate|payment|ach|nach|ecs)[^.]{0,20}bounce|bounce charges?|ach debit return|return charges?"),
    ("Running SIP debit failed", "Old, closed or changed bank account", "Yes",
     BANK_CHANGE + r"|(old|previous|closed|inactive|dormant) (bank )?account|switch\w* (my )?bank"),
    ("Fund-house side", "Fund suspended or fund limit reached", "No",
     r"overseas (investment )?limit|investment limit|suspend\w*|stopped (taking|accepting)"
     r"|not accepting (new|fresh|sip|lump)|offshore.{0,120}(cancel|reject)"),
    ("Money taken, SIP not invested", "Debited but units not allotted, or refund pending", "No",
     r"(debited|deducted) but|(debited|deducted)[^.]{0,40}(not|no) (units|allot|invest|reflect|order)"
     r"|refund|not allot|units? (not|weren'?t|were not) (allot|credit)|deducted (money )?(twice|thrice|multiple)"),
    ("First SIP never went live", "Mandate or autopay setup failed or pending", "No",
     _SETUP + r"[^.]{0,60}" + _FAIL + r"|(unable|can'?t|cannot|not able|fail\w*|problem|issue|error|tried|spent hours)[^.]{0,60}"
     + _SETUP),
    ("First SIP never went live", "First order or first instalment rejected", "No",
     r"(first|1st|initial)[^.]{0,40}(instal\w*|order|sip|payment|txn|transaction|invest\w*)[^.]{0,80}(fail|reject|declin)"
     r"|(order|sip)s? (is |was |were |got |has been )?(rejected|declined)|rejected [^.]{0,20}(after|due)"),
    ("Cancelled or stopped without user action", "SIP cancelled, stopped or skipped by itself", "No",
     r"cancell?ed (automatically|without)|got cancell?ed|(stopped|cancell?ed) automatically|got stopped"
     r"|without (my |any )?(action|permission)|never requested|(sip|instal\w*|autopay)[^.]{0,20}skipped"),
    ("System failure", "Server, payment gateway, bank or app error", "Yes (merged)",
     r"server|gateway|technical (issue|error|glitch|problem)|glitch|\bbugs?\b|buggy|outage|downtime|\bnpci\b"
     r"|razorpay|app (is )?(down|crash)|something went wrong|(showing|throw\w*|got an?) error"),
    ("Running SIP debit failed", "Failed, no reason given", "—",
     r"fail|reject|declin|not deducted|not (being )?debited|not working|stopped working"),
]
NO_FAILURE = ("No failure described", "Question about how autopay or mandates work", "—")
ACTIVATION_GROUP = "First SIP never went live"


def payment_subcause(df: pd.DataFrame) -> pd.Series:
    """Index into PAYMENT_SUBCAUSES of each post's first matching sub-cause (-1 = no failure described)."""
    out = pd.Series(-1, index=df.index)
    text = df["post_clean"].str.lower()
    for i, (*_, pattern) in enumerate(PAYMENT_SUBCAUSES):
        out = out.mask((out == -1) & text.str.contains(pattern, regex=True), i)
    return out


# Order matters: first match becomes the post's primary reason (most specific first).
STOP_REASONS: list[Theme] = [
    Theme(
        "activation_failure",
        "First SIP never went live (KYC, mandate setup, first order)",
        "|".join(p for g, *_, p in PAYMENT_SUBCAUSES if g == ACTIVATION_GROUP),
        "The first SIP is created but never gets a successful instalment: KYC isn't validated, the mandate "
        "never registers, or the first order is rejected. The investor never becomes active.",
        "involuntary",
    ),
    Theme(
        "involuntary_system",
        "Payment, mandate or platform stopped the SIP",
        r"mandate|autopay|auto[- ]?pay|auto[- ]?debit|declined|rejected|sip failed|failed (sip|debit|instal)"
        r"|bounce|not deducted|cancell?ed (automatically|without)|never requested|without (my )?(permission|consent)"
        r"|got cancelled|(stopped|cancelled) automatically",
        "A debit fails, a mandate is revoked, or the platform closes the SIP. The investor did not "
        "decide to stop, and often only notices weeks later.",
        "involuntary",
    ),
    Theme(
        "fund_suspension",
        "Fund house paused or closed new instalments",
        r"overseas|investment limit|stopped (taking|the further|further) |(paused|suspend\w*|stop\w*) "
        r"(active|new|existing|ongoing|fresh) (sips?|subscriptions|investments)|temporary pause"
        r"|will stop ongoing|stopped the further investment|front running|stopped taking sip",
        "The AMC suspends a fund (e.g. overseas limits). The SIP ends even though the investor "
        "wanted to continue, and they are left asking what to do instead.",
        "involuntary",
    ),
    Theme(
        "control_friction",
        "Can't pause, resume, skip or cancel as expected",
        r"how (do i|to|can i) (stop|cancel|pause|un-?pause|resume|skip|close|change)|un-?pause|can'?t (see|find)"
        r"|only (see )?(a |the )?pause option|still (deducted|debited|active|showing)|sip status|skip (next|one|a|the)"
        r"|what happens if (we|i) (miss|skip)|(option|possible) to pause|is it possible to pause"
        r"|pause .{0,30}for .{0,10}months|take (a )?month to stop|deducted despite",
        "The investor wants a temporary or partial change, but the controls are unclear (pause vs "
        "stop, no resume, status mismatch). A pause becomes a permanent stop by default.",
        "friction",
    ),
    Theme(
        "cash_or_goal",
        "Needs the money / goal reached / life change",
        r"down ?payment|emergency|family reasons|personal reasons|lost (my )?job|job loss|laid off|can'?t afford"
        r"|need (the )?money|higher stud|wedding|medical|target reached|reached the required|financial (crunch|trouble)"
        r"|don'?t have (the )?(money|balance)|salary (delay|cut)|home loan|relocat|expenses are increasing",
        "Cash flow changes (loan, relocation, emergency) or a goal is met. The SIP is the most "
        "visible flexible expense, so it is cut entirely instead of reduced or skipped.",
        "exit",
    ),
    Theme(
        "poor_returns",
        "Unhappy with returns",
        r"poor (performance|returns?)|not (at all )?giving returns|negative returns?|in (the )?red|\bin loss|losses"
        r"|underperform|barely|cheated|no returns|low returns|returns (are|is) (not|bad|poor|shit)"
        r"|structural failure|down \d+ ?%|benchmark|i'?m done",
        "Returns fall short of what the investor expected when starting. Without a reference for "
        "what is normal at this stage, a slow year reads as 'not working'.",
        "exit",
    ),
    Theme(
        "platform_move",
        "Moving the SIP to another platform / AMC",
        r"mf ?central|amc website|directly (with|from) (the )?amc|moving out of|move (my )?sips?"
        r"|shift (the same|my sips?)|from groww (to|and)|change from .{0,30}groww",
        "The investor wants to keep investing but through another channel. The stop/restart is "
        "manual, so a gap (or a forgotten restart) is likely.",
        "redirect",
    ),
    Theme(
        "fund_switch",
        "Stopping one SIP to invest elsewhere",
        r"switch(ed|ing)? (to|from|it|my)|instead of|suggest .{0,25}funds?|which (sips?|funds?) to stop"
        r"|stop\w* .{0,60}(and|&) (increase|start|invest|put|shift)|reallocat|trim|consolidat"
        r"|too many (funds|sips)|overlap|regular (plan|fund|sip)s?|direct (plan|fund|mutual)|replace"
        r"|better (fund|option)|(into|as) (a )?lump ?sum|change .{0,20}lump ?sum|tiny amounts",
        "The investor stays committed to investing but changes the vehicle (consolidate, regular to "
        "direct, lump sum). The old SIP is cancelled first; the new one depends on a second decision.",
        "redirect",
    ),
    Theme(
        "market_timing",
        "Trying to time the market",
        r"market (is |becomes )?(at )?(all[- ]time )?(high|expensive)|crash|correction|right time"
        r"|wait(ing)? for (the )?(dip|crash|correction)|recogni[sz]e a dip|book(ing)? profit|p/?e ratio|time the market",
        "The investor considers pausing to wait for a better entry. This turns a habit into a "
        "repeated decision, and each decision is a chance to not restart.",
        "undecided",
    ),
    Theme(
        "decision_doubt",
        "Unsure whether to continue (asks others)",
        r"should (i|you|we) (continue|stop|pause|exit)|continue or stop|stop or continue"
        r"|when (would|should) (you|i) (consider )?stop|need to pause|is it time to|thinking (of|to) stop"
        r"|do i (need|have) to stop",
        "The investor has no framework to judge whether their SIP is on track, so they ask strangers. "
        "The answer they get, not their own goal, decides whether they continue.",
        "undecided",
    ),
    Theme(
        "confusion",
        "Doesn't understand returns / statements",
        r"xirr|\bnav\b|confus|what does .{0,30} mean|why (is|does) my|explain|how does .{0,30} work"
        r"|don'?t understand|not reflect|missing from",
        "Metrics like XIRR, NAV or unit allotment are hard to read, so the investor cannot tell "
        "whether something is wrong or normal.",
        "undecided",
    ),
]

STOP_CLASSES: dict[str, tuple[str, str]] = {
    "involuntary": (
        "Didn't choose to stop",
        "Lost to the metric without any user decision; the cheapest SIPs to save.",
    ),
    "friction": (
        "Wanted control, product got in the way",
        "User wanted to adjust, not quit; unclear controls convert a pause into a lapse.",
    ),
    "redirect": (
        "Stopping one SIP to invest elsewhere",
        "Intent to stay invested is intact; risk is the gap between cancel and restart.",
    ),
    "undecided": (
        "Unsure whether to continue",
        "Decision is still open; outcome depends on the information they find.",
    ),
    "exit": (
        "Deliberate exit (returns or cash need)",
        "Hardest to change; only flexibility or expectation-setting help, never pressure.",
    ),
    "unclear": ("Reason not stated", "Not enough text to infer a cause."),
}

START_MOTIVES: list[Theme] = [
    Theme("long_term_wealth", "Long-term wealth / retirement", r"long[- ]term|wealth|retire|\bfire\b|crore|1 ?cr\b|compounding",
          "Starts with a multi-year horizon; expects compounding to show up eventually."),
    Theme("expected_return", "Has a specific return number in mind", r"\d{1,2} ?% (return|cagr|xirr|annual|p\.?a)|expected return|how much (will|can) i (get|earn)",
          "Anchors on a % (often 12–15%) that early statements rarely match."),
    Theme("tax_saving", "Tax saving (ELSS / 80C)", r"\belss\b|tax sav|80c",
          "Started for a tax deadline; the SIP may end when the tax need is met."),
    Theme("influenced", "Influenced by YouTube / friends / family", r"youtube|friend|colleague|influencer|family|father|told me",
          "Borrowed conviction; when the source goes quiet or results lag, conviction fades."),
    Theme("start_small_learn", "Starting small to learn", r"small amount|start(ing)? (with|small)|beginner|new to|first (sip|time|investment)|\blearn",
          "Low commitment by design; an easy SIP to drop if it feels unimportant."),
    Theme("goal_specific", "Saving for a specific goal", r"\bgoal|house|car\b|wedding|education|child",
          "A concrete goal anchors continuation better than a vague 'returns' expectation."),
    Theme("portfolio_validation", "Wants someone to validate the choice", r"portfolio review|rate my|review my|how am i doing|is this (good|okay|ok)|suggest(ions)?|roast",
          "Needs external confirmation from day one; the same dependency drives later doubt."),
]

CHECKING: list[Theme] = [
    Theme("returns_value", "Checks returns / current value", r"returns?|xirr|cagr|current value|profit|gain",
          "Judges the SIP on a % that is noisy in the first months."),
    Theme("frequent_checking", "Checks very often", r"daily|every day|everyday|weekly|keep checking|check(ing)? (it|my|the)|every (morning|week)",
          "Frequent checks amplify normal volatility into perceived failure."),
    Theme("status_units", "Checks status / units / statements", r"statement|status|reflect|units|transaction|allot|pending",
          "Operational uncertainty (did it go through?) rather than performance."),
    Theme("benchmark_compare", "Compares with index / peers", r"benchmark|nifty|index|compare|peer|than (my )?(fd|friend)",
          "Relative comparison makes a positive return feel like a loss."),
    Theme("metric_confusion", "Confused by XIRR / NAV / absolute return", r"xirr|\bnav\b|absolute return|confus|what does .{0,30} mean|explain",
          "Cannot interpret what they see, so cannot tell normal from wrong."),
]

OUTSIDE_APP: list[Theme] = [
    Theme("validate_portfolio", "Asks strangers to review their portfolio", r"portfolio review|rate my|review my|roast|how am i doing|is this (good|okay|ok)|feedback",
          "Seeks reassurance that the plan is right; the app gives no such signal."),
    Theme("should_i_stop", "Asks whether to continue or stop", STOP_REASONS[8].pattern,
          "The continue/stop decision is outsourced to a forum."),
    Theme("fund_suggestion", "Asks which fund to pick / switch to", r"suggest|recommend|which fund|best fund|good fund",
          "Choice anxiety; tends to trigger switching (stop-then-restart)."),
    Theme("how_to_operational", "Asks how to do something in the app", r"how (do i|to|can i)|steps to|option to|unable to|not able to",
          "Product mechanics are unclear enough to need a forum."),
    Theme("explain_metrics", "Asks what a number means", r"xirr|\bnav\b|cagr|absolute return|explain|what does .{0,30} mean",
          "Numbers on screen don't explain themselves."),
]


@dataclass(frozen=True)
class Opportunity:
    code: str
    label: str
    reasons: tuple[str, ...]
    metric_branch: str
    severity: float
    addressability: float
    product_direction: str
    business_metric: str
    experiment: str


OPPORTUNITIES: list[Opportunity] = [
    Opportunity(
        "first_sip_success", "First SIP goes live (first instalment succeeds)",
        ("activation_failure",), "0 — Activation",
        0.9, 0.8,
        "Track every new SIP until its first instalment succeeds; name the blocker (KYC pending, mandate not "
        "registered, order rejected) with the one fix; let the first instalment go through by UPI while the "
        "mandate registers.",
        "% of first SIPs whose first instalment succeeds within 7 days of setup; days from setup to first success.",
        "Show a 'first instalment' tracker with the blocking reason and a one-tap fix to half of new first-time "
        "SIPs; compare first-instalment success within 7 days and active SIPs at 90 days.",
    ),
    Opportunity(
        "payment_reliability", "Payment & mandate reliability",
        ("involuntary_system",), "A — Involuntary lapse",
        0.9, 0.9,
        "Detect failed / revoked mandates instantly; one-tap retry or mandate repair; plain reason for the failure.",
        "Failed-instalment recovery within 7 days; share of SIP lapses preceded by a failed debit.",
        "Show the failure reason with a one-tap 'pay now / repair mandate' for half of failed instalments; compare "
        "recovery within 7 days and active SIPs after two cycles.",
    ),
    Opportunity(
        "control_clarity", "Pause / skip / resume controls that behave as expected",
        ("control_friction",), "B — Pause becomes stop",
        0.7, 0.9,
        "Clear pause vs stop vs skip; auto-resume date with reminder; status that matches the AMC.",
        "Share of paused SIPs that resume on time; support contacts about SIP status.",
        "Add an auto-resume date and reminder to the pause flow for half of users; compare the resume rate 90 days later.",
    ),
    Opportunity(
        "switch_without_gap", "Change fund or platform without a gap",
        ("fund_switch", "platform_move", "fund_suspension"), "B/C — Stop before restart",
        0.5, 0.7,
        "Treat 'stop + start another' as one flow (no fund recommendation): keep the instalment date, "
        "confirm the new SIP before cancelling the old one; guided next step when an AMC suspends.",
        "Share of cancellations followed by a new SIP within 30 days; days without an active SIP after a switch.",
        "Offer 'start the new SIP first' inside the cancel flow for half of users; compare the 30-day restart rate.",
    ),
    Opportunity(
        "decision_support", "Neutral support at the 'should I stop?' moment",
        ("decision_doubt", "market_timing", "poor_returns"), "C — Deliberate stop",
        0.8, 0.5,
        "At cancel: recap the user's own goal and horizon, show what stopping changes, offer pause/reduce. "
        "Informative only — no advice, no fear or urgency.",
        "Share of stop intents that become reduce / skip / continue; cancellations in the 2 weeks after a market fall.",
        "Show a neutral 'before you stop' sheet (own goal, horizon, alternatives) to half of users who tap Stop SIP; "
        "compare completed cancellations and 90-day active SIPs.",
    ),
    Opportunity(
        "understandability", "'Is my SIP working?' in plain language",
        ("confusion",), "D — Confusion drift",
        0.5, 0.8,
        "Explain XIRR / NAV / units in context; show progress against the user's own plan, not market noise.",
        "Returns-related support contacts; first-year stop rate.",
        "Show an in-context explainer when a holding's return first turns negative; compare the first-year stop rate.",
    ),
    Opportunity(
        "cash_flexibility", "Cash-flow flexibility",
        ("cash_or_goal",), "B — Pause becomes stop",
        0.7, 0.6,
        "Skip a month, lower the amount or move the date instead of cancelling.",
        "Share of cancellations that used skip / reduce first; 12-month active-SIP rate.",
        "Put skip, reduce and change-date at the top of the cancel flow for half of users; compare cancellations.",
    ),
]

SMALL_SAMPLE = 30
MIN_EVIDENCE = 3

FIRST_TIMER_SEGMENTS = ("first_mf_investor", "young_22_30", "first_job", "small_ticket")

LIFECYCLE_ORDER: dict[str, str] = {
    "considering_start": "Considering a first SIP",
    "just_started": "First instalments",
    "months_2_6": "Months 2–6",
    "months_7_12": "Months 7–12",
    "paused": "Paused",
    "payment_failed": "Payment failed",
    "stopped": "Stopped",
}


# ---------------------------------------------------------------- tagging

def _dedupe_title(text: str) -> str:
    """Posts are stored as 'Title Title body'; drop the repeated title."""
    for i in range(10, min(220, len(text) // 2)):
        if text[i] == " " and text[:i] == text[i + 1: 2 * i + 1]:
            return text[i + 1:]
    return text


def tag_themes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["post_clean"] = out["post_text"].map(_dedupe_title)
    blob = (out["post_text"] + " " + out["unmet_need"] + " " + out["quote"]).str.lower()
    for lens in (STOP_REASONS, START_MOTIVES, CHECKING, OUTSIDE_APP):
        for t in lens:
            out[f"t_{t.code}"] = blob.str.contains(t.pattern, regex=True)
    # Activation is a stage of payment failure: only payment posts whose sub-cause is in that group count.
    out["payment_subcause"] = payment_subcause(out).where(out["t_involuntary_system"], -1)
    groups = [g for g, *_ in PAYMENT_SUBCAUSES]
    out["t_activation_failure"] = out["payment_subcause"].map(lambda i: i >= 0 and groups[i] == ACTIVATION_GROUP)

    def primary(row: pd.Series) -> str:
        for t in STOP_REASONS:
            if row[f"t_{t.code}"]:
                return t.code
        return "unclear"

    out["primary_reason"] = out.apply(primary, axis=1)
    out["stop_class"] = out.apply(_stop_class, axis=1)
    out["is_first_timer"] = out["segment"].str.contains("|".join(FIRST_TIMER_SEGMENTS)) | out[
        "lifecycle"
    ].isin(["considering_start", "just_started"])
    out["is_at_risk"] = _at_risk_mask(out)
    out["is_continuer"] = out["behaviour"].eq("continued")
    out["_blob"] = blob
    return out


def _stop_class(row: pd.Series) -> str:
    if row["t_activation_failure"] or row["t_involuntary_system"] or row["t_fund_suspension"]:
        return "involuntary"
    if row["t_control_friction"]:
        return "friction"
    if row["t_fund_switch"] or row["t_platform_move"]:
        return "redirect"
    if row["t_cash_or_goal"] or row["t_poor_returns"]:
        return "exit"
    if row["t_decision_doubt"] or row["t_market_timing"] or row["t_confusion"]:
        return "undecided"
    return "unclear"


AT_RISK_STATUS = ("running_considering_stop", "paused_or_reduced", "stopped", "stopped_and_switched",
                  "failed_without_choice")


def _at_risk_mask(df: pd.DataFrame) -> pd.Series:
    """The problem-fit SIP status decides where it was coded; the older codebook rule is the fallback.

    Fund-choice posts (portfolio reviews, trimming or swapping funds) are never at risk: the metric counts any
    active SIP, so reshuffling funds doesn't threaten it."""
    legacy = (
        df["behaviour"].isin(
            ["stopped_sip", "paused_sip", "reduced_amount", "lump_sum_instead", "switched_platform"]
        )
        | df["lifecycle"].isin(["stopped", "paused", "payment_failed"])
        | df["link_to_lapse"].eq("high")
    )
    status = df["status"] if "status" in df else pd.Series("", index=df.index)
    concern = df["concern"] if "concern" in df else pd.Series("", index=df.index)
    has_status = ~status.isin(["", "unknown"])
    return ((has_status & status.isin(AT_RISK_STATUS)) | (~has_status & legacy)) & concern.ne("which_fund")


# ---------------------------------------------------------------- routing

@dataclass
class Intent:
    code: str
    title: str
    keywords: tuple[str, ...]
    lens: list[Theme]
    population: str
    base: str = "all posts"
    notes: list[str] = field(default_factory=list)


# Keywords are regexes matched at a word start; ties go to the earlier intent.
INTENTS: list[Intent] = [
    Intent("segment", "How patterns differ by segment",
           ("segment", "age", "income", "salaried", "young", "first[- ]time", "influenc", "student", "differ"),
           STOP_REASONS, "at-risk posts, split into first-time vs other investors"),
    Intent("contrast", "What separates continuers from drop-offs",
           ("continu", "separate", "stay", "retain", "keep going", "versus", "vs"),
           STOP_REASONS, "at-risk posts", base="posts from investors who continued"),
    Intent("stop", "Why SIPs are paused, stopped or lapse",
           ("stop", "pause", "cancel", "quit", "lapse", "drop", "discontinu", "exit", "halt", "skip", "leave"),
           STOP_REASONS, "posts about stopping, pausing, lapsing or a high link to lapse"),
    Intent("payment", "Payment and mandate failures",
           ("mandate", "autopay", "debit", "payment", "fail", "bounce", "declin", "reject", "salary account",
            "bank account", "new (salary |bank )?account", "account chang", r"chang\w* (\w+ )?(job|bank|account)",
            r"job chang", r"(new|switch\w*|chang\w*) bank"),
           STOP_REASONS, "posts about failed debits, mandates or system cancellations"),
    Intent("journey", "How behaviour changes after starting",
           ("months after", "over time", r"behaviou?r", "after starting", "first year", "journey", "lifecycle"),
           STOP_REASONS, "posts with a known stage after starting (first instalments → paused/stopped)"),
    Intent("check", "What investors check and how often",
           ("check", "monitor", "track", "how often", "look at", "statement", "dashboard"),
           CHECKING, "posts mentioning returns, statements, status or checking habits"),
    Intent("start", "Why people start and what they expect",
           (r"(why|how) (do|did) (people|investors|they|i|first[- ]time \w+) start", r"start(ing)? (a|an|their|my) ",
            "begin", "expect", "motivat", "new to", "why invest"),
           START_MOTIVES, "posts from people considering or just starting a SIP"),
    Intent("confusion", "What confuses or worries investors",
           ("confus", "worr", "understand", "unclear", "anxious", "scared", "doubt"),
           CHECKING, "posts tagged confused / anxious or mentioning hard-to-read metrics"),
    Intent("outside", "What investors do outside the app",
           ("outside", "youtube", "reddit", "friend", "family", "forum", "ask", "advice"),
           OUTSIDE_APP, "posts where the investor sought help outside the app"),
    Intent("needs", "Unmet needs",
           ("unmet", "need", "wish", "want", "missing", "gap", "opportunit"),
           STOP_REASONS, "at-risk posts with a stated unmet need"),
]


SCOPE_TERMS = re.compile(
    r"\b(sips?|mutual|funds?|mf|invest\w*|mandate|auto-?pay|auto-?debit|debit|instal\w*|nav|xirr|cagr|returns?"
    r"|market|portfolio|groww|app|bank|salary|money|saving\w*|wealth|equity|index|elss|kyc|redeem\w*|payment\w*"
    r"|stop\w*|paus\w*|cancel\w*|laps\w*|start\w*|continu\w*|drop|churn|retention|first[- ]time|beginner|panic|crash"
    r"|fall\w*|confus\w*|statement|platform|ui|features?|unmet)\b",
    re.I,
)


def is_in_scope(question: str) -> bool:
    """The corpus only covers SIP / mutual fund investing behaviour; anything else must not be answered from it."""
    return bool(SCOPE_TERMS.search(question))


def route_question(question: str) -> Intent:
    q = question.lower()
    best, best_score = None, 0
    for intent in INTENTS:
        score = sum(1 for k in intent.keywords if re.search(rf"\b{k}", q))
        if score > best_score:
            best, best_score = intent, score
    return best or Intent("general", "General discovery", (), STOP_REASONS,
                          "posts matching the question's keywords")


def _population(df: pd.DataFrame, intent: Intent, question: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df
    if intent.code in ("stop", "segment"):
        pop = df[df["is_at_risk"]]
    elif intent.code == "contrast":
        pop, base = df[df["is_at_risk"]], df[df["is_continuer"]]
    elif intent.code == "payment":
        pop = df[
            df["t_involuntary_system"]
            | df["trigger"].isin(["failed_debit", "mandate_autopay_issue"])
            | df["lifecycle"].eq("payment_failed")
        ]
    elif intent.code == "journey":
        pop = df[df["lifecycle"].isin(list(LIFECYCLE_ORDER)[1:])]
    elif intent.code == "start":
        pop = df[df["lifecycle"].isin(["considering_start", "just_started"])]
    elif intent.code == "check":
        pop = df[df[[f"t_{t.code}" for t in CHECKING]].any(axis=1) & ~df["lifecycle"].eq("considering_start")]
    elif intent.code == "confusion":
        pop = df[
            df["emotion"].isin(["confused", "anxious"])
            | df["trigger"].eq("confusion_metrics")
            | df["t_metric_confusion"]
        ]
    elif intent.code == "outside":
        pop = df[df["behaviour"].isin(["posted_reddit_forum", "asked_friend_family", "searched_youtube"])]
    elif intent.code == "needs":
        pop = df[df["is_at_risk"] & (df["unmet_need"].str.len() > 3)]
    else:
        terms = _terms(question)
        mask = df["_blob"].apply(lambda b: any(t in b for t in terms)) if terms else df["is_at_risk"]
        pop = df[mask] if mask.any() else df[df["is_at_risk"]]
    return pop, base


STOPWORDS = frozenset(
    "the and for are was with have this that from your what when why how who does do did people "
    "their they sip sips users user investors investor".split()
)


def _terms(question: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) > 2 and t not in STOPWORDS]


# ---------------------------------------------------------------- question focus

# Question wording → cause themes the question is about (matched at word start).
FOCUS_ALIASES: dict[str, tuple[str, ...]] = {
    "poor_returns": ("negative", "loss", r"red\b", "return", "underperform", "perform", "profit", "xirr is low"),
    "cash_or_goal": ("cash", "money", "afford", "salary", "income", r"job\b", "emergency", "loan", "expense", "goal", "financ"),
    "involuntary_system": ("payment", "mandate", "autopay", "auto-pay", "debit", "bank", "fail", "bounce", "declin",
                           "reject", "salary account", "account chang"),
    "activation_failure": ("kyc", "first instal", "first sip", "set up", "setup", "activat", "registration",
                           "go live", "went live"),
    "control_friction": ("resume", "un-pause", "unpause", "skip", "control", "option", "status"),
    "fund_switch": ("switch", "rebalanc", "direct", "regular", "consolidat", "reallocat", "lump"),
    "platform_move": ("platform", "move", "migrat", "app to", "amc website"),
    "market_timing": ("market", "crash", "timing", "volatil", "fall", r"dip\b", "correction", "panic", "scare", "fear"),
    "decision_doubt": ("unsure", "doubt", "should i", "decide", "decision", "hesita", "uncertain"),
    "confusion": ("confus", "understand", r"nav\b", "xirr", "statement", "unclear", "explain"),
    "fund_suspension": (r"amc\b", "overseas", "suspend", "fund house", "international"),
}

# Question wording → behaviour the question is about (narrows the group of posts).
ACTION_FOCUS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "pause": (("pause", "paus", "skip"), ("paused_sip",), r"\bpaus|\bskip"),
    "reduce": (("reduc", "lower", "decreas", r"cut\b"), ("reduced_amount",), r"reduc|lower (the )?amount|decreas"),
    "cancel": (("cancel", "stop", "quit", "discontinu", "exit", "lapse", "drop"), ("stopped_sip",), r"\bstop|cancel|discontinu"),
}

HYPOTHESIS_START = re.compile(r"^\s*(do|does|did|is|are|was|were|can|could|will|would|has|have)\b", re.I)

# A job or bank change breaks the debit mandate; it is a payment question, not a cash-need question.
ACCOUNT_CHANGE = re.compile(
    r"salary account|bank account|new (salary |bank )?account|account chang|job chang|chang\w* (\w+ )?(job|bank|account)"
    r"|(new|switch\w*|chang\w*) bank|new job",
    re.I,
)
CASH_NEED = re.compile(r"afford|emergency|loan|lost (my |a |their )?job|job loss|laid off|expense|cash", re.I)


def detect_focus(question: str) -> tuple[list[str], str | None, bool]:
    q = question.lower()
    themes = [
        code for code, aliases in FOCUS_ALIASES.items()
        if any(re.search(rf"\b{a}", q) for a in aliases)
    ]
    if ACCOUNT_CHANGE.search(q):
        themes = ["involuntary_system"] + [
            t for t in themes
            if t != "involuntary_system"
            and (t != "cash_or_goal" or CASH_NEED.search(q))
            and (t != "fund_switch" or re.search(r"switch\w* (to|from|fund|sip|plan)", q))
        ]
    if "activation_failure" in themes and not re.search(r"mandate|auto-?pay|debit|bounce|bank|salary", q):
        themes = [t for t in themes if t != "involuntary_system"]
    named = [
        code for code, (aliases, _, _) in ACTION_FOCUS.items()
        if any(re.search(rf"\b{a}", q) for a in aliases)
    ]
    # "pause, reduce or stop" asks about stopping in general, not one specific action.
    action = named[0] if len(named) == 1 else None
    return themes, action, bool(HYPOTHESIS_START.match(question))


def _action_mask(df: pd.DataFrame, action: str) -> pd.Series:
    _, behaviours, text_pat = ACTION_FOCUS[action]
    lifecycle = {"pause": ["paused"], "cancel": ["stopped"], "reduce": []}[action]
    return (
        df["behaviour"].isin(behaviours)
        | df["lifecycle"].isin(lifecycle)
        | df["post_clean"].str.lower().str.contains(text_pat, regex=True)
    )


def focus_analysis(df: pd.DataFrame, pop: pd.DataFrame, code: str) -> dict[str, Any]:
    """Test one theme against outcomes: does mentioning it go with stopping?"""
    labels = {t.code: t.label for t in STOP_REASONS}
    col = f"t_{code}"
    with_t, without_t = df[df[col]], df[~df[col]]
    risk_with = _pct(int(with_t["is_at_risk"].sum()), len(with_t))
    risk_without = _pct(int(without_t["is_at_risk"].sum()), len(without_t))
    in_pop = pop[pop[col]]
    risk = df[df["is_at_risk"]]
    cont = df[df["is_continuer"]]
    rank_tbl = risk[[f"t_{t.code}" for t in STOP_REASONS]].sum().sort_values(ascending=False)
    rank = list(rank_tbl.index).index(col) + 1

    co = []
    for t in STOP_REASONS:
        if t.code == code or in_pop.empty:
            continue
        n = int(in_pop[f"t_{t.code}"].sum())
        if n:
            co.append((t.label, n, _pct(n, len(in_pop))))
    co.sort(key=lambda x: -x[1])

    behaviour = (
        in_pop["behaviour"].str.replace("_", " ").value_counts().head(4).to_dict() if not in_pop.empty else {}
    )
    stop_kind = in_pop["stop_class"].value_counts().to_dict() if not in_pop.empty else {}
    needs = [n for n in in_pop["unmet_need"].tolist() if len(n) > 3][:4]

    rr = round(risk_with / risk_without, 1) if risk_without else None
    share_risk = _pct(int(risk[col].sum()), len(risk))
    share_cont = _pct(int(cont[col].sum()), len(cont))
    if rr is not None and rr >= 1.3 and share_risk > share_cont:
        verdict = "Supported"
    elif (rr is None or rr <= 0.9) and share_cont >= share_risk:
        verdict = "Not supported"
    else:
        verdict = "Partly supported"
    return {
        "code": code,
        "label": labels.get(code, code),
        "mentions": len(with_t),
        "in_group": len(in_pop),
        "risk_rate_with": risk_with,
        "risk_rate_without": risk_without,
        "risk_ratio": rr,
        "share_of_at_risk": share_risk,
        "share_of_continuers": share_cont,
        "rank_among_causes": rank,
        "n_causes": len(STOP_REASONS),
        "co_occurring": co[:3],
        "behaviour": behaviour,
        "stop_kind": stop_kind,
        "needs": needs,
        "verdict": verdict,
    }


SYNONYMS: dict[str, tuple[str, ...]] = {
    "stop": ("stop", "cancel", "discontinu", "quit", "exit"),
    "pause": ("pause", "skip"),
    "negative": ("negative", "loss", "red", "down"),
    "returns": ("return", "xirr", "cagr", "profit"),
    "payment": ("payment", "debit", "mandate", "autopay"),
    "confus": ("confus", "understand", "unclear"),
    "cash": ("cash", "money", "afford", "salary"),
}


def relevant_posts(pop: pd.DataFrame, question: str, focus: list[str], k: int = 8) -> pd.DataFrame:
    """Rank posts by how directly they speak to this question (terms + synonyms + focus themes)."""
    terms = set(_terms(question))
    for t in list(terms):
        for key, syns in SYNONYMS.items():
            if t.startswith(key) or key.startswith(t):
                terms.update(syns)
    if pop.empty:
        return pop
    scored = pop.copy()
    scored["_rel"] = scored["_blob"].apply(lambda b: sum(min(b.count(t), 3) for t in terms))
    for code in focus:
        scored["_rel"] += scored[f"t_{code}"].astype(int) * 4
    scored["_rel"] += scored["link_to_lapse"].map({"high": 1, "medium": 0.5}).fillna(0)
    return scored.sort_values("_rel", ascending=False).head(k)


# ---------------------------------------------------------------- analysis

def _pct(n: int | float, d: int | float) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def _lift(a: float, b: float) -> float | None:
    if b <= 0:
        return None
    return round(a / b, 1)


def theme_profile(
    pop: pd.DataFrame, base: pd.DataFrame, continuers: pd.DataFrame, lens: list[Theme]
) -> pd.DataFrame:
    rows = []
    for t in lens:
        col = f"t_{t.code}"
        n = int(pop[col].sum())
        p_pop = _pct(n, len(pop))
        p_base = _pct(int(base[col].sum()), len(base))
        p_cont = _pct(int(continuers[col].sum()), len(continuers))
        rows.append(
            {
                "code": t.code,
                "Theme": t.label,
                "Posts": n,
                "% of group": p_pop,
                "% comparison": p_base,
                "Lift": _lift(p_pop, p_base),
                "% among continuers": p_cont,
                "Mechanism": t.mechanism,
            }
        )
    return pd.DataFrame(rows).sort_values("Posts", ascending=False).reset_index(drop=True)


def stop_class_split(pop: pd.DataFrame) -> pd.DataFrame:
    vc = pop["stop_class"].value_counts()
    rows = []
    for code, (label, impact) in STOP_CLASSES.items():
        n = int(vc.get(code, 0))
        rows.append({"code": code, "Kind of stop": label, "Posts": n,
                     "% of group": _pct(n, len(pop)), "What it means for the metric": impact})
    return pd.DataFrame(rows)


def pathways(pop: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    labels = {t.code: t.label for t in STOP_REASONS}
    labels["unclear"] = "Reason not stated"
    g = (
        pop.groupby(["primary_reason", "behaviour", "link_to_lapse"])
        .size()
        .reset_index(name="Posts")
        .sort_values("Posts", ascending=False)
    )
    g = g[g["primary_reason"] != "unclear"].head(top)
    g["Cause"] = g["primary_reason"].map(labels)
    g["What they did"] = g["behaviour"].str.replace("_", " ")
    g["Link to lapse"] = g["link_to_lapse"]
    return g[["Cause", "What they did", "Link to lapse", "Posts"]]


def segment_split(pop: pd.DataFrame, lens: list[Theme]) -> pd.DataFrame:
    ft, other = pop[pop["is_first_timer"]], pop[~pop["is_first_timer"]]
    rows = []
    for t in lens:
        col = f"t_{t.code}"
        rows.append({
            "Theme": t.label,
            f"First-time investors (n={len(ft)}) %": _pct(int(ft[col].sum()), len(ft)),
            f"Others (n={len(other)}) %": _pct(int(other[col].sum()), len(other)),
        })
    return pd.DataFrame(rows)


def lifecycle_table(pop: pd.DataFrame) -> pd.DataFrame:
    """Per journey stage: size, dominant cause and kind of stop."""
    labels = {t.code: t.label for t in STOP_REASONS}
    rows = []
    for code, label in LIFECYCLE_ORDER.items():
        stage = pop[pop["lifecycle"] == code]
        if stage.empty:
            continue
        causes = stage[stage["primary_reason"] != "unclear"]["primary_reason"].value_counts()
        rows.append({
            "Stage": label,
            "Posts": len(stage),
            "Top cause": labels.get(causes.index[0], "—") if not causes.empty else "—",
            "Top cause %": _pct(int(causes.iloc[0]), len(stage)) if not causes.empty else 0.0,
            "High link to lapse %": _pct(int(stage["link_to_lapse"].eq("high").sum()), len(stage)),
        })
    return pd.DataFrame(rows)


def need_themes(pop: pd.DataFrame, top_examples: int = 2) -> pd.DataFrame:
    needs = pop[pop["unmet_need"].str.len() > 3]
    rows = []
    for t in STOP_REASONS:
        hit = needs[needs["unmet_need"].str.lower().str.contains(t.pattern, regex=True)]
        if hit.empty:
            continue
        rows.append({
            "Need theme": t.label,
            "Posts": len(hit),
            "% of stated needs": _pct(len(hit), len(needs)),
            "Examples": " · ".join(hit["unmet_need"].head(top_examples).tolist()),
        })
    if not rows:
        return pd.DataFrame(columns=["Need theme", "Posts", "% of stated needs", "Examples"])
    return pd.DataFrame(rows).sort_values("Posts", ascending=False)


def opportunity_table(
    df: pd.DataFrame,
    lapse_rate: float = 0.40,
    recovery_rate: float = 0.30,
    overrides: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Rank opportunity areas on the at-risk corpus.

    share        = at-risk posts whose primary reason belongs to the area
    lapse_share  = share (posts about stopping are the proxy for lapses; posts with no sized cause stay unsized)
    leverage     = lapse_share × addressability
    est_pp       = lapse_rate × lapse_share × addressability × recovery_rate (percentage points)
    Severity is shown as a judgement but does not weight the sizing.
    """
    overrides = overrides or {}
    risk = df[df["is_at_risk"]]
    n_risk = len(risk)
    rows = []
    for o in OPPORTUNITIES:
        sev = overrides.get(o.code, {}).get("severity", o.severity)
        addr = overrides.get(o.code, {}).get("addressability", o.addressability)
        ev = risk[risk["primary_reason"].isin(o.reasons)]
        any_hit = df[df[[f"t_{r}" for r in o.reasons]].any(axis=1)]
        p_risk_any = _pct(int(risk[[f"t_{r}" for r in o.reasons]].any(axis=1).sum()), n_risk)
        p_all_any = _pct(len(any_hit), len(df))
        cont = df[df["is_continuer"]]
        p_cont_any = _pct(int(cont[[f"t_{r}" for r in o.reasons]].any(axis=1).sum()), len(cont))
        n_sources = ev["source"].nunique()
        n = len(ev)
        confidence = "High" if n >= 25 and n_sources >= 2 else "Medium" if n >= 10 else "Low"
        rows.append({
            "code": o.code,
            "Opportunity": o.label,
            "Metric branch": o.metric_branch,
            "Evidence posts": n,
            "share": n / n_risk if n_risk else 0.0,
            "% of at-risk posts": _pct(n, n_risk),
            "Mentioned in at-risk %": p_risk_any,
            "Lift vs all posts": _lift(p_risk_any, p_all_any),
            "% among continuers": p_cont_any,
            "First-timer share %": _pct(int(ev["is_first_timer"].sum()), n),
            "Severity": sev,
            "Addressability": addr,
            "Confidence": confidence,
            "Product direction": o.product_direction,
            "Business metric": o.business_metric,
            "Experiment": o.experiment,
            "Sources": ev["source"].value_counts().to_dict(),
            "evidence_ids": ev.sort_values("opportunity_score", ascending=False)["item_id"].head(6).tolist(),
        })
    out = pd.DataFrame(rows)
    out["Lapse share %"] = (out["share"] * 100).round(1)
    out["Leverage index"] = (out["Lapse share %"] * out["Addressability"]).round(1)
    out["Est. uplift (pp)"] = (
        lapse_rate * 100 * (out["Lapse share %"] / 100) * out["Addressability"] * recovery_rate
    ).round(2)
    return out.sort_values("Leverage index", ascending=False).reset_index(drop=True)


def _snippet(text: str, n: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n] + ("…" if len(text) > n else "")


def _evidence_for(
    pop: pd.DataFrame, code: str, question: str, used: set[str], k: int = 3
) -> list[dict[str, str]]:
    col = f"t_{code}"
    if col not in pop:
        return []
    hits = pop[pop[col] & ~pop["item_id"].isin(used)].copy()
    if hits.empty:
        return []
    terms = _terms(question)
    hits["_q"] = hits["_blob"].apply(lambda b: sum(b.count(t) for t in terms))
    hits["_l"] = hits["link_to_lapse"].map({"high": 3, "medium": 2, "low": 1}).fillna(0)
    hits = hits.sort_values(["_q", "_l", "opportunity_score"], ascending=False).head(k)
    used.update(hits["item_id"])
    return [
        {"id": r["item_id"], "source": r["source"], "text": _snippet(r["post_clean"]), "url": r["url"]}
        for _, r in hits.iterrows()
    ]


def build_reasoned_analysis(
    df: pd.DataFrame,
    question: str,
    lapse_rate: float = 0.40,
    recovery_rate: float = 0.30,
    overrides: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    intent = route_question(question)
    pop, base = _population(df, intent, question)
    focus_codes, action, is_hypothesis = detect_focus(question)
    action_n, action_count = None, 0
    if action and intent.lens is STOP_REASONS:
        narrowed = pop[_action_mask(pop, action)]
        action_count = len(narrowed)
        if action_count >= 10:
            pop, action_n = narrowed, action_count
    continuers = df[df["is_continuer"]]
    lens = intent.lens

    profile = theme_profile(pop, base, continuers, lens)
    stop_profile = theme_profile(pop, df, continuers, STOP_REASONS)
    opps = opportunity_table(df, lapse_rate, recovery_rate, overrides)

    reason_lift = dict(zip(stop_profile["code"], stop_profile["Lift"].fillna(0)))
    reason_pct = dict(zip(stop_profile["code"], stop_profile["% of group"]))
    opps["Relevant to question"] = opps["code"].map(
        lambda c: any(
            reason_pct.get(r, 0) >= 10 or (reason_lift.get(r, 0) or 0) >= 1.5
            for r in next(o.reasons for o in OPPORTUNITIES if o.code == c)
        )
    )

    focus = [focus_analysis(df, pop, c) for c in focus_codes]
    if focus_codes:
        focus_opps = {o.code for o in OPPORTUNITIES if set(o.reasons) & set(focus_codes)}
        opps["Relevant to question"] = opps["code"].isin(focus_opps)

    lens_codes = [t.code for t in lens]
    top_codes = [c for c in focus_codes if c in lens_codes]
    top_codes += [c for c in profile[profile["Posts"] > 0]["code"] if c not in top_codes]
    top_codes = top_codes[:4]
    used: set[str] = set()
    evidence = {c: _evidence_for(pop, c, question, used) for c in top_codes}
    rel = relevant_posts(pop, question, focus_codes)
    relevant = [
        {"id": r["item_id"], "source": r["source"], "text": _snippet(r["post_clean"], 240), "url": r["url"],
         "cause": r["primary_reason"], "did": r["behaviour"].replace("_", " ")}
        for _, r in rel.iterrows()
    ]

    source_mix = pop["source"].value_counts(normalize=True).mul(100).round(1).to_dict()
    return {
        "question": question,
        "intent": intent,
        "pop": pop,
        "pop_n": len(pop),
        "base_n": len(base),
        "corpus_n": len(df),
        "continuers_n": len(continuers),
        "profile": profile,
        "stop_profile": stop_profile,
        "stop_classes": stop_class_split(pop),
        "pathways": pathways(pop),
        "segments": segment_split(pop, lens),
        "lifecycle": lifecycle_table(pop),
        "first_timer_n": int(pop["is_first_timer"].sum()),
        "needs": need_themes(pop),
        "opportunities": opps,
        "evidence": evidence,
        "relevant_posts": relevant,
        "focus": focus,
        "action": action,
        "action_n": action_n,
        "action_count": action_count,
        "is_hypothesis": is_hypothesis,
        "source_mix": source_mix,
        "explained_pct": _pct(int((pop["primary_reason"] != "unclear").sum()), len(pop)),
        "assumptions": {"lapse_rate": lapse_rate, "recovery_rate": recovery_rate},
    }


# ---------------------------------------------------------------- LLM evidence pack

def _focus_lines(a: dict[str, Any]) -> list[str]:
    lines = []
    for f in a["focus"]:
        lines.append(
            f"- {f['label']}: mentioned in {f['mentions']} posts; {f['risk_rate_with']}% of those are at-risk vs "
            f"{f['risk_rate_without']}% of posts without it (risk ratio {f['risk_ratio']}x). "
            f"Share of at-risk posts {f['share_of_at_risk']}% vs share of continuers {f['share_of_continuers']}%. "
            f"Rank {f['rank_among_causes']} of {f['n_causes']} causes. Verdict by rule: {f['verdict']}."
        )
        if f["co_occurring"]:
            lines.append("  co-occurs with: " + "; ".join(f"{l} {p}%" for l, _, p in f["co_occurring"]))
        if f["behaviour"]:
            lines.append("  what they did: " + "; ".join(f"{k} {v}" for k, v in f["behaviour"].items()))
        if f["needs"]:
            lines.append("  stated needs: " + " | ".join(n[:90] for n in f["needs"][:3]))
    return lines


def format_compact_pack(a: dict[str, Any], max_themes: int = 5, max_opps: int = 3) -> str:
    """~3–5k chars. Reasoning models think in proportion to input, so send only what the answer needs."""
    intent: Intent = a["intent"]
    lines = [
        f"QUESTION: {a['question']}",
        f"QUESTION TYPE: {'yes/no hypothesis — give a verdict' if a['is_hypothesis'] else 'explain why/what'}"
        f" | topic: {intent.title}"
        + (f" | action asked about: {a['action']} ({a['action_n']} posts)" if a["action_n"] else ""),
        "METRIC: % first-time MF investors with an active SIP at month 12.",
        f"GROUP: {a['pop_n']} {intent.population}, of {a['corpus_n']} coded public posts. "
        f"Comparison base: {a['base_n']} {intent.base}. Continuer posts in corpus: {a['continuers_n']}"
        + (" (small — treat continuer contrasts as weak)" if a["continuers_n"] < 30 else "")
        + f". Reddit {a['source_mix'].get('Reddit', 0)}%. First-time tagged in group: {a['first_timer_n']}.",
    ]
    if a["focus"]:
        lines.append("FOCUS THEMES NAMED IN THE QUESTION:")
        lines += _focus_lines(a)
    lines.append("POSTS MOST RELEVANT TO THIS QUESTION (primary evidence; cite as [#id]):")
    for e in a["relevant_posts"]:
        lines.append(f"[#{e['id']}] cause={e['cause']}, did={e['did']}: {e['text'][:220]}")
    lines.append("THEMES IN GROUP (n | % group | lift vs base | % among continuers):")
    for _, r in a["profile"][a["profile"]["Posts"] > 0].head(max_themes).iterrows():
        lines.append(f"- {r['Theme']}: {r['Posts']} | {r['% of group']}% | {r['Lift']}x | {r['% among continuers']}%")
    if intent.lens is STOP_REASONS:
        lines.append("KIND OF STOP (% group): " + "; ".join(
            f"{r['Kind of stop']} {r['% of group']}%" for _, r in a["stop_classes"].iterrows() if r["Posts"]
        ))
    opps = a["opportunities"]
    opps = pd.concat([opps[opps["Relevant to question"]], opps[~opps["Relevant to question"]]]).head(max_opps)
    lines.append("OPPORTUNITIES LINKED TO THIS QUESTION (leverage | est. pp uplift | % at-risk | confidence | direction):")
    for _, o in opps.iterrows():
        lines.append(
            f"- {o['Opportunity']}: {o['Leverage index']} | +{o['Est. uplift (pp)']}pp | "
            f"{o['% of at-risk posts']}% | {o['Confidence']} | {o['Product direction'][:120]}"
        )
    return "\n".join(lines)


LLM_REFUSAL_PREFIX = "I can't answer this:"

LLM_INSTRUCTIONS = f"""You are the reasoning layer of a product discovery engine for Groww's growth team. You explain
how investors behave, using an evidence pack built from public posts. You are not an investment adviser or a support agent.

Step 1 - check the question. If it asks for any of the following, reply with exactly one line,
"{LLM_REFUSAL_PREFIX} <short reason>. This engine only analyses how investors behave in public posts.", and stop:
- personal investment advice: which fund, stock, bank or app to choose, or whether someone should start, stop,
  continue, pause or redeem;
- account or app support: OTP, login, KYC, refunds, a failed payment on the asker's own account;
- fear, FOMO, urgency or other tactics to pressure investors;
- anything unrelated to SIP / mutual fund investor behaviour.

Step 2 - otherwise answer THIS exact question from the pack. Do not summarise reviews. Use exactly this
markdown and no other headings:

**Answer**
Two sentences. For a yes/no question start with Yes / No / Partly and the deciding numbers.

**Why**
- Three bullets, one line each (max 30 words): claim, number, and one [#id] with 3-6 quoted words.
  One bullet must be counter-evidence or a reason the obvious explanation is incomplete.

**What it means for the 12-month metric**
One line: the most relevant opportunity, its est. pp uplift, and why.

**Validate next**
- One interview question that could overturn the answer.
- The main data limitation.

Rules: only numbers from the pack; never recommend funds, stocks, banks or apps; no fear, FOMO or urgency
framing; say 'public posts' when quantifying, and never claim how common something is among all investors.
If the question asks whether a product change would work, the Answer must first say public posts cannot test
that, then describe the need behind it. If the pack's group has fewer than 30 posts, say it is a small sample.
Max 170 words. Lead with what is specific to this question."""
