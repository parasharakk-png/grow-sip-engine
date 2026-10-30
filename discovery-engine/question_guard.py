"""Decide whether a question is a research question the engine may answer from public posts.

The engine is a product-discovery tool: it explains how investors behave. It must not give personal
investment advice (which fund, whether to stop), act as account support (OTP, login, KYC), or help
design fear / FOMO / urgency tactics. Anything else off-topic is declined too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reasoning import is_in_scope

GUARD_TESTS_CSV = Path(__file__).resolve().parent / "data" / "question_guard_tests.csv"

RESEARCH, ADVICE, SUPPORT, UNSAFE, OFF_TOPIC, PREVALENCE = (
    "research", "advice", "support", "unsafe", "off_topic", "prevalence"
)

POPULATION = r"(all |the |most |groww('s)? )?(first[- ]time |new |young |indian |retail |groww )*(investors?|users?|people|customers|sip (holders|investors))"
PREVALENCE_RE = re.compile(
    rf"\b(what|which)( is the)? (percentage|percent|%|share|proportion|fraction|number|rate) of {POPULATION}\b"
    rf"|\bhow many {POPULATION}\b(?!.{{0,40}}\b(mention|say|post|talk|write|report|complain|describe)\w*)"
    rf"|\bhow (common|widespread|prevalent|likely) is\b.{{0,60}}\bamong {POPULATION}"
    r"|\b(retention|churn|lapse|drop[- ]?off|stop|cancellation|continuation) rate\b"
    r"|\b(what|how much) (%|percent)",
    re.I,
)
INTERVENTION_RE = re.compile(
    r"\b(would|will|could|does|do|can|might)\b.{0,80}\b(help\w*|work|reduce\w*|increase\w*|improve\w*|prevent\w*"
    r"|stop (them|people|users|investors)|keep (them|people|users|investors)|retain|change (their )?behaviou?r)\b"
    r".{0,40}$"
    r"|\bwhat if (we|groww)\b|\b(impact|effect) of (showing|adding|a |an |the )"
    r"|\bshould groww (show|add|build|offer|send|launch|introduce)\b",
    re.I,
)
PRODUCT_CHANGE = re.compile(
    r"\b(show\w*|display\w*|add\w*|nudge\w*|bottom ?sheet|notification\w*|reminder\w*|message\w*|banner|feature"
    r"|prompt\w*|sheet|flow|screen|button|retry|social proof|alert\w*|explainer)\b",
    re.I,
)

MANIPULATION = re.compile(
    r"\b(how to|how (can|do|could) we|ways to|help me|let'?s)\b.{0,40}\b(scare\w*|frighten\w*|guilt\w*|pressur\w*"
    r"|trick\w*|manipulat\w*|fomo|urgency|panic|fear|afraid)"
    r"|\b(create|use|add|induce|exploit|leverage|trigger)\w*\b.{0,20}\b(urgency|fomo|fear|panic|scarcity|guilt)"
    r"|\bmake (them|users|investors|people|customers) (panic|afraid|scared|fear)|\bdark patterns?\b",
    re.I,
)
RESEARCH_SUBJECT = re.compile(
    r"\b(investors?|people|users?|customers?|first[- ]time\w*|beginners?|newbies?|they|their|drop[- ]?offs?"
    r"|continuers?|stayers?|segments?|cohorts?|posts?|reviews?|respondents?|young|groww users)\b",
    re.I,
)
RESEARCH_FRAME = re.compile(
    r"\b(affect\w*|impact\w*|lead(s)? to|drive[sn]?|cause[sd]?|correlat\w*|relat\w* to|pattern\w*|trend\w*"
    r"|how often|how common|what (leads|makes|drives|causes|separates)|why do|why does|continuation|retention"
    r"|drop[- ]?off|lapse\w*|make\w* (\w+ ){0,2}(investors?|people|users?|them|customers|first[- ]time\w*))\b",
    re.I,
)
FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|my|me|mine)\b", re.I)
INVEST_ACTION = (r"(stop|pause|continue|start|invest|buy|sell|redeem|switch|open|choose|pick|increase|reduce"
                 r"|cancel|withdraw|exit|put|keep|hold)")
GENERIC_ADVICE = re.compile(
    rf"\b(should|must|ought to)\b (investors?|people|users?|one|we|beginners?|first[- ]time\w*( investors?)?|young\w*)\b"
    rf".{{0,40}}\b{INVEST_ACTION}\w*",
    re.I,
)
ADVICE_RE = re.compile(
    rf"\b(should|shall|can|could|must) i\b.{{0,60}}\b{INVEST_ACTION}\w*"
    rf"|\b(should|shall) i\b.{{0,40}}\b(sip|fund|stock|bank|account|app|money)"
    r"|\bwhich (mutual )?(fund|stock|share|scheme|bank|account|app|platform|amc|sip|etf|index)s?\b"
    r"|\b(best|top|good|safest|better) (mutual )?(fund|stock|share|bank|app|sip|scheme|platform|etf)s?\b"
    r"|\b(recommend|suggest)\w*\b.{0,30}\b(fund|stock|sip|bank|scheme|portfolio|app)"
    r"|\bwhat (mutual )?(fund|stock)s? (to|should|can)\b"
    r"|\bis (it|now|this) (a )?(good|right|safe|bad) time\b"
    r"|\bhow much (should|to|can) (i )?invest\b"
    r"|\b(is|are)\b.{0,50}\b(funds?|(flexi|multi|mid|small|large) ?caps?|index|etfs?|elss|nifty\w*|stocks?|shares?)\b"
    r".{0,20}\b(good|bad|safe|worth\w*|better|right|risky)\b"
    r"|\bfor me\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\botp\b|\blog ?in\b|\bsign ?in\b|\bpassword\b|\bpin\b|\bkyc\b|\brefund\b|customer (care|support)|helpline"
    r"|\bcontact (groww|support)\b|\bnot (coming|working|loading|received|receiving|showing|reflecting|opening)\b"
    r"|\bisn'?t (coming|working|loading)\b|\bunable to\b|\bcan'?t (log|sign|pay|see|find|withdraw|cancel|link|add|open)"
    r"|\b(app|website) (is )?(crash\w*|stuck|slow|down)\b|\berror\b"
    r"|\bhow (do|can) i (cancel|pause|change|update|link|add|withdraw|redeem|delete|contact|reset|set ?up|modify)"
    r"|\bmy\b.{0,40}\b(failed|rejected|deducted|debited|stuck|pending|missing|blocked)",
    re.I,
)

REFRAMES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"otp|log ?in|sign ?in|password|kyc|app|error|crash|not (coming|working|loading)", re.I),
     "How do app, login or OTP problems affect whether first-time investors keep their SIP?"),
    (re.compile(r"bank|account|mandate|autopay|auto-?debit|payment|debit|deduct|instal", re.I),
     "How do payment and mandate problems affect whether investors keep their SIP?"),
    (re.compile(r"\b(fall|crash|red|loss|negative|down|market)\b", re.I),
     "Do market falls make first-time investors stop their SIPs?"),
    (re.compile(r"stop|pause|cancel|continue|redeem|withdraw|exit|sell", re.I),
     "What leads investors to pause, reduce or stop a SIP?"),
    (re.compile(r"fund|stock|scheme|buy|invest|start|amount|how much", re.I),
     "Why do people start their first SIP and what do they expect?"),
]


@dataclass(frozen=True)
class GuardResult:
    kind: str
    reason: str
    reframe: str | None = None

    @property
    def allowed(self) -> bool:
        return self.kind == RESEARCH


MESSAGES = {
    ADVICE: ("This asks for personal investment advice, which the engine does not give.",
             "It analyses what investors say in public posts to find product problems. It cannot tell anyone "
             "which fund, bank or app to choose, or whether to stop, start or change a SIP. For a personal "
             "decision, speak to a SEBI-registered investment adviser."),
    SUPPORT: ("This is an account or app support request, not a research question.",
              "The engine cannot see anyone's account. For OTP, login, KYC or payment problems, use Groww's "
              "official Help & Support in the app or at groww.in/help."),
    UNSAFE: ("This asks for fear, FOMO or urgency tactics, which the project rules out.",
             "The engine only supports neutral, informative product ideas. Ask about what investors need instead."),
    OFF_TOPIC: ("This question is outside the scope of the engine.",
                "It only answers questions about SIP and mutual fund investor behaviour: why people start, check, "
                "pause or stop SIPs, what confuses them, and how first-time investors differ."),
    PREVALENCE: ("Public posts can't measure how common this is among investors.",
                 "People who post are a self-selected group that over-represents problems, so any percentage here "
                 "would describe posts, not investors. Measure this with Groww's own data (for example, a cohort "
                 "retention report of first-time SIPs). The engine can tell you what share of posts mention a "
                 "behaviour and why it happens."),
}


def is_intervention_question(question: str) -> bool:
    """Asks whether a product change would work: public posts can describe the need, not test the fix."""
    return bool(INTERVENTION_RE.search(question) and PRODUCT_CHANGE.search(question))


def _reframe(question: str) -> str | None:
    return next((q for pat, q in REFRAMES if pat.search(question)), None)


def classify_question(question: str) -> GuardResult:
    q = question.strip()
    if not q:
        return GuardResult(OFF_TOPIC, "empty")
    if MANIPULATION.search(q):
        return GuardResult(UNSAFE, "asks for fear / FOMO / urgency tactics",
                           "What do first-time investors need at the moment they consider stopping?")
    if PREVALENCE_RE.search(q) and is_in_scope(q):
        return GuardResult(PREVALENCE, "asks how common something is among all investors",
                           "What leads investors to pause, reduce or stop a SIP?")
    if GENERIC_ADVICE.search(q):
        return GuardResult(ADVICE, "asks what investors should do, which is advice", _reframe(q))
    support, advice = bool(SUPPORT_RE.search(q)), bool(ADVICE_RE.search(q))
    first_person = bool(FIRST_PERSON.search(q))
    frame = bool(RESEARCH_FRAME.search(q))
    if not first_person and (RESEARCH_SUBJECT.search(q) or frame) and (is_in_scope(q) or support) and (
        frame or not support
    ):
        return GuardResult(RESEARCH, "about investors' behaviour")
    if advice:
        return GuardResult(ADVICE, "asks for a personal investment decision", _reframe(q))
    if support:
        return GuardResult(SUPPORT, "asks for account or app help", _reframe(q))
    if not is_in_scope(q):
        return GuardResult(OFF_TOPIC, "no SIP / mutual fund topic")
    if first_person:
        return GuardResult(ADVICE, "a personal question about the asker's own investing", _reframe(q))
    return GuardResult(RESEARCH, "a general question about SIP behaviour")


def run_guard_tests(path: Path = GUARD_TESTS_CSV) -> pd.DataFrame:
    tests = pd.read_csv(path, dtype=str).fillna("")
    tests["got"] = tests["question"].map(lambda q: classify_question(q).kind)
    tests["pass"] = tests["got"].eq(tests["expected"])
    return tests


if __name__ == "__main__":
    res = run_guard_tests()
    print(f"{int(res['pass'].sum())}/{len(res)} passed")
    print(res[~res["pass"]].to_string(index=False))
