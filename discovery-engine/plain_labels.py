"""Human-readable labels for metrics shown in the Streamlit app."""

from __future__ import annotations

TRIGGER_PLAIN: dict[str, str] = {
    "failed_debit": "Payment failed / not deducted",
    "mandate_autopay_issue": "Autopay or mandate issue",
    "cash_crunch": "Short on cash / salary timing",
    "red_returns": "Worried about negative returns",
    "expectations_not_met": "Returns not as expected",
    "confusion_metrics": "Confused (NAV, XIRR, statements)",
    "market_volatility": "Market fall / volatility",
    "life_event": "Life event (job, emergency)",
    "platform_bug": "App or platform problem",
    "fees_charges": "Fees or charges",
    "none_obvious": "No clear single reason",
    "other": "Other",
    "unknown": "Unknown",
}

BRANCH_PLAIN: dict[str, str] = {
    "A_involuntary_lapse": "Payment failed (user did not mean to stop)",
    "B_pause_becomes_stop": "Paused SIP — may never resume",
    "C_deliberate_stop_expectations": "Chose to stop / unhappy with returns",
    "D_confusion_drift": "Confused or asking others for help",
    "other": "General mention (weak SIP signal)",
}

LAPSE_PLAIN: dict[str, str] = {
    "high": "Strong link to stopping SIP",
    "medium": "Some link to stopping SIP",
    "low": "Weak link",
    "none": "Not about stopping",
    "unknown": "Unclear",
}


def plain_trigger(code: str) -> str:
    return TRIGGER_PLAIN.get(code, code.replace("_", " ").title())


def plain_branch(code: str) -> str:
    return BRANCH_PLAIN.get(code, code)


def opportunity_score_plain(score: float) -> str:
    if score >= 1.5:
        return "High"
    if score >= 0.8:
        return "Medium"
    return "Low"
