"""Problem-fit coding: one LLM pass whose labels mirror the screener (Q1 months since start, Q4 SIP status)
and the working hypothesis (what the investor's concern is, and what they judge their SIP against).

Used by build_corpus.py --fit. Results are cached in data/problem_fit.checkpoint.jsonl with the prompt version,
so a changed prompt re-codes everything and an unchanged one never re-calls the model.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import code_posts

ENGINE = Path(__file__).resolve().parent
CHECKPOINT = ENGINE / "data" / "problem_fit.checkpoint.jsonl"

ENUMS: dict[str, tuple[str, ...]] = {
    "investor_stage": ("not_started", "first_year", "experienced", "unknown"),
    "months_since_start": ("lt3", "3_6", "6_12", "12_18", "gt18", "unknown"),
    "sip_status": ("not_started", "running_no_doubt", "running_considering_stop", "paused_or_reduced", "stopped",
                   "stopped_and_switched", "failed_without_choice", "unknown"),
    "core_concern": ("how_it_runs", "is_it_working", "cash_or_life", "which_fund", "payment_or_platform_failure",
                     "other"),
    "yardstick": ("none_stated", "own_goal_or_horizon", "return_number", "other_people", "index_or_benchmark",
                  "fd_or_savings"),
    "decides_outside": ("yes", "no"),
    "life_change": ("job_change", "salary_account_change", "income_drop", "big_expense_or_loan", "relocation",
                    "none_stated"),
    "age_band": ("under_25", "25_30", "over_30", "unknown"),
    "confidence": ("high", "low"),
}

LABELS: dict[str, dict[str, str]] = {
    "months_since_start": {"lt3": "< 3 months", "3_6": "3–6 months", "6_12": "6–12 months", "12_18": "12–18 months",
                           "gt18": "> 18 months", "unknown": "Not stated"},
    "sip_status": {"not_started": "Not started yet", "running_no_doubt": "Running, no doubt",
                   "running_considering_stop": "Running, considering stopping", "paused_or_reduced": "Paused or reduced",
                   "stopped": "Stopped", "stopped_and_switched": "Stopped and started another",
                   "failed_without_choice": "Failed / stopped without their choice", "unknown": "Not stated"},
    "core_concern": {"how_it_runs": "How the SIP runs (dates, debits, skip, status)",
                     "is_it_working": "Is it working / is this normal?", "cash_or_life": "Cash need or life change",
                     "which_fund": "Which fund / portfolio choice",
                     "payment_or_platform_failure": "Payment or platform failure", "other": "Other"},
    "yardstick": {"none_stated": "Nothing stated", "own_goal_or_horizon": "Own goal or time horizon",
                  "return_number": "A return number (e.g. 12%)", "other_people": "What other people do or earn",
                  "index_or_benchmark": "An index or benchmark", "fd_or_savings": "FD / savings account"},
    "life_change": {"job_change": "Changed or lost job", "salary_account_change": "Salary / bank account changed",
                    "income_drop": "Income fell or became irregular", "big_expense_or_loan": "Big expense or loan",
                    "relocation": "Moved city", "none_stated": "None stated"},
}
AT_RISK_STATUS = ("running_considering_stop", "paused_or_reduced", "stopped", "stopped_and_switched",
                  "failed_without_choice")

PROMPT = """You code public posts for a study of FIRST-TIME mutual fund SIP investors in India (first 12-18 months).
Judge only from the author's own words. Do not guess: use "unknown" / "none_stated" when the post doesn't say.

Fields:
- own_experience: true if the author describes THEIR OWN investing situation or question (app reviews count);
  false for news, guides, promotions, or commentary about others.
- investor_stage: "not_started" (about to start), "first_year" (started within ~12 months, or calls themselves a
  beginner / new and gives no longer history), "experienced" (investing more than ~1 year), "unknown".
- months_since_start: since their FIRST SIP / MF investment: "lt3", "3_6", "6_12", "12_18", "gt18", "unknown".
- sip_status: "not_started", "running_no_doubt", "running_considering_stop" (asks whether to stop/pause, or doubts
  continuing), "paused_or_reduced", "stopped", "stopped_and_switched" (stopped one SIP to start another fund,
  platform or lump sum), "failed_without_choice" (an instalment failed, or the SIP stopped without them choosing),
  "unknown". A portfolio-review or "which fund should I pick" post with no doubt about continuing is
  "running_no_doubt" or "not_started", never a stop.
- core_concern: the ONE main concern: "how_it_runs" (debit dates, mandate setup, skip/pause mechanics, order status,
  units not showing, what happens if...), "is_it_working" (returns, losses, whether this is normal, whether to keep
  going), "cash_or_life" (money is tight, a loan, a job or life change), "which_fund" (choosing or switching funds,
  portfolio construction), "payment_or_platform_failure" (debit failed, mandate rejected, app or platform error),
  "other".
- yardstick: what the author judges their SIP against, if anything: "none_stated", "own_goal_or_horizon",
  "return_number", "other_people", "index_or_benchmark", "fd_or_savings".
- decides_outside: "yes" if they ask strangers / a forum / friends to decide what they should do with the SIP.
- life_change: a change in the author's life since starting that the post mentions: "job_change" (switched, lost
  or between jobs), "salary_account_change" (new salary or bank account, old account closed), "income_drop",
  "big_expense_or_loan", "relocation", "none_stated". If several, pick the one closest to the SIP's fate.
- age_band: "under_25", "25_30", "over_30", "unknown" (only from a stated age, studies or first job).
- evidence_quote: the 5-25 words, copied exactly from the post, that best support sip_status and core_concern.
- confidence: "high" if the post states it explicitly, "low" if inferred.

Return a JSON array only, one object per post, with keys: item_id, own_experience, investor_stage,
months_since_start, sip_status, core_concern, yardstick, decides_outside, life_change, age_band, evidence_quote,
confidence.

POSTS:
"""
PROMPT_VERSION = hashlib.sha1(PROMPT.encode()).hexdigest()[:8]


def load() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.open(encoding="utf-8"):
            if line.strip():
                obj = json.loads(line)
                if obj.get("prompt_version") == PROMPT_VERSION:
                    out[str(obj["item_id"])] = obj
    return out


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def validate(item: dict[str, Any], source_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"own_experience": str(item.get("own_experience")).lower() == "true"}
    for field, allowed in ENUMS.items():
        val = str(item.get(field, "")).strip().lower()
        out[field] = val if val in allowed else allowed[-1] if field == "confidence" else (
            "none_stated" if field in ("yardstick", "life_change") else "no" if field == "decides_outside"
            else "other" if field == "core_concern" else "unknown")
    quote = code_posts.clip(str(item.get("evidence_quote", "")), 240)
    out["evidence_quote"] = quote
    out["quote_verbatim"] = bool(quote) and _norm(quote) in _norm(source_text)
    return out


def _code_batch(client: Any, model: str, batch: list[dict[str, str]]) -> list[dict[str, Any]]:
    posts = [{"item_id": r["item_id"], "source": r["source"], "text": code_posts.clip(code_posts.post_text(r), 1200)}
             for r in batch]
    body = PROMPT + json.dumps(posts, ensure_ascii=False)
    last: Exception | None = None
    for attempt in range(3):
        extra = "\n\nReply with one valid JSON array only." if attempt else ""
        try:
            msg = client.messages.create(model=model, max_tokens=8000, temperature=0,
                                         messages=[{"role": "user", "content": body + extra}])
            parsed = {str(p.get("item_id")): p for p in
                      code_posts.parse_llm_json_array(code_posts.anthropic_response_text(msg))}
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        out = []
        for r in batch:
            if r["item_id"] not in parsed:
                continue
            coded = validate(parsed[r["item_id"]], code_posts.post_text(r))
            coded.update(item_id=r["item_id"], fit_model=model, prompt_version=PROMPT_VERSION)
            out.append(coded)
        return out
    raise last or RuntimeError("batch failed")


def code(rows: list[dict[str, str]], client: Any, model: str, batch_size: int = 8,
         workers: int = 6) -> dict[str, dict[str, Any]]:
    done = load()
    todo = [r for r in rows if r["item_id"] not in done]
    if not todo:
        return done
    batches = [todo[i: i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"Problem-fit coding {len(todo)} posts with {model}, prompt {PROMPT_VERSION} ({len(batches)} batches)")
    CHECKPOINT.parent.mkdir(exist_ok=True)
    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, CHECKPOINT.open("a", encoding="utf-8") as f:
        futures = {pool.submit(_code_batch, client, model, b): b for b in batches}
        for fut in as_completed(futures):
            finished += 1
            try:
                coded = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  batch failed: {str(e)[:160]}", file=sys.stderr)
                continue
            for c in coded:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                done[c["item_id"]] = c
            f.flush()
            if finished % 10 == 0 or finished == len(batches):
                print(f"  {finished}/{len(batches)} batches")
    return done
