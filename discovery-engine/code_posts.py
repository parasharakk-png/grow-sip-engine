"""Codebook-based LLM coding of posts (fixed enums in codebook.json). Used by build_corpus.py --code."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

ENGINE = Path(__file__).resolve().parent
CODEBOOK_PATH = ENGINE / "codebook.json"

with CODEBOOK_PATH.open(encoding="utf-8") as f:
    CODEBOOK = json.load(f)

ENUMS = {
    k: CODEBOOK["fields"][k]["values"]
    for k in CODEBOOK["fields"]
    if CODEBOOK["fields"][k].get("type") == "single"
}
MULTI_ENUMS = {
    k: CODEBOOK["fields"][k]["values"]
    for k in CODEBOOK["fields"]
    if CODEBOOK["fields"][k].get("type") == "multi"
}


def post_text(row: dict[str, str]) -> str:
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()
    if title and body and title not in body:
        return f"{title}\n{body}"
    return body or title


def clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def pick_quote(text: str) -> str:
    for part in re.split(r"(?<=[.!?])\s+", text):
        part = part.strip()
        if 40 <= len(part) <= 280:
            return part
    return clip(text, 200)


def validate_coding(item: dict[str, Any]) -> dict[str, str]:
    """Normalize and enforce codebook enums."""
    out: dict[str, str] = {}
    for field, allowed in ENUMS.items():
        val = str(item.get(field, "unknown")).strip().lower()
        if val not in allowed:
            val = "unknown" if "unknown" in allowed else allowed[0]
        out[field] = val

    for field, allowed in MULTI_ENUMS.items():
        raw = item.get(field, [])
        if isinstance(raw, str):
            parts = [p.strip().lower() for p in raw.replace(",", "|").split("|")]
        elif isinstance(raw, list):
            parts = [str(p).strip().lower() for p in raw]
        else:
            parts = []
        clean = []
        for p in parts:
            if p in allowed and p not in clean:
                clean.append(p)
        out[field] = "|".join(clean) if clean else "unknown"

    out["workaround"] = clip(str(item.get("workaround", "")), 200)
    out["unmet_need"] = clip(str(item.get("unmet_need", "")), 200)
    out["quote"] = clip(str(item.get("quote", "")), 280)
    if not out["quote"]:
        out["quote"] = pick_quote(str(item.get("_source_text", "")))
    return out


def llm_prompt(batch: list[dict[str, str]]) -> str:
    codebook_summary = json.dumps(
        {k: CODEBOOK["fields"][k].get("values") or CODEBOOK["fields"][k].get("type") for k in CODEBOOK["fields"]},
        indent=2,
    )
    posts = []
    for row in batch:
        posts.append(
            {
                "item_id": row.get("item_id"),
                "source": row.get("source"),
                "query": row.get("query"),
                "text": clip(post_text(row), 1200),
            }
        )
    return f"""You are a qualitative researcher coding public posts about SIP mutual fund investing in India.

Use ONLY the allowed enum values in this codebook (segment_hints is multi-value array):
{codebook_summary}

Rules:
- lifecycle_stage: infer best single stage from text; use unknown if unclear.
- link_to_lapse: how strongly this post relates to stopping or lapsing a SIP (high/medium/low/none).
- workaround / unmet_need: short phrases (max 200 chars); empty string if not stated.
- quote: verbatim short phrase from the post (max 280 chars).
- confidence: high if explicit, low if inferred.

Return JSON array only, one object per post, keys: item_id, lifecycle_stage, trigger, emotion, behaviour, segment_hints (array), link_to_lapse, workaround, unmet_need, quote, confidence.

POSTS:
{json.dumps(posts, ensure_ascii=False, indent=2)}
"""


def parse_llm_json_array(text: str) -> list[dict[str, Any]]:
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError(f"No JSON array in model response: {text[:500]}")
    return json.loads(m.group(0))


def anthropic_response_text(message: Any) -> str:
    """Join TextBlock parts; skip ThinkingBlock from extended-thinking models."""
    parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(block.text)
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError(f"No text in model response blocks: {[b.type for b in message.content]}")
    return text


def code_llm_batch_anthropic(client: Any, batch: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    user_content = llm_prompt(batch)
    last_err: Exception | None = None
    for attempt in range(3):
        extra = ""
        if attempt:
            extra = "\n\nCRITICAL: Reply with a single valid JSON array only. No markdown, no commentary."
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=0,
            messages=[{"role": "user", "content": user_content + extra}],
        )
        try:
            text = anthropic_response_text(message)
            parsed = parse_llm_json_array(text)
            return _merge_batch_results(batch, parsed, model)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            time.sleep(2 * attempt)
    raise last_err or RuntimeError("LLM batch failed")


def _merge_batch_results(
    batch: list[dict[str, str]], parsed: list[dict[str, Any]], model: str
) -> list[dict[str, str]]:
    by_id = {str(p["item_id"]): p for p in parsed}
    out = []
    for row in batch:
        iid = str(row.get("item_id"))
        src_text = post_text(row)
        item = by_id.get(iid, {})
        item["_source_text"] = src_text
        coded = validate_coding(item)
        coded["item_id"] = iid
        coded["coding_method"] = "llm"
        coded["coding_model"] = model
        out.append(coded)
    return out


def code_llm_batch(client: Any, batch: list[dict[str, str]], model: str, provider: str = "anthropic") -> list[dict[str, str]]:
    return code_llm_batch_anthropic(client, batch, model)
