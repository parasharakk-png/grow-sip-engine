#!/usr/bin/env python3
"""
Rebuild the discovery corpus: first-person conversations from first-time and young investors.

A post is kept only if it passes all three filters:
  1. relevant    — about SIPs / mutual funds (not stock trading, F&O, loans, generic app praise)
  2. user_voice  — the author talks about their own situation (first person, EN + Hinglish),
                   not news, PSAs, promotions or essays about other people
  3. target      — first-time investor OR early-age investor (18–30, student, first job)

Every pool row is written to data/selection_log.csv with the signals that matched, so the
selection itself is auditable for the AI Evidence Log.

Usage:
  python3 build_corpus.py            # select + reuse existing LLM codes
  python3 build_corpus.py --code     # also LLM-code kept posts that only have rules codes / no codes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
ROOT = ENGINE.parent
POOL_CSV = ROOT / "NL GROWW - cleaned_SIP_data_paste_into_sheets.backup.csv"
RAW_CSV = ROOT / "raw_posts.csv"
MAIN_CHECKPOINT = ENGINE / "coded_posts.checkpoint.jsonl"
CORPUS_CHECKPOINT = ENGINE / "data" / "corpus_first_time.checkpoint.jsonl"
OUT_CORPUS = ENGINE / "data" / "corpus_first_time.csv"
OUT_LOG = ENGINE / "data" / "selection_log.csv"
OUT_MERGED = ENGINE / "data" / "grow_sip_956.csv"
MERGE_BASE = ENGINE / "data" / "grow_sip_956.original.csv"

RAW_SOURCE = {"play_store": "Play Store", "app_store": "App Store", "youtube": "YouTube",
              "reddit": "Reddit", "tradingqna": "TradingQ&A"}

RELEVANT = re.compile(
    r"\bsips?\b|systematic investment|mutual ?funds?|\bmfs?\b|index fund|\belss\b|flexi ?cap|small ?cap"
    r"|mid ?cap|large ?cap|nifty ?50|\bnav\b|\bxirr\b|autopay|auto[- ]?debit|mandate|step[- ]?up"
    r"|lump ?sum|\bamc\b|direct plan|regular plan|\bfunds?\b.{0,30}\binvest|invest\w*.{0,30}\bfunds?\b",
    re.I,
)
NOT_RELEVANT_ONLY = re.compile(
    r"\bmtf\b|\bf&o\b|\boptions? trading|intraday|\bfutures?\b|\bmcx\b|commodit|crypto|\bipo\b|us stocks?"
    r"|credit card|personal loan|loan app",
    re.I,
)

FIRST_PERSON = re.compile(
    r"\b(i|i'm|im|i've|ive|i'd|i'll|my|me|mine|myself|maine|mujhe|mera|meri|mere|main|mai|hamara|humne)\b",
    re.I,
)
NOT_USER_VOICE = re.compile(
    r"^\s*\[?(psa|news|breaking|announcement|nfo)\]?|press release|let me tell you about|here'?s the actual data"
    r"|informative perspective|time ?stamp|\bsubscribe\b|join (my|our) (channel|telegram)|use my referral"
    r"|referral code|vivekkaul\.com",
    re.I,
)

FIRST_TIME = re.compile(
    r"first (ever )?(sip|investment|mutual fund|\bmf\b|portfolio|fund)|first time (invest|sip|in mutual|buying|starting)"
    r"|first[- ]time investor|my first (sip|investment|mutual|mf|fund"
    r"|year|portfolio|salary|job|time)|new to (investing|investment|mutual|mf|sips?|stock|the market|this)"
    r"|\bbeginners?\b|\bnewbie|\bnovice|\bnoob\b|just started|recently started|started (a |my first )sip"
    r"|started (investing|my sips?|sips?|a sip|mf|mutual funds?) .{0,20}(months? ago|weeks? ago|this year|recently|202[5-6])"
    r"|started .{0,25}\b(\d|few|couple of|two|three|six) (months?|weeks?) ago|new investor"
    r"|never invested|no (prior )?experience|zero knowledge|complete(ly)? new|don'?t know much"
    r"|(want|planning|plan|thinking of|about) to start (my |a |an )?(first )?(sip|investing|investment|mf|mutual)"
    r"|starting (my|a) (first )?(sip|investment|mutual|mf)|just starting out|how (do|should) i start|where (do|should) i start"
    r"|pehli baar|naya hu|abhi start|shuru (kiya|karna)",
    re.I,
)
YOUNG = re.compile(
    r"\b(1[89]|2[0-9]) ?(y/?o|yo|yrs? old|years? old)\b|\b(i am|i'm|im|age[d:]?|aged) ?(1[89]|2[0-9])\b"
    r"|\(?\b(1[89]|2[0-9]) ?[mf]\b\)?|\b[mf] ?(1[89]|2[0-9])\b|\bstudent\b|\bcollege\b|\bfresher\b|internship"
    r"|\bintern\b|first job|first salary|just graduated|recent(ly)? graduat|early (20s|twenties)|in my 20s"
    r"|\b(1[89]|2[0-9]) saal",
    re.I,
)
OLDER = re.compile(
    r"\b([3-6][1-9]|[4-6]0) ?(y/?o|yo|yrs? old|years? old)\b|\b(i am|i'm|im|age[d:]?|aged) ?([3-6][1-9]|[4-6]0)\b"
    r"|\(?\b([3-6][1-9]|[4-6]0) ?[mf]\b\)?|\b[mf] ?([3-6][1-9]|[4-6]0)\b",
    re.I,
)
EXPERIENCED = re.compile(
    r"(started|been) (investing|sips?|my sips?|in mutual funds?) (in|since|from) (19\d\d|20[01]\d|202[0-3])"
    r"|(investing|sips?) since (19\d\d|20[01]\d|202[0-3])|\b(\d{1,2}|several|many) years (of|into) (investing|sips?)"
    r"|been investing for (\d{1,2}|several|many) years|long[- ]time (investor|lurker)",
    re.I,
)
LLM_SEGMENT_FIRST = ("first_mf_investor",)
LLM_SEGMENT_YOUNG = ("young_22_30", "first_job")


def load_pool() -> pd.DataFrame:
    pool = pd.read_csv(POOL_CSV, dtype=str).fillna("")
    pool = pool[["item_id", "source", "platform_detail", "date_clean", "query", "title", "text", "url"]]
    raw = pd.read_csv(RAW_CSV, dtype=str).fillna("")
    raw = raw[~raw["url"].isin(pool["url"]) & (raw["text"].str.len() >= 40)].copy()
    raw["source"] = raw["source"].map(RAW_SOURCE).fillna(raw["source"])
    raw["date_clean"] = raw["date"]
    start = pool["item_id"].astype(int).max() + 1
    raw["item_id"] = [str(start + i) for i in range(len(raw))]
    raw = raw[["item_id", "source", "platform_detail", "date_clean", "query", "title", "text", "url"]]
    return pd.concat([pool, raw], ignore_index=True)


def load_codes() -> dict[str, dict[str, str]]:
    codes: dict[str, dict[str, str]] = {}
    for path in (MAIN_CHECKPOINT, CORPUS_CHECKPOINT):
        if path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    obj = json.loads(line)
                    codes[str(obj["item_id"])] = obj
    return codes


def select(pool: pd.DataFrame, codes: dict[str, dict[str, str]]) -> pd.DataFrame:
    text = (pool["title"] + " " + pool["text"]).str.strip()
    seg = pool["item_id"].map(lambda i: codes.get(i, {}).get("segment_hints", "")).fillna("")

    def matches(pattern: re.Pattern[str]) -> pd.Series:
        return text.apply(lambda t: ", ".join(sorted({m.group(0).lower().strip() for m in pattern.finditer(t)}))[:120])

    log = pool[["item_id", "source", "url"]].copy()
    log["relevant_terms"] = matches(RELEVANT)
    log["off_topic_terms"] = matches(NOT_RELEVANT_ONLY)
    log["first_person_hits"] = text.apply(lambda t: len(FIRST_PERSON.findall(t)))
    log["not_user_voice"] = matches(NOT_USER_VOICE)
    log["first_time_terms"] = matches(FIRST_TIME)
    log["young_terms"] = matches(YOUNG)
    log["llm_segment"] = seg

    log["relevant"] = (log["relevant_terms"] != "") & ~(
        (log["off_topic_terms"] != "") & (log["relevant_terms"].str.count(",") < 1)
    )
    # App reviews are always the reviewer's own experience; comments and forum posts must say "I/my".
    min_fp = pool["source"].map({"Play Store": 0, "App Store": 0, "YouTube": 1}).fillna(2)
    log["user_voice"] = (log["first_person_hits"] >= min_fp) & (log["not_user_voice"] == "")
    log["older_terms"] = matches(OLDER)
    log["experienced_terms"] = matches(EXPERIENCED)
    log["first_time"] = (
        (log["first_time_terms"] != "") | seg.str.contains("|".join(LLM_SEGMENT_FIRST))
    ) & (log["experienced_terms"] == "")
    log["young"] = (
        (log["young_terms"] != "") | seg.str.contains("|".join(LLM_SEGMENT_YOUNG))
    ) & (log["older_terms"] == "")
    log["keep"] = log["relevant"] & log["user_voice"] & (log["first_time"] | log["young"])

    def reason(r: pd.Series) -> str:
        if r["keep"]:
            return "kept: " + "+".join(x for x, ok in (("first_time", r["first_time"]), ("young", r["young"])) if ok)
        if not r["relevant"]:
            return "dropped: not about SIPs / mutual funds"
        if not r["user_voice"]:
            return "dropped: not the author's own experience"
        return "dropped: not first-time or young investor"

    log["decision"] = log.apply(reason, axis=1)
    return log


def _client() -> tuple[object, str]:
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ENGINE / ".env")
    model = os.environ.get("ANTHROPIC_MODEL", "MiniMax-M2.7-highspeed").strip()
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"].strip(),
        base_url=(os.environ.get("ANTHROPIC_BASE_URL") or "").strip() or None,
        timeout=180,
        max_retries=1,
    )
    return client, model


SCREEN_PROMPT = """You screen public posts for a study of FIRST-TIME and YOUNG mutual fund / SIP investors in India.
For each post decide, from the author's own words only:
- own_experience: true if the author describes THEIR OWN investing situation or question; false for news,
  guides, essays, promotions, or commentary about other people / a video.
- investor_stage: "not_started" (about to start), "first_year" (started within ~12 months or calls
  themselves a beginner/new), "experienced" (investing > ~1 year), or "unknown".
- age_band: "under_25", "25_30", "over_30" or "unknown" (only if age, studies or first job is stated).
- reason: max 12 words.
Return a JSON array only: [{"item_id": "...", "own_experience": true, "investor_stage": "...", "age_band": "...", "reason": "..."}]

POSTS:
"""
SCREEN_CHECKPOINT = ENGINE / "data" / "screening.checkpoint.jsonl"


def load_screening() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    if SCREEN_CHECKPOINT.exists():
        for line in SCREEN_CHECKPOINT.open(encoding="utf-8"):
            if line.strip():
                obj = json.loads(line)
                out[str(obj["item_id"])] = obj
    return out


def screen_with_llm(rows: pd.DataFrame, batch_size: int) -> dict[str, dict[str, object]]:
    import code_posts

    done = load_screening()
    todo = [r for r in rows.to_dict("records") if r["item_id"] not in done]
    if not todo:
        return done
    client, model = _client()
    print(f"LLM screening {len(todo)} candidates with {model} (batch {batch_size})")
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        posts = [{"item_id": r["item_id"], "source": r["source"], "text": code_posts.clip(code_posts.post_text(r), 900)} for r in batch]
        try:
            msg = client.messages.create(  # type: ignore[attr-defined]
                model=model, max_tokens=8000,
                messages=[{"role": "user", "content": SCREEN_PROMPT + json.dumps(posts, ensure_ascii=False)}],
            )
            parsed = code_posts.parse_llm_json_array(code_posts.anthropic_response_text(msg))
        except Exception as e:  # noqa: BLE001
            print(f"  batch {i} failed: {str(e)[:160]}", file=sys.stderr)
            continue
        with SCREEN_CHECKPOINT.open("a", encoding="utf-8") as f:
            for p in parsed:
                p["item_id"] = str(p.get("item_id"))
                p["screen_model"] = model
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                done[p["item_id"]] = p
        print(f"  screened {min(i + batch_size, len(todo))}/{len(todo)}")
    return done


def llm_keep(s: dict[str, object] | None, rule_first: bool, rule_young: bool) -> bool | None:
    """The LLM judges whether the post is the author's own story and can veto stage/age; when it
    answers "unknown" for stage or age, the explicit rule match (e.g. "I am 25") stands."""
    if not s:
        return None
    stage, age = s.get("investor_stage"), s.get("age_band")
    first = stage in ("not_started", "first_year") or (stage == "unknown" and rule_first)
    young = age in ("under_25", "25_30") or (age == "unknown" and rule_young)
    return str(s.get("own_experience")).lower() == "true" and (first or young)


def code_missing(rows: pd.DataFrame, codes: dict[str, dict[str, str]], batch_size: int) -> None:
    import code_posts

    client, model = _client()
    todo = [r for r in rows.to_dict("records") if codes.get(r["item_id"], {}).get("coding_method") != "llm"]
    if not todo:
        return
    print(f"LLM coding {len(todo)} posts with {model} (batch {batch_size})")
    CORPUS_CHECKPOINT.parent.mkdir(exist_ok=True)
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        try:
            coded = code_posts.code_llm_batch(client, batch, model, "anthropic")
        except Exception as e:  # noqa: BLE001
            print(f"  batch {i} failed: {str(e)[:160]}", file=sys.stderr)
            continue
        with CORPUS_CHECKPOINT.open("a", encoding="utf-8") as f:
            for c in coded:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                codes[c["item_id"]] = c
        print(f"  coded {min(i + batch_size, len(todo))}/{len(todo)}")
        time.sleep(0.3)


def to_content(row: pd.Series, code: dict[str, str], selection: str) -> str:
    post = row["text"] if not row["title"] or row["title"] in row["text"] else f"{row['title']} {row['text']}"
    seg = code.get("segment_hints", "")
    if isinstance(seg, list):
        seg = "|".join(seg)
    return "\n".join([
        f"ID: {row['item_id']}",
        f"SOURCE: {row['source']}",
        f"LIFECYCLE: {code.get('lifecycle_stage', 'unknown')}",
        f"TRIGGER: {code.get('trigger', 'unknown')}",
        f"EMOTION: {code.get('emotion', 'unknown')}",
        f"BEHAVIOUR: {code.get('behaviour', 'unknown')}",
        f"SEGMENT: {seg}",
        f"LINK_TO_LAPSE: {code.get('link_to_lapse', 'unknown')}",
        f"WORKAROUND: {code.get('workaround', '')}",
        f"UNMET_NEED: {code.get('unmet_need', '')}",
        f"QUOTE: {code.get('quote', '')}",
        f"SEARCH_QUERY: {row['query']}",
        f"SELECTION: {selection}",
        f"POST: {re.sub(chr(10), ' ', post)}",
        f"URL: {row['url']}",
    ])


def _tag_comparison(content: str) -> str:
    if "\nSELECTION:" in content:
        return content
    line = "SELECTION: comparison (not identified as first-time / young)"
    return content.replace("\nPOST:", f"\n{line}\nPOST:", 1) if "\nPOST:" in content else f"{content}\n{line}"


def write_merged(target: pd.DataFrame) -> None:
    """Target corpus + earlier posts (comparison group) -> the file the app reads.
    The earlier file is preserved once as MERGE_BASE, so reruns never duplicate rows."""
    if not MERGE_BASE.exists():
        if not OUT_MERGED.exists():
            return
        shutil.copy(OUT_MERGED, MERGE_BASE)
    base = pd.read_csv(MERGE_BASE, dtype=str).fillna("")
    extra = base[~base["item_id"].isin(target["item_id"])].copy()
    extra["content"] = extra["content"].map(_tag_comparison)
    merged = pd.concat([target, extra[["item_id", "url", "content"]]], ignore_index=True)
    merged.to_csv(OUT_MERGED, index=False)
    print(f"Wrote {OUT_MERGED.name}: {len(merged)} posts ({len(target)} target + {len(extra)} comparison from {MERGE_BASE.name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true", help="LLM-verify rule candidates (own story, stage, age)")
    ap.add_argument("--code", action="store_true", help="LLM-code kept posts without LLM codes")
    ap.add_argument("--batch-size", type=int, default=5)
    args = ap.parse_args()

    pool = load_pool()
    codes = load_codes()
    log = select(pool, codes)
    log = log.rename(columns={"keep": "rule_keep"})
    candidates = pool[log["rule_keep"].values]
    screening = screen_with_llm(candidates, 10) if args.screen else load_screening()

    log["llm_own_experience"] = log["item_id"].map(lambda i: screening.get(i, {}).get("own_experience", ""))
    log["llm_investor_stage"] = log["item_id"].map(lambda i: screening.get(i, {}).get("investor_stage", ""))
    log["llm_age_band"] = log["item_id"].map(lambda i: screening.get(i, {}).get("age_band", ""))
    log["llm_reason"] = log["item_id"].map(lambda i: screening.get(i, {}).get("reason", ""))
    llm = [llm_keep(screening.get(r.item_id), r.first_time, r.young) for r in log.itertuples()]
    log["keep"] = log["rule_keep"] & pd.Series([True if v is None else v for v in llm], index=log.index)
    log.loc[log["rule_keep"] & ~log["keep"], "decision"] = "dropped by LLM screen: " + log["llm_reason"].astype(str)
    log.to_csv(OUT_LOG, index=False)
    kept = pool[log["keep"].values]

    screened = log[log["rule_keep"] & log["llm_own_experience"].ne("")]
    print(f"Pool: {len(pool)} posts | rule candidates: {int(log['rule_keep'].sum())} | LLM-screened: {len(screened)}")
    print(log["decision"].str.replace(r"dropped by LLM screen: .*", "dropped by LLM screen", regex=True).value_counts().to_string())
    print("\nKept by source:\n" + kept["source"].value_counts().to_string())

    if args.code:
        code_missing(kept, codes, args.batch_size)

    selection = log.set_index("item_id")["decision"].str.replace("kept: ", "", regex=False)
    uncoded = [i for i in kept["item_id"] if codes.get(i, {}).get("coding_method") != "llm"]
    out = pd.DataFrame({
        "item_id": kept["item_id"].values,
        "url": kept["url"].values,
        "content": [to_content(r, codes.get(r["item_id"], {}), selection[r["item_id"]]) for _, r in kept.iterrows()],
    })
    out.to_csv(OUT_CORPUS, index=False)
    print(f"\nWrote {OUT_CORPUS.name}: {len(out)} posts ({len(out) - len(uncoded)} LLM-coded, {len(uncoded)} not yet)")
    write_merged(out)
    print(f"Wrote {OUT_LOG.name}: every pool row with its decision and matched signals")


if __name__ == "__main__":
    main()
