#!/usr/bin/env python3
"""
Rebuild the discovery corpus: first-person conversations about SIPs, split into first-time investors (target)
and everyone else (comparison), with the same filters for both.

A post is kept only if it passes:
  1. relevant    — about SIPs / mutual funds (not stock trading, F&O, loans, generic app praise)
  2. user_voice  — the author talks about their own situation (first person, EN + Hinglish),
                   not news, PSAs, promotions or essays about other people
  3. not a duplicate of an earlier post (same body text, e.g. a forum post collected with and without its title)
  4. the problem-fit LLM pass (problem_fit.py) confirms it is the author's own experience

Cohort: target if the LLM puts the author before or in their first 12 months of investing (or, without an LLM
verdict, a first-time phrase matches); comparison otherwise, including anyone 12+ months in. Age alone no longer
makes a post "target".

Every pool row is written to data/selection_log.csv with the signals that matched, so the
selection itself is auditable for the AI Evidence Log.

Usage:
  python3 build_corpus.py                # select, reusing saved LLM results (offline)
  python3 build_corpus.py --fit --code   # LLM problem-fit pass on new candidates + codebook-code new posts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
ROOT = ENGINE.parent
POOL_CSV = ROOT / "NL GROWW - cleaned_SIP_data_paste_into_sheets.backup.csv"
RAW_CSV = ROOT / "raw_posts.csv"
RAW_TARGETED_CSV = ROOT / "raw_posts_targeted.csv"
MAIN_CHECKPOINT = ENGINE / "coded_posts.checkpoint.jsonl"
CORPUS_CHECKPOINT = ENGINE / "data" / "corpus_first_time.checkpoint.jsonl"
OUT_CORPUS = ENGINE / "data" / "corpus_first_time.csv"
OUT_LOG = ENGINE / "data" / "selection_log.csv"
OUT_MERGED = ENGINE / "data" / "grow_sip_956.csv"

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
    raw = pd.concat([pd.read_csv(p, dtype=str).fillna("") for p in (RAW_CSV, RAW_TARGETED_CSV) if p.exists()],
                    ignore_index=True).drop_duplicates("url")
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
    log["duplicate_of"] = duplicate_of(pool, log["relevant"] & log["user_voice"])
    log["keep"] = log["relevant"] & log["user_voice"] & (log["duplicate_of"] == "")

    def reason(r: pd.Series) -> str:
        if not r["relevant"]:
            return "dropped: not about SIPs / mutual funds"
        if not r["user_voice"]:
            return "dropped: not the author's own experience"
        if r["duplicate_of"]:
            return f"dropped: duplicate of #{r['duplicate_of']}"
        return "candidate"

    log["decision"] = log.apply(reason, axis=1)
    return log


def _body_key(title: str, text: str) -> str:
    body = text.strip() if len(text) >= 80 else f"{title} {text}".strip()
    title = title.strip()
    while title and body.startswith(title):
        body = body[len(title):].strip()
    return re.sub(r"[^a-z0-9]", "", body.lower())[:200]


def duplicate_of(pool: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    """item_id of an earlier eligible post with the same body (title stripped), else ''.

    Also catches reposts: a post whose text sits inside another post of similar length ("Posting it again…").
    Similar length only, so a reply that quotes part of another post is not dropped."""
    first: dict[str, str] = {}
    out = pd.Series("", index=pool.index)
    bodies: list[tuple[object, str, str]] = []
    for idx, r in pool[eligible].iterrows():
        key = _body_key(r["title"], r["text"])
        if len(key) < 60:
            continue
        if key in first:
            out[idx] = first[key]
            continue
        first[key] = r["item_id"]
        bodies.append((idx, r["item_id"], re.sub(r"[^a-z0-9]", "", f"{r['title']} {r['text']}".lower())))
    for i, (idx_i, _, body_i) in enumerate(bodies):
        if len(body_i) < 300:
            continue
        window = body_i[len(body_i) // 2 - 100: len(body_i) // 2 + 100]
        for idx_j, id_j, body_j in bodies[:i]:
            if out[idx_j] == "" and 0.6 <= len(body_i) / len(body_j) <= 1.67 and window in body_j:
                out[idx_i] = id_j
                break
    return out


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


FIRST_STAGES = ("not_started", "first_year")
PAST_FIRST_YEAR = ("12_18", "gt18")


def assign(fit: dict | None, screen: dict | None, rule_first: bool) -> tuple[bool, str]:
    """(keep, selection). The problem-fit verdict wins, then the older screen, then the first-time phrase rule."""
    verdict = fit or screen
    if verdict and str(verdict.get("own_experience")).lower() != "true":
        return False, "dropped by LLM: not the author's own experience"
    stage = str((verdict or {}).get("investor_stage", "unknown"))
    if fit and fit.get("months_since_start") in PAST_FIRST_YEAR:
        return True, f"comparison: past first 12 months (llm: {fit['months_since_start']})"
    if stage in FIRST_STAGES:
        return True, f"first_time (llm: {stage})"
    if stage == "experienced":
        return True, "comparison: experienced investor (llm)"
    if rule_first:
        return True, "first_time (rule: first-time phrase, llm: unknown)" if verdict else "first_time (rule only)"
    return True, "comparison: stage not stated"


def code_missing(rows: pd.DataFrame, codes: dict[str, dict[str, str]], batch_size: int, workers: int = 6) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import code_posts

    todo = [r for r in rows.to_dict("records") if codes.get(r["item_id"], {}).get("coding_method") != "llm"]
    if not todo:
        return
    client, model = _client()
    batches = [todo[i: i + batch_size] for i in range(0, len(todo), batch_size)]
    print(f"Codebook coding {len(todo)} posts with {model} ({len(batches)} batches)")
    CORPUS_CHECKPOINT.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, CORPUS_CHECKPOINT.open("a", encoding="utf-8") as f:
        futures = [pool.submit(code_posts.code_llm_batch, client, b, model, "anthropic") for b in batches]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                coded = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  batch failed: {str(e)[:160]}", file=sys.stderr)
                continue
            for c in coded:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                codes[c["item_id"]] = c
            f.flush()
            if n % 10 == 0 or n == len(batches):
                print(f"  {n}/{len(batches)} batches")


def to_content(row: pd.Series, code: dict[str, str], fit: dict, selection: str) -> str:
    post = row["text"] if not row["title"] or row["title"] in row["text"] else f"{row['title']} {row['text']}"
    seg = code.get("segment_hints", "")
    if isinstance(seg, list):
        seg = "|".join(seg)
    branch = row["query"].split("|", 1)[0] if "|" in row["query"] else ""
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
        f"STAGE: {fit.get('investor_stage', '')}",
        f"MONTHS: {fit.get('months_since_start', '')}",
        f"STATUS: {fit.get('sip_status', '')}",
        f"CONCERN: {fit.get('core_concern', '')}",
        f"YARDSTICK: {fit.get('yardstick', '')}",
        f"DECIDES_OUTSIDE: {fit.get('decides_outside', '')}",
        f"LIFE_CHANGE: {fit.get('life_change', '')}",
        f"AGE_BAND: {fit.get('age_band', '')}",
        f"FIT_QUOTE: {fit.get('evidence_quote', '') if fit.get('quote_verbatim') else ''}",
        f"QUERY_BRANCH: {branch}",
        f"SEARCH_QUERY: {row['query']}",
        f"SELECTION: {selection}",
        f"POST: {re.sub(chr(10), ' ', post)}",
        f"URL: {row['url']}",
    ])


def main() -> None:
    import problem_fit

    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="LLM problem-fit pass on candidates without a verdict")
    ap.add_argument("--code", action="store_true", help="LLM codebook-code kept posts without LLM codes")
    ap.add_argument("--batch-size", type=int, default=5)
    args = ap.parse_args()

    pool = load_pool()
    codes = load_codes()
    log = select(pool, codes)
    candidates = pool[log["keep"].values]
    if args.fit:
        client, model = _client()
        fits = problem_fit.code(candidates.to_dict("records"), client, model)
    else:
        fits = problem_fit.load()
    screening = load_screening()

    for col in ("investor_stage", "months_since_start", "sip_status", "core_concern"):
        log[f"fit_{col}"] = log["item_id"].map(lambda i, c=col: fits.get(i, {}).get(c, ""))
    log["fit_own_experience"] = log["item_id"].map(lambda i: fits.get(i, {}).get("own_experience", ""))
    decided = [assign(fits.get(r.item_id), screening.get(r.item_id), r.first_time) if r.keep else (False, r.decision)
               for r in log.itertuples()]
    log["keep"] = [k for k, _ in decided]
    log["decision"] = [d if not k else f"kept: {d}" for k, d in decided]
    log.to_csv(OUT_LOG, index=False)
    kept = pool[log["keep"].values]

    print(f"Pool: {len(pool)} posts | candidates: {len(candidates)} | with problem-fit verdict: "
          f"{int(candidates['item_id'].isin(fits).sum())}")
    print(log["decision"].str.replace(r"duplicate of #\d+", "duplicate", regex=True).value_counts().to_string())

    if args.code:
        code_missing(kept, codes, args.batch_size)

    selection = log.set_index("item_id")["decision"].str.replace("kept: ", "", regex=False)
    out = pd.DataFrame({
        "item_id": kept["item_id"].values,
        "url": kept["url"].values,
        "content": [to_content(r, codes.get(r["item_id"], {}), fits.get(r["item_id"], {}), selection[r["item_id"]])
                    for _, r in kept.iterrows()],
    })
    target = out[~out["content"].str.contains("\nSELECTION: comparison", regex=False)]
    target.to_csv(OUT_CORPUS, index=False)
    out.to_csv(OUT_MERGED, index=False)
    uncoded = int((~kept["item_id"].isin([i for i, c in codes.items() if c.get("coding_method") == "llm"])).sum())
    print(f"\nWrote {OUT_MERGED.name}: {len(out)} posts ({len(target)} first-time target, {len(out) - len(target)} "
          f"comparison; {uncoded} without codebook codes)")
    print("Kept by source:\n" + pool[log["keep"].values]["source"].value_counts().to_string())
    print(f"Wrote {OUT_LOG.name}: every pool row with its decision and matched signals")


if __name__ == "__main__":
    main()
