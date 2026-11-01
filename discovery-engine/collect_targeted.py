#!/usr/bin/env python3
"""Second collection round, aimed at the 12-month continuation problem.

Every query is tagged with the metric branch it probes (see metric_decomposition.py), and every branch gets
queries, so the round can also disconfirm the working hypothesis. The query is stored as "<branch>|<query>".

Run:  python3 collect_targeted.py            (adds to ../raw_posts_targeted.csv, ~20 min)
      python3 collect_targeted.py --only reddit,tradingqna,youtube --branches life_transition
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
from google_play_scraper import Sort, reviews as gp_reviews
from youtube_comment_downloader import YoutubeCommentDownloader

from collect import (GROWW_PLAY_ID, Post, collect_app_store, dedupe, fetch_url, write_csv,
                     youtube_search_video_ids)

OUTPUT = Path(__file__).resolve().parent.parent / "raw_posts_targeted.csv"

QUERIES: dict[str, list[str]] = {
    "start_expectation": ["started my first sip", "first sip expectations", "started sip months ago",
                          "beginner sip returns expectation"],
    "A_involuntary": ["sip installment failed", "sip not deducted", "sip mandate", "autopay sip failed",
                      "change bank account sip"],
    "B_pause_cash": ["skip sip", "pause sip", "reduce sip amount", "stop sip need money", "sip date change"],
    "C_deliberate": ["should i continue sip", "stop sip start new", "sip negative returns", "sip in loss"],
    "D_confusion": ["xirr negative", "sip returns low", "is it normal sip", "sip units nav confused"],
    "drift": ["forgot sip", "sip stopped automatically", "sip cancelled without"],
    "life_transition": ["job change sip", "changed job sip mandate", "salary account changed sip",
                        "notice period sip", "between jobs sip", "new job stop sip"],
}
YOUTUBE_QUERIES: dict[str, list[str]] = {
    "C_deliberate": ["should I stop my SIP", "SIP negative returns kya kare", "SIP loss first year"],
    "B_pause_cash": ["SIP pause kaise kare", "SIP skip kaise kare groww"],
    "A_involuntary": ["SIP autopay failed groww"],
    "start_expectation": ["first SIP mistakes beginners India"],
    "life_transition": ["job change SIP kya kare", "SIP bank account change"],
}
SUBREDDITS = ["mutualfunds", "IndiaInvestments", "personalfinanceindia", "FIREIndia"]
SIP_TERMS = re.compile(r"\bsips?\b|mutual ?funds?|\bmf\b|mandate|autopay|auto[- ]?debit|instal+ment|\bnav\b|xirr",
                       re.I)


def play_store(newest: int, relevant: int) -> list[Post]:
    """Groww reviews mention stocks far more than SIPs, so pull many and keep only SIP-related ones."""
    out: list[Post] = []
    for sort, target, tag in ((Sort.NEWEST, newest, "newest"), (Sort.MOST_RELEVANT, relevant, "most_relevant")):
        token, seen = None, 0
        while seen < target:
            try:
                batch, token = gp_reviews(GROWW_PLAY_ID, lang="en", country="in", sort=sort, count=200,
                                          continuation_token=token)
            except Exception as e:  # noqa: BLE001
                print(f"  play store {tag} stopped at {seen}: {str(e)[:100]}")
                break
            if not batch:
                break
            seen += len(batch)
            for r in batch:
                text = str(r.get("content") or "")
                if len(text) < 40 or not SIP_TERMS.search(text):
                    continue
                out.append(Post(
                    source="play_store", platform_detail="Groww Android",
                    date=r["at"].strftime("%Y-%m-%d") if r.get("at") else "", rating=str(r.get("score", "")),
                    author="", title="", text=text,
                    url=f"https://play.google.com/store/apps/details?id={GROWW_PLAY_ID}&reviewId={r.get('reviewId', '')}",
                    query=f"app_review|{tag}",
                ))
            if not token:
                break
            time.sleep(0.2)
        print(f"  play store {tag}: scanned {seen}, kept {sum(p.query.endswith(tag) for p in out)} SIP reviews")
    return out


def app_store() -> list[Post]:
    rows = [p for p in collect_app_store(500) if SIP_TERMS.search(f"{p.title} {p.text}")]
    for p in rows:
        p.author, p.query = "", "app_review|most_recent"
    return rows


def reddit() -> list[Post]:
    """Unauthenticated RSS is rate-limited to roughly 10 requests a minute; stop after repeated refusals."""
    out: list[Post] = []
    failures = 0
    for branch, qs in QUERIES.items():
        for q in qs:
            for sub in SUBREDDITS:
                if failures >= 6:
                    print("  reddit: rate-limited repeatedly, stopping early")
                    return out
                url = f"https://www.reddit.com/r/{sub}/search.rss?q={quote_plus(q)}&restrict_sr=on&sort=relevance&t=all"
                try:
                    parsed = feedparser.parse(fetch_url(url, retries=2))
                    failures = 0
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"  reddit {sub} '{q}': {str(e)[:80]}")
                    time.sleep(30)
                    continue
                for e in parsed.entries:
                    date = datetime(*e.published_parsed[:6]).strftime("%Y-%m-%d") if getattr(e, "published_parsed", None) else ""
                    out.append(Post(source="reddit", platform_detail=f"r/{sub}", date=date, rating="", author="",
                                    title=getattr(e, "title", ""), text=re.sub(r"<[^>]+>", " ", getattr(e, "summary", "")),
                                    url=getattr(e, "link", ""), query=f"{branch}|{q}"))
                time.sleep(7)
        print(f"  reddit after {branch}: {len(out)}")
    return out


def tradingqna() -> list[Post]:
    out: list[Post] = []
    seen: set[int] = set()
    for branch, qs in QUERIES.items():
        for q in qs:
            for page in (1, 2):
                try:
                    payload = json.loads(fetch_url(f"https://tradingqna.com/search.json?q={quote_plus(q)}&page={page}"))
                except Exception:  # noqa: BLE001
                    break
                topics = {t["id"]: t.get("title", "") for t in payload.get("topics") or []}
                for p in payload.get("posts") or []:
                    tid = p.get("topic_id")
                    if not tid or tid in seen:
                        continue
                    seen.add(tid)
                    out.append(Post(source="tradingqna", platform_detail="TradingQ&A",
                                    date=(p.get("created_at") or "")[:10], rating="", author="",
                                    title=topics.get(tid, ""), text=re.sub(r"\s+", " ", p.get("blurb") or "").strip(),
                                    url=f"https://tradingqna.com/t/{tid}", query=f"{branch}|{q}"))
                time.sleep(0.5)
    print(f"  tradingqna: {len(out)}")
    return out


def youtube(per_video: int = 150) -> list[Post]:
    out: list[Post] = []
    ycd = YoutubeCommentDownloader()
    seen: set[str] = set()
    for branch, qs in YOUTUBE_QUERIES.items():
        for q in qs:
            for vid in youtube_search_video_ids(q, limit=3):
                if vid in seen:
                    continue
                seen.add(vid)
                url = f"https://www.youtube.com/watch?v={vid}"
                try:
                    for i, c in enumerate(ycd.get_comments_from_url(url, sort_by=0)):
                        if i >= per_video:
                            break
                        text = (c.get("text") or "").strip()
                        if len(text) >= 40:
                            out.append(Post(source="youtube", platform_detail="YouTube comments",
                                            date=str(c.get("time", "")), rating="", author="", title="", text=text,
                                            url=f"{url}&lc={c.get('cid', '')}", query=f"{branch}|{q}"))
                except Exception as e:  # noqa: BLE001
                    print(f"  youtube {vid}: {str(e)[:80]}")
    print(f"  youtube: {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="play_store,app_store,reddit,tradingqna,youtube")
    ap.add_argument("--play-newest", type=int, default=30000)
    ap.add_argument("--play-relevant", type=int, default=6000)
    ap.add_argument("--branches", default="", help="comma-separated query branches to run (default: all)")
    args = ap.parse_args()
    if args.branches:
        wanted = set(args.branches.split(","))
        for qs in (QUERIES, YOUTUBE_QUERIES):
            for b in [b for b in qs if b not in wanted]:
                del qs[b]
    steps = {
        "play_store": lambda: play_store(args.play_newest, args.play_relevant),
        "app_store": app_store, "reddit": reddit, "tradingqna": tradingqna, "youtube": youtube,
    }
    for name in args.only.split(","):
        print(f"Collecting {name}…", flush=True)
        save(steps[name]())


def save(new: list[Post]) -> None:
    """Add to what earlier runs collected, so every source is kept as soon as it finishes."""
    import csv

    posts: list[Post] = []
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8") as f:
            posts = [Post(**r) for r in csv.DictReader(f)]
    posts = dedupe(posts + new)
    write_csv(posts, OUTPUT)
    by_source: dict[str, int] = {}
    for p in posts:
        by_source[p.source] = by_source.get(p.source, 0) + 1
    print(f"Wrote {len(posts)} rows to {OUTPUT.name}: {by_source}", flush=True)


if __name__ == "__main__":
    main()
