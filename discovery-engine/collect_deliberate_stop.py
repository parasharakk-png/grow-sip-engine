#!/usr/bin/env python3
"""Pull at most 200 posts about deliberately stopping a SIP.

Reasons covered: a market fall, flat or low returns, moving to another asset,
and frustration with averaging down. Written to data/deliberate_stop_candidates.csv. build_corpus.py reads this file,
runs the same filters and coding as the rest of the pool, and merges what it keeps.

    python3 collect_deliberate_stop.py
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
from google_play_scraper import Sort, reviews as gp_reviews
from youtube_comment_downloader import YoutubeCommentDownloader

from collect import GROWW_PLAY_ID, Post, collect_app_store, dedupe, fetch_url, write_csv, youtube_search_video_ids

OUT = Path(__file__).resolve().parent / "data" / "deliberate_stop_candidates.csv"
CAP = 200
QUERIES = [
    "stopped sip market crash",
    "stop sip flat returns",
    "stopped sip switched to stocks",
    "tired of averaging sip loss",
]
SUBS = ["mutualfunds", "IndiaInvestments", "personalfinanceindia"]
YT = ["should I stop SIP market fall", "SIP flat returns stop or continue", "stop SIP invest in stocks instead"]
STOP = re.compile(r"stop|stopp|pause|paus|cancel|quit|discontinu|switch|moved|shifting|gave up|fed up|tired", re.I)
WHY = re.compile(r"market|crash|fall|down|negative|loss|flat|return|averag|gold|stock|fd\b|asset|underperform", re.I)
SIP = re.compile(r"\bsips?\b|mutual fund", re.I)


def keep(text: str) -> bool:
    return bool(SIP.search(text) and STOP.search(text) and WHY.search(text) and len(text) >= 40)


def reddit() -> list[Post]:
    out: list[Post] = []
    for q in QUERIES:
        for sub in SUBS:
            url = f"https://www.reddit.com/r/{sub}/search.rss?q={quote_plus(q)}&restrict_sr=on&sort=relevance&t=all"
            try:
                parsed = feedparser.parse(fetch_url(url, retries=2))
            except Exception as e:  # noqa: BLE001
                print(f"  reddit {sub} '{q}': {str(e)[:80]}")
                time.sleep(20)
                continue
            for e in parsed.entries:
                text = re.sub(r"<[^>]+>", " ", getattr(e, "summary", ""))
                title = getattr(e, "title", "")
                if not keep(f"{title} {text}"):
                    continue
                date = datetime(*e.published_parsed[:6]).strftime("%Y-%m-%d") if getattr(e, "published_parsed", None) else ""
                out.append(Post("reddit", f"r/{sub}", date, "", "", title, text, getattr(e, "link", ""), q))
            time.sleep(7)
        print(f"  reddit after '{q}': {len(out)}")
    return out


def tradingqna() -> list[Post]:
    out, seen = [], set()
    for q in QUERIES:
        try:
            payload = json.loads(fetch_url(f"https://tradingqna.com/search.json?q={quote_plus(q)}"))
        except Exception as e:  # noqa: BLE001
            print(f"  tradingqna '{q}': {str(e)[:80]}")
            continue
        titles = {t["id"]: t.get("title", "") for t in payload.get("topics") or []}
        for p in payload.get("posts") or []:
            tid = p.get("topic_id")
            if not tid or tid in seen:
                continue
            text = re.sub(r"\s+", " ", p.get("blurb") or "")
            if not keep(f"{titles.get(tid, '')} {text}"):
                continue
            seen.add(tid)
            out.append(Post("tradingqna", "TradingQ&A", (p.get("created_at") or "")[:10], "", "",
                            titles.get(tid, ""), text, f"https://tradingqna.com/t/{tid}", q))
        time.sleep(0.5)
    print(f"  tradingqna: {len(out)}")
    return out


def youtube() -> list[Post]:
    out, seen = [], set()
    ycd = YoutubeCommentDownloader()
    for q in YT:
        for vid in youtube_search_video_ids(q, limit=2):
            if vid in seen:
                continue
            seen.add(vid)
            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                for i, c in enumerate(ycd.get_comments_from_url(url, sort_by=0)):
                    if i >= 80:
                        break
                    text = (c.get("text") or "").strip()
                    if keep(text):
                        out.append(Post("youtube", "YouTube comments", str(c.get("time", "")), "", "", "",
                                        text, f"{url}&lc={c.get('cid', '')}", q))
            except Exception as e:  # noqa: BLE001
                print(f"  youtube {vid}: {str(e)[:80]}")
    print(f"  youtube: {len(out)}")
    return out


def stores() -> list[Post]:
    out = []
    token, seen = None, 0
    while seen < 2000:
        try:
            batch, token = gp_reviews(GROWW_PLAY_ID, lang="en", country="in", sort=Sort.NEWEST, count=200,
                                      continuation_token=token)
        except Exception as e:  # noqa: BLE001
            print(f"  play store stopped at {seen}: {str(e)[:80]}")
            break
        if not batch:
            break
        seen += len(batch)
        for r in batch:
            text = str(r.get("content") or "")
            if keep(text):
                out.append(Post("play_store", "Groww Android",
                                r["at"].strftime("%Y-%m-%d") if r.get("at") else "", str(r.get("score", "")),
                                "", "", text,
                                f"https://play.google.com/store/apps/details?id={GROWW_PLAY_ID}&reviewId={r.get('reviewId', '')}",
                                "play_newest"))
        if not token:
            break
        time.sleep(0.2)
    print(f"  play store: scanned {seen}, kept {len(out)}")
    app = []
    for p in collect_app_store(300):
        if keep(f"{p.title} {p.text}"):
            p.query = "app_store"
            app.append(p)
    print(f"  app store: kept {len(app)}")
    return out + app


def main() -> None:
    posts: list[Post] = []
    for step in (reddit, tradingqna, youtube, stores):
        posts.extend(step())
        if len(dedupe(posts)) >= CAP:
            break
    posts = dedupe(posts)[:CAP]
    write_csv(posts, OUT)
    print(f"wrote {len(posts)} to {OUT}")


if __name__ == "__main__":
    main()
