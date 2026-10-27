#!/usr/bin/env python3
"""Collect public posts about SIP / first-time investing for Groww discovery engine."""

from __future__ import annotations

import csv
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

import json
import urllib.request

import feedparser
import requests
from google_play_scraper import Sort, reviews as gp_reviews
from youtube_comment_downloader import YoutubeCommentDownloader

OUTPUT = Path(__file__).resolve().parent.parent / "raw_posts.csv"
GROWW_PLAY_ID = "com.nextbillion.groww"
GROWW_APPSTORE_ID = "1404871703"  # Groww Stocks, Mutual Fund, IPO

REDDIT_SUBREDDITS = [
    "mutualfunds",
    "IndiaInvestments",
    "personalfinanceindia",
    "IndianStreetBets",
    "FIREIndia",
]
REDDIT_QUERIES = [
    "stop sip",
    "pause sip",
    "sip failed",
    "first sip",
    "groww sip",
    "sip autopay",
    "cancel sip",
    "sip not deducted",
]

TRADINGQNA_QUERIES = [
    "stop sip",
    "pause sip",
    "sip failed",
    "first sip",
    "groww sip",
    "sip autopay",
    "cancel sip",
    "sip mandate",
]

YOUTUBE_VIDEO_QUERIES = [
    "should I stop SIP mutual fund",
    "pause SIP groww",
    "SIP failed payment",
    "first SIP mutual fund India",
    "SIP kab band kare",
    "mutual fund SIP mistakes",
    "Groww SIP tutorial",
    "SIP autopay UPI",
]

NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class Post:
    source: str
    platform_detail: str
    date: str
    rating: str
    author: str
    title: str
    text: str
    url: str
    query: str

    def dedupe_key(self) -> str:
        if self.url:
            return self.url.strip().lower()
        blob = f"{self.source}|{self.title}|{self.text[:200]}"
        return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    return s


def collect_play_store(target: int = 4000) -> list[Post]:
    rows: list[Post] = []
    token = None
    while len(rows) < target:
        batch, token = gp_reviews(
            GROWW_PLAY_ID,
            lang="en",
            country="in",
            sort=Sort.NEWEST,
            count=min(200, target - len(rows)),
            continuation_token=token,
        )
        if not batch:
            break
        for r in batch:
            rows.append(
                Post(
                    source="play_store",
                    platform_detail="Groww Android",
                    date=r.get("at", datetime.now(timezone.utc)).strftime("%Y-%m-%d")
                    if r.get("at")
                    else "",
                    rating=str(r.get("score", "")),
                    author=str(r.get("userName", "")),
                    title="",
                    text=str(r.get("content", "")),
                    url=f"https://play.google.com/store/apps/details?id={GROWW_PLAY_ID}&reviewId={r.get('reviewId', '')}",
                    query="groww_app_reviews",
                )
            )
        if not token:
            break
        time.sleep(0.3)
    return rows[:target]


def collect_app_store(target: int = 500) -> list[Post]:
    rows: list[Post] = []
    sess = session()
    for page in range(1, 11):
        url = (
            f"https://itunes.apple.com/in/rss/customerreviews/"
            f"id={GROWW_APPSTORE_ID}/sortby=mostrecent/page={page}/json"
        )
        resp = sess.get(url, timeout=30)
        if resp.status_code != 200:
            break
        data = resp.json()
        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            break
        # First entry on page 1 is app metadata, not a review
        start = 1 if page == 1 and entries and "im:rating" not in entries[0] else 0
        for e in entries[start:]:
            if len(rows) >= target:
                break
            rating = ""
            if "im:rating" in e:
                rating = str(e["im:rating"].get("label", ""))
            title = e.get("title", {}).get("label", "")
            text = e.get("content", {}).get("label", "")
            author = e.get("author", {}).get("name", {}).get("label", "")
            link = ""
            raw_links = e.get("link", [])
            if isinstance(raw_links, dict):
                raw_links = [raw_links]
            for link_obj in raw_links:
                if not isinstance(link_obj, dict):
                    continue
                attrs = link_obj.get("attributes", {})
                if attrs.get("rel") == "related":
                    link = attrs.get("href", "")
            updated = e.get("updated", {}).get("label", "")[:10]
            rows.append(
                Post(
                    source="app_store",
                    platform_detail="Groww iOS",
                    date=updated,
                    rating=rating,
                    author=author,
                    title=title,
                    text=text,
                    url=link or f"https://apps.apple.com/in/app/id{GROWW_APPSTORE_ID}",
                    query="groww_app_reviews",
                )
            )
        time.sleep(0.5)
    return rows[:target]


def fetch_url(url: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
            if "429" in str(exc) and attempt < retries - 1:
                time.sleep(8 * (attempt + 1))
                continue
            raise
    raise last_err or RuntimeError("fetch failed")


def collect_reddit() -> list[Post]:
    rows: list[Post] = []
    for sub in REDDIT_SUBREDDITS:
        for q in REDDIT_QUERIES:
            feed_url = (
                f"https://www.reddit.com/r/{sub}/search.rss?"
                f"q={quote_plus(q)}&restrict_sr=on&sort=relevance&t=all"
            )
            try:
                parsed = feedparser.parse(fetch_url(feed_url))
            except Exception:
                continue
            for entry in parsed.entries:
                published = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                text = re.sub(r"<[^>]+>", "", summary)
                link = getattr(entry, "link", "")
                author = getattr(entry, "author", "")
                rows.append(
                    Post(
                        source="reddit",
                        platform_detail=f"r/{sub}",
                        date=published,
                        rating="",
                        author=author,
                        title=title,
                        text=text,
                        url=link,
                        query=q,
                    )
                )
            time.sleep(6)
    return rows


def collect_tradingqna() -> list[Post]:
    rows: list[Post] = []
    base = "https://tradingqna.com"
    seen_topics: set[int] = set()
    for q in TRADINGQNA_QUERIES:
        page = 1
        per_query = 0
        while per_query < 50 and page <= 5:
            search_url = f"{base}/search.json?q={quote_plus(q)}&page={page}"
            try:
                payload = json.loads(fetch_url(search_url).decode("utf-8"))
            except Exception:
                break
            posts = payload.get("posts") or []
            if not posts:
                break
            for p in posts:
                topic_id = p.get("topic_id")
                if not topic_id or topic_id in seen_topics:
                    continue
                seen_topics.add(topic_id)
                created = (p.get("created_at") or "")[:10]
                blurb = re.sub(r"\s+", " ", (p.get("blurb") or "")).strip()
                topic_url = f"{base}/t/{topic_id}"
                rows.append(
                    Post(
                        source="tradingqna",
                        platform_detail="TradingQ&A",
                        date=created,
                        rating="",
                        author=str(p.get("username", "")),
                        title="",
                        text=blurb,
                        url=topic_url,
                        query=q,
                    )
                )
                per_query += 1
                if per_query >= 50:
                    break
            page += 1
            time.sleep(0.4)
    return rows[:350]


def youtube_search_video_ids(query: str, limit: int = 3) -> list[str]:
    sess = session()
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    try:
        html = sess.get(url, timeout=30).text
    except requests.RequestException:
        return []
    ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
    seen: list[str] = []
    for vid in ids:
        if vid not in seen:
            seen.append(vid)
        if len(seen) >= limit:
            break
    return seen


def collect_youtube() -> list[Post]:
    rows: list[Post] = []
    ycd = YoutubeCommentDownloader()
    video_ids: list[str] = []
    for q in YOUTUBE_VIDEO_QUERIES:
        video_ids.extend(youtube_search_video_ids(q, limit=3))
    # Dedupe video ids
    unique_vids = list(dict.fromkeys(video_ids))[:25]

    for vid in unique_vids:
        video_url = f"https://www.youtube.com/watch?v={vid}"
        try:
            comments = ycd.get_comments_from_url(video_url, sort_by=1)
        except Exception:
            continue
        count = 0
        for c in comments:
            text = (c.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                Post(
                    source="youtube",
                    platform_detail="YouTube comments",
                    date=str(c.get("time", "")),
                    rating=str(c.get("votes", "")),
                    author=str(c.get("author", "")),
                    title="",
                    text=text,
                    url=video_url,
                    query="sip_investing_comments",
                )
            )
            count += 1
            if count >= 120:
                break
        time.sleep(0.5)
    return rows


def dedupe(posts: Iterable[Post]) -> list[Post]:
    seen: set[str] = set()
    out: list[Post] = []
    for p in posts:
        key = p.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def write_csv(posts: list[Post], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "platform_detail",
        "date",
        "rating",
        "author",
        "title",
        "text",
        "url",
        "query",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in posts:
            w.writerow(asdict(p))


def main() -> None:
    print("Collecting Play Store reviews…")
    play = collect_play_store(4000)
    print(f"  {len(play)} rows")

    print("Collecting App Store reviews…")
    app = collect_app_store(500)
    print(f"  {len(app)} rows")

    print("Collecting Reddit (RSS)…")
    reddit = collect_reddit()
    print(f"  {len(reddit)} rows")

    print("Collecting TradingQ&A…")
    tqna = collect_tradingqna()
    print(f"  {len(tqna)} rows")

    print("Collecting YouTube comments…")
    yt = collect_youtube()
    print(f"  {len(yt)} rows")

    all_posts = dedupe(play + app + reddit + tqna + yt)
    write_csv(all_posts, OUTPUT)
    print(f"Wrote {len(all_posts)} unique rows to {OUTPUT}")


if __name__ == "__main__":
    main()
