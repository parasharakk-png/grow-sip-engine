"""Write data/post_dates.csv: the publication date of every corpus post, joined from the source files by URL.

YouTube only gives relative dates ("3 weeks ago"), so those are kept at month precision, counted back from the
day the raw file was collected (its modification date).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ENGINE_DIR = Path(__file__).resolve().parent
ROOT = ENGINE_DIR.parent
CORPUS = ENGINE_DIR / "data" / "grow_sip_956.csv"
BACKUP = ROOT / "NL GROWW - cleaned_SIP_data_paste_into_sheets.backup.csv"
RAW_FILES = (ROOT / "raw_posts.csv", ROOT / "raw_posts_targeted.csv")
OUT = ENGINE_DIR / "data" / "post_dates.csv"

RELATIVE = re.compile(r"(\d+)\s*(minute|hour|day|week|month|year)s?\s*ago", re.I)
UNIT_DAYS = {"minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30, "year": 365}


def _key(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower()


def _raw_dates(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str).fillna("")
    collected = datetime.fromtimestamp(path.stat().st_mtime)

    def parse(d: str) -> tuple[str, str]:
        if re.match(r"^\d{4}-\d{2}", d):
            return d[:10], "day"
        m = RELATIVE.search(d)
        if m:
            when = collected - timedelta(days=int(m.group(1)) * UNIT_DAYS[m.group(2).lower()])
            return when.strftime("%Y-%m"), "month"
        return "", ""

    parsed = raw["date"].map(parse)
    return pd.DataFrame({"u": _key(raw["url"]), "published": parsed.str[0], "precision": parsed.str[1]})


def main() -> None:
    corpus = pd.read_csv(CORPUS, dtype=str).fillna("")
    backup = pd.read_csv(BACKUP, dtype=str).fillna("")
    dates = pd.concat([
        pd.DataFrame({"u": _key(backup["url"]), "published": backup["date_clean"],
                      "precision": backup["date_type"].map({"exact": "day", "relative": "month"}).fillna("day")}),
        *[_raw_dates(p) for p in RAW_FILES if p.exists()],
    ])
    dates = dates[dates["published"].str.match(r"^\d{4}-\d{2}")].drop_duplicates("u")
    out = corpus.assign(u=_key(corpus["url"])).merge(dates, on="u", how="left")[["item_id", "published", "precision"]]
    out.fillna("").to_csv(OUT, index=False)
    print(f"{(out['published'].fillna('') != '').sum()} of {len(out)} posts dated -> {OUT.name}")


if __name__ == "__main__":
    main()
