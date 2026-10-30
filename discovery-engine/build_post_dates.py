"""Write data/post_dates.csv: the publication date of every corpus post, joined from the source files by URL.

YouTube only gives relative dates ("3 weeks ago"), so those are kept at month precision.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ENGINE_DIR = Path(__file__).resolve().parent
ROOT = ENGINE_DIR.parent
CORPUS = ENGINE_DIR / "data" / "grow_sip_956.csv"
BACKUP = ROOT / "NL GROWW - cleaned_SIP_data_paste_into_sheets.backup.csv"
RAW = ROOT / "raw_posts.csv"
OUT = ENGINE_DIR / "data" / "post_dates.csv"


def _key(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower()


def main() -> None:
    corpus = pd.read_csv(CORPUS, dtype=str).fillna("")
    backup = pd.read_csv(BACKUP, dtype=str).fillna("")
    raw = pd.read_csv(RAW, dtype=str).fillna("")
    dates = pd.concat([
        pd.DataFrame({"u": _key(backup["url"]), "published": backup["date_clean"],
                      "precision": backup["date_type"].map({"exact": "day", "relative": "month"}).fillna("day")}),
        pd.DataFrame({"u": _key(raw["url"]), "published": raw["date"], "precision": "day"}),
    ])
    dates = dates[dates["published"].str.match(r"^\d{4}-\d{2}")].drop_duplicates("u")
    out = corpus.assign(u=_key(corpus["url"])).merge(dates, on="u", how="left")[["item_id", "published", "precision"]]
    out.fillna("").to_csv(OUT, index=False)
    print(f"{(out['published'].fillna('') != '').sum()} of {len(out)} posts dated -> {OUT.name}")


if __name__ == "__main__":
    main()
