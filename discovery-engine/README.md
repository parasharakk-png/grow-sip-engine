# Groww SIP Discovery Engine

Reasoned discovery (Streamlit) over **499 LLM-coded public posts**: 255 from first-time and young (≤30) SIP / mutual fund investors describing their own situation (target group), plus 244 from other investors (comparison group).

## Run locally

```bash
cd discovery-engine
pip install -r requirements.txt
cp .env.example .env          # then set ANTHROPIC_API_KEY (needed only for AI reasoning / re-coding)
python3 -m streamlit run app.py
```

Ask a question in **Ask the engine** (or click one of the nine discovery questions). Answers need at least 3 matching posts and are labelled a small sample below 30; every answer shows the dataset, rules and prompt versions it came from. **Computed reasoning** is instant and needs no key; **AI reasoning** streams from `ANTHROPIC_MODEL` (use `MiniMax-M2.7-highspeed` on the PenPencil gateway).

## Rebuild the corpus

```bash
python3 build_corpus.py                  # re-apply filters using saved LLM results (offline, seconds)
python3 build_corpus.py --screen --code  # LLM-screen new candidates + LLM-code new posts (network, minutes)
```

Pipeline over 2,669 pooled posts (backup cleaned sheet + extra `raw_posts.csv`):

1. **About SIPs / mutual funds** — not only trading, IPO, crypto or loans.
2. **Author's own experience** — first-person ("I/my/main/mera"); not news, guides or promos. App reviews always count.
3. **First-time or young** — phrases like "my first SIP", "beginner", "started 3 months ago", "I'm 23", "first job"; a stated age 31+ or "investing since 2018" does not count.
4. **LLM screen** — confirms it is the author's own story and vetoes experienced / over-30 investors; where the LLM says "unknown", an explicit rule match decides.

Outputs:

- `data/grow_sip_956.csv` — the file the app reads (`DATA_CSV` in `app.py`): the target corpus plus the earlier posts as a comparison group (tagged `SELECTION: comparison`). Rebuilt on every run from `data/grow_sip_956.original.csv` (your original 284 rows, kept untouched).
- `data/corpus_first_time.csv` — the target group only (first-time / young investors).
- `data/selection_log.csv` — every pooled post with its decision, matched phrases and the LLM verdict/reason (evidence log for selection).
- `data/screening.checkpoint.jsonl`, `data/corpus_first_time.checkpoint.jsonl` — saved LLM results, so reruns resume.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit UI |
| `reasoning.py` | Question routing, cause taxonomy, lift, opportunity sizing, evidence pack |
| `rule_based_narrative.py` | Computed (non-AI) answer |
| `hypotheses.py` | Tests the two PM problem hypotheses (evidence for / against, mechanism, reframing) |
| `golden_set_eval.py`, `data/golden_set.csv` | Hand-labelled golden set (35 posts, both hypotheses) and engine-vs-human scoring; run `python3 golden_set_eval.py` |
| `question_guard.py`, `data/question_guard_tests.csv` | Checks each question before answering: declines personal advice, account support, fear/FOMO tactics, off-topic and "how common among all investors" questions, and flags "would this feature help?" questions; run `python3 question_guard.py` |
| `build_post_dates.py`, `data/post_dates.csv` | Publication date of every corpus post, joined by URL from the source files; rerun after rebuilding the corpus |
| `llm_client.py` | Gateway client + streaming for AI reasoning |
| `discovery_analytics.py`, `plain_labels.py` | Metric branches and plain-English labels |
| `build_corpus.py` | Builds the dataset (filters, LLM screen, merge) |
| `code_posts.py`, `codebook.json` | Codebook-based LLM coding used by `build_corpus.py --code` |
| `collect.py` | Original scraper that produced `../raw_posts.csv` |
| `coded_posts.checkpoint.jsonl` | Saved LLM codes reused by `build_corpus.py` |

Inputs outside this folder: `../NL GROWW - cleaned_SIP_data_paste_into_sheets.backup.csv` and `../raw_posts.csv`.

## Share with mentors (Streamlit Community Cloud)

1. Push this repo to GitHub (keep `.env` out of git).
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → repo `NextLeapProject`.
3. **Main file path:** `discovery-engine/app.py`
4. **Secrets** (optional, for AI reasoning only):

```toml
ANTHROPIC_API_KEY = "..."
ANTHROPIC_BASE_URL = "https://llm-gateway-azure.penpencil.guru/anthropic"
ANTHROPIC_MODEL = "MiniMax-M2.7-highspeed"
```

The app works without secrets; it then shows computed reasoning only.
