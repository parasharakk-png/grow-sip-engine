# Groww SIP Discovery Engine

A Streamlit app that tests two problem statements against **1,907 AI-coded public posts** (495 from first-time SIP investors, 1,412 from other investors as a comparison group) and **5 user interviews**.

Goal metric: % of first-time mutual fund investors with an active SIP at 12 months. No fund advice; no fear, FOMO or urgency.

- **Problem 1 · Failed payments:** first-time investors lose SIPs they meant to keep: the first SIP never goes live, or a later instalment fails, and they aren't told why or how to fix it.
- **Problem 2 · The stop decision:** in months 6–12, first-time investors judge their SIP only by red or green against a one-year expectation. With no way to tell what's normal, they ask whoever got them started, then stop, reduce, or move the money out of SIPs.

**Focus decision:** the stop decision was the risky moment in 5 of 5 interviews and a failed instalment in 0 of 5, so the MVP builds on Problem 2. Failed payments stay a parallel operations track.

## Run locally

```bash
cd discovery-engine
pip install -r requirements.txt
python3 -m streamlit run app.py
```

The app needs no API key: every answer is computed from the coded posts with auditable rules. A key (`ANTHROPIC_API_KEY` in `.env`, see `.env.example`) is needed only to screen or code new posts when rebuilding the corpus.

## Tabs

| Tab | What it shows |
|---|---|
| **Overview** | The two problem statements, their share of posts about stopping, estimated uplift, interview result and the focus decision |
| **Metric decomposition** | 12-month continuation broken into product outcomes, behaviours and leading indicators, sized for all vs first-time investors |
| **Problem validation** | Each original hypothesis tested against the posts and the interviews, ending in the problem statements and final prioritisation |
| **Ask the engine** | Ask a discovery question and get a computed answer with its evidence posts. Answers need at least 3 matching posts and are labelled a small sample below 30. Questions asking for advice, fear/FOMO tactics or "how common among all investors" are declined |
| **Opportunity sizing** | Opportunities compared by uplift, stoppers vs continuers and addressability; a deep dive per opportunity; metric branches |
| **How it works & evidence** | The 5-step pipeline, sources and volume, and agreement with the hand-labelled golden set |
| **Evidence explorer** | Every coded post, with filters |

Sizes assume 40% of first SIPs lapse by month 12 and 30% of each cause is recoverable. The full selection funnel, data-quality checks, codebook and golden-set disagreements are in the AI Evidence Log (`python3 evidence_log.py`).

## Rebuild the corpus

```bash
python3 build_corpus.py                  # re-apply filters using saved LLM results (offline, seconds)
python3 build_corpus.py --screen --code  # LLM-screen new candidates + LLM-code new posts (network, minutes)
```

Pipeline over 4,522 pooled posts:

1. **Collect** — Play Store, App Store, Reddit, YouTube and TradingQ&A (`collect.py`, `collect_targeted.py`).
2. **Select** — about SIPs / mutual funds; the author's own experience; duplicates dropped; first-time investors identified by phrases ("my first SIP", "started 3 months ago") and an LLM screen.
3. **Code** — each post tagged with `codebook.json` (journey stage, SIP status, behaviour, link to lapse) by an LLM.
4. **Find causes** — text rules give each post one cause, and each payment post one sub-cause (`reasoning.py`).
5. **Size & validate** — causes → opportunities → estimated uplift; checked against the golden set and the interviews.

Outputs:

- `data/grow_sip_956.csv` — the file the app reads (`DATA_CSV` in `app.py`).
- `data/corpus_first_time.csv` — the first-time investor posts only.
- `data/selection_log.csv` — every pooled post with its selection decision and the LLM verdict.
- `data/*.checkpoint.jsonl` — saved LLM results, so reruns resume without calling the model again.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit UI |
| `reasoning.py` | Question routing, cause and payment sub-cause rules, lift, opportunity sizing |
| `rule_based_narrative.py` | Computed answer text |
| `hypotheses.py` | Tests the two problem hypotheses against the posts (evidence for / against, sub-causes) |
| `interviews.py` | Reads the anonymised interview scorecard and screener (`../user-research/`) and applies the focus decision rule |
| `metric_decomposition.py`, `exports/` | Metric tree and sizing; run `python3 metric_decomposition.py` to regenerate the slide SVG |
| `golden_set_eval.py`, `data/golden_set.csv` | Hand-labelled golden set (35 posts) and engine-vs-human scoring; run `python3 golden_set_eval.py` |
| `question_guard.py`, `data/question_guard_tests.csv` | Checks each question before answering; run `python3 question_guard.py` |
| `evidence_log.py` | Writes the AI Evidence Log from the live data |
| `build_corpus.py`, `code_posts.py`, `problem_fit.py`, `codebook.json` | Corpus build: filters, LLM screening and coding |
| `build_post_dates.py`, `data/post_dates.csv` | Publication date of every post |
| `collect.py`, `collect_targeted.py` | Scrapers |
| `discovery_analytics.py`, `plain_labels.py` | Metric branches and plain-English labels |
| `llm_client.py` | LLM gateway client (not used by the app) |

## Share with mentors (Streamlit Community Cloud)

1. Push this repo to GitHub (keep `.env` out of git).
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → repo `NextLeapProject`.
3. **Main file path:** `discovery-engine/app.py`

No secrets are needed to run the app.
