# AI Evidence Log — Groww SIP Discovery Engine
Generated 03 Nov 2026 from the live data by `discovery-engine/evidence_log.py`. Engine: https://groww-sip-discovery-engine-ve6n6mkspofse9xbwkzrsh.streamlit.app · Code and data: https://github.com/parasharakk-png/groww-sip-discovery-engine

## 1. Sources and volume
The engine analyses **1,917 public posts** (526 from first-time investors, 1,391 from other investors as a comparison group), selected from 4,546 pooled posts. 447 posts describe stopping, pausing, a failed payment or a strong link to lapse; 421 describe continuing.

| Source | Pooled | In corpus | First-time target | Dated | Years | How collected |
|---|---|---|---|---|---|---|
| Play Store | 1674 | 786 | 127 | 100% | 2018–2026 | Unofficial google-play-scraper library (Groww app reviews) |
| Reddit | 941 | 697 | 236 | 100% | 2014–2026 | Public subreddit search RSS feeds, keyword queries |
| TradingQ&A | 974 | 226 | 84 | 100% | 2016–2026 | The forum's public search endpoint, keyword queries |
| YouTube | 893 | 197 | 77 | 100% | 2023–2026 | Unofficial youtube-comment-downloader library |
| App Store | 64 | 11 | 2 | 100% | 2026–2026 | Apple's public customer-reviews RSS feed (Groww app) |
| **Total** | 4546 | 1917 | 526 | 100% |  |  |

**How posts were selected**

| Step | Posts | Method |
|---|---|---|
| 1. Pooled public posts (after cleaning) | 4546 | Scrapers, then cleaned sheet |
| 2. About SIPs / mutual funds | 3131 | Keyword rules |
| 3. Author's own experience | 2305 | First-person rules |
| 4. Duplicates removed | 2209 | Same body text |
| 5. Own experience confirmed by LLM | 1917 | Problem-fit LLM pass rejected 292 |
| 6. First-time investors (target group) | 526 | LLM investor stage; first-time phrase rule where the LLM can't tell |

**AI used at each step**

- **First-pass screening** (is this a first-time investor's own experience?): MiniMax-M2.7-highspeed, 326 candidates.
- **Problem-fit coding** (2,209 posts: investor stage, months since start, SIP status, core concern): MiniMax-M2.7-highspeed through an Anthropic-compatible gateway; prompt versioned and cached.
- **Codebook coding** (lifecycle, trigger, emotion, behaviour, link to lapse, quote): 1,984 posts coded by the LLM (latest pass MiniMax-M2.7-highspeed), cached in checkpoints so reruns reuse them.
- **Answer layer** (Ask the engine): the LLM reasons over a numeric evidence pack and cites post IDs; every question is checked by a rule-based guard first.
- **Not AI:** the cause taxonomy, opportunity sizing and golden-set scoring are fixed, auditable rules.
- **Quote check:** 1,886 AI-extracted quotes were found verbatim in their post; 31 were not.

## 2. Golden set: hand labels vs the engine
**20 posts** (13 for Problem 2 — stopping in a market fall, my priority; 7 for Problem 1 — payment / mandate failure) are labelled against a written codebook with 19 fields. Rows were chosen to be hard: core cases, hard negatives that fool keyword rules, early signals and counter-evidence to my own solution.

- **Hand check:** all 20 labels checked by me, reading each post.
- **Selection screen check:** the pipeline's keep/drop decision matched my *own experience* label on **20 of 20** golden posts. It kept every post I marked as the author's own experience. The tables use 19 rows: post #398 is correctly excluded, and #2547 is scored as its kept duplicate #1405.

**How often the engine agreed with me**

| Problem | What is checked | Engine method | Rows | Agreement | Precision | Recall |
|---|---|---|---|---|---|---|
| P2 | Own portfolio is falling | Text rule (regex) | 12 | 92% | 90% | 100% |
| P2 | Anxious / panicking | Text rule (regex) | 12 | 67% | 60% | 60% |
| P2 | Anxious / panicking | LLM codebook | 12 | 75% | 75% | 60% |
| P2 | Stopping, pausing or reducing | Engine (rules + LLM) | 12 | 75% | 100% | 67% |
| P2 | Stop caused by the fall | Text rule (regex) | 12 | 67% | 57% | 80% |
| P2 | Stop caused by the fall | LLM codebook | 12 | 83% | 80% | 80% |
| P2 | Asks what others are doing | Text rule (regex) | 12 | 50% | 20% | 33% |
| P2 | Why they stop (driver) | Engine (rules + LLM) | 9 | 33% | — | — |
| P1 | Instalment failed | Text rule (regex) | 7 | 57% | 75% | 60% |
| P1 | Instalment failed | LLM codebook | 7 | 86% | 83% | 100% |
| P1 | Cause of failure | Text rule (regex) | 5 | 40% | — | — |
| P1 | Job / bank / salary-account change | Text rule (regex) | 7 | 71% | 50% | 50% |

**Where it failed** (lowest agreement first, with the labelling rationale)

- **Why they stop (driver) — Engine (rules + LLM): 33% agreement on 9 rows.**
  - Post #279: hand label *fall_emotion*, engine *fund_specific*. First-time investor (2.5 yrs) whose XIRR fell from 16% to 0.6%. Long-term plan stated, but the recent fall overrides it. Asks for a decision, not what others do; a '% still continuing' sheet helps a little but does not explain why XIRR swings.
  - Post #1033: hand label *fall_emotion*, engine *market_timing*. The clearest case for the proposed nudge: portfolio 'bleeding', first instinct is to pause, and explicitly wants to know what other investors are doing right now.
- **Cause of failure — Text rule (regex): 40% agreement on 5 rows.**
  - Post #35: hand label *platform_technical*, engine *setup_incomplete*. Autopay failure with no retry. The workaround (manual payment) only works if the user opens the app.
  - Post #895: hand label *insufficient_balance*, engine *bank_account_issue*. Salary lands in one bank, SIPs debit from another. A manual transfer was delayed by bank holidays and failed SIPs cost over Rs 3,000 in charges.
- **Asks what others are doing — Text rule (regex): 50% agreement on 12 rows.**
  - Post #307: hand label *no*, engine *yes*. Fear of a future recession, not a fall that has happened. The push comes from a trusted family member. A stranger-statistic nudge competes with dad's 20 years of experience, so only partial fit.
  - Post #1027: hand label *no*, engine *yes*. No fall at all (45% up). The market-fall pattern matches only because of the word 'correction'. A wish to time dips, not a panic stop.
- **Instalment failed — Text rule (regex): 57% agreement on 7 rows.**
  - Post #259: hand label *no*, engine *yes*. The exact Problem 1 moment: bank changed, old mandate removed, and no way to see whether the new mandate is live. No failure yet, so at risk. The engine's bank-change pattern misses 'change my primary bank'.
  - Post #1405: hand label *yes*, engine *no*. First-timer whose SIP never ran for 6 months because a setup step was not explained.

**What the golden set changed**

- Of 9 hand-labelled stops in a fall, only 4 are fear or an expectation gap; 2 are market timing, 1 fund-specific and 1 cash needs. "Panic" was too narrow.
- My original nudge (share of investors continuing in this fund) fully fits 1 posts and partly fits 4; it does not fit 4.
- Of 5 payment failures, 2 were not notified in time and 6 were fixable by the user with the right information.

## 3. Where I rejected or corrected AI output

### 1. The LLM's stop trigger could not explain stopping — replaced with an auditable cause taxonomy
- **AI output:** the codebook LLM gives each post one trigger. For 82 of 447 posts about stopping (18%) it said *none obvious / other / unknown*, and the stated ones were broad (top: platform_bug, failed_debit, mandate_autopay_issue).
- **Why I rejected it:** a vague label per post can't separate a failed mandate from a fund switch or a cash need, so trigger counts said nothing about *why* SIPs end.
- **Correction:** a fixed, versioned text-pattern taxonomy of 10 causes (reasoning.py) assigns each post a primary cause; it covers 272 of 447 stop posts. Patterns are visible in the app and scored against the golden set.

### 2. The AI's reading of why people stop in a fall didn't match the posts — not used for sizing
- **AI output:** the LLM codebook found only 80% of the stops that a fall caused (text rules: 80%), and the engine's stop driver matched the hand labels on only 33% of stops, mixing up fear, market timing and fund-specific reasons.
- **Why I rejected it:** sizing from the LLM label would under-count fall-driven stops, and a driver that can't tell fear from timing can't say whether 'panic' is the problem. Reading the posts, fear was a minority (4 of 9 stops, with expectation gaps); the common thread was having no way to judge whether the SIP was on track.
- **Correction:** sizing uses the text-pattern causes, not the LLM's fall-causation label; Problem 2 is reframed from "panic" to "no way to judge my own SIP at the stop moment", and fund-level social proof was dropped from the solution.

### 3. The AI answered personal-advice and support questions — added a question guard
- **AI output:** when I tested it with real user questions — *"Should I stop SIP?"*, *"Which mutual fund to buy?"*, *"Which bank account should I open?"*, *"OTP is not coming"* — the answer layer answered them from the evidence pack instead of declining.
- **Why I rejected it:** the brief forbids fund recommendations, and an evidence engine has nothing valid to say about one person's account.
- **Correction:** a rule-based guard classifies every question first (research, advice, support, unsafe, off-topic, prevalence) and declines with a reason and a research reframe; 58 of 58 test questions pass (6 kinds). The LLM also has a refusal step, and below 3 matching posts the engine abstains.

## 4. Limitations
- Public posts are not a random sample of investors; shares are of posts, not people.
- Few first-time investors post about stopping (68 posts), so first-time splits are hypotheses for the interviews, not measurements.
- Uplift estimates combine post shares with fixed assumptions (40% lapse rate, 30% recovery, addressability per opportunity); Groww cohort data would replace them.
- The AI selection screen loses some real stories (0 of 19 in the golden set), so counts are slightly conservative.
- The golden set is small (20 rows) and checked by one person.
