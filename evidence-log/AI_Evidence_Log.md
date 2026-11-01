# AI Evidence Log — Groww SIP Discovery Engine
Generated 01 Nov 2026 from the live data by `discovery-engine/evidence_log.py`. Engine: https://groww-sip-discovery-engine-ve6n6mkspofse9xbwkzrsh.streamlit.app · Code and data: https://github.com/parasharakk-png/groww-sip-discovery-engine

## 1. Sources and volume
The engine analyses **1,907 public posts** (495 from first-time investors, 1,412 from other investors as a comparison group), selected from 4,522 pooled posts. 414 posts describe stopping, pausing, a failed payment or a strong link to lapse; 376 describe continuing.

| Source | Pooled | In corpus | First-time target | Dated | Years | How collected |
|---|---|---|---|---|---|---|
| Play Store | 1674 | 786 | 127 | 100% | 2018–2026 | Unofficial google-play-scraper library (Groww app reviews) |
| Reddit | 919 | 687 | 205 | 100% | 2014–2026 | Public subreddit search RSS feeds, keyword queries |
| TradingQ&A | 973 | 226 | 84 | 100% | 2016–2026 | The forum's public search endpoint, keyword queries |
| YouTube | 892 | 197 | 77 | 100% | 2023–2026 | Unofficial youtube-comment-downloader library |
| App Store | 64 | 11 | 2 | 100% | 2026–2026 | Apple's public customer-reviews RSS feed (Groww app) |
| **Total** | 4522 | 1907 | 495 | 100% |  |  |

**How posts were selected**

| Step | Posts | Method |
|---|---|---|
| 1. Pooled public posts (after cleaning) | 4522 | Scrapers, then cleaned sheet |
| 2. About SIPs / mutual funds | 3108 | Keyword rules |
| 3. Author's own experience | 2284 | First-person rules |
| 4. Duplicates removed | 2188 | Same body text |
| 5. Own experience confirmed by LLM | 1907 | Problem-fit LLM pass rejected 281 |
| 6. First-time investors (target group) | 495 | LLM investor stage; first-time phrase rule where the LLM can't tell |

**AI used at each step**

- **First-pass screening** (is this a first-time investor's own experience?): MiniMax-M2.7-highspeed, 326 candidates.
- **Problem-fit coding** (1,927 posts: investor stage, months since start, SIP status, core concern): MiniMax-M2.7-highspeed through an Anthropic-compatible gateway; prompt versioned and cached.
- **Codebook coding** (lifecycle, trigger, emotion, behaviour, link to lapse, quote): 1,803 posts coded by the LLM (latest pass MiniMax-M2.7-highspeed), cached in checkpoints so reruns reuse them.
- **Answer layer** (Ask the engine): the LLM reasons over a numeric evidence pack and cites post IDs; every question is checked by a rule-based guard first.
- **Not AI:** the cause taxonomy, opportunity sizing and golden-set scoring are fixed, auditable rules.
- **Quote check:** 1,715 AI-extracted quotes were found verbatim in their post; 27 were not.

## 2. Golden set: hand labels vs the engine
**35 posts** (22 for Problem 2 — stopping in a market fall, my priority; 13 for Problem 1 — payment / mandate failure) are labelled against a written codebook with 19 fields. Rows were chosen to be hard: core cases, hard negatives that fool keyword rules, early signals and counter-evidence to my own solution.

- **Hand check:** labels were drafted with AI help; **0 of 35** checked by me so far (column `human_verified`). Scores below are against the current labels.
- 5 golden posts (#936, #920, #398, #2547, #1106) left the corpus in the latest rebuild and are excluded from scoring.

**How often the engine agreed with me**

| Problem | What is checked | Engine method | Rows | Agreement | Precision | Recall |
|---|---|---|---|---|---|---|
| P2 | Own portfolio is falling | Text rule (regex) | 19 | 90% | 94% | 94% |
| P2 | Anxious / panicking | Text rule (regex) | 19 | 74% | 67% | 57% |
| P2 | Anxious / panicking | LLM codebook | 19 | 74% | 67% | 57% |
| P2 | Stopping, pausing or reducing | Engine (rules + LLM) | 19 | 58% | 100% | 43% |
| P2 | Stop caused by the fall | Text rule (regex) | 19 | 63% | 58% | 78% |
| P2 | Stop caused by the fall | LLM codebook | 19 | 68% | 80% | 44% |
| P2 | Asks what others are doing | Text rule (regex) | 19 | 63% | 33% | 40% |
| P2 | Why they stop (driver) | Engine (rules + LLM) | 14 | 29% | — | — |
| P1 | Instalment failed | Text rule (regex) | 11 | 73% | 78% | 88% |
| P1 | Instalment failed | LLM codebook | 11 | 64% | 75% | 75% |
| P1 | Cause of failure | Text rule (regex) | 8 | 50% | — | — |
| P1 | Job / bank / salary-account change | Text rule (regex) | 11 | 82% | 67% | 67% |

**Where it failed** (lowest agreement first, with the labelling rationale)

- **Why they stop (driver) — Engine (rules + LLM): 29% agreement on 14 rows.**
  - Post #279: hand label *fall_emotion*, engine *fund_specific*. First-time investor (2.5 yrs) whose XIRR fell from 16% to 0.6%. Long-term plan stated, but the recent fall overrides it. Asks for a decision, not what others do; a '% still continuing' sheet helps a little but does not explain why XIRR swings.
  - Post #1033: hand label *fall_emotion*, engine *market_timing*. The clearest case for the proposed nudge: portfolio 'bleeding', first instinct is to pause, and explicitly wants to know what other investors are doing right now.
- **Cause of failure — Text rule (regex): 50% agreement on 8 rows.**
  - Post #35: hand label *platform_technical*, engine *setup_incomplete*. Autopay failure with no retry. The workaround (manual payment) only works if the user opens the app.
  - Post #146: hand label *platform_technical*, engine *unknown*. Autopay SIPs cancelled despite enough balance; user gives up on SIPs on the app ('don't start sip through this app').
- **Stopping, pausing or reducing — Engine (rules + LLM): 58% agreement on 19 rows.**
  - Post #517: hand label *yes*, engine *no*. New investor who picked funds from 'grow app rating'; cut SIPs from 5-6K to 2K each after a fall. A reduce, not a stop, so a nudge only on the stop/pause flow would miss it. Also asks for fund picks, which we cannot answer.
  - Post #992: hand label *yes*, engine *no*. One year in, in the red, compares with FD. Asks whether everyone is in the same position, which a 'most investors in this fund kept their SIP' fact answers. Medium confidence because 'sentiment' is not the same as behaviour.
- **Stop caused by the fall — Text rule (regex): 63% agreement on 19 rows.**
  - Post #1027: hand label *no*, engine *yes*. No fall at all (45% up). The market-fall pattern matches only because of the word 'correction'. A wish to time dips, not a panic stop.
  - Post #278: hand label *no*, engine *yes*. An analytical fund-specific exit while the index is up. The panic pattern matches only because the post quotes the advice 'don't panic'. Not a market-fall stop.

**What the golden set changed**

- Of 15 hand-labelled stops in a fall, only 6 are fear or an expectation gap; 4 are market timing, 2 fund-specific and 2 cash needs. "Panic" was too narrow.
- My original nudge (share of investors continuing in this fund) fully fits 2 posts and partly fits 6; it does not fit 9.
- Of 10 payment failures, 5 were not notified in time and 9 were fixable by the user with the right information.

## 3. Where I rejected or corrected AI output

### 1. The LLM's stop trigger could not explain stopping — replaced with an auditable cause taxonomy
- **AI output:** the codebook LLM gives each post one trigger. For 81 of 414 posts about stopping (20%) it said *none obvious / other / unknown*, and the stated ones were broad (top: platform_bug, failed_debit, mandate_autopay_issue).
- **Why I rejected it:** a vague label per post can't separate a failed mandate from a fund switch or a cash need, so trigger counts said nothing about *why* SIPs end.
- **Correction:** a fixed, versioned text-pattern taxonomy of 10 causes (reasoning.py) assigns each post a primary cause; it covers 246 of 414 stop posts. Patterns are visible in the app and scored against the golden set.

### 2. The AI's reading of why people stop in a fall didn't match the posts — not used for sizing
- **AI output:** the LLM codebook found only 44% of the stops that a fall caused (text rules: 78%), and the engine's stop driver matched the hand labels on only 29% of stops, mixing up fear, market timing and fund-specific reasons.
- **Why I rejected it:** sizing from the LLM label would under-count fall-driven stops, and a driver that can't tell fear from timing can't say whether 'panic' is the problem. Reading the posts, fear was a minority (6 of 15 stops, with expectation gaps); the common thread was having no way to judge whether the SIP was on track.
- **Correction:** sizing uses the text-pattern causes, not the LLM's fall-causation label; Problem 2 is reframed from "panic" to "no way to judge my own SIP at the stop moment", and fund-level social proof was dropped from the solution.

### 3. The AI answered personal-advice and support questions — added a question guard
- **AI output:** when I tested it with real user questions — *"Should I stop SIP?"*, *"Which mutual fund to buy?"*, *"Which bank account should I open?"*, *"OTP is not coming"* — the answer layer answered them from the evidence pack instead of declining.
- **Why I rejected it:** the brief forbids fund recommendations, and an evidence engine has nothing valid to say about one person's account.
- **Correction:** a rule-based guard classifies every question first (research, advice, support, unsafe, off-topic, prevalence) and declines with a reason and a research reframe; 58 of 58 test questions pass (6 kinds). The LLM also has a refusal step, and below 3 matching posts the engine abstains.

## 4. Limitations
- Public posts are not a random sample of investors; shares are of posts, not people.
- Few first-time investors post about stopping (62 posts), so first-time splits are hypotheses for the interviews, not measurements.
- Uplift estimates combine post shares with editable assumptions (lapse rate, recovery, severity, addressability); Groww cohort data would replace them.
- The golden set is small (35 rows) and checked by one person; 35 rows still await that check.
