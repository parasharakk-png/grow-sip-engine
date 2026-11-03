# Golden set: knowledge base

The golden set is 20 public posts that I labelled by hand against a written codebook. It is the answer key used to check whether the AI discovery engine reads posts the way a careful human would.

- File: `discovery-engine/data/golden_set.csv` (one row per post)
- Scoring code: `discovery-engine/golden_set_eval.py` (run `python3 golden_set_eval.py`)
- Results: section 2 of `evidence-log/AI_Evidence_Log.md`, and the "How it works & evidence" tab in the engine
- Per-post review sheet: `evidence-log/golden_set_review.md`

## 1. Why it exists (assessment requirement)

Part 1 of the brief asks for an AI Evidence Log with about 20 hand-labelled items, agreement scores, and the places where the AI failed. Part 5 asks for an MVP that is eval-gated and measured against the same golden set. The golden set is the single source of truth for both.

## 2. What is in it

| | Problem 2: stopping in a fall | Problem 1: failed payment |
|---|---|---|
| Posts | 13 | 7 |
| First-time / young investors | 7 | 2 |
| Other investors (comparison) | 6 | 5 |
| Core cases | 4 | 6 |
| Hard negatives (posts that fool keyword rules) | 6 | 1 |
| Counter-evidence (posts that argue against my solution) | 1 | 0 |
| Early signals (doubt before stopping) | 2 | 0 |
| High-confidence labels | 12 | 7 |
| Sources | Reddit 13 | Reddit 3, TradingQ&A 2, App Store 1, Play Store 1 |

All 20 rows were hand-labelled and checked by me (`human_verified = yes`).

## 3. The codebook (what each label means)

| Field | Question it answers | Used for |
|---|---|---|
| `own_experience` | Is the writer describing their own SIP? | Checks the selection screen |
| `fall_experienced` | Is their own portfolio falling? | Problem 2 check |
| `emotion` | Anxious / panicking, calm, confused… | Problem 2 check |
| `action` | Stopped, paused, reduced, continued, considering | Problem 2 check |
| `fall_drove_action` | Did the fall cause the stop? | Problem 2 check |
| `stop_reason` | Fear, expectation gap, market timing, fund under-performance, cash need, family | Problem 2 driver check |
| `seeks_peer_behaviour` | Do they ask what others are doing? | Tests the social-proof idea |
| `user_need`, `nudge_fit` | What would have helped; does my nudge fit? | Solution fit |
| `payment_failure`, `failure_cause` | Did an instalment fail, and why? | Problem 1 check |
| `account_change`, `two_account_setup` | Job, bank or salary-account change; two accounts | Problem 1 check |
| `failure_notified`, `user_fixable` | Were they told in time; could they fix it? | Problem 1 solution fit |
| `lapse_outcome` | Did the SIP actually end? | Context |
| `label_confidence`, `evidence_quote`, `rationale` | How sure I am, the quote, and why | Audit trail |
| `cohort`, `source`, `url`, `Platform` | Who wrote it and where | Metadata only; not scored |

## 4. How the golden set is used

1. **Scoring the engine.** For each of 12 checks, the engine's label (text rule, LLM codebook, or both) is compared with my label to give agreement, precision and recall. Disagreements are listed with my rationale.
2. **Scoring the selection screen.** Each post's keep/drop decision in `selection_log.csv` is compared with my `own_experience` label.
3. **Duplicates.** A golden post the pipeline dropped as a duplicate is scored as the copy the pipeline kept (#2547 is scored as #1405).
4. **In the app.**
   - Every post card shows whether it is in the golden set.
   - The evidence explorer can be filtered to golden posts only.
   - Each opportunity card shows how well the engine scored on the check behind it.
   - The "How it works & evidence" tab shows the headline agreement numbers.
5. **Evidence Log.** The "where it failed" examples and the "where I rejected AI output" sections are built from the disagreements.
6. **MVP (Part 5).** Golden posts become test inputs for the "Is my SIP okay?" check-in:
   - Problem 2 posts test whether the reply explains the user's own SIP without advice or FOMO.
   - Hard negatives and counter-evidence test that it does not over-reach.
   - Problem 1 posts test the routing to the payment and mandate path.

## 5. Current results (19 scorable posts)

| Problem | Check | Method | Agreement |
|---|---|---|---|
| P2 | Own portfolio is falling | Text rule | 92% |
| P2 | Anxious / panicking | Text rule / LLM | 67% / 75% |
| P2 | Stopping, pausing or reducing | Engine | 75% (precision 100%, recall 67%) |
| P2 | Stop caused by the fall | Text rule / LLM | 67% / 83% |
| P2 | Asks what others are doing | Text rule | 50% |
| P2 | Why they stop (driver) | Engine | 33% |
| P1 | Instalment failed | Text rule / LLM | 57% / 86% |
| P1 | Cause of failure | Text rule | 40% |
| P1 | Job / bank / salary-account change | Text rule | 71% |

The selection screen agrees with my `own_experience` label on 20 of 20 posts. Post #398 is correctly excluded, and #2547 is scored as its kept duplicate #1405, so the tables use 19 rows.

**What this tells me**

- The engine reliably spots a falling portfolio (92%).
- It catches about two thirds of real stops (recall 67%), and every stop it flags is correct.
- It cannot tell *why* people stop (33%). This is why sizing uses text-pattern causes, and why "panic" was reframed as "no way to judge my own SIP".
- On payment failures, the LLM (86%) beats the text rule (57%).

## 6. How to keep it valid

- Change labels only in the CSV, set `human_verified = yes` after reading the post, and rerun `golden_set_eval.py` and `evidence_log.py`.
- Never tune the text rules or prompts on the golden set and then report the same score as if it were unseen. If rules are changed, add new held-out posts.
