# User research kit: validating the stop-moment opportunity (Part 3)

**Goal of this round:** find out why first-time investors' SIP continuation breaks down in the first 12 months,
and test whether the engine is right that the *stop moment* (pause, reduce, skip, stop or switch) matters more
for them than payment failures.

## Files, in the order you use them

| Step | File | When |
|---|---|---|
| 1 | `01_screener.md` | Recruit: paste into a Google Form, share with the recruitment message |
| 2 | `02_interview_guide.md` | Run each 45-minute interview (consent script is at the top) |
| 2a | `concept_stop_sheet.md` | Stimulus for the last 4 minutes of the interview only |
| 3 | `03_notes_template.md` | Copy once per participant (`notes/P1.md` …), fill in within 2 hours of the call |
| 4 | `04_synthesis.md` + `scorecard.csv` | After all interviews: score hypotheses, compare with the engine |

## The hypotheses being tested

These come from the engine's **Metric decomposition** tab. Every guide question maps to one of them.

| ID | Branch | Hypothesis |
|---|---|---|
| H1 | C · Deliberate stop | First-timers who consider stopping lack a way to judge their own SIP (goal, horizon, what is normal). The fall is the trigger, not the cause. |
| H2 | C · Deliberate stop | "Are others continuing?" is a proxy need; fund-level stats would not resolve it. |
| H3 | C · Deliberate stop | Many first-timer "stops" are switches, and the new SIP depends on a second decision that often slips. |
| H4 | B · Pause becomes stop | A cash squeeze ends in a cancel because reduce / skip is unknown or not trusted. |
| H5 | A · Involuntary lapse | Failed instalments go unnoticed; a bank or salary-account change triggers them for first-timers too. |
| H6 | D · Confusion drift | Not understanding returns turns normal volatility into "it isn't working". |

## Decision rule, written before the interviews

- **Keep the stop-moment focus** if at least 3 of 6 participants describe a stop decision (H1–H4) as the moment
  continuation was at risk.
- **Switch focus to payment reliability (Problem 1)** if at least 3 of 6 describe a failed or missed instalment
  (H5) as that moment, and fewer describe a stop decision.
- **Rethink the solution** if H1 is challenged by most participants: they knew how to judge their SIP but stopped anyway.

## Privacy rules (from the brief, plus good practice)

- Never ask for or record PAN, account numbers, folio numbers or portfolio screenshots.
- No screen-sharing of the investing app. Ask participants to describe what they see.
- Ask for amounts as ranges only.
- Refer to people as P1–P6 everywhere. Names and contact details live only in `private/`, which is gitignored,
  as are recordings in `recordings/`.
