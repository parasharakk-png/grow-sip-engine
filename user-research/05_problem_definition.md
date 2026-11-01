# Part 4 · Problem definition

Goal metric: % of first-time mutual fund investors with an active SIP (one successful instalment in the last 30 days) 12 months after starting their first SIP.

## Problem statement (MVP focus)

> In months 6–12, first-time investors judge their SIP only by red or green against a one-year expectation. With no way to tell what's normal, they ask whoever got them started, then stop, reduce, or move the money out of SIPs.

Parallel operations track (not the MVP): *first-time investors lose SIPs they meant to keep: the first SIP never goes live, or a later instalment fails, and they aren't told why or how to fix it.*

Two parts of the statement go further than the interviews show: "ask whoever got them started" (2 of 5; others asked YouTubers, searched online or asked a friend) and "months 6–12" (3 of 5; two decided at months 12–15). Suggested wording: "…they ask someone they trust…" and "In their first year…".

## My starting hypotheses

From my own understanding of the product, SIPs end in one of two ways: the payment fails, or the investor stops on purpose.

**Hypothesis 1 · Failed payments.** Many SIPs end without the investor choosing it, because an instalment fails. I think it fails for one of three reasons:

1. Groww's systems or the payment gateway are down on the debit date.
2. The bank account behind the autopay doesn't have enough money on that day.
3. The autopay is still set on an old bank account the investor no longer uses (for example after a job change), and they never updated it in Groww.

**Hypothesis 2 · Intentional stops.** Other investors stop their SIP on purpose. I think they do it for one of three reasons:

1. They picked a fund without understanding it (its type, lock-in period or costs such as the expense ratio) and stop when they find out.
2. The market keeps falling, the money they put in shows a loss, and they stop.
3. They put extra cash in during a fall to average out the loss, the market keeps falling, and with no cash left they stop.

### What the evidence says about each reason

Posts = posts about stopping or lapsing (414 in all; 62 from first-time investors). Interviews = P1–P5.

| Reason | Posts | Interviews | Verdict |
|---|---|---|---|
| 1.1 Systems or gateway down | 8 posts (0 first-time) | Not seen | Real but small |
| 1.2 Not enough balance | 9 posts (1 first-time) | P1: one skipped instalment, SIP continued | Real, rarely ends a SIP |
| 1.3 Autopay on an old account | 0 posts (3 payment posts mention it) | P2 and P5 changed account; no failure | Not seen; needs Groww's bank-return data |
| *Missing from my list* | The biggest groups: failure with no reason given (29), money debited but units not allotted (17), first SIP never went live (12) | — | The real gap is that investors aren't told why it failed |
| 2.1 Fund not understood (type, lock-in, costs) | 22 posts mention lock-in, ELSS, expense ratio or exit load (5 first-time) | P4 stopped on learning each ELSS instalment is locked for 3 years | Supported |
| 2.2 Market keeps falling, investment in loss | Poor returns or market timing is the main cause for 6 first-time posts; falls are mentioned *more* by continuers | 4 of 5 judged only by red / green; P2 stopped, P5 reduced, P1 and P3 considered stopping | Supported, but the fall is the trigger, not the cause |
| 2.3 Averaged down, ran out of cash | 4 posts mention averaging (0 first-time) | P3 understood averaging and was still discouraged; no one ran out of cash | Not seen |

Two reasons I hadn't listed came up in interviews: money redirected outside SIPs (FD / RD, lump sum, a gold scheme: 3 of 5), and doubt settled by someone they trust rather than by data.

## The seven elements

| Element | Answer | Evidence |
|---|---|---|
| **Target segment** | First-time SIP investors, 22–32, mostly salaried, ₹1k–10k+ a month, who started on someone else's word (a colleague, family member or YouTuber) and are 6–15 months in | Interviews P1–P5; 495 first-time investor posts |
| **Moment** | The first time the SIP has looked red or flat for months against what they expected after a year, and they open the app wondering whether to keep going | Risky moment was a stop decision for 5 of 5; doubt peaked at months 6–15 |
| **Product outcome** | At the "should I stop?" moment, the investor continues, reduces, skips or changes the date instead of cancelling (branches B and C of the metric tree) | Metric decomposition |
| **Root cause** | They have no yardstick of their own. They started with a borrowed goal and a one-year expectation, and the app shows returns, not progress against their own goal and horizon. So they borrow a verdict from someone else | 4 of 5 judged only by red / green against a one-year expectation; P3: "everything seems very vast and not able to get any conclusions" |
| **Workarounds** | Ask the person who got them started; watch YouTubers; search articles; ask an LLM; move new money to FD / RD, lump sums or a gold scheme | Doubt settled by a colleague (P1), cousin (P4), YouTubers (P2), a friend (P5); 3 of 5 moved money outside SIPs |
| **User value** | They make the decision with an understanding of their own SIP, at an amount they can afford, instead of on someone else's word; stopping stays an informed choice | P5 wanted "a view of their SIP journey"; P3 wanted help for "people like me" to understand SIPs |
| **Business value** | Money that stays in SIPs raises assets under management and lifetime value per investor; investors who reduce or skip stay active for the metric; fewer "should I stop?" support contacts | 3 of 5 still invest, but outside SIPs: lost to the metric, not to investing |

## How the thinking evolved

| Stage | What I believed | What changed it |
|---|---|---|
| **Business metric** | 12-month continuation is one number | An active SIP needs one successful instalment in 30 days, so a SIP can end by a decision *or* by a failed payment, and a lapse can be recovered |
| **Product outcomes** | Five branches: activation, instalments succeed, a pause doesn't become a stop, the investor keeps choosing to continue, the investor can tell it's working | Metric decomposition tree |
| **AI discovery** | My two hypotheses: (1) failed payments, from systems down, low balance or an old bank account; (2) intentional stops, from a fund not understood, a falling market or averaging out of cash | 1,907 coded posts. In first-time posts the two problems are about level (25.8% vs 29.0% of posts about stopping; +2.7 vs +2.3 pp). The old-account cause wasn't visible (0 posts link a job change to a mandate); most payment posts say the failure had no reason given; falls were mentioned *more* by continuers, so a falling market alone didn't separate stoppers |
| **Primary research** | The engine leaned towards payment failure for first-timers | 5 interviews: the risky moment was a stop decision for 5 of 5 and a failed instalment for 0 of 5. Two had a failure; both found it only by opening the app and neither stopped over it. Peer stats mattered to 1 of 5 |
| **Problem definition** | "Investors panic and stop when markets fall" | The fall is only the trigger. The cause is having no yardstick of their own, so the decision is borrowed from someone else |

## Where research validated or challenged the engine

| Engine said | Interviews found | Verdict |
|---|---|---|
| Failed payments are the bigger lever for first-timers | 0 of 5 risky moments were failed instalments | Challenged |
| Doubt comes from having no way to judge their own SIP | 4 of 5 judged only by red / green against a one-year expectation | Validated |
| "Are others continuing?" is a proxy need | 1 of 5 asked for peer stats; a trusted person settled the doubt | Nuanced |
| Many stops are switches | 3 of 5 moved money outside SIPs (FD / RD, lump sum, gold) | Reframed |
| Cash squeezes end in cancels | No one stopped for cash; a low balance caused one skip | Challenged |
| Failed instalments go unnoticed | Both failures were found only by opening the app | Validated |
| — (missed) | A product rule (ELSS per-instalment lock-in) caused a stop; investors already ask LLMs about their SIP | New |

## Limits

Five interviews from my network, selected for being at risk: they show what the moment looks like, not how common it is. No continuer was interviewed. P3 uses Zerodha Coin. Post shares describe public posts, not Groww's users, and uplift estimates assume a 40% lapse rate and 30% recovery.
