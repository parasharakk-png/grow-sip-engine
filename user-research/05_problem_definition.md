# Part 4 · Problem definition

Goal metric: % of first-time mutual fund investors with an active SIP (one successful instalment in the last 30 days) 12 months after starting their first SIP.

Live engine: [Groww SIP Discovery Engine](https://grow-sip-engine-nlems472oc7cfzz4ripgxr.streamlit.app) · source: [grow-sip-engine](https://github.com/parasharakk-png/grow-sip-engine)

## Problem statement (MVP · Problem 1)

> In their first year, first-time investors judge their SIP only by red or green against a one-year expectation. With no way to tell what's normal, they ask someone they trust, then stop, reduce, or move the money out of SIPs.

Parallel operations track (Problem 2, not the MVP): *first-time investors lose SIPs they meant to keep: the first SIP never goes live, or a later instalment fails, and they aren't told why or how to fix it.*

The engine still stores the first draft of Problem 1 ("months 6–12" and "whoever got them started"). Interviews tightened two clauses: 3 of 5 peaked in months 6–12, two decided at 12–15; 2 of 5 asked the person who started them, the others asked YouTubers, searched, or asked a friend. The statement above is the one I am building on.

---

## What the discovery engine did

I used the engine to test two ways a first SIP can be gone by month 12, before I picked which one to build.

| Step | What happened |
|---|---|
| Collect | 4,546 public posts (Play Store, App Store, Reddit, YouTube, TradingQ&A) |
| Keep | 1,917 about the author's own SIP, duplicates removed. **526** from first-time investors |
| Tag | Stage, months in, SIP status, trigger, a quote. The LLM did this |
| Cause | One visible cause per stop post, from **text rules**, mapped to a metric branch. I did not let the LLM pick the cause: on the 20 posts I labelled, its "why they stop" label matched me only 33% of the time |
| Size | Share of stop posts × 40% lapse × 30% recoverable. First-timer stop posts: **68** |

How I checked the AI (20 hard posts I labelled by hand):

- "Instalment failed" matched me **86%** (keywords alone: 57%).
- **1,886 of 1,917** AI-picked quotes appear word for word in the post.
- Q&A once answered "which fund should I buy?", so I added a guard: **58 of 58** test questions handled.

Posts are public complaints, not Groww users. Treat shares as a ranking of what people write about, then take that ranking into interviews.

---

## My starting hypotheses

From the product, a first SIP is lost in one of two ways. The engine now labels them the same way as this file.

**Problem 1 · The stop decision.** People stop on purpose. I thought they did it for one of three reasons:

1. They picked a fund without understanding it (type, lock-in, expense ratio) and stop when they find out.
2. The market keeps falling, the money they put in shows a loss, and they stop.
3. They put extra cash in during a fall to average out the loss, it keeps falling, and with no cash left they stop.

**Problem 2 · Failed payments.** Many SIPs end without anyone choosing to stop. I thought the debit fails for one of three reasons:

1. Groww's systems or the payment gateway are down on the debit date.
2. The bank account behind the autopay doesn't have enough money that day.
3. Autopay is still on an old salary account after a job change, and they never updated it in Groww.

### What the engine (and interviews) say about each reason

Posts = posts about stopping or lapsing (**447** in all; **68** from first-time investors). Interviews = P1–P5.

| Reason | Posts | Interviews | Verdict |
|---|---|---|---|
| 1.1 Fund not understood (type, lock-in, costs) | 22 posts mention lock-in, ELSS, expense ratio or exit load (5 first-time) | P4 stopped on learning each ELSS instalment is locked for 3 years | Supported |
| 1.2 Market keeps falling, investment in loss | Poor returns or market timing is the main cause for 6 first-time posts; falls are mentioned *more* by people who kept going | 4 of 5 judged only by red / green; P2 stopped, P5 reduced, P1 and P3 considered stopping | Supported, but the fall is the trigger, not the cause |
| 1.3 Averaged down, ran out of cash | 4 posts mention averaging (0 first-time) | P3 understood averaging and was still discouraged; no one ran out of cash | Not seen |
| 2.1 Systems or gateway down | 8 posts (0 first-time) | Not seen | Real but small |
| 2.2 Not enough balance | 9 posts (1 first-time) | P1: one skipped instalment, SIP continued | Real, rarely ends a SIP |
| 2.3 Autopay on an old account | 0 posts (3 payment posts mention it) | P2 and P5 changed account; no failure | Not seen; needs Groww's bank-return data |
| *Missing from my payment list* | Biggest groups: failure with no reason given (29), money debited but units not allotted (17), first SIP never went live (12) | Two people had a failure; both found it only by opening the app; neither stopped over it | The real gap is that they aren't told why it failed |

Two reasons I hadn't listed came up in interviews: money redirected outside SIPs (FD / RD, lump sum, a gold scheme: 3 of 5), and doubt settled by someone they trust rather than by data.

The 20-post golden set already pointed this way before interviews: of the stop cases, most needed a way to judge their own SIP; only a minority were "just fear". The original "x% of investors kept their SIP" nudge fully fitted **1 of 9** posts, so I dropped it.

---

## What the engine sized (first-time investors)

On **68** first-timer posts about a SIP ending, the two problems come out a tie. That is why interviews, not the bars, picked the MVP.

| Problem | Share of first-timer stop posts | Est. uplift if fixed | Interviews (risky moment) | What I did |
|---|---|---|---|---|
| **1 · The stop decision** (pause, skip, reduce, cancel, switch) | **29.4%** | **+2.3 pp** | **5 of 5** | Built the MVP |
| **2 · Failed payments** (first SIP never live, or a later debit fails) | **27.9%** | **+2.9 pp** | **0 of 5** (2 had a failure, none stopped over it) | Ops track |

Uplift = 40% of first SIPs gone by month 12 × share of stop posts × how fixable that cause looks × 30% can come back. Change those two assumptions in the engine sidebar and the bars move. The other ~34% of first-timer stop posts had no clear cause.

The engine's "If Groww fixed one thing" tab ranks six product areas the same way. Payments and the stop screen sit at the top; I still needed people, because posts can't tell you which moment actually ends a first SIP.

Rule I wrote before talking to anyone: stay on the stop screen if at least 3 of 5 describe that moment. That's what happened.

---

## The seven elements

| Element | Answer | Evidence |
|---|---|---|
| **Target segment** | First-time SIP investors, 22–32, mostly salaried, ₹1k–10k+ a month, who started on someone else's word (a colleague, family member or YouTuber) and are 6–15 months in | Interviews P1–P5; 526 first-time investor posts in the engine |
| **Moment** | The first time the SIP has looked red or flat for months against what they expected after a year, and they open the app wondering whether to keep going | Risky moment was a stop decision for 5 of 5; doubt peaked at months 6–15 |
| **Product outcome** | At the "should I stop?" moment, the investor continues, reduces, skips or changes the date instead of cancelling (branches B and C of the metric tree) | Engine → Metric decomposition |
| **Root cause** | They have no yardstick of their own. They started with a borrowed goal and a one-year expectation, and the app shows returns, not progress against their own goal and horizon. So they borrow a verdict from someone else | 4 of 5 judged only by red / green against a one-year expectation; P3: "everything seems very vast and not able to get any conclusions" |
| **Workarounds** | Ask the person who got them started; watch YouTubers; search articles; ask an LLM; move new money to FD / RD, lump sums or a gold scheme | Doubt settled by a colleague (P1), cousin (P4), YouTubers (P2), a friend (P5); 3 of 5 moved money outside SIPs |
| **User value** | They make the decision with an understanding of their own SIP, at an amount they can afford, instead of on someone else's word; stopping stays an informed choice | P5 wanted "a view of their SIP journey"; P3 wanted help for "people like me" to understand SIPs |
| **Business value** | Money that stays in SIPs raises assets under management and lifetime value per investor; investors who reduce or skip stay active for the metric; fewer "should I stop?" support contacts | 3 of 5 still invest, but outside SIPs: lost to the metric, not to investing |

---

## How the thinking evolved

| Stage | What I believed | What changed it |
|---|---|---|
| **Business metric** | 12-month continuation is one number | An active SIP needs one successful instalment in 30 days, so a SIP can end by a decision *or* by a failed payment, and a lapse can be recovered. Reduce or skip still counts as active |
| **Product outcomes** | Five branches: activation, instalments succeed, a pause doesn't become a stop, the investor keeps choosing to continue, the investor can tell it's working | Engine → Metric decomposition |
| **AI discovery** | Problem 1 would be panic-stops; Problem 2 would be low balance, an old account, or the gateway being down | 1,917 coded posts. In first-timer stop posts the two problems are about level (29.4% vs 27.9%; +2.3 vs +2.9 pp). The old-account cause isn't in the posts (0 link a job change to a mandate). Most payment posts give no reason. Falls are mentioned *more* by people who kept going, so a fall alone doesn't separate quitters from stayers. I overruled the LLM on cause labels (33% match) and used visible rules instead |
| **Primary research** | Posts were too close to call, so whichever 3 of 5 people described would win | 5 interviews: the risky moment was a stop decision for 5 of 5 and a failed instalment for 0 of 5. Two had a failure; both found it only by opening the app and neither stopped over it. Peer stats mattered to 1 of 5 |
| **Problem definition** | "Investors panic and stop when markets fall" | The fall is only the trigger. The cause is having no yardstick of their own, so the decision is borrowed from someone they trust |

---

## Where research validated or challenged the engine

| Engine said | Interviews found | Verdict |
|---|---|---|
| On first-timer posts, failed payments and the stop screen are a tie (+2.9 vs +2.3 pp) | 0 of 5 risky moments were failed instalments; 5 of 5 were a stop decision | Posts ranked them; interviews picked the MVP |
| Doubt comes from having no way to judge their own SIP | 4 of 5 judged only by red / green against a one-year expectation | Validated |
| "Are others continuing?" is a proxy need | 1 of 5 asked for peer stats; a trusted person settled the doubt | Nuanced — dropped the "x% kept going" nudge |
| Many stops are switches | 3 of 5 moved money outside SIPs (FD / RD, lump sum, gold) | Reframed: they still invest, just not in this SIP |
| Cash squeezes end in cancels | No one stopped for cash; a low balance caused one skip | Challenged |
| Failed instalments go unnoticed | Both failures were found only by opening the app | Validated |
| Job-change → dead mandate is a main payment cause | 0 posts, 0 interviews with a mandate miss after a job change | Not in this set; needs Groww's bank-return file |
| — (missed) | A product rule (ELSS per-instalment lock-in) caused a stop; people already ask LLMs about their SIP | New |

---

## Limits

Five interviews from my network, all already at risk: they show what the moment looks like, not how common it is. No continuer was interviewed. P3 uses Zerodha Coin. Post shares describe public posts, not Groww users. Uplift rests on two assumptions (40% lapse, 30% recoverable) that cohort data would replace. The golden set was labelled by one person. Full pipeline, disagreements and the 58 guard questions are in the [AI Evidence Log](https://github.com/parasharakk-png/grow-sip-engine/blob/main/evidence-log/AI_Evidence_Log.md).
