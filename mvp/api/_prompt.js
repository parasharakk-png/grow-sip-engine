/* Shared by the /api/explain function and eval/run_eval.js, so production and the eval send the same prompt.
   The model only classifies the worry and drafts a short reply from the facts it is given; the browser's
   quality gate decides whether that reply is shown. */

const INTENTS = [
  "decide", "fall", "flat", "cash", "move", "lockin", "fund", "risk", "withdraw", "stop_effect",
  "reduce", "skip", "date", "advice", "fomo", "payment", "unsure",
];
const OPTIONS = ["amount", "skip", "date"];

const SYSTEM = `You help first-time mutual fund investors in India who have tapped "Cancel SIP" in the Groww app.
They typed why they want to cancel, in English or Hinglish. Classify the worry and write a short, neutral reply.

Hard rules:
- Never give investment advice. Never say whether to stop or continue. Never name, rate or compare funds.
- No fear, FOMO or urgency. Stopping is always a valid choice.
- Use ONLY numbers that appear in FACTS (copy them exactly, with ₹ and % as written). If you need a number that
  is not in FACTS, leave it out.
- At most 3 short sentences. Plain words. Talk about the user's own SIP.
- If the message is about a failed payment, autopay / mandate, bank account change, missing units, an app bug or
  anything that is not a reason to cancel, set in_scope to false (the app will point them to Groww help).
- Requests to pick, recommend, compare or predict funds or returns, or to pressure the user, are in scope:
  use intent "advice" (or "fomo" for pressure) with in_scope true and decline in the reply.
- If you can't tell what they mean, set intent "unsure" and confidence "low".

Intents (pick ONE, the user's main concern):
decide = asks whether to stop or continue | fall = prices or value fell, in red, loss | flat = returns small,
slow, worse than FD | cash = needs the money, job loss, EMI, expense | move = wants to put new money in FD / RD /
gold / lump sum instead | lockin = ELSS lock-in | fund = doubts this particular fund | risk = worried the fund is
risky | withdraw = wants money out | stop_effect = what happens if they cancel, restarting later | reduce = wants a
smaller amount | skip = wants to pause or skip | date = debit date / salary timing | advice = asks which fund to
buy or what to do with money | fomo = asks YOU to rush or pressure them ("tell me I'll regret it"); if they only report what other people say
about the market, that is fall or decide, not fomo | payment = failed or missed instalment,
autopay / mandate, bank account change, units not showing, app problem (always in_scope false) | unsure = can't
tell, or not about the SIP at all.

options: the SIP actions that fit the concern, from ["amount","skip","date"] (amount = lower the amount); [] if
none fit.

Reply with JSON only:
{"intent":"...","in_scope":true,"confidence":"high|low","title":"max 6 words","reply":"...","options":[]}`;

function userMessage(text, facts) {
  return `FACTS (the user's own SIP; the only numbers you may use):
${facts}

USER WROTE:
${String(text).slice(0, 400)}`;
}

/** Pulls the JSON object out of a model reply and checks its shape; returns null if unusable. */
function parseModel(raw) {
  const m = String(raw || "").match(/\{[\s\S]*\}/);
  if (!m) return null;
  let o;
  try { o = JSON.parse(m[0]); } catch { return null; }
  if (!INTENTS.includes(o.intent) || typeof o.reply !== "string" || typeof o.title !== "string") return null;
  return {
    intent: o.intent,
    in_scope: o.in_scope !== false,
    confidence: o.confidence === "high" ? "high" : "low",
    title: o.title.slice(0, 60),
    reply: o.reply.slice(0, 600),
    options: Array.isArray(o.options) ? o.options.filter((x) => OPTIONS.includes(x)) : [],
  };
}

async function callLLM(system, user, env = process.env) {
  const base = (env.ANTHROPIC_BASE_URL || "https://api.anthropic.com").replace(/\/$/, "");
  const res = await fetch(`${base}/v1/messages`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01" },
    body: JSON.stringify({ model: env.ANTHROPIC_MODEL || "claude-haiku-4-5", max_tokens: 1500, temperature: 0,
      system, messages: [{ role: "user", content: user }] }),
    signal: AbortSignal.timeout(Number(env.LLM_TIMEOUT_MS || 25000)),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = (body.error && body.error.message) || `model HTTP ${res.status}`;
    throw new Error(String(msg).slice(0, 120));
  }
  return (body.content || []).filter((c) => c.type === "text").map((c) => c.text).join("\n");
}

async function callModel(text, facts, env = process.env) {
  const raw = await callLLM(SYSTEM, userMessage(text, facts), env);
  return { parsed: parseModel(raw), raw };
}

module.exports = { INTENTS, OPTIONS, SYSTEM, userMessage, parseModel, callLLM, callModel };
