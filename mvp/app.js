/* Groww-style Mutual Funds prototype: Dashboard, SIPs, SIP Details and the "before you cancel" check-in.
   Only users who tap Cancel SIP see the check-in; Dashboard and SIPs carry no nudges.
   Figures are samples for the prototype, not live Groww data. Every reply passes a number check and an
   advice / urgency check before it is shown; a failed check shows a plain fact template instead. */

const FUNDS = [
  { id: "absl-psu", scheme: "Aditya Birla Sun Life PSU Equity Fund Direct Growth", logo: "AB", logoBg: "bg-[#c8102e]",
    monthly: 200, months: 14, value: 3240, sipDay: 5, sipId: "SIP1140919244119463",
    category: "thematic", navs: [31.8, 29.6, 27.9], elss: false },
  { id: "groww-total", scheme: "Groww Nifty Total Market Index Fund Direct Growth", logo: "G",
    logoBg: "bg-gradient-to-br from-sky-400 to-groww", monthly: 500, months: 9, value: 4710, sipDay: 5,
    sipId: "SIP1140919244120118", category: "broad_index", navs: [11.9, 11.4, 11.1], elss: false },
  { id: "bandhan-n50", scheme: "Bandhan Nifty 50 Index Fund Direct Plan Growth", logo: "B", logoBg: "bg-[#e2231a]",
    monthly: 500, months: 18, value: 10350, sipDay: 28, sipId: "SIP1140919244121702",
    category: "large_index", navs: [52.4, 51.0, 50.2], elss: false },
  { id: "quant-small", scheme: "Quant Small Cap Fund Direct Plan Growth", logo: "q", logoBg: "bg-black border border-white/30",
    monthly: 2000, months: 12, value: 22600, sipDay: 28, sipId: "SIP1140919244122377",
    category: "small_cap", navs: [268.1, 251.4, 239.7], elss: false },
  { id: "sbi-psu", scheme: "SBI PSU Direct Plan Growth", logo: "SBI", logoBg: "bg-[#1a75bc]",
    monthly: 500, months: 6, value: 2870, sipDay: 28, sipId: "SIP1140919244123051",
    category: "thematic", navs: [36.2, 33.9, 32.5], elss: false },
];

/* Sample history of SIPs at each age, by category: share below the amount invested, and the usual range.
   The live version would compute this from rolling SIP returns for the category. */
const CATEGORY = {
  large_index: { label: "large-cap index funds", ages: { 6: [35, -12, 22], 12: [30, -15, 30], 18: [25, -14, 32] } },
  broad_index: { label: "broad market index funds", ages: { 6: [36, -13, 24], 12: [31, -16, 32], 18: [26, -15, 35] } },
  small_cap: { label: "small-cap funds", ages: { 6: [40, -20, 35], 12: [38, -25, 45], 18: [33, -24, 55] } },
  thematic: { label: "sector and theme funds", ages: { 6: [42, -22, 40], 12: [40, -28, 55], 18: [35, -27, 60] } },
};

const REASONS = [
  { id: "fall", label: "Market fell" },
  { id: "flat", label: "Returns flat or low" },
  { id: "cash", label: "Need the money" },
  { id: "move", label: "Moving to FD / RD / gold" },
  { id: "lockin", label: "Locked in (ELSS)", elssOnly: true },
  { id: "fund", label: "Unsure about this fund" },
];

FUNDS.forEach((f) => { f.invested = f.monthly * f.months; });
const ORIGINAL = FUNDS.map((f) => ({ monthly: f.monthly, sipDay: f.sipDay }));
const TABS = ["Explore", "Dashboard", "SIPs", "Watchlist"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const EVENTS_KEY = "sipCancelEvents";

const state = {
  tab: "Dashboard",
  screen: "tab",
  fundId: null,
  sheet: false,
  cancelled: new Set(),
  skipped: new Set(),
  toast: "",
  reason: null,
  draft: "",
  answer: null,
  option: null,
  pickAmount: null,
  pickDay: null,
};

const $ = (id) => document.getElementById(id);
const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");
const pctStr = (n) => (n > 0 ? "+" : n < 0 ? "−" : "") + Math.abs(n).toFixed(1) + "%";
const today = () => { const d = new Date(); d.setHours(0, 0, 0, 0); return d; };
const fmt = (d, year = false) => `${String(d.getDate()).padStart(2, "0")} ${MONTHS[d.getMonth()]}${year ? " " + d.getFullYear() : ""}`;
const ordinal = (n) => {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

function active() { return FUNDS.filter((f) => !state.cancelled.has(f.id)); }
function current() { return FUNDS.find((f) => f.id === state.fundId); }

function sipFacts(p) {
  const invested = p.invested;
  const change = p.value - invested;
  return { invested, change, pct: invested ? (change / invested) * 100 : 0 };
}

/** Next instalment on the SIP day, from tomorrow onwards; a skipped instalment moves it a month. */
function nextDue(f) {
  const t = today();
  const d = new Date(t.getFullYear(), t.getMonth(), f.sipDay);
  if (d <= t) d.setMonth(d.getMonth() + 1);
  if (state.skipped.has(f.id)) d.setMonth(d.getMonth() + 1);
  return d;
}

function daysUntil(d) { return Math.round((d - today()) / 86400000); }

/** Orders are placed on the next working day when the SIP date falls on a weekend. */
function holidayNote(d) {
  const day = d.getDay();
  if (day !== 6 && day !== 0) return "";
  const nav = new Date(d);
  nav.setDate(d.getDate() + (day === 6 ? 2 : 1));
  const last = new Date(nav);
  last.setDate(nav.getDate() - 1);
  return day === 6
    ? `NAV date will be ${fmt(nav)} as ${fmt(d)} to ${fmt(last)} are holidays`
    : `NAV date will be ${fmt(nav)} as ${fmt(d)} is a holiday`;
}

/* ---------- Facts the check-in may quote. Every number in a reply must come from here. ---------- */

function normalRange(f) {
  const cat = CATEGORY[f.category];
  const age = [18, 12, 6].find((a) => f.months >= a) || 6;
  const [redShare, lo, hi] = cat.ages[age];
  return { label: cat.label, age, redShare, lo, hi };
}

function unitsLine(f) {
  const [oldest, , latest] = f.navs;
  const uOld = f.monthly / oldest, uNew = f.monthly / latest;
  const more = uNew > uOld;
  return {
    text: `Your latest ${inr(f.monthly)} instalment bought ${uNew.toFixed(2)} units at ₹${latest.toFixed(2)} a unit. `
      + `Two months ago the same amount bought ${uOld.toFixed(2)} units at ₹${oldest.toFixed(2)}.`
      + (more ? " Lower prices meant more units for the same money." : " Higher prices meant fewer units."),
  };
}

function factPack(f) {
  const x = sipFacts(f), r = normalRange(f), u = unitsLine(f);
  const where = x.pct < r.lo ? "lower than" : x.pct > r.hi ? "higher than" : "within";
  const so = `You have put in ${inr(x.invested)} over ${f.months} months. It is worth ${inr(f.value)} today (${pctStr(x.pct)}).`;
  const range = `SIPs in ${r.label} often move up and down in the first year or two. `
    + `At ${f.months} months, your ${pctStr(x.pct)} is ${where} the usual range for these funds.`;
  return { so, range, units: u.text, x, r };
}

/* ---------- Reason -> reply (rules preview; the model call replaces this step) ---------- */

/* Order matters: the first matching intent wins, so specific asks come before broad worries. */
const INTENTS = [
  ["advice", /which fund|should i buy|best fund|recommend|suggest (a |some )?fund|kahan invest|kaunsa fund|better fund/],
  ["fomo", /fomo|urgent|don'?t miss|everyone is (buying|selling)|sab log/],
  ["decide", /should i (stop|continue|cancel|keep|exit)|stop or continue|continue or stop|band karu|band kar doon/],
  ["withdraw", /withdraw|redeem|take (my |the )?money out|get my money|paisa nikal|sell (my )?units/],
  ["stop_effect", /what happens if|if i (stop|cancel)|restart|start again|later when|will i lose/],
  ["reduce", /reduce|lower (the |my )?amount|smaller amount|less amount|decrease|amount kam|kam karna/],
  ["skip", /\bskip|\bpause|break for|one month off/],
  ["date", /\bdate\b|salary (comes|credited|is late)|debit day|tareekh/],
  ["lockin", /lock[- ]?in|\belss\b|teen saal/],
  ["cash", /need (the )?money|emergency|paisa chahiye|\bloan\b|\bemi\b|expense|\bjob\b|laid off|\brent\b|\bbills?\b|wedding|medical|tight|afford/],
  ["flat", /\bflat\b|no returns?|not seeing (any )?returns?|returns? (are|is) (bad|low|poor|less|nothing|zero)|not growing|same value|kuch nahi badh|too slow|waiting|too long/],
  ["move", /\bfd\b|\brd\b|fixed deposit|\bgold\b|jewell|lump ?sum|band karke/],
  ["risk", /risky|\brisk\b|\bsafe\b|volatile|dangerous/],
  ["fall", /\bfall|\bfell\b|crash|\bred\b|\bloss|\blosing\b|\bdown\b|negative|minus|\bgir(a|e|i|ta|te)\b|doob|ghat/],
  ["fund", /this fund|the fund|fund (is )?(bad|wrong|good)|wrong fund|galat fund/],
];

function classify(text) {
  const t = text.toLowerCase();
  const hit = INTENTS.find(([, re]) => re.test(t));
  return hit ? hit[0] : "unsure";
}

/** One reply per concern. Each answers that concern only; the SIP numbers are already on the card above. */
function reply(kind, f) {
  const k = factPack(f), due = fmt(nextDue(f), true), value = inr(f.value);
  const avgMonths = Math.floor((f.months + 1) / 2);
  const cancelEffect = `Cancelling stops the ${due} instalment and every one after it. Your ${value} stays invested; nothing is sold.`;
  const replies = {
    advice: { ok: false, title: "I can't suggest funds",
      body: "This check-in doesn't say which fund to pick or buy. It only explains your own SIP and the options on this screen." },
    fomo: { ok: false, title: "No pressure here",
      body: "This check-in doesn't use fear or urgency. Cancelling is always available below." },
    decide: { ok: true, title: "That's your call", options: ["amount", "skip", "date"],
      body: `This screen won't tell you to stop or continue. ${cancelEffect} Lowering the amount or skipping a month keeps the SIP running.` },
    fall: { ok: true, title: "Lower prices, more units", options: ["amount", "skip"], body: k.units },
    flat: { ok: true, title: "Why early SIP returns look small", options: ["amount"],
      body: `Your money has been invested for about ${avgMonths} months on average, not ${f.months}: each instalment only starts from its own date. The newest instalments have had little time to change.` },
    cash: { ok: true, title: "Cancelling doesn't give you this money", options: ["skip", "amount", "date"],
      body: `Cancelling only stops future instalments. Your ${value} stays invested until you withdraw it, which is a separate step. If money is tight for a while, skipping a month or lowering the amount frees up cash without ending the SIP.` },
    move: { ok: true, title: "Saving in an FD, RD or gold instead", options: ["amount"],
      body: `That is still saving, and it's your choice. ${cancelEffect} If you'd like to keep some money here as well, you can lower the amount instead.` },
    lockin: { ok: true, title: "The lock-in rule", options: [],
      body: "In an ELSS fund each instalment is locked for 3 years from its own date, not from the day you started. Cancelling stops future instalments; money already invested stays locked until each instalment's date." },
    fund: { ok: true, title: "This screen can't judge the fund", options: [],
      body: "Whether a fund suits you depends on your goal and how much ups and downs you can live with. The fund page shows what it invests in and its riskometer; a SEBI-registered adviser can review the choice with you." },
    risk: { ok: true, title: "How much this fund can move", options: [],
      body: "The riskometer on the fund page shows this fund's risk level. Whether that level suits you depends on when you will need the money." },
    withdraw: { ok: true, title: "Taking money out is a separate step", options: [],
      body: "Cancelling doesn't pay anything out. To take money out, redeem units from the fund page; the money usually reaches your bank in a few working days. The fund page shows any exit load." },
    stop_effect: { ok: true, title: "What cancelling changes", options: [],
      body: `${cancelEffect} You can start a new SIP later whenever you choose.` },
    reduce: { ok: true, title: "You can lower the amount", options: ["amount"],
      body: `Pick a smaller monthly amount below. The SIP stays active and the ${due} instalment uses the new amount.` },
    skip: { ok: true, title: "You can skip one month", options: ["skip"],
      body: `No money is taken on ${due}. The SIP continues the month after.` },
    date: { ok: true, title: "You can move the debit date", options: ["date"],
      body: "If your salary or other money arrives later in the month, pick a date after it. The SIP keeps running on the new date." },
    unsure: { ok: true, title: "I couldn't tell what's worrying you", options: ["amount", "skip", "date"],
      body: `Try a reason above, or ask about skipping, lowering the amount, withdrawing or what cancelling changes. ${cancelEffect}` },
  };
  return { kind, ...replies[kind] };
}

/** The quality gate: numbers must match the fact pack; no advice, fund names or urgency. */
const BLOCKED = /\b(buy|switch to|invest in|best fund|recommend|hurry|last chance|don'?t miss|everyone is|guarantee)/i;
function numbersIn(s) { return s.match(/₹[\d,]+(?:\.\d+)?|[+−-]?\d+(?:\.\d+)?%|\d+\.\d{2} units/g) || []; }

function gate(r, f) {
  if (!r.ok) return { ...r, check: "declined" };
  const k = factPack(f);
  const allowed = new Set(numbersIn([k.so, k.range, k.units, inr(f.monthly)].join(" ")));
  const badNumber = numbersIn(r.body).find((n) => !allowed.has(n));
  const otherFund = FUNDS.find((o) => o.id !== f.id && r.body.includes(o.scheme));
  if (badNumber || otherFund || BLOCKED.test(r.body)) {
    return { ...reply("stop_effect", f), check: "fallback",
      why: badNumber ? `a number (${badNumber}) did not match your SIP` : otherFund ? "it named another fund" : "it read like advice" };
  }
  return { ...r, check: r.kind === "unsure" ? "unsure" : "passed" };
}

/* ---------- Event log for testing (mentor view only) ---------- */

function events() { try { return JSON.parse(localStorage.getItem(EVENTS_KEY)) || []; } catch { return []; } }
function logEvent(type, extra = {}) {
  const all = events();
  all.push({ type, fund: state.fundId, reason: state.reason, at: new Date().toISOString(), ...extra });
  localStorage.setItem(EVENTS_KEY, JSON.stringify(all));
}
function eventSummary() {
  const e = events(), n = (t) => e.filter((x) => x.type === t).length;
  const intents = n("cancel_tap"), cancelled = n("cancelled");
  const changed = n("amount_lowered") + n("date_changed") + n("skipped");
  return { intents, cancelled, changed, kept: n("kept") };
}

/* ---------- Screens ---------- */

function renderControls() {
  const s = eventSummary();
  const btn = (action, label) =>
    `<button data-action="${action}" class="text-xs font-semibold px-3 py-1.5 rounded-full border bg-white text-neutral-800 border-neutral-300">${label}</button>`;
  $("controls").innerHTML = [
    btn("tab:Dashboard", "Open Dashboard"),
    btn("tab:SIPs", "Open SIPs"),
    btn("reset", "Reset demo"),
  ].join("") + `<p class="w-full text-[11px] text-neutral-600 mt-1">Test log · cancel taps ${s.intents} · changed instead ${s.changed} · kept ${s.kept} · cancelled ${s.cancelled}</p>`;
}

function header() {
  const pill = (t) => `<button data-action="tab:${t}" class="shrink-0 px-4 py-1.5 rounded-full text-[14px] font-medium border ${
    state.tab === t ? "border-white text-white bg-white/10" : "border-white/20 text-white/80"}">${t}</button>`;
  return `<header class="pt-4 pb-2">
    <div class="flex gap-2 overflow-x-auto px-4 no-scrollbar">${TABS.map(pill).join("")}</div>
  </header>`;
}

function logo(f, size = "w-9 h-9") {
  return `<div class="${size} ${f.logoBg} rounded-md flex items-center justify-center text-white text-[11px] font-bold shrink-0">${f.logo}</div>`;
}

function toast() {
  return state.toast
    ? `<div class="mx-4 mt-3 rounded-xl bg-groww/15 border border-groww/40 px-3 py-2 text-[13px] text-groww">${state.toast}</div>`
    : "";
}

function dashboard() {
  const funds = active();
  const invested = funds.reduce((s, f) => s + sipFacts(f).invested, 0);
  const value = funds.reduce((s, f) => s + f.value, 0);
  const change = value - invested, up = change >= 0;
  const pct = invested ? Math.abs(change / invested) * 100 : 0;
  const color = up ? "text-groww" : "text-red-400";
  return `${header()}${toast()}
    <section class="px-4 mt-3">
      <div class="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
        <p class="text-[11px] tracking-wide text-white/50">HOLDINGS (${funds.length})</p>
        <p class="text-[28px] font-semibold mt-1">${inr(value)}</p>
        <div class="mt-4 space-y-2.5 text-[13px]">
          <div class="flex justify-between"><span class="text-white/60">Total returns</span><span class="${color}">${up ? "+" : "−"}${inr(Math.abs(change))} (${pct.toFixed(1)}%)</span></div>
          <div class="flex justify-between"><span class="text-white/60">Invested</span><span>${inr(invested)}</span></div>
        </div>
      </div>
      <div class="mt-4 divide-y divide-white/10">
        ${funds.map((f) => {
          const x = sipFacts(f), c = x.change >= 0 ? "text-groww" : "text-red-400";
          return `<button data-action="open-sip:${f.id}" class="w-full text-left py-4 flex justify-between gap-3">
            <div class="flex gap-3 min-w-0">${logo(f)}<div class="min-w-0">
              <p class="text-[14px] font-medium leading-snug">${f.scheme}</p>
              <p class="text-[12px] text-white/50 mt-1">SIP · ${inr(f.monthly)}/month</p></div></div>
            <div class="text-right shrink-0"><p class="${c} text-[14px] font-medium">${inr(f.value)}</p>
              <p class="text-[12px] text-white/40">(${inr(x.invested)})</p></div>
          </button>`;
        }).join("")}
      </div>
    </section>`;
}

function sipsList() {
  const funds = active().map((f) => ({ f, due: nextDue(f) })).sort((a, b) => a.due - b.due);
  return `${header()}${toast()}
    <section class="px-4 mt-3">
      <div class="flex justify-between items-center">
        <p class="text-[15px] font-semibold">Active SIPs (${funds.length})</p>
        <p class="text-[13px] text-white/80"><span class="text-white/60">☰</span> <span class="underline decoration-dotted">Sort by: Due date</span></p>
      </div>
      ${funds.length ? "" : `<p class="mt-8 text-center text-sm text-white/50">No active SIPs.</p>`}
      <div class="mt-2 divide-y divide-white/10">
        ${funds.map(({ f, due }) => {
          const days = daysUntil(due), note = holidayNote(due);
          return `<button data-action="open-sip:${f.id}" class="w-full text-left py-4">
            <div class="flex gap-3">
              ${logo(f)}
              <div class="flex-1 min-w-0">
                <p class="text-[15px] leading-snug">${f.scheme}</p>
                <p class="text-[14px] text-white/70 mt-2">${inr(f.monthly)}${days <= 7 ? ` <span class="text-white/40">•</span> <span class="text-white/60 text-[13px]">▦ Due in ${days} day${days === 1 ? "" : "s"}</span>` : ""}</p>
              </div>
              <div class="w-12 h-14 rounded-lg border border-white/20 flex flex-col items-center justify-center shrink-0">
                <span class="text-[17px] font-semibold">${String(due.getDate()).padStart(2, "0")}</span>
                <span class="text-[12px] text-white/60">${MONTHS[due.getMonth()]}</span>
              </div>
            </div>
            ${note ? `<p class="ml-12 mt-3 rounded-lg bg-white/[0.06] px-3 py-2 text-[12px] text-white/70">${note}</p>` : ""}
          </button>`;
        }).join("")}
      </div>
    </section>`;
}

function placeholder(t) {
  return `${header()}<p class="px-4 mt-10 text-center text-sm text-white/50">${t} is not part of this prototype.</p>`;
}

function sipDetails() {
  const f = current(), due = nextDue(f), note = holidayNote(due);
  const paid = [1, 2].map((k) => { const d = new Date(due); d.setMonth(d.getMonth() - k); return d; });
  const step = (dot, title, date, extra = "") => `<li class="relative pl-7 pb-6">
      <span class="absolute left-0 top-1 w-3 h-3 rounded-full ${dot}"></span>
      <p class="text-[14px]">${title}</p><p class="text-[12px] text-white/50 mt-1">${date}</p>${extra}</li>`;
  return `${toast()}<section class="px-4 pt-5 pb-28 min-h-full">
    <div class="flex items-center justify-between">
      <button data-action="back-tab" class="flex items-center gap-3 text-[17px]"><span class="text-xl">←</span> SIP Details</button>
      <button data-action="cancel-sip" class="text-[13px] font-semibold tracking-wide text-white/70">CANCEL SIP</button>
    </div>
    <p class="mt-6 text-[14px] uppercase tracking-wide text-white/80">${f.scheme}</p>
    <p class="mt-4 text-[26px] font-semibold">${inr(f.monthly)}</p>
    <p class="text-[13px] text-white/70 mt-1">Next installment date: <span class="font-semibold text-white">${fmt(due, true)}</span></p>
    <p class="text-[13px] text-white/70">SIP ID: ${f.sipId}</p>
    <div class="mt-5 py-4 border-y border-white/10 flex justify-between items-center">
      <p class="text-[16px] font-medium">AutoPay Active <span class="text-groww">✓</span></p><span class="text-white/40">⌄</span>
    </div>
    <p class="mt-5 text-[16px] font-medium">Payments &amp; other events</p>
    <ol class="mt-4 border-l border-white/10 ml-1.5">
      ${step("bg-amber-400 -translate-x-1.5", `${ordinal(f.months + 1)} installment: Upcoming in ${daysUntil(due)} days`, fmt(due, true),
        note ? `<p class="mt-2 rounded-lg bg-white/[0.06] px-3 py-2 text-[12px] text-white/70">${note}</p>` : "")}
      ${paid.map((d, i) => step("bg-white/40 -translate-x-1.5", `${ordinal(f.months - i)} installment: Paid ›`, fmt(d, true))).join("")}
    </ol>
    <div class="fixed bottom-[72px] left-1/2 -translate-x-1/2 w-full max-w-[430px] px-4 grid grid-cols-2 gap-3">
      <button class="border border-groww text-groww font-semibold rounded-md py-3 text-[14px]">EDIT SIP</button>
      <button class="bg-groww text-black font-semibold rounded-md py-3 text-[14px]">PAY ONE-TIME</button>
    </div>
  </section>`;
}

/* ---------- The "before you cancel" sheet ---------- */

function answerCard(a) {
  const badge = a.check === "declined" ? "Declined"
    : a.check === "fallback" ? `Fallback shown: the reply failed a check (${a.why})`
    : a.check === "unsure" ? "Not sure what you meant · showing the basics"
    : "Checked: numbers match your SIP · no advice";
  const tone = a.check === "declined" ? "border-white/20" : "border-groww/40";
  return `<article class="mt-3 rounded-2xl border ${tone} p-4">
    <p class="text-[11px] ${a.check === "passed" ? "text-groww" : "text-white/60"}">${badge}</p>
    <h3 class="font-semibold mt-1 text-[15px]">${a.title}</h3>
    <p class="text-[13px] text-white/80 mt-2 leading-relaxed">${a.body}</p>
  </article>`;
}

function optionsBlock(f) {
  const amounts = [100, 250, 500, 1000, 1500].filter((a) => a < f.monthly);
  const row = (id, title, sub) => `<button data-action="option:${id}" class="w-full text-left py-3 flex justify-between items-center">
      <span><span class="block text-[14px]">${title}</span><span class="block text-[12px] text-white/50 mt-0.5">${sub}</span></span>
      <span class="text-white/40">${state.option === id ? "⌃" : "›"}</span></button>`;
  const panel = (id, inner) => (state.option === id ? `<div class="pb-3">${inner}</div>` : "");
  const save = (action, label) => `<button data-action="${action}" class="mt-2 w-full border border-groww text-groww font-semibold rounded-full py-2 text-[13px]">${label}</button>`;
  const due = nextDue(f);
  const blocks = {
    amount: row("amount", "Lower the amount", "Keeps the SIP active")
      + panel("amount", `<select data-action="pick-amount" class="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-[14px]">
          ${amounts.map((a) => `<option value="${a}" ${state.pickAmount === a ? "selected" : ""}>${inr(a)} / month</option>`).join("")}
        </select>${save("save-amount", "Save new amount")}`),
    date: row("date", "Change the debit date", "If the timing is the problem")
      + panel("date", `<select data-action="pick-day" class="w-full bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-[14px]">
          ${Array.from({ length: 28 }, (_, i) => i + 1).map((d) => `<option value="${d}" ${(state.pickDay || f.sipDay) === d ? "selected" : ""}>${ordinal(d)} of every month</option>`).join("")}
        </select>${save("save-date", "Save new date")}`),
    skip: row("skip", "Skip the next instalment", `Only ${fmt(due, true)}; the SIP continues after that`)
      + panel("skip", `<p class="text-[12px] text-white/60">No money is taken on ${fmt(due, true)}. The following instalment runs as usual.</p>${save("save-skip", "Skip this instalment")}`),
  };
  const show = (state.answer?.options ?? ["amount", "date", "skip"]).filter((id) => id !== "amount" || amounts.length);
  if (!show.length) return "";
  return `<div class="mt-4">
    <p class="text-[13px] font-semibold">Other options</p>
    <div class="divide-y divide-white/10">${show.map((id) => blocks[id]).join("")}</div>
  </div>`;
}

function fundSheet() {
  if (!state.sheet) return "";
  const f = current(), k = factPack(f);
  const chips = REASONS.filter((r) => !r.elssOnly || f.elss).map((r) =>
    `<button data-action="reason:${r.id}" class="px-3 py-1.5 rounded-full border text-[12px] ${
      state.reason === r.id ? "border-groww text-groww bg-groww/10" : "border-white/20 text-white/80"}">${r.label}</button>`).join("");
  return `<div class="fixed inset-0 z-[60] max-w-[430px] mx-auto">
    <button data-action="keep" class="absolute inset-0 bg-black/60" aria-label="Close"></button>
    <div class="absolute bottom-0 w-full bg-[#16161c] rounded-t-3xl max-h-[90vh] flex flex-col">
      <div class="overflow-y-auto px-5 pt-3 pb-4">
        <div class="w-10 h-1 rounded-full bg-white/20 mx-auto mb-4"></div>
        <h2 class="text-[18px] font-semibold">Before you cancel</h2>
        <div class="flex gap-3 items-start mt-3">${logo(f, "w-8 h-8")}<p class="text-[13px] text-white/70 leading-snug">${f.scheme}</p></div>

        <div class="mt-4 rounded-2xl bg-white/[0.04] border border-white/10 p-4">
          <p class="text-[12px] text-white/50">Your SIP so far</p>
          <div class="grid grid-cols-3 gap-2 mt-2 text-[13px]">
            <div><p class="text-white/50 text-[11px]">Invested</p><p class="font-semibold">${inr(k.x.invested)}</p></div>
            <div><p class="text-white/50 text-[11px]">Worth today</p><p class="font-semibold">${inr(f.value)}</p></div>
            <div><p class="text-white/50 text-[11px]">Change</p><p class="font-semibold">${pctStr(k.x.pct)}</p></div>
          </div>
          <p class="text-[12px] text-white/50 mt-3">Is this normal at ${f.months} months?</p>
          <p class="text-[13px] text-white/80 mt-1 leading-relaxed">${k.range}</p>
        </div>

        <p class="mt-5 text-[13px] font-semibold">What's making you cancel? <span class="font-normal text-white/50">Optional</span></p>
        <div class="flex flex-wrap gap-2 mt-2">${chips}</div>
        <div class="mt-3 flex gap-2">
          <input id="worry" value="${state.draft.replace(/"/g, "&quot;")}" placeholder="Or say it in your words (English or Hinglish)"
            class="flex-1 bg-white/5 border border-white/10 rounded-full px-3 py-2 text-[13px] outline-none focus:border-groww" />
          <button data-action="ask" class="px-3 rounded-full bg-white/10 text-[13px]">Explain</button>
        </div>
        ${state.answer ? answerCard(state.answer) : ""}

        ${optionsBlock(f)}

        <p class="text-[11px] text-white/50 mt-4">Information about your own SIP, not investment advice. Past ranges don't predict future returns. Sample figures, not live Groww data.</p>
      </div>
      <div class="grid grid-cols-2 gap-3 px-5 pt-3 pb-6 border-t border-white/10 bg-[#16161c]">
        <button data-action="keep" class="border border-white/30 font-semibold rounded-full py-3 text-sm">Keep SIP</button>
        <button data-action="confirm-cancel" class="border border-white/30 font-semibold rounded-full py-3 text-sm">Cancel SIP</button>
      </div>
    </div>
  </div>`;
}

function screen() {
  if (state.screen === "details") return sipDetails();
  if (state.tab === "Dashboard") return dashboard();
  if (state.tab === "SIPs") return sipsList();
  return placeholder(state.tab);
}

function render() {
  renderControls();
  $("screen").innerHTML = screen();
  $("sheet").innerHTML = fundSheet();
  $("tabbar").innerHTML = ["Stocks", "Mutual Funds", "UPI", "Credit"]
    .map((t) => `<span class="flex flex-col items-center w-16 gap-1 ${t === "Mutual Funds" ? "text-sky-400" : "text-white/40"}"><span class="text-lg leading-none">${t === "Mutual Funds" ? "◉" : "○"}</span>${t}</span>`)
    .join("");
}

function closeSheet(toastText) {
  state.sheet = false;
  state.option = null;
  state.toast = toastText || "";
}

document.body.addEventListener("click", (e) => {
  const b = e.target.closest("[data-action]");
  if (!b || b.tagName === "SELECT" || b.tagName === "INPUT") return;
  const [action, arg] = b.dataset.action.split(":");
  const f = current();
  if (!["ask", "reason", "option"].includes(action)) state.toast = "";

  if (action === "tab") { state.tab = arg; state.screen = "tab"; state.sheet = false; }
  if (action === "open-sip") { state.fundId = arg; state.screen = "details"; }
  if (action === "back-tab") state.screen = "tab";
  if (action === "cancel-sip") {
    Object.assign(state, { sheet: true, reason: null, draft: "", answer: null, option: null, pickAmount: null, pickDay: null });
    logEvent("cancel_tap");
  }
  if (action === "reason") {
    state.reason = arg;
    state.answer = gate(reply(arg, f), f);
    logEvent("reason_chosen", { check: state.answer.check });
  }
  if (action === "ask" && state.draft.trim()) {
    const kind = classify(state.draft);
    state.reason = REASONS.some((r) => r.id === kind) ? kind : null;
    state.answer = gate(reply(kind, f), f);
    logEvent("free_text", { kind, check: state.answer.check });
  }
  if (action === "option") state.option = state.option === arg ? null : arg;
  if (action === "save-amount") {
    const amt = state.pickAmount || [100, 250, 500, 1000, 1500].filter((a) => a < f.monthly)[0];
    logEvent("amount_lowered", { from: f.monthly, to: amt });
    f.monthly = amt;
    closeSheet(`SIP amount changed to ${inr(amt)} a month. Your SIP stays active.`);
  }
  if (action === "save-date") {
    const day = state.pickDay || f.sipDay;
    logEvent("date_changed", { from: f.sipDay, to: day });
    f.sipDay = day;
    closeSheet(`Debit date changed to the ${ordinal(day)} of every month.`);
  }
  if (action === "save-skip") {
    const due = nextDue(f);
    state.skipped.add(f.id);
    logEvent("skipped");
    closeSheet(`The ${fmt(due, true)} instalment will be skipped. Your SIP continues after that.`);
  }
  if (action === "keep") { logEvent("kept"); closeSheet(); }
  if (action === "confirm-cancel") {
    logEvent("cancelled");
    state.cancelled.add(state.fundId);
    closeSheet("SIP cancelled. Units you already hold stay invested.");
    state.screen = "tab";
    state.tab = "SIPs";
  }
  if (action === "reset") {
    state.cancelled.clear();
    state.skipped.clear();
    FUNDS.forEach((x, i) => Object.assign(x, ORIGINAL[i]));
    localStorage.removeItem(EVENTS_KEY);
    closeSheet();
    state.screen = "tab";
  }
  render();
});

document.body.addEventListener("input", (e) => {
  if (e.target.id === "worry") state.draft = e.target.value;
});

document.body.addEventListener("keydown", (e) => {
  if (e.target.id === "worry" && e.key === "Enter") document.querySelector('[data-action="ask"]').click();
});

document.body.addEventListener("change", (e) => {
  if (e.target.dataset.action === "pick-amount") state.pickAmount = Number(e.target.value);
  if (e.target.dataset.action === "pick-day") state.pickDay = Number(e.target.value);
});

render();
