/* Groww-style Mutual Funds prototype: Dashboard, SIPs, SIP Details and the cancel sheet.
   Fund figures are samples for the prototype, not live Groww data. The check-in never
   names a fund to buy and declines fear or urgency requests. */

const FUNDS = [
  { id: "absl-psu", scheme: "Aditya Birla Sun Life PSU Equity Fund Direct Growth", logo: "AB", logoBg: "bg-[#c8102e]",
    monthly: 200, months: 14, value: 3240, sipDay: 5, sipId: "SIP1140919244119463",
    returns: { 5: 18.4, 3: 24.1 }, investors: "21.6 lakh", activeSips: "6.4 lakh", avgSip: 4200 },
  { id: "groww-total", scheme: "Groww Nifty Total Market Index Fund Direct Growth", logo: "G",
    logoBg: "bg-gradient-to-br from-sky-400 to-groww", monthly: 500, months: 9, value: 4710, sipDay: 5,
    sipId: "SIP1140919244120118", returns: { 5: null, 3: 13.2 }, investors: "1.9 lakh", activeSips: "78,000",
    avgSip: 1800 },
  { id: "bandhan-n50", scheme: "Bandhan Nifty 50 Index Fund Direct Plan Growth", logo: "B", logoBg: "bg-[#e2231a]",
    monthly: 500, months: 18, value: 10350, sipDay: 28, sipId: "SIP1140919244121702",
    returns: { 5: 14.6, 3: 12.9 }, investors: "4.3 lakh", activeSips: "1.6 lakh", avgSip: 2500 },
  { id: "quant-small", scheme: "Quant Small Cap Fund Direct Plan Growth", logo: "q", logoBg: "bg-black border border-white/30",
    monthly: 2000, months: 12, value: 22600, sipDay: 28, sipId: "SIP1140919244122377",
    returns: { 5: 31.2, 3: 22.8 }, investors: "28.4 lakh", activeSips: "9.1 lakh", avgSip: 3100 },
  { id: "sbi-psu", scheme: "SBI PSU Direct Plan Growth", logo: "SBI", logoBg: "bg-[#1a75bc]",
    monthly: 500, months: 6, value: 2870, sipDay: 28, sipId: "SIP1140919244123051",
    returns: { 5: 27.5, 3: 30.4 }, investors: "3.7 lakh", activeSips: "1.2 lakh", avgSip: 2300 },
];

const TABS = ["Explore", "Dashboard", "SIPs", "Watchlist"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const state = {
  tab: "Dashboard",
  screen: "tab",
  fundId: null,
  sheet: false,
  years: 5,
  cancelled: new Set(),
  toast: "",
  draft: "",
  answer: null,
};

const $ = (id) => document.getElementById(id);
const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");
const today = () => { const d = new Date(); d.setHours(0, 0, 0, 0); return d; };
const fmt = (d, year = false) => `${String(d.getDate()).padStart(2, "0")} ${MONTHS[d.getMonth()]}${year ? " " + d.getFullYear() : ""}`;
const ordinal = (n) => {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

function active() { return FUNDS.filter((f) => !state.cancelled.has(f.id)); }
function current() { return FUNDS.find((f) => f.id === state.fundId); }

function sipFacts(p) {
  const invested = p.monthly * p.months;
  const change = p.value - invested;
  return { invested, change, pct: invested ? (change / invested) * 100 : 0 };
}

/** Next instalment on the SIP day, from tomorrow onwards. */
function nextDue(f) {
  const t = today();
  const d = new Date(t.getFullYear(), t.getMonth(), f.sipDay);
  if (d <= t) d.setMonth(d.getMonth() + 1);
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

function classify(text) {
  const t = text.toLowerCase();
  if (/which fund|should i buy|best fund|recommend|kahan invest|kaunsa fund/.test(t)) return "advice";
  if (/fear|fomo|urgent|don't miss|everyone is buying|sab log/.test(t)) return "fomo";
  if (/lock|elss|3 year|teen saal/.test(t)) return "lockin";
  if (/fail|bounce|mandate|autopay|debit|nahi kata|balance/.test(t)) return "payment";
  if (/fd|rd|gold|jewell|lump\s?sum|band karke/.test(t)) return "switch";
  if (/red|loss|gir|negative|xirr|return/.test(t)) return "returns";
  return "unsure";
}

function reply(kind, p, f) {
  const sign = f.change >= 0 ? "+" : "−";
  const facts = `You have put in ${inr(f.invested)} over ${p.months} months. It is worth ${inr(p.value)} today (${sign}${inr(Math.abs(f.change))}, ${f.pct.toFixed(1)}%).`;
  const options = "You can continue, reduce the amount, skip one month, change the debit date, or stop. Stopping stays your choice.";
  const base = {
    advice: { ok: false, title: "This asks for a fund recommendation", body: "This check-in doesn't suggest funds or say what to buy. It only explains your own SIP. Ask about your SIP, not which fund to pick.", source: null },
    fomo: { ok: false, title: "This asks for a fear or urgency nudge", body: "This check-in doesn't use fear, FOMO or urgency. It explains your own SIP and lists the options, including stopping.", source: null },
    lockin: { ok: true, title: "This is about a lock-in rule", body: facts + " In an ELSS (tax-saving) fund, each instalment is locked for 3 years from the date it was invested, not from the day you started. A stop ends future instalments; money already invested stays locked until each instalment's own date.", source: "AMFI investor education · scheme information document" },
    payment: { ok: true, title: "This is about a failed instalment", body: facts + " An instalment usually fails for one of three reasons: the account was short on the debit date, the autopay is still on an old account, or the bank or gateway rejected it. The app should name which one and offer pay-now or update-account. Nothing here can see your bank.", source: "NPCI e-mandate guidance" },
    switch: { ok: true, title: "This is about moving the money elsewhere", body: facts + " Moving new savings to an FD, RD or gold scheme means the SIP stops counting as active, even though you are still saving. Reducing the SIP keeps it active at an amount you can afford.", source: "AMFI investor education · SIP basics" },
    returns: { ok: true, title: "This is about returns looking red", body: facts + " A SIP buys units every month. When the value is below what you put in, the recent units were bought at lower prices. Whether that is a problem depends on when you need the money, not on the colour on the dashboard. This screen cannot say what happens next.", source: "AMFI investor education · rupee cost averaging" },
    unsure: { ok: true, title: "Not sure what to do", body: facts + " The dashboard shows a gain or loss. It doesn't show whether that is normal for how long you have been investing or for when you need the money. " + options, source: "AMFI investor education · SIP basics" },
  };
  const r = { ...base[kind] };
  if (r.ok && kind !== "unsure") r.body += " " + options;
  return r;
}

function renderControls() {
  const btn = (action, label) =>
    `<button data-action="${action}" class="text-xs font-semibold px-3 py-1.5 rounded-full border bg-white text-neutral-800 border-neutral-300">${label}</button>`;
  $("controls").innerHTML = [
    btn("tab:Dashboard", "Open Dashboard"),
    btn("tab:SIPs", "Open SIPs"),
    btn("reset", "Restore cancelled SIPs"),
  ].join("");
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
  return `<section class="px-4 pt-5 pb-28 min-h-full">
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

function checkin() {
  const p = current();
  return `<section class="px-4 pt-5">
    <button data-action="back-details" class="text-white/60 text-sm">← SIP Details</button>
    <h2 class="text-xl font-semibold mt-3">Check my SIP</h2>
    <p class="text-sm text-white/60 mt-1">${p.scheme}</p>
    <p class="text-[11px] text-white/40 mt-3">This is not investment advice. It explains your own SIP and lists options. It never names a fund to buy.</p>
    <textarea id="worry" rows="4" placeholder="What's on your mind? English or Hinglish."
      class="mt-4 w-full bg-white/5 border border-white/10 rounded-xl p-3 text-sm outline-none focus:border-groww">${state.draft}</textarea>
    <button data-action="ask" class="mt-3 w-full bg-groww text-black font-semibold rounded-full py-3 text-sm">Check my SIP</button>
    ${state.answer ? answerCard(state.answer) : ""}
  </section>`;
}

function answerCard(a) {
  const badge = a.ok ? "No advice · Numbers verified" + (a.source ? " · " + a.source : "") : "Declined";
  return `<article class="mt-4 rounded-2xl border ${a.ok ? "border-groww/40" : "border-red-400/40"} p-4">
    <p class="text-[11px] ${a.ok ? "text-groww" : "text-red-300"}">${badge}</p>
    <h3 class="font-semibold mt-1">${a.title}</h3>
    <p class="text-sm text-white/80 mt-2 leading-relaxed">${a.body}</p>
    <p class="text-[11px] text-white/40 mt-3">Rules-based preview. The model call is the next step; the number and advice checks stay.</p>
  </article>`;
}

function pastReturn(f) {
  const r = f.returns[state.years];
  const picker = `<select data-action="years" aria-label="Return period"
      class="bg-white/10 border border-white/20 rounded-md text-[13px] px-1.5 py-0.5 outline-none">
      ${[5, 3].map((y) => `<option value="${y}" ${state.years === y ? "selected" : ""}>${y} years</option>`).join("")}
    </select>`;
  const value = r == null
    ? `<span class="text-white/60 text-[13px]">Not available: fund is under ${state.years} years old</span>`
    : `<span class="font-semibold">${r.toFixed(1)}%</span><span class="text-white/60 text-[13px]">over</span>`;
  return `<div class="py-3 border-b border-white/10">
      <p class="text-[12px] text-white/60">Past return of this fund</p>
      <div class="flex items-center gap-2 mt-1 flex-wrap">${value}${picker}</div>
    </div>`;
}

function fundSheet() {
  if (!state.sheet) return "";
  const f = current();
  const line = (k, v) => `<div class="py-3 border-b border-white/10">
      <p class="text-[12px] text-white/60">${k}</p><p class="font-semibold mt-1">${v}</p></div>`;
  return `<div class="fixed inset-0 z-[60] max-w-[430px] mx-auto">
    <button data-action="close-sheet" class="absolute inset-0 bg-black/60" aria-label="Close"></button>
    <div class="absolute bottom-0 w-full bg-[#16161c] rounded-t-3xl px-5 pt-3 pb-8 max-h-[88vh] overflow-y-auto">
      <div class="w-10 h-1 rounded-full bg-white/20 mx-auto mb-4"></div>
      <div class="flex gap-3 items-start">${logo(f)}<div>
        <h2 class="text-[16px] font-semibold leading-snug">${f.scheme}</h2>
        <p class="text-[12px] text-white/50 mt-1">Your SIP: ${inr(f.monthly)}/month · ${f.months} instalments paid</p></div></div>
      <div class="mt-3">
        ${pastReturn(f)}
        ${line("People invested in this fund", f.investors)}
        ${line("Active monthly SIPs", f.activeSips)}
        ${line("Average SIP of active users", inr(f.avgSip) + " / month")}
        ${line("Minimum horizon", "3 years to see the returns")}
      </div>
      <p class="text-[11px] text-amber-200/80 mt-3">Sample figures for this prototype. They are not live Groww data.</p>
      <p class="text-[12px] text-white/70 mt-2">Past returns don't guarantee future returns. Mutual fund investments are subject to market risk. This is not investment advice.</p>
      <div class="mt-5 grid grid-cols-2 gap-3">
        <button data-action="close-sheet" class="border border-white/30 font-semibold rounded-full py-3 text-sm">Keep SIP</button>
        <button data-action="confirm-cancel" class="border border-white/30 font-semibold rounded-full py-3 text-sm">Cancel SIP</button>
      </div>
      <button data-action="open-checkin" class="mt-4 w-full text-center text-[13px] text-groww">Not sure? Check my SIP</button>
    </div>
  </div>`;
}

function screen() {
  if (state.screen === "details") return sipDetails();
  if (state.screen === "checkin") return checkin();
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

document.body.addEventListener("click", (e) => {
  const b = e.target.closest("[data-action]");
  if (!b || b.tagName === "SELECT") return;
  const [action, arg] = b.dataset.action.split(":");
  if (action !== "ask") state.toast = "";
  if (action === "tab") { state.tab = arg; state.screen = "tab"; state.sheet = false; }
  if (action === "open-sip") { state.fundId = arg; state.screen = "details"; state.answer = null; state.draft = ""; }
  if (action === "back-tab") state.screen = "tab";
  if (action === "cancel-sip") { state.sheet = true; state.years = 5; }
  if (action === "close-sheet") state.sheet = false;
  if (action === "confirm-cancel") {
    state.cancelled.add(state.fundId);
    state.sheet = false;
    state.screen = "tab";
    state.tab = "SIPs";
    state.toast = "SIP cancelled. Units you already hold stay invested.";
  }
  if (action === "open-checkin") { state.sheet = false; state.screen = "checkin"; }
  if (action === "back-details") state.screen = "details";
  if (action === "reset") { state.cancelled.clear(); state.screen = "tab"; state.sheet = false; }
  if (action === "ask") {
    state.draft = $("worry") ? $("worry").value : state.draft;
    const p = current();
    state.answer = reply(classify(state.draft), p, sipFacts(p));
  }
  render();
  const box = $("worry");
  if (box) box.value = state.draft;
});

document.body.addEventListener("change", (e) => {
  if (e.target.dataset.action !== "years") return;
  state.years = Number(e.target.value);
  render();
});

render();
