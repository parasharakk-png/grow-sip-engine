/* Groww-style SIP check-in. Numbers come only from sipFacts(). Answers are rules
   until the model call is added; advice requests are declined. */

const PROFILES = {
  P1: { name: "P1", scheme: "Parag Parikh Flexi Cap", monthly: 3000, months: 12, value: 34200,
    pastReturn: "18.4% over 5 years", investors: "21.6 lakh", activeSips: "6.4 lakh", avgSip: 4200,
    note: "Red after 12 months. A colleague said to hold." },
  P4: { name: "P4", scheme: "Axis ELSS Tax Saver", monthly: 10000, months: 10, value: 104000,
    pastReturn: "14.1% over 5 years", investors: "9.8 lakh", activeSips: "3.1 lakh", avgSip: 3500,
    note: "ELSS. Learned each instalment is locked for 3 years." },
  P5: { name: "P5", scheme: "Mirae Asset Large Cap", monthly: 15000, months: 15, value: 230000,
    pastReturn: "15.6% over 5 years", investors: "14.2 lakh", activeSips: "4.7 lakh", avgSip: 5000,
    note: "Flat after 15 months. A friend suggested a gold scheme." },
};

const state = {
  profile: "P1",
  moment: "dashboard",
  screen: "dashboard",
  sheet: false,
  draft: "",
  answer: null,
};

const $ = (id) => document.getElementById(id);
const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");

function sipFacts(p) {
  const invested = p.monthly * p.months;
  const change = p.value - invested;
  return { invested, change, pct: invested ? (change / invested) * 100 : 0 };
}

function current() { return PROFILES[state.profile]; }

function classify(text) {
  const t = text.toLowerCase();
  if (/which fund|should i buy|best fund|recommend|kahan invest|kaunsa fund/.test(t)) return "advice";
  if (/fear|fomo|urgent|don't miss|everyone is buying|sab log/.test(t)) return "fomo";
  if (/lock|elss|3 year|teen saal/.test(t)) return "lockin";
  if (/fail|bounce|mandate|autopay|debit|nahi kata|balance/.test(t)) return "payment";
  if (/fd|rd|gold|jewell|lump\s?sum|band karke/.test(t)) return "switch";
  if (/red|loss|gir|negative|loss|xirr|return/.test(t)) return "returns";
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
  const r = base[kind];
  if (r.ok && kind !== "unsure") r.body += " " + options;
  return r;
}

function renderControls() {
  const btn = (action, label, on) =>
    `<button data-action="${action}" class="text-xs font-semibold px-3 py-1.5 rounded-full border ${on ? "bg-ink text-white border-ink" : "bg-white text-neutral-800 border-neutral-300"}">${label}</button>`;
  $("controls").innerHTML = [
    ...Object.keys(PROFILES).map((id) => btn("profile:" + id, id + " · " + PROFILES[id].note.split(".")[0], state.profile === id)),
    btn("moment:dashboard", "Dashboard", state.moment === "dashboard"),
    btn("moment:stop", "You tapped Stop", state.moment === "stop"),
  ].join("");
}

function header() {
  return `<header class="px-4 pt-5 pb-2">
    <div class="flex items-center gap-2">
      <div class="w-8 h-8 rounded-full bg-gradient-to-br from-sky-400 to-groww"></div>
      <h1 class="text-[17px] font-semibold flex-1">Mutual Funds</h1>
      <span class="text-white/80 text-lg">⌕</span>
    </div>
    <div class="flex gap-5 mt-4 text-[13px] text-white/50">
      <span>Explore</span><span class="text-white font-semibold border-b-2 border-white pb-2">Dashboard</span>
      <span>SIPs</span><span>Watchlist</span>
    </div></header>`;
}

function row(k, v, c = "") {
  return `<div class="flex justify-between"><span class="text-white/60">${k}</span><span class="${c}">${v}</span></div>`;
}

function checkin() {
  const p = current();
  return `<section class="px-4 pt-5">
    <button data-action="back" class="text-white/60 text-sm">← Dashboard</button>
    <h2 class="text-xl font-semibold mt-3">Before you stop</h2>
    <p class="text-sm text-white/60 mt-1">${p.name}'s SIP · ${p.note}</p>
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

function fundSheet() {
  if (!state.sheet) return "";
  const p = current();
  const line = (k, v) => `<div class="flex justify-between gap-4 py-3 border-b border-white/10 text-sm"><span class="text-white/60">${k}</span><span class="font-medium text-right">${v}</span></div>`;
  return `<div class="fixed inset-0 z-[60] max-w-[430px] mx-auto">
    <button data-action="close-sheet" class="absolute inset-0 bg-black/60" aria-label="Close"></button>
    <div class="absolute bottom-0 w-full bg-[#16161c] rounded-t-3xl px-5 pt-3 pb-8 max-h-[85vh] overflow-y-auto">
      <div class="w-10 h-1 rounded-full bg-white/20 mx-auto mb-4"></div>
      <h2 class="text-lg font-semibold">${p.scheme}</h2>
      <p class="text-[12px] text-white/50 mt-1">The fund this SIP is already in. Updated daily.</p>
      <div class="mt-2">
        ${line("Past return of this fund", p.pastReturn)}
        ${line("People invested in this fund", p.investors)}
        ${line("Active monthly SIPs", p.activeSips)}
        ${line("Average SIP of active users", inr(p.avgSip) + " / month")}
        ${line("Minimum horizon", "3 years to see the returns")}
      </div>
      <p class="text-[11px] text-amber-200/80 mt-3">Sample figures for this prototype. They are not live Groww data.</p>
      <p class="text-[12px] text-white/70 mt-3">Mutual fund investments are subject to market risk. This is not investment advice.</p>
      <button data-action="close-sheet" class="mt-5 w-full bg-white text-black font-semibold rounded-full py-3 text-sm">Close</button>
    </div>
  </div>`;
}

function render() {
  renderControls();
  $("screen").innerHTML = state.screen === "checkin" ? checkin() : rawScreen();
  $("sheet").innerHTML = fundSheet();
  $("tabbar").innerHTML = ["Stocks", "F&O", "Mutual Funds", "UPI", "Loans"]
    .map((t) => `<span class="flex flex-col items-center w-14 ${t === "Mutual Funds" ? "text-sky-400" : "text-white/40"}"><span class="text-base">${t === "Mutual Funds" ? "◉" : "○"}</span>${t}</span>`)
    .join("");
}

function rawScreen() {
  if (state.screen === "checkin") return checkin();
  const p = current(), f = sipFacts(p), up = f.change >= 0, color = up ? "text-groww" : "text-red-400";
  return `${header()}
    <section class="px-4 mt-3">
      <div class="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
        <p class="text-[11px] tracking-wide text-white/50">HOLDINGS (1)</p>
        <p class="text-[28px] font-semibold mt-1">${inr(p.value)}</p>
        <div class="mt-4 space-y-2.5 text-[13px]">
          ${row("Total returns", `${up ? "+" : "−"}${inr(Math.abs(f.change))} (${Math.abs(f.pct).toFixed(1)}%)`, color)}
          ${row("Invested", inr(f.invested))}
        </div>
        <p class="text-[11px] text-white/40 mt-3">1-day returns and XIRR aren't shown. This demo only displays numbers it calculates from the amount and the months.</p>
      </div>
      <button data-action="open-sheet" class="w-full text-left mt-6 flex justify-between items-start">
        <div>
          <p class="text-[15px] font-medium">${p.scheme}</p>
          <p class="text-[12px] text-white/50 mt-0.5">${p.name} · ₹${p.monthly.toLocaleString("en-IN")}/month · month ${p.months}</p>
          <p class="text-[11px] text-white/40 mt-1">SIP · tap for details</p>
        </div>
        <div class="text-right">
          <p class="${color} text-[15px] font-medium">${inr(p.value)}</p>
          <p class="text-[12px] text-white/40">(${inr(f.invested)})</p>
        </div>
      </button>
      ${state.moment === "stop" ? `<div class="mt-5 rounded-xl bg-white/5 border border-white/10 p-3 text-sm">You tapped Stop. <button data-action="open-sheet" class="text-groww font-semibold">See the fund details →</button></div>` : ""}
    </section>`;
}

document.body.addEventListener("click", (e) => {
  const b = e.target.closest("[data-action]");
  if (!b) return;
  const [action, arg] = b.dataset.action.split(":");
  if (action === "profile") { state.profile = arg; state.answer = null; state.sheet = false; }
  if (action === "moment") { state.moment = arg; state.screen = "dashboard"; state.sheet = arg === "stop"; state.answer = null; }
  if (action === "open-sheet") state.sheet = true;
  if (action === "close-sheet") state.sheet = false;
  if (action === "open-checkin") state.screen = "checkin";
  if (action === "back") state.screen = "dashboard";
  if (action === "ask") {
    state.draft = $("worry") ? $("worry").value : state.draft;
    const p = current();
    state.answer = reply(classify(state.draft), p, sipFacts(p));
  }
  render();
  const box = $("worry");
  if (box) box.value = state.draft;
});

render();
