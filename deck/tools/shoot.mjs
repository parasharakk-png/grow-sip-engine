// Screenshots of the MVP for the deck. Needs `node mvp/dev_server.js` running on :3000.
// Run: node deck/tools/shoot.mjs
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const OUT = new URL("../img/", import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9333;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, "--user-data-dir=/tmp/chrome-deck",
  "--no-first-run", "--hide-scrollbars", "about:blank"], { stdio: "ignore" });

let target;
for (let i = 0; i < 40 && !target; i++) {
  await sleep(250);
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === "page"); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const js = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(OUT + name, Buffer.from(r.result.data, "base64"));
  console.log("saved", name);
};
const click = (sel) => js(`document.querySelector(${JSON.stringify(sel)}).click()`);
const ask = async (text) => {
  await js(`(() => { const i = document.getElementById("worry"); i.value = ${JSON.stringify(text)};
    i.dispatchEvent(new Event("input", { bubbles: true })); document.querySelector('[data-action="ask"]').click(); })()`);
  for (let i = 0; i < 120; i++) {
    await sleep(500);
    const loading = await js(`[...document.querySelectorAll("article p")].some((p) => p.textContent.includes("Checking"))`);
    if (!loading) break;
  }
  await js(`(() => { const a = [...document.querySelectorAll("article")].pop(); a && a.scrollIntoView({ block: "center" }); })()`);
  await sleep(300);
};

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
await send("Page.navigate", { url: "http://127.0.0.1:3000/" });
await sleep(1500);
await js(`localStorage.clear()`);
await send("Page.reload");
await sleep(1500);

await click('[data-action="tab:SIPs"]');
await sleep(300);
await click('[data-action="open-sip:quant-small"]');
await sleep(400);
await shot("01_sip_details.png");

await click('[data-action="cancel-sip"]');
await sleep(500);
await shot("02_sheet.png");

await ask("12 months and still in red. Everyone in office says market will fall more. Should I stop and put it in RD?");
await shot("03_ai_answer.png");

await ask("My autopay failed again this month");
await shot("04_payment_help.png");

await ask("Which fund should I switch to instead?");
await shot("05_advice_declined.png");

ws.close();
chrome.kill();
