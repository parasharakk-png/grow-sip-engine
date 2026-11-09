// Checks the deck (minimum font, nothing outside a slide), writes deck/preview/sN.png and deck/NL_Groww.pdf.
// Run: node deck/tools/build.mjs
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const DECK = new URL("../", import.meta.url).pathname;
const MIN_PX = 19; // 1280 px slides print at 960 pt wide, so 19 px ≈ 14.25 pt
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9334;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, "--user-data-dir=/tmp/chrome-deck-build",
  "--no-first-run", "--hide-scrollbars", "--allow-file-access-from-files", "about:blank"], { stdio: "ignore" });

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

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1328, height: 820, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: `file://${DECK}deck.html` });
await sleep(1500);

const problems = await js(`(() => {
  const out = [];
  document.querySelectorAll(".slide").forEach((slide, i) => {
    const box = slide.getBoundingClientRect();
    slide.querySelectorAll("*").forEach((el) => {
      const text = [...el.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent.trim()).join("");
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) return;
      const size = parseFloat(getComputedStyle(el).fontSize);
      if (text && size < ${MIN_PX}) out.push("slide " + (i + 1) + ": " + size + "px  " + text.slice(0, 50));
      if (r.right > box.right + 1 || r.bottom > box.bottom + 1)
        out.push("slide " + (i + 1) + ": outside the slide  <" + el.tagName.toLowerCase() + "> " + (el.textContent || "").trim().slice(0, 50));
    });
    const foot = slide.querySelector(".foot");
    if (foot) {
      const fTop = foot.getBoundingClientRect().top;
      [...slide.children].filter((c) => !c.classList.contains("foot") && !c.classList.contains("note")).forEach((c) => {
        if (c.getBoundingClientRect().bottom > fTop - 4)
          out.push("slide " + (i + 1) + ": runs into the footer  " + (c.textContent || "").trim().slice(0, 50));
      });
    }
  });
  return out;
})()`);

if (problems.length) {
  console.log(problems.join("\n"));
  ws.close(); chrome.kill(); process.exit(1);
}
console.log(`fonts ≥ ${MIN_PX}px and every slide fits`);

mkdirSync(`${DECK}preview`, { recursive: true });
await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
for (let n = 1; n <= 10; n++) {
  await send("Page.navigate", { url: `file://${DECK}deck.html?s=${n}` });
  await sleep(500);
  const r = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: 1280, height: 720, scale: 1 } });
  writeFileSync(`${DECK}preview/s${n}.png`, Buffer.from(r.result.data, "base64"));
}
console.log("previews written");

await send("Page.navigate", { url: `file://${DECK}deck.html` });
await sleep(1000);
const pdf = await send("Page.printToPDF", { preferCSSPageSize: true, printBackground: true });
writeFileSync(`${DECK}NL_Groww.pdf`, Buffer.from(pdf.result.data, "base64"));
console.log("wrote deck/NL_Groww.pdf");

ws.close();
chrome.kill();
