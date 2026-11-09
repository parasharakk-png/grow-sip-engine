// Local preview with the AI endpoint: node mvp/dev_server.js  ->  http://localhost:3000
// Reads the key from discovery-engine/.env; the key never reaches the browser.
const http = require("http");
const fs = require("fs");
const path = require("path");
const { callModel } = require("./api/_prompt.js");

const ROOT = __dirname;
const envFile = path.join(ROOT, "../discovery-engine/.env");
if (fs.existsSync(envFile)) {
  for (const line of fs.readFileSync(envFile, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*"?([^"\n]*)"?\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
  }
}
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".css": "text/css" };

http.createServer(async (req, res) => {
  const url = req.url.split("?")[0];
  if (url === "/api/explain" && req.method === "POST") {
    let body = "";
    for await (const chunk of req) body += chunk;
    try {
      const { text, facts } = JSON.parse(body);
      const { parsed } = await callModel(String(text).slice(0, 600), String(facts), process.env);
      res.writeHead(parsed ? 200 : 502, { "content-type": "application/json" });
      res.end(JSON.stringify(parsed || { error: "unusable model reply" }));
    } catch (e) {
      const detail = [e.message, e.cause && e.cause.message].filter(Boolean).join(": ");
      res.writeHead(String(e.message).includes("not configured") ? 503 : 502, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: String(detail || e).slice(0, 160) }));
    }
    console.log(`explain ${res.statusCode}`);
    return;
  }
  const file = path.join(ROOT, url === "/" ? "index.html" : path.normalize(url));
  if (!file.startsWith(ROOT) || file.includes(`${path.sep}api${path.sep}`) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404); res.end("Not found"); return;
  }
  res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-store" });
  fs.createReadStream(file).pipe(res);
}).listen(3000, () => console.log("MVP with AI: http://localhost:3000"));
