/* POST /api/explain  { text, facts }  ->  { intent, in_scope, confidence, title, reply, options }
   Serverless function (Vercel). The key stays on the server; the browser gates whatever comes back. */

const { callModel } = require("./_prompt");

module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  if (!process.env.ANTHROPIC_API_KEY) return res.status(503).json({ error: "model not configured" });
  const { text, facts } = req.body || {};
  if (typeof text !== "string" || !text.trim() || typeof facts !== "string" || facts.length > 2000) {
    return res.status(400).json({ error: "text and facts are required" });
  }
  try {
    const { parsed } = await callModel(text, facts);
    if (!parsed) return res.status(502).json({ error: "unusable model reply" });
    return res.status(200).json(parsed);
  } catch (e) {
    return res.status(502).json({ error: String(e.message || e).slice(0, 120) });
  }
};
