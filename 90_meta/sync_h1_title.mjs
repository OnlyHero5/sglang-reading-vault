#!/usr/bin/env node
/** Sync H1 with frontmatter title for batch-doc / module-moc files */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const READING = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "sglang_reading");
const SKIP = new Set(["_archive", "_TEMPLATE", "_qa", "progress.md", "PLAN.md"]);

function process(text) {
  const fm = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  if (!fm) return text;
  const titleM = fm[1].match(/^title:\s*"(.+)"\s*$/m);
  if (!titleM) return text;
  const title = titleM[1];
  const rest = text.slice(fm[0].length);
  const newRest = rest.replace(/^#\s+.+\r?\n/, `# ${title}\n`);
  if (newRest === rest) return text;
  return fm[0] + newRest;
}

function walk(d) {
  let n = 0;
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    if (SKIP.has(e.name)) continue;
    const f = path.join(d, e.name);
    if (e.isDirectory()) n += walk(f);
    else if (e.name.endsWith(".md")) {
      const t = fs.readFileSync(f, "utf8");
      const nxt = process(t);
      if (nxt !== t) {
        fs.writeFileSync(f, nxt, "utf8");
        n++;
      }
    }
  }
  return n;
}

console.log("Synced H1 in", walk(READING), "files");
