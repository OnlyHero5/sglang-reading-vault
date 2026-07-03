import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const root = path.dirname(fileURLToPath(import.meta.url));
const vaultRoot = path.join(root, '..');

function walk(dir, acc = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory() && e.name !== '_TEMPLATE') walk(p, acc);
    else if (e.name.endsWith('-05-checkpoint.md')) acc.push(p);
  }
  return acc;
}

function modulePrefix(filePath) {
  const base = path.basename(filePath);
  const m = base.match(/^(.+)-05-checkpoint\.md$/);
  return m ? m[1] : null;
}

function fixCheckpoint(filePath) {
  let text = fs.readFileSync(filePath, 'utf8');
  const mod = modulePrefix(filePath);
  if (!mod) return { file: filePath, changed: false };

  let changed = false;
  const lines = text.split(/\r?\n/);
  const out = [];
  let inReader = false;

  for (const line of lines) {
    let next = line;
    if (/^##\s+读者自测/.test(next)) inReader = true;
    else if (/^##\s+/.test(next) && !/^##\s+读者自测/.test(next)) inReader = false;

    if (inReader && /^- \[x\]/i.test(next)) {
      next = next.replace(/^- \[x\]/i, '- [ ]');
      changed = true;
    }

    const replacements = [
      ['README.md', `${mod}-00-MOC.md`],
      ['01-核心概念.md', `${mod}-01-核心概念.md`],
      ['02-源码走读.md', `${mod}-02-源码走读.md`],
      ['03-数据流与交互.md', `${mod}-03-数据流与交互.md`],
      ['04-关键问题.md', `${mod}-04-关键问题.md`],
    ];
    for (const [from, to] of replacements) {
      if (next.includes(from)) {
        const nl = next.replaceAll(from, to);
        if (nl !== next) {
          next = nl;
          changed = true;
        }
      }
    }

    out.push(next);
  }

  const newText = out.join('\n');
  if (changed) fs.writeFileSync(filePath, newText, 'utf8');
  return { file: filePath, changed };
}

const results = [];
for (const r of ['sglang_reading', 'slime_reading']) {
  const dir = path.join(vaultRoot, r);
  if (!fs.existsSync(dir)) continue;
  for (const f of walk(dir)) results.push(fixCheckpoint(f));
}

console.log(`Modified: ${results.filter((x) => x.changed).length}`);
for (const x of results.filter((x) => x.changed)) {
  console.log(' ', path.relative(vaultRoot, x.file));
}
