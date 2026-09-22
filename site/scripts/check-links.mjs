// 檢查 dist/client 內所有站內連結都指向存在的頁面或檔案（或 _redirects 中的舊網址）。
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = 'dist/client';
const redirects = new Set(readFileSync(join(ROOT, '_redirects'), 'utf8').split('\n').filter((l) => l && !l.startsWith('#')).map((l) => l.split(' ')[0]));
const dynamic = [/^\/api\//, /^\/auth\//, /^\/account\//, /^\/admin\//, /^\/camp\/courses\/[^/]+\/(forum|materials)\//];
const broken = [];
function walk(dir) {
  for (const n of readdirSync(dir)) {
    const p = join(dir, n);
    if (statSync(p).isDirectory()) walk(p);
    else if (n.endsWith('.html')) check(p);
  }
}
function exists(path) {
  const clean = decodeURI(path.split('#')[0].split('?')[0]);
  if (!clean || dynamic.some((re) => re.test(clean)) || redirects.has(clean)) return true;
  const f = join(ROOT, clean);
  return existsSync(f) && statSync(f).isFile() || existsSync(join(f, 'index.html'));
}
function check(file) {
  const html = readFileSync(file, 'utf8');
  for (const m of html.matchAll(/\s(?:href|src)="(\/[^"]*)"/g)) {
    const url = m[1];
    if (url.startsWith('//') || url.startsWith('/_astro/')) continue;
    if (!exists(url)) broken.push(`${file.slice(ROOT.length)} → ${url}`);
  }
}
walk(ROOT);
if (broken.length) { console.error(`站內斷連結 ${broken.length} 個：\n${[...new Set(broken)].join('\n')}`); process.exit(1); }
console.log('站內連結檢查通過。');
