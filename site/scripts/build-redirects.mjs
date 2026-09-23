// 從 data/legacy-map.json 產生 public/_redirects（ADR 0009）。
// - 舊中文網址：兩種寫法（有／無結尾斜線）都 301 到新網址。
// - 舊翻譯語系網址 /{locale}/…：先 301 回中文路徑，再由上面的規則轉到新網址。
import { readFileSync, writeFileSync } from 'node:fs';

const LOCALES = ['zh-cn', 'zh-hk', 'en', 'ja', 'ko', 'es', 'pt', 'hi', 'id', 'ar'];
const map = JSON.parse(readFileSync('data/legacy-map.json', 'utf8'));

const lines = ['# 由 scripts/build-redirects.mjs 產生，勿手動修改'];
const seen = new Set();
const add = (from, to) => {
  if (from === to || seen.has(from)) return;
  seen.add(from);
  lines.push(`${from} ${to} 301`);
};
for (const e of map) {
  if (e.old === '/') continue;
  add(e.old, e.redirect);
  add(e.old.replace(/\/$/, ''), e.redirect);
}
for (const l of LOCALES) {
  add(`/${l}`, '/');
  add(`/${l}/`, '/');
  lines.push(`/${l}/* /:splat 301`);
}
writeFileSync('public/_redirects', `${lines.join('\n')}\n`);

// robots.txt：正式站開放索引，staging 一律擋掉
const production = process.env.SITE_ENV === 'production';
writeFileSync('public/robots.txt', production
  ? `User-agent: *\nAllow: /\nDisallow: /account/\nDisallow: /admin/\nDisallow: /auth/\nDisallow: /api/\n\nSitemap: https://www.weiqi.kids/sitemap-index.xml\n`
  : `# staging：不開放索引\nUser-agent: *\nDisallow: /\n`);
console.log(production ? 'robots.txt：正式站版本' : 'robots.txt：staging 版本（Disallow: /）');
console.log(`轉址規則 ${lines.length - 1} 條已寫入 public/_redirects。`);
