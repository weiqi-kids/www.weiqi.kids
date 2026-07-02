#!/usr/bin/env node
// IndexNow 推送：把改動頁 URL 通知 IndexNow（即時收錄，Bing / Yandex / Seznam 等；Google 不採 IndexNow，靠 sitemap 自然收錄）。
// 驗證：金鑰檔部署在 https://www.weiqi.kids/<key>.txt（static/<key>.txt），IndexNow 會抓它比對。
// 用法：node analytics/scripts/indexnow-ping.mjs <url> [<url> ...]
// 由大腦層 seo-brain-cron.sh 於改動頁 push 後呼叫。

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(readFileSync(join(here, 'indexnow.config.json'), 'utf8'));

const urls = process.argv.slice(2).filter(Boolean);
if (!urls.length) {
  console.error('用法：indexnow-ping.mjs <url> [<url> ...]');
  process.exit(2);
}

const body = { host: cfg.host, key: cfg.key, keyLocation: cfg.keyLocation, urlList: urls };
const res = await fetch('https://api.indexnow.org/indexnow', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json; charset=utf-8' },
  body: JSON.stringify(body),
});
const txt = await res.text().catch(() => '');
// 200 OK / 202 Accepted 皆為成功；422 多為 keyLocation 尚未部署可讀（部署後即可）。
console.log(`IndexNow → ${res.status} ${res.statusText}（${urls.length} 個 URL）${txt ? ' ' + txt.slice(0, 120) : ''}`);
process.exit(res.ok ? 0 : 1);
