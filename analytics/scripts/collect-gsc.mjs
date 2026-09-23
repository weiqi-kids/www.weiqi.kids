#!/usr/bin/env node
// 收集 Google Search Console 的搜尋成效（曝光／點擊／CTR／平均排名、熱門查詢、熱門頁面）。
//
// 憑證（不進 repo）：服務帳號 JSON 金鑰
//   ~/.config/weiqi-kids/google-sa.json，或環境變數 GOOGLE_SA_JSON（整份 JSON 字串）
// 設定：analytics/config.json 的 gsc.siteUrl（例：sc-domain:weiqi.kids）
//
// 服務帳號必須被加入該 GSC 資源的使用者（完整或受限皆可），否則會 403。
// GSC 資料有 2～3 天延遲，預設抓「結束日 = 今天往前 gsc.lagDays 天」的 7 天區間。
//
// 用法：node analytics/scripts/collect-gsc.mjs [YYYY-MM-DD]（指定區間結束日）

import {
  loadConfig, taipeiDate, writeRaw, appendHistory, writeCurrent,
  cleanupRaw, skipMissing, markConnected,
} from './lib/collector.mjs';
import { findCredentials, gscQuery } from './lib/google-data.mjs';

const SOURCE = 'gsc';
const cfg = loadConfig();
const rawDays = cfg.retention?.rawDays ?? 30;
const lagDays = cfg.gsc?.lagDays ?? 3;
const siteUrl = process.env.GSC_SITE_URL || cfg.gsc?.siteUrl || '';
const endDate = process.argv[2] || taipeiDate(-lagDays);

const DOCS = '請參照 analytics/README.md「Search Console 授權步驟」。';

if (!findCredentials()) {
  skipMissing(
    SOURCE,
    `尚未授權，缺服務帳號金鑰（~/.config/weiqi-kids/google-sa.json 或環境變數 GOOGLE_SA_JSON）。${DOCS}`,
    'Google Cloud Console 建立服務帳號 → 下載 JSON 金鑰 → 到 Search Console 設定 → 使用者和權限，把服務帳號 email 加為使用者。',
  );
}
if (!siteUrl) {
  skipMissing(SOURCE, `尚未設定 GSC 站台（analytics/config.json 的 gsc.siteUrl）。${DOCS}`);
}

const start = new Date(`${endDate}T00:00:00Z`);
start.setUTCDate(start.getUTCDate() - 6);
const startDate = start.toISOString().slice(0, 10);

const query = (body) => gscQuery(siteUrl, { startDate, endDate, ...body });

let byDate, byQuery, byPage;
try {
  [byDate, byQuery, byPage] = await Promise.all([
    query({ dimensions: ['date'], rowLimit: 100 }),
    query({ dimensions: ['query'], rowLimit: 25 }),
    query({ dimensions: ['page'], rowLimit: 25 }),
  ]);
} catch (e) {
  skipMissing(SOURCE, `GSC API 呼叫失敗：${e.message}${e.message.includes('403') ? '（多半是服務帳號還沒被加到 Search Console 資源的使用者）' : ''}`);
}

const rows = (res, key) =>
  (res.rows ?? []).map((r) => ({
    [key]: r.keys?.[0] ?? '',
    clicks: r.clicks ?? 0,
    impressions: r.impressions ?? 0,
    ctr: Number(((r.ctr ?? 0) * 100).toFixed(2)),
    position: Number((r.position ?? 0).toFixed(1)),
  }));

const dailyRows = rows(byDate, 'date');
const topQueries = rows(byQuery, 'query');
const topPages = rows(byPage, 'page');

writeRaw(SOURCE, endDate, {
  collectedAt: new Date().toISOString(),
  siteUrl, range: { startDate, endDate },
  daily: dailyRows, topQueries, topPages,
});
for (const row of dailyRows) appendHistory(SOURCE, row);
writeCurrent('gsc-summary', {
  siteUrl,
  range: { startDate, endDate },
  clicks: dailyRows.reduce((s, r) => s + r.clicks, 0),
  impressions: dailyRows.reduce((s, r) => s + r.impressions, 0),
  avgPosition: dailyRows.length
    ? Number((dailyRows.reduce((s, r) => s + r.position, 0) / dailyRows.length).toFixed(1))
    : null,
});
writeCurrent('gsc-top-queries', { siteUrl, range: { startDate, endDate }, queries: topQueries });
writeCurrent('gsc-top-pages', { siteUrl, range: { startDate, endDate }, pages: topPages });
cleanupRaw(SOURCE, rawDays);
markConnected(SOURCE, { siteUrl });

console.log(
  `[gsc] ${startDate}~${endDate}｜曝光 ${dailyRows.reduce((s, r) => s + r.impressions, 0)}、` +
    `點擊 ${dailyRows.reduce((s, r) => s + r.clicks, 0)}、熱門查詢 ${topQueries.length} 筆`,
);
