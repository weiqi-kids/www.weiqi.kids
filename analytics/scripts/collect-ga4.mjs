#!/usr/bin/env node
// 收集 GA4（評量 ID G-16V1KSEH6W，新舊站共用）的每日網站指標。
//
// 抓三份報表：
//   1. 每日總量：工作階段、活躍使用者、瀏覽、平均參與時間
//   2. 熱門頁面：pagePath × 瀏覽數（Top 20）
//   3. 站外點擊事件：eventName（見 analytics/config.json 的 ga4.outboundEventNames）× link_url
//
// 憑證（不進 repo）：服務帳號 JSON 金鑰
//   ~/.config/weiqi-kids/google-sa.json，或環境變數 GOOGLE_SA_JSON（整份 JSON 字串）
// 設定：analytics/config.json 的 ga4.propertyId（數值資源 ID，不是 G- 開頭的評量 ID）
//
// 服務帳號必須被加入該 GA4 資源的「檢視者」權限，否則會 403。
//
// 用法：node analytics/scripts/collect-ga4.mjs [YYYY-MM-DD]

import {
  loadConfig, taipeiDate, writeRaw, appendHistory, writeCurrent,
  cleanupRaw, skipMissing, markConnected,
} from './lib/collector.mjs';
import { findCredentials, ga4RunReport } from './lib/google-data.mjs';

const SOURCE = 'ga4';
const date = process.argv[2] || taipeiDate(-1); // GA4 當日資料未完整，預設抓前一天
const cfg = loadConfig();
const rawDays = cfg.retention?.rawDays ?? 30;
const propertyId = process.env.GA4_PROPERTY_ID || cfg.ga4?.propertyId || '';

const DOCS = '請參照 analytics/README.md「GA4 授權步驟」。';

if (!findCredentials()) {
  skipMissing(
    SOURCE,
    `尚未授權，缺服務帳號金鑰（~/.config/weiqi-kids/google-sa.json 或環境變數 GOOGLE_SA_JSON）。${DOCS}`,
    'Google Cloud Console 建立服務帳號 → 下載 JSON 金鑰 → 到 GA4 管理 → 資源存取管理，把服務帳號 email 加為「檢視者」。',
  );
}
if (!propertyId) {
  skipMissing(SOURCE, `尚未設定 GA4 資源 ID（analytics/config.json 的 ga4.propertyId）。${DOCS}`);
}

// 以 date 為結束日，往前 7 天的區間。
const start = new Date(`${date}T00:00:00Z`);
start.setUTCDate(start.getUTCDate() - 6);
const startDate = start.toISOString().slice(0, 10);
const dateRanges = [{ startDate, endDate: date }];

/** 把 runReport 回應攤平成 [{dim1, dim2, metric1, ...}] */
function flatten(res) {
  const dims = (res.dimensionHeaders ?? []).map((h) => h.name);
  const mets = (res.metricHeaders ?? []).map((h) => h.name);
  return (res.rows ?? []).map((row) => {
    const out = {};
    dims.forEach((d, i) => { out[d] = row.dimensionValues?.[i]?.value ?? ''; });
    mets.forEach((m, i) => { out[m] = Number(row.metricValues?.[i]?.value ?? 0); });
    return out;
  });
}

const METRICS = ['sessions', 'activeUsers', 'screenPageViews', 'userEngagementDuration'].map((name) => ({ name }));
const outboundEvents = cfg.ga4?.outboundEventNames ?? ['click'];

let daily, pages, events;
try {
  [daily, pages, events] = await Promise.all([
    ga4RunReport(propertyId, {
      dateRanges,
      dimensions: [{ name: 'date' }],
      metrics: METRICS,
      orderBys: [{ dimension: { dimensionName: 'date' } }],
    }),
    ga4RunReport(propertyId, {
      dateRanges,
      dimensions: [{ name: 'pagePath' }, { name: 'pageTitle' }],
      metrics: [{ name: 'screenPageViews' }, { name: 'activeUsers' }],
      orderBys: [{ metric: { metricName: 'screenPageViews' }, desc: true }],
      limit: 20,
    }),
    ga4RunReport(propertyId, {
      dateRanges,
      dimensions: [{ name: 'eventName' }, { name: 'linkUrl' }],
      metrics: [{ name: 'eventCount' }],
      dimensionFilter: {
        filter: { fieldName: 'eventName', inListFilter: { values: outboundEvents } },
      },
      orderBys: [{ metric: { metricName: 'eventCount' }, desc: true }],
      limit: 30,
    }),
  ]);
} catch (e) {
  skipMissing(SOURCE, `GA4 API 呼叫失敗：${e.message}${e.message.includes('403') ? '（多半是服務帳號還沒被加到 GA4 資源的檢視者）' : ''}`);
}

const dailyRows = flatten(daily).map((r) => ({
  date: `${r.date.slice(0, 4)}-${r.date.slice(4, 6)}-${r.date.slice(6, 8)}`,
  sessions: r.sessions,
  activeUsers: r.activeUsers,
  pageViews: r.screenPageViews,
  engagementSeconds: Math.round(r.userEngagementDuration),
}));
const topPages = flatten(pages).map((r) => ({
  path: r.pagePath, title: r.pageTitle, views: r.screenPageViews, users: r.activeUsers,
}));
const outboundClicks = flatten(events).map((r) => ({
  eventName: r.eventName, linkUrl: r.linkUrl, count: r.eventCount,
}));

writeRaw(SOURCE, date, {
  collectedAt: new Date().toISOString(),
  date, propertyId, range: { startDate, endDate: date },
  daily: dailyRows, topPages, outboundClicks,
});
for (const row of dailyRows) appendHistory(SOURCE, row); // 一次補齊整個區間，缺漏的日子會自動回填
writeCurrent('ga4-summary', {
  date,
  range: { startDate, endDate: date },
  sessions: dailyRows.reduce((s, r) => s + r.sessions, 0),
  activeUsers: dailyRows.reduce((s, r) => s + r.activeUsers, 0),
  pageViews: dailyRows.reduce((s, r) => s + r.pageViews, 0),
});
writeCurrent('ga4-top-pages', { date, range: { startDate, endDate: date }, pages: topPages });
writeCurrent('ga4-outbound-clicks', { date, range: { startDate, endDate: date }, events: outboundClicks });
cleanupRaw(SOURCE, rawDays);
markConnected(SOURCE, { propertyId });

console.log(
  `[ga4] ${startDate}~${date}｜工作階段 ${dailyRows.reduce((s, r) => s + r.sessions, 0)}、` +
    `熱門頁 ${topPages.length} 筆、站外點擊事件 ${outboundClicks.length} 筆`,
);
