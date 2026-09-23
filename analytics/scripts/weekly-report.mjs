#!/usr/bin/env node
// 把各管道的收集結果彙整成一份繁體中文 Markdown 週報。
//
// 輸出：analytics/reports/YYYY-Www.md（ISO 週次，週一起算）
// 內容：網站（GA4）、搜尋（GSC）、YouTube、LINE@、GitHub repo 流量
//
// 原則：沒有資料就寫「尚未接上」並附上原因，絕不編造數字。
//
// 用法：
//   node analytics/scripts/weekly-report.mjs             # 上一個完整週（週一～週日）
//   node analytics/scripts/weekly-report.mjs 2026-09-23  # 指定日期所屬的那一週

import {
  loadJSON, loadConfig, taipeiDate, isoWeek,
  historyPath, currentPath, REPORTS_DIR,
} from './lib/collector.mjs';
import { join } from 'node:path';
import { mkdirSync, writeFileSync } from 'node:fs';

const cfg = loadConfig();

// 預設產出「上一個完整週」：以今天往前 7 天所屬的那一週。
const anchor = process.argv[2] || taipeiDate(-7);
const week = isoWeek(anchor);
const inWeek = (d) => d >= week.monday && d <= week.sunday;

const num = (v) => (typeof v === 'number' ? v.toLocaleString('zh-TW') : v);
const signed = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${num(v)}`);

/** 取某來源在本週的每日資料；同時回傳接上狀態。 */
function source(name) {
  const status = loadJSON(currentPath(`${name}-status`));
  const history = loadJSON(historyPath(name));
  const daily = (history?.daily ?? []).filter((e) => inWeek(e.date));
  const connected = status ? status.connected !== false : (history?.daily ?? []).length > 0;
  return { connected, reason: status?.reason ?? null, hint: status?.hint ?? null, daily, history };
}

/** 尚未接上時的統一寫法。 */
function notConnected(s, fallbackReason) {
  const lines = [`> **尚未接上** — ${s.reason ?? fallbackReason}`];
  if (s.hint) lines.push('>', `> 取得方式：${s.hint}`);
  return lines.join('\n');
}

function table(headers, rows) {
  if (!rows.length) return '（本週無資料）';
  return [
    `| ${headers.join(' | ')} |`,
    `|${headers.map(() => '---').join('|')}|`,
    ...rows.map((r) => `| ${r.join(' | ')} |`),
  ].join('\n');
}

const out = [];
out.push(`# 流量週報 ${week.label}`);
out.push('');
out.push(`**期間**：${week.monday} ～ ${week.sunday}（週一起算）`);
out.push('');
out.push(`**產生時間**：${new Date().toISOString().replace('T', ' ').slice(0, 19)} UTC`);
out.push('');
out.push('---');
out.push('');

// ---------- 0. 資料來源狀態 ----------

const ga4 = source('ga4');
const gsc = source('gsc');
const yt = source('youtube');
const line = source('line');

out.push('## 資料來源狀態');
out.push('');
out.push(
  table(
    ['管道', '狀態', '本週資料筆數'],
    [
      ['網站（GA4）', ga4.connected ? '已接上' : '尚未接上', String(ga4.daily.length)],
      ['搜尋（Search Console）', gsc.connected ? '已接上' : '尚未接上', String(gsc.daily.length)],
      ['YouTube', yt.connected ? '已接上' : '尚未接上', String(yt.daily.length)],
      ['LINE 官方帳號', line.connected ? '已接上' : '尚未接上', String(line.daily.length)],
    ],
  ),
);
out.push('');

// ---------- 1. 網站（GA4）----------

out.push('## 一、網站（GA4）');
out.push('');
if (!ga4.connected || !ga4.daily.length) {
  out.push(notConnected(ga4, 'GA4 服務帳號尚未授權，或本週尚無資料。'));
  out.push('');
  out.push(`評量 ID：\`${cfg.site?.ga4MeasurementId ?? '未設定'}\`（新舊站共用）。正式網域切換前，資料仍以舊站為主。`);
} else {
  const sum = (k) => ga4.daily.reduce((s, r) => s + (r[k] ?? 0), 0);
  out.push(`**本週合計**：工作階段 ${num(sum('sessions'))}、活躍使用者 ${num(sum('activeUsers'))}、瀏覽 ${num(sum('pageViews'))}`);
  out.push('');
  out.push('### 每日趨勢');
  out.push('');
  out.push(
    table(
      ['日期', '工作階段', '活躍使用者', '瀏覽'],
      ga4.daily.map((r) => [r.date, num(r.sessions), num(r.activeUsers), num(r.pageViews)]),
    ),
  );
  out.push('');
  out.push('### 熱門頁面');
  out.push('');
  const pages = loadJSON(currentPath('ga4-top-pages'))?.pages ?? [];
  out.push(
    table(
      ['排名', '頁面', '標題', '瀏覽', '使用者'],
      pages.slice(0, 10).map((p, i) => [String(i + 1), `\`${p.path}\``, p.title ?? '', num(p.views), num(p.users)]),
    ),
  );
  out.push('');
  out.push('### 站外點擊事件');
  out.push('');
  const clicks = loadJSON(currentPath('ga4-outbound-clicks'))?.events ?? [];
  out.push(
    clicks.length
      ? table(
          ['事件', '連結', '次數'],
          clicks.slice(0, 15).map((e) => [e.eventName, `\`${e.linkUrl || '(未帶 link_url)'}\``, num(e.count)]),
        )
      : '（本週未收到站外點擊事件；若剛埋完事件，GA4 需要一段時間才會有量）',
  );
}
out.push('');

// ---------- 2. 搜尋（GSC）----------

out.push('## 二、搜尋（Google Search Console）');
out.push('');
if (!gsc.connected || !gsc.daily.length) {
  out.push(notConnected(gsc, 'Search Console 服務帳號尚未授權，或本週尚無資料。'));
} else {
  const clicks = gsc.daily.reduce((s, r) => s + (r.clicks ?? 0), 0);
  const impressions = gsc.daily.reduce((s, r) => s + (r.impressions ?? 0), 0);
  const ctr = impressions ? ((clicks / impressions) * 100).toFixed(2) : '0.00';
  const pos = gsc.daily.length
    ? (gsc.daily.reduce((s, r) => s + (r.position ?? 0), 0) / gsc.daily.length).toFixed(1)
    : '—';
  out.push(`**本週合計**：曝光 ${num(impressions)}、點擊 ${num(clicks)}、CTR ${ctr}%、平均排名 ${pos}`);
  out.push('');
  out.push('### 熱門查詢');
  out.push('');
  const queries = loadJSON(currentPath('gsc-top-queries'))?.queries ?? [];
  out.push(
    table(
      ['排名', '查詢', '曝光', '點擊', 'CTR', '平均排名'],
      queries.slice(0, 15).map((q, i) => [String(i + 1), q.query, num(q.impressions), num(q.clicks), `${q.ctr}%`, String(q.position)]),
    ),
  );
  out.push('');
  out.push('### 熱門著陸頁');
  out.push('');
  const gp = loadJSON(currentPath('gsc-top-pages'))?.pages ?? [];
  out.push(
    table(
      ['排名', '頁面', '曝光', '點擊', '平均排名'],
      gp.slice(0, 10).map((p, i) => [String(i + 1), p.page, num(p.impressions), num(p.clicks), String(p.position)]),
    ),
  );
}
out.push('');

// ---------- 3. YouTube ----------

out.push('## 三、YouTube');
out.push('');
if (!yt.connected || !yt.daily.length) {
  out.push(notConnected(yt, 'YouTube OAuth 權杖尚未設定，或本週尚無資料。'));
} else {
  const first = yt.daily[0];
  const last = yt.daily.at(-1);
  const subDelta = last.subscribers - first.subscribers;
  const viewDelta = last.totalViews - first.totalViews;
  out.push(`**頻道**：${last.title}（${last.handle}）`);
  out.push('');
  out.push(
    table(
      ['指標', '週初', '週末', '變化'],
      [
        ['訂閱數', num(first.subscribers), num(last.subscribers), signed(subDelta)],
        ['影片數', num(first.videoCount), num(last.videoCount), signed(last.videoCount - first.videoCount)],
        ['總觀看數', num(first.totalViews), num(last.totalViews), signed(viewDelta)],
      ],
    ),
  );
  out.push('');
  out.push('> 註：YouTube Data API 只給累計值，「本週觀看」是用週初／週末總觀看數相減估算。');
  out.push('');
  out.push('### 熱門影片（累計觀看 Top 10）');
  out.push('');
  const videos = loadJSON(currentPath('youtube-videos'))?.topByViews ?? [];
  out.push(
    table(
      ['排名', '影片', '觀看', '按讚', '留言'],
      videos.slice(0, 10).map((v, i) => [
        String(i + 1),
        `[${(v.title ?? '').replace(/\|/g, '\\|')}](https://youtu.be/${v.id})`,
        num(v.views), num(v.likes), num(v.comments),
      ]),
    ),
  );
  out.push('');
  const ytStatus = loadJSON(currentPath('youtube-channel'));
  if (ytStatus && ytStatus.analyticsAvailable === false) {
    out.push(`> **YouTube Analytics 尚未接上**：${(ytStatus.analyticsNote ?? '缺 yt-analytics.readonly 權限').replace(/[。\s]+$/, '')}。`);
  } else {
    const ytAnalytics = loadJSON(currentPath('youtube-analytics'));
    const traffic = ytAnalytics?.trafficSources ?? [];
    if (traffic.length) {
      out.push('### 流量來源（近 28 天）');
      out.push('');
      out.push(
        table(
          ['來源', '觀看', '觀看分鐘'],
          traffic.slice(0, 10).map((t) => [
            t.insightTrafficSourceType, num(t.views), num(t.estimatedMinutesWatched),
          ]),
        ),
      );
    }
  }
}
out.push('');

// ---------- 4. LINE@ ----------

out.push('## 四、LINE 官方帳號');
out.push('');
if (!line.connected || !line.daily.length) {
  out.push(notConnected(line, 'LINE Messaging API 權杖尚未設定，或本週尚無資料。'));
} else {
  const ready = line.daily.filter((r) => r.followers != null);
  if (!ready.length) {
    out.push(`> **本週無可用資料** — LINE insight 尚未產出（好友數未達門檻時 LINE 不提供統計）。帳號：${cfg.line?.basicId ?? ''}`);
  } else {
    const first = ready[0];
    const last = ready.at(-1);
    out.push(`**帳號**：${cfg.line?.displayName ?? ''}（${cfg.line?.basicId ?? ''}）`);
    out.push('');
    out.push(
      table(
        ['指標', '週初', '週末', '變化'],
        [
          ['好友數', num(first.followers), num(last.followers), signed(last.followers - first.followers)],
          ['有效觸及', num(first.targetedReaches), num(last.targetedReaches), signed(last.targetedReaches - first.targetedReaches)],
          ['封鎖數', num(first.blocks), num(last.blocks), signed(last.blocks - first.blocks)],
        ],
      ),
    );
    out.push('');
    out.push('### 每日好友數');
    out.push('');
    out.push(
      table(
        ['日期', '好友數', '有效觸及', '封鎖', '訊息送達'],
        ready.map((r) => [r.date, num(r.followers), num(r.targetedReaches), num(r.blocks), r.messagesDelivered == null ? '—' : num(r.messagesDelivered)]),
      ),
    );
    out.push('');
    out.push('> 註：LINE insight 有一日延遲，所以本週最後一天的資料會在下週初才補齊。');
  }
}
out.push('');

// ---------- 5. GitHub repo 流量（既有來源）----------

out.push('## 五、GitHub Repo 流量');
out.push('');
const views = loadJSON(historyPath('daily-views'))?.views ?? [];
const weekViews = views.filter((v) => inWeek((v.timestamp ?? '').slice(0, 10)));
if (!weekViews.length) {
  out.push('（本週無資料）');
} else {
  out.push(
    `**本週合計**：${num(weekViews.reduce((s, v) => s + v.count, 0))} 瀏覽 / ` +
      `${num(weekViews.reduce((s, v) => s + v.uniques, 0))} 獨立訪客`,
  );
}
out.push('');

// ---------- 結尾 ----------

out.push('---');
out.push('');
out.push('本報告由 `analytics/scripts/weekly-report.mjs` 自動產生。缺少的管道一律標示「尚未接上」，不以估算值補齊。');
out.push('');
out.push(`跨管道連結請依 [UTM 命名規則](../README.md#utm-命名規則) 加參數；站內連結不加 UTM。`);
out.push('');

const file = join(REPORTS_DIR, `${week.label}.md`);
mkdirSync(REPORTS_DIR, { recursive: true });
writeFileSync(file, out.join('\n'));

console.log(`[report] 已產生 ${file}（${week.monday} ～ ${week.sunday}）`);
