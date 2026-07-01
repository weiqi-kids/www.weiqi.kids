#!/usr/bin/env node
// weiqi.kids 資料層「數據心跳」：讀當日 analytics/seo-daily/<台灣日期>.json，把 GA4（+GSC 若有）
// 摘要成人話發到 Slack C0BEGBQLCD7。純讀取、無 AI、無 commit。當作「資料層有在跑」的每日心跳。
// 由 /etc/cron.d/weiqi-seo 於每日台灣 05:00 呼叫（排在收集之後、大腦之前）。
// 用法：node analytics/scripts/seo-report-slack.mjs

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { execFileSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const CHANNEL = process.env.WEIQI_SLACK_CHANNEL || 'C0BEGBQLCD7';
const NOTIFY = join(here, 'weiqi-slack-notify.sh');
const twDate = () => new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Taipei' });
const date = twDate();
const file = join(here, '..', 'seo-daily', `${date}.json`);

function send(text) {
  try { execFileSync('bash', [NOTIFY, CHANNEL], { input: text, stdio: ['pipe', 'inherit', 'inherit'] }); }
  catch { /* notify 自身已印錯誤；心跳不因發送失敗中斷 */ }
}

if (!existsSync(file)) {
  send(`🟡 *weiqi.kids SEO 數據心跳 · ${date.slice(5)}*\n今天還沒有數據檔（資料層可能尚未執行或失敗）。`);
  process.exit(0);
}

const d = JSON.parse(readFileSync(file, 'utf8'));
const ga = d.ga4 || {};
const gsc = d.gsc || {};
const lines = [];
lines.push(`📈 *weiqi.kids SEO 數據心跳 · ${date.slice(5)}*`);
lines.push('');
lines.push('【最近 7 天流量（GA4）】');
if (ga.error) {
  lines.push(`・GA4 讀取失敗：${ga.error}`);
} else {
  lines.push(`・造訪：${ga.sessions ?? 0} 次、訪客 ${ga.users ?? 0} 人、瀏覽 ${ga.views ?? 0} 次`);
  lines.push(`・台灣 Google 搜尋來的訪客：${ga.taiwanOrganicSessions ?? 0} 人`);
  const tp = (ga.topPages || [])[0];
  if (tp) lines.push(`・最熱門頁：${tp.path}（${tp.views} 次瀏覽）`);
}
lines.push('');
lines.push('【搜尋表現（Google Search Console）】');
if (gsc.error) {
  lines.push('・尚未接上（服務帳號待加入 GSC 屬性）——接上後這裡會有關鍵字排名與可優化清單。');
} else {
  const t = gsc.totals || {};
  lines.push(`・被看到 ${t.impressions ?? 0} 次、有人點 ${t.clicks ?? 0} 次`);
  lines.push(`・快擠進第一頁的關鍵字（第5–15名）：${(gsc.strikingDistance || []).length} 個`);
  lines.push(`・很多人看到卻沒人點的頁：${(gsc.highImpZeroClick || []).length} 個`);
  lines.push('　（Google 數據有 2–3 天延遲）');
}
lines.push('');
lines.push(`📄 資料：analytics/seo-daily/${date}.json`);

send(lines.join('\n'));
console.log('心跳已發送');
