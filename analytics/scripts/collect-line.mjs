#!/usr/bin/env node
// 收集 LINE 官方帳號（@685hqmrm）的每日指標。
//
// 端點（Messaging API insight）：
//   /v2/bot/insight/followers?date=YYYYMMDD    好友數／有效觸及／封鎖數
//   /v2/bot/insight/demographic                受眾輪廓（性別／年齡／地區，門檻不足時 available=false）
//   /v2/bot/insight/message/delivery?date=…    當日各類訊息送達數
//   /v2/bot/info                               帳號基本資料
//
// 注意：LINE insight 有資料延遲，所以預設抓「前一天」。
// 若當天狀態回 unready，會自動再往前退一天重試（最多退 3 天）。
//
// 憑證（不進 repo）：
//   ~/.config/weiqi-kids/line-token.txt（chmod 600），或環境變數 LINE_CHANNEL_ACCESS_TOKEN
//
// 用法：node analytics/scripts/collect-line.mjs [YYYY-MM-DD]

import {
  loadConfig, taipeiDate, compact, writeRaw, appendHistory, writeCurrent,
  cleanupRaw, skipMissing, markConnected, readSecret, fetchJSON, historyPath, loadJSON,
} from './lib/collector.mjs';

const SOURCE = 'line';
const cfg = loadConfig();
const rawDays = cfg.retention?.rawDays ?? 30;

const secret = readSecret({ env: 'LINE_CHANNEL_ACCESS_TOKEN', file: 'line-token.txt' });
if (!secret?.value) {
  skipMissing(
    SOURCE,
    '找不到 LINE Messaging API 存取權杖。',
    'LINE Developers Console → 該 Channel → Messaging API → Channel access token，' +
      '存到 ~/.config/weiqi-kids/line-token.txt（chmod 600）；CI 請設 LINE_CHANNEL_ACCESS_TOKEN secret。',
  );
}
const TOKEN = secret.value;

const call = async (path) => {
  try {
    return await fetchJSON(`https://api.line.me/v2/bot/${path}`, {
      headers: { Authorization: `Bearer ${TOKEN}` },
    });
  } catch (e) {
    // insight 端點在沒有資料時可能回 400/404，視為「當日無資料」而非失敗。
    if (e.status === 400 || e.status === 404) return { status: 'unavailable', error: e.message };
    throw e;
  }
};

// ---------- 決定要抓哪一天 ----------

// 使用者可指定日期；否則從前一天開始往前找到第一筆 status === 'ready'。
const requested = process.argv[2] || null;
const candidates = requested ? [requested] : [taipeiDate(-1), taipeiDate(-2), taipeiDate(-3)];

let date = candidates[0];
let followers = null;
for (const d of candidates) {
  const res = await call(`insight/followers?date=${compact(d)}`);
  date = d;
  followers = res;
  if (res?.status === 'ready') break;
}

const ready = followers?.status === 'ready';

const [demographic, delivery, info] = await Promise.all([
  call('insight/demographic'),
  call(`insight/message/delivery?date=${compact(date)}`),
  call('info'),
]);

// ---------- 整理 ----------

const deliveryTotal = ['broadcast', 'targeting', 'autoResponse', 'welcomeResponse', 'chat', 'apiBroadcast', 'apiPush', 'apiMulticast', 'apiNarrowcast', 'apiReply']
  .reduce((sum, k) => sum + Number(delivery?.[k] ?? 0), 0);

const prev = (loadJSON(historyPath(SOURCE))?.daily ?? []).filter((e) => e.date < date).at(-1);

const historyEntry = {
  date,
  status: followers?.status ?? 'unavailable',
  followers: ready ? Number(followers.followers ?? 0) : null,
  targetedReaches: ready ? Number(followers.targetedReaches ?? 0) : null,
  blocks: ready ? Number(followers.blocks ?? 0) : null,
  followersDelta: ready && prev?.followers != null ? Number(followers.followers ?? 0) - prev.followers : null,
  messagesDelivered: delivery?.status === 'ready' ? deliveryTotal : null,
};

writeRaw(SOURCE, date, {
  collectedAt: new Date().toISOString(),
  date,
  basicId: info?.basicId ?? cfg.line?.basicId ?? null,
  displayName: info?.displayName ?? cfg.line?.displayName ?? null,
  followers,
  demographic,
  messageDelivery: delivery,
});
appendHistory(SOURCE, historyEntry);
writeCurrent('line-oa', {
  ...historyEntry,
  basicId: info?.basicId ?? cfg.line?.basicId ?? null,
  displayName: info?.displayName ?? cfg.line?.displayName ?? null,
  demographicAvailable: demographic?.available === true,
});
if (demographic?.available === true) writeCurrent('line-demographic', { date, ...demographic });
cleanupRaw(SOURCE, rawDays);
markConnected(SOURCE, { dataStatus: historyEntry.status });

if (ready) {
  console.log(
    `[line] ${date}｜好友 ${historyEntry.followers}` +
      `${historyEntry.followersDelta === null ? '' : `（${historyEntry.followersDelta >= 0 ? '+' : ''}${historyEntry.followersDelta}）`}` +
      `、有效觸及 ${historyEntry.targetedReaches}、封鎖 ${historyEntry.blocks}`,
  );
} else {
  console.log(`[line] ${date}｜LINE 尚未產出該日 insight（status=${historyEntry.status}），已記錄空值，明日會補上。`);
}
