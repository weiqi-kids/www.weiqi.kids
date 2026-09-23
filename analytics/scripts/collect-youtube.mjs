#!/usr/bin/env node
// 收集協會 YouTube 頻道（@goodmoveassociation）的每日指標。
//
// 資料來源一：YouTube Data API v3（現有 OAuth 權杖的 scope 就夠）
//   - 頻道層級：訂閱數、影片數、總觀看數
//   - 影片層級：每支影片的觀看／按讚／留言數
// 資料來源二（選配）：YouTube Analytics API
//   - 需要另一份帶 yt-analytics.readonly scope 的權杖
//     （~/.config/weiqi-kids/youtube-analytics-token.json）
//   - 有的話額外抓每日觀看時間與流量來源；沒有就略過並在輸出中註明。
//
// 憑證（不進 repo）：
//   ~/.config/weiqi-kids/youtube-client.json  {"client_id","client_secret"}
//   ~/.config/weiqi-kids/youtube-token.json   {"refresh_token",...}
//   CI 可改用環境變數 YOUTUBE_CLIENT_JSON / YOUTUBE_TOKEN_JSON（整份 JSON 字串）
//
// 用法：node analytics/scripts/collect-youtube.mjs [YYYY-MM-DD]

import { existsSync, readFileSync } from 'node:fs';
import {
  credPath, loadConfig, taipeiDate, writeRaw, appendHistory, writeCurrent,
  cleanupRaw, skipMissing, markConnected, fetchJSON, historyPath, loadJSON,
} from './lib/collector.mjs';

const SOURCE = 'youtube';
const date = process.argv[2] || taipeiDate();
const cfg = loadConfig();
const rawDays = cfg.retention?.rawDays ?? 30;

// ---------- 憑證 ----------

function readJsonSecret(envName, fileName) {
  if (process.env[envName]) {
    try {
      return JSON.parse(process.env[envName]);
    } catch {
      return null;
    }
  }
  const p = credPath(fileName);
  return existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null;
}

const client = readJsonSecret('YOUTUBE_CLIENT_JSON', 'youtube-client.json');
const token = readJsonSecret('YOUTUBE_TOKEN_JSON', 'youtube-token.json');

if (!client?.client_id || !client?.client_secret || !token?.refresh_token) {
  skipMissing(
    SOURCE,
    '找不到 YouTube OAuth 憑證（youtube-client.json / youtube-token.json）。',
    '在本機執行 site/scripts/youtube-upload.py auth 完成裝置授權；CI 請設 YOUTUBE_CLIENT_JSON、YOUTUBE_TOKEN_JSON secret。',
  );
}

/** 用 refresh token 換短效存取權杖。 */
async function accessToken(refreshToken) {
  const data = await fetchJSON('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: client.client_id,
      client_secret: client.client_secret,
      refresh_token: refreshToken,
      grant_type: 'refresh_token',
    }),
  });
  return data.access_token;
}

const api = async (path, tok) =>
  fetchJSON(`https://www.googleapis.com/youtube/v3/${path}`, {
    headers: { Authorization: `Bearer ${tok}` },
  });

// ---------- Data API：頻道 + 影片 ----------

async function collectChannel(tok) {
  const res = await api('channels?part=snippet,statistics,contentDetails&mine=true', tok);
  const ch = res.items?.[0];
  if (!ch) throw new Error('這組權杖查不到任何頻道（channels.mine 回空）。');
  return ch;
}

/** 走訪上傳播放清單，取回全部影片 ID。 */
async function listVideoIds(tok, uploadsPlaylistId) {
  const ids = [];
  let pageToken = '';
  do {
    const res = await api(
      `playlistItems?part=contentDetails&maxResults=50&playlistId=${uploadsPlaylistId}` +
        (pageToken ? `&pageToken=${pageToken}` : ''),
      tok,
    );
    for (const item of res.items ?? []) ids.push(item.contentDetails.videoId);
    pageToken = res.nextPageToken ?? '';
  } while (pageToken);
  return ids;
}

/** 每 50 支一批取影片統計。 */
async function listVideoStats(tok, ids) {
  const videos = [];
  for (let i = 0; i < ids.length; i += 50) {
    const batch = ids.slice(i, i + 50).join(',');
    const res = await api(`videos?part=snippet,statistics,contentDetails&id=${batch}`, tok);
    for (const v of res.items ?? []) {
      videos.push({
        id: v.id,
        title: v.snippet?.title ?? '',
        publishedAt: v.snippet?.publishedAt ?? null,
        duration: v.contentDetails?.duration ?? null,
        views: Number(v.statistics?.viewCount ?? 0),
        likes: Number(v.statistics?.likeCount ?? 0),
        comments: Number(v.statistics?.commentCount ?? 0),
      });
    }
  }
  return videos;
}

// ---------- Analytics API（選配）----------

async function collectAnalytics(channelId) {
  const analyticsToken = readJsonSecret('YOUTUBE_ANALYTICS_TOKEN_JSON', 'youtube-analytics-token.json');
  if (!analyticsToken?.refresh_token) {
    return {
      available: false,
      note:
        '未設定 YouTube Analytics 權杖（~/.config/weiqi-kids/youtube-analytics-token.json），' +
        '本次略過每日觀看時間與流量來源。需要 https://www.googleapis.com/auth/yt-analytics.readonly scope。',
    };
  }

  try {
    const tok = await accessToken(analyticsToken.refresh_token);
    const endDate = date;
    const startDate = new Date(`${date}T00:00:00Z`);
    startDate.setUTCDate(startDate.getUTCDate() - 27);
    const start = startDate.toISOString().slice(0, 10);
    const base = `https://youtubeanalytics.googleapis.com/v2/reports?ids=channel%3D%3D${channelId}&startDate=${start}&endDate=${endDate}`;
    const get = (qs) => fetchJSON(`${base}&${qs}`, { headers: { Authorization: `Bearer ${tok}` } });

    const [daily, traffic] = await Promise.all([
      get('metrics=views,estimatedMinutesWatched,averageViewDuration,subscribersGained,subscribersLost&dimensions=day&sort=day'),
      get('metrics=views,estimatedMinutesWatched&dimensions=insightTrafficSourceType&sort=-views'),
    ]);

    const toRows = (r) =>
      (r.rows ?? []).map((row) =>
        Object.fromEntries(row.map((v, i) => [r.columnHeaders?.[i]?.name ?? `col${i}`, v])),
      );

    return { available: true, range: { start, end: endDate }, daily: toRows(daily), trafficSources: toRows(traffic) };
  } catch (e) {
    return { available: false, note: `YouTube Analytics API 呼叫失敗，已略過：${e.message}` };
  }
}

// ---------- 主流程 ----------

const tok = await accessToken(token.refresh_token);
const channel = await collectChannel(tok);
const uploads = channel.contentDetails?.relatedPlaylists?.uploads;
const videoIds = uploads ? await listVideoIds(tok, uploads) : [];
const videos = await listVideoStats(tok, videoIds);
const analytics = await collectAnalytics(channel.id);

const stats = channel.statistics ?? {};
const channelSummary = {
  date,
  channelId: channel.id,
  title: channel.snippet?.title ?? '',
  handle: channel.snippet?.customUrl ?? cfg.youtube?.handle ?? '',
  subscribers: Number(stats.subscriberCount ?? 0),
  videoCount: Number(stats.videoCount ?? 0),
  totalViews: Number(stats.viewCount ?? 0),
};

// 與前一筆歷史比較，算出增減（週報要用）。
const prev = (loadJSON(historyPath(SOURCE))?.daily ?? []).filter((e) => e.date < date).at(-1);
const historyEntry = {
  ...channelSummary,
  subscribersDelta: prev ? channelSummary.subscribers - prev.subscribers : null,
  totalViewsDelta: prev ? channelSummary.totalViews - prev.totalViews : null,
  videoCountDelta: prev ? channelSummary.videoCount - prev.videoCount : null,
};

writeRaw(SOURCE, date, {
  collectedAt: new Date().toISOString(),
  date,
  channel: { id: channel.id, snippet: channel.snippet, statistics: stats },
  videos,
  analytics,
});
appendHistory(SOURCE, historyEntry);
writeCurrent('youtube-channel', { ...historyEntry, analyticsAvailable: analytics.available, analyticsNote: analytics.note ?? null });
writeCurrent('youtube-videos', {
  date,
  total: videos.length,
  topByViews: [...videos].sort((a, b) => b.views - a.views).slice(0, 20),
});
if (analytics.available) writeCurrent('youtube-analytics', analytics);
cleanupRaw(SOURCE, rawDays);
markConnected(SOURCE, { analyticsAvailable: analytics.available, analyticsNote: analytics.note ?? null });

console.log(
  `[youtube] ${date}｜訂閱 ${channelSummary.subscribers}` +
    `${historyEntry.subscribersDelta === null ? '' : `（${historyEntry.subscribersDelta >= 0 ? '+' : ''}${historyEntry.subscribersDelta}）`}` +
    `、影片 ${channelSummary.videoCount} 支、總觀看 ${channelSummary.totalViews}`,
);
if (!analytics.available) console.log(`[youtube] ${analytics.note}`);
