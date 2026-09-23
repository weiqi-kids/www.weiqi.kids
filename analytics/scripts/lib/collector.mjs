// 多管道收集器共用工具：目錄慣例、JSON 讀寫、每日快照落地、缺憑證時的優雅結束。
//
// 資料夾慣例沿用既有 GitHub Traffic 收集：
//   analytics/raw/<來源>-YYYY-MM-DD.json   原始 API 回應（保留 30 天）
//   analytics/history/<來源>.json          每日累積（以日期去重）
//   analytics/current/<檔名>.json          最新快照
//
// 設計原則：憑證缺少時「印出中文說明並 exit 0」，不要讓每日排程整包失敗。

import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, unlinkSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { homedir } from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));

export const SCRIPTS_DIR = join(here, '..');
export const ANALYTICS_DIR = join(SCRIPTS_DIR, '..');
export const RAW_DIR = join(ANALYTICS_DIR, 'raw');
export const HISTORY_DIR = join(ANALYTICS_DIR, 'history');
export const CURRENT_DIR = join(ANALYTICS_DIR, 'current');
export const REPORTS_DIR = join(ANALYTICS_DIR, 'reports');

/** 憑證目錄：所有機密只放這裡（或 CI 的 GitHub Secrets），絕不進 repo。 */
export const CRED_DIR = process.env.WEIQI_CRED_DIR || join(homedir(), '.config', 'weiqi-kids');
export const credPath = (name) => join(CRED_DIR, name);

// ---------- JSON 讀寫 ----------

export function loadJSON(filepath, fallback = null) {
  try {
    if (existsSync(filepath)) return JSON.parse(readFileSync(filepath, 'utf8'));
  } catch {
    console.warn(`提醒：${filepath} 無法解析，視為無資料。`);
  }
  return fallback;
}

export function saveJSON(filepath, data) {
  mkdirSync(dirname(filepath), { recursive: true });
  writeFileSync(filepath, `${JSON.stringify(data, null, 2)}\n`);
}

/** 讀 analytics/config.json（非機密設定）。 */
export function loadConfig() {
  return loadJSON(join(ANALYTICS_DIR, 'config.json'), {}) ?? {};
}

// ---------- 日期 ----------

/** 以 Asia/Taipei 計算的 YYYY-MM-DD；offsetDays 為 -1 代表昨天。 */
export function taipeiDate(offsetDays = 0) {
  const d = new Date(Date.now() + offsetDays * 86400000);
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Taipei' }).format(d);
}

/** YYYY-MM-DD → YYYYMMDD（LINE insight 用的格式）。 */
export const compact = (isoDate) => isoDate.replace(/-/g, '');

/** ISO 週次：回 { year, week, label: 'YYYY-Www', monday, sunday }（週一起算）。 */
export function isoWeek(dateStr = taipeiDate()) {
  const d = new Date(`${dateStr}T00:00:00Z`);
  const day = (d.getUTCDay() + 6) % 7; // 週一 = 0
  const monday = new Date(d);
  monday.setUTCDate(d.getUTCDate() - day);
  const thursday = new Date(monday);
  thursday.setUTCDate(monday.getUTCDate() + 3); // ISO 週次由該週的週四決定年份
  const year = thursday.getUTCFullYear();
  const jan1 = new Date(Date.UTC(year, 0, 1));
  const week = Math.floor((thursday - jan1) / 604800000) + 1;
  const sunday = new Date(monday);
  sunday.setUTCDate(monday.getUTCDate() + 6);
  const iso = (x) => x.toISOString().slice(0, 10);
  return {
    year,
    week,
    label: `${year}-W${String(week).padStart(2, '0')}`,
    monday: iso(monday),
    sunday: iso(sunday),
  };
}

/** 回傳 dateStr 往前 n 天（含當天）的日期字串陣列，由舊到新。 */
export function lastNDays(n, endDate = taipeiDate()) {
  const end = new Date(`${endDate}T00:00:00Z`);
  return Array.from({ length: n }, (_, i) => {
    const d = new Date(end);
    d.setUTCDate(end.getUTCDate() - (n - 1 - i));
    return d.toISOString().slice(0, 10);
  });
}

// ---------- 落地 ----------

export const rawPath = (source, date) => join(RAW_DIR, `${source}-${date}.json`);
export const historyPath = (source) => join(HISTORY_DIR, `${source}.json`);
export const currentPath = (name) => join(CURRENT_DIR, `${name}.json`);

/** 寫入原始快照 analytics/raw/<source>-<date>.json */
export function writeRaw(source, date, payload) {
  const file = rawPath(source, date);
  saveJSON(file, payload);
  return file;
}

/**
 * 併入 analytics/history/<source>.json。
 * entry 需含 date 欄位；同一天重複執行會覆蓋當天那筆，不會長出重複列。
 */
export function appendHistory(source, entry) {
  const file = historyPath(source);
  const history = loadJSON(file, null) ?? { source, daily: [], lastUpdated: null };
  if (!Array.isArray(history.daily)) history.daily = [];
  const map = new Map(history.daily.map((e) => [e.date, e]));
  map.set(entry.date, entry);
  history.daily = Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date));
  history.lastUpdated = new Date().toISOString();
  saveJSON(file, history);
  return history;
}

/** 寫入最新快照 analytics/current/<name>.json */
export function writeCurrent(name, payload) {
  const file = currentPath(name);
  saveJSON(file, payload);
  return file;
}

/** 清掉超過保留天數的 raw 檔（預設 30 天，沿用既有 merge-traffic-data.js 的規則）。 */
export function cleanupRaw(source, days = 30) {
  if (!existsSync(RAW_DIR)) return 0;
  const cutoff = new Date(Date.now() - days * 86400000);
  let deleted = 0;
  for (const filename of readdirSync(RAW_DIR)) {
    if (!filename.startsWith(`${source}-`)) continue;
    const match = filename.match(/\d{4}-\d{2}-\d{2}/);
    if (!match) continue;
    if (new Date(`${match[0]}T00:00:00Z`) < cutoff) {
      unlinkSync(join(RAW_DIR, filename));
      deleted++;
    }
  }
  if (deleted) console.log(`已清除 ${deleted} 個超過 ${days} 天的 ${source} 原始檔。`);
  return deleted;
}

// ---------- 缺憑證時的處理 ----------

/**
 * 憑證不足時呼叫：印出清楚的中文說明後 exit 0，讓每日排程的其他來源照常跑完。
 * 同時把「尚未接上」的狀態寫進 current/<source>-status.json，週報才知道要標示。
 */
export function skipMissing(source, message, hint) {
  writeCurrent(`${source}-status`, {
    source,
    connected: false,
    reason: message,
    hint: hint ?? null,
    checkedAt: new Date().toISOString(),
  });
  console.log(`[${source}] 尚未接上：${message}`);
  if (hint) console.log(`[${source}] 取得方式：${hint}`);
  process.exit(0);
}

/** 收集成功時標記來源已接上。 */
export function markConnected(source, extra = {}) {
  writeCurrent(`${source}-status`, {
    source,
    connected: true,
    checkedAt: new Date().toISOString(),
    ...extra,
  });
}

/**
 * 讀機密：優先吃環境變數（CI 用 GitHub Secrets），否則讀本機檔案。
 * 回 { value, from } 或 null。
 */
export function readSecret({ env, file }) {
  if (env && process.env[env]) return { value: process.env[env], from: `環境變數 ${env}` };
  if (file) {
    const p = credPath(file);
    if (existsSync(p)) return { value: readFileSync(p, 'utf8').trim(), from: p };
  }
  return null;
}

/** 通用 JSON API 呼叫，非 2xx 時丟出帶回應內容的錯誤。 */
export async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const err = new Error(`${res.status} ${res.statusText}：${text.slice(0, 300)}`);
    err.status = res.status;
    err.body = data;
    throw err;
  }
  return data;
}
