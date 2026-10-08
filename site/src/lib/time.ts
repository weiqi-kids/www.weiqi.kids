// 台北時間（UTC+8，無日光節約）。資料庫一律存 UTC ISO，畫面一律顯示台北時間。
const OFFSET = 8 * 3_600_000;
const WEEKDAYS = ['日', '一', '二', '三', '四', '五', '六'];

const tp = (iso: string | Date) => new Date((typeof iso === 'string' ? Date.parse(iso) : iso.getTime()) + OFFSET);
const pad = (n: number) => String(n).padStart(2, '0');

export const weekdayLabel = (d: number) => `週${WEEKDAYS[d]}`;

export function fmtDate(iso: string | Date) {
  const d = tp(iso);
  return `${d.getUTCFullYear()}/${pad(d.getUTCMonth() + 1)}/${pad(d.getUTCDate())}`;
}

export function fmtDateTime(iso: string | Date) {
  const d = tp(iso);
  return `${fmtDate(iso)}（${weekdayLabel(d.getUTCDay())}）${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

export const fmtTime = (iso: string) => { const d = tp(iso); return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`; };

// <input type="datetime-local"> 的值視為台北時間。
export function parseTaipeiLocal(value: string): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value);
  if (!m) return null;
  return new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) - OFFSET).toISOString();
}

export function toTaipeiLocal(iso: string | null | undefined) {
  if (!iso) return '';
  const d = tp(iso);
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}

// 某個每週時段在 notBefore 之後最早的一次（台北時間）。
export function nextSlotAfter(weekday: number, startTime: string, notBefore: number) {
  const [h, m] = startTime.split(':').map(Number);
  const local = new Date(notBefore + OFFSET);
  const base = Date.UTC(local.getUTCFullYear(), local.getUTCMonth(), local.getUTCDate(), h, m) - OFFSET;
  for (let i = 0; i <= 7; i++) {
    const t = base + i * 86_400_000;
    if (new Date(t + OFFSET).getUTCDay() === weekday && t >= notBefore) return new Date(t).toISOString();
  }
  return new Date(base + 14 * 86_400_000).toISOString();
}

export const addDaysIso = (iso: string, days: number) => new Date(Date.parse(iso) + days * 86_400_000).toISOString();
export const addMonthIso = (iso: string) => { const d = new Date(iso); d.setUTCMonth(d.getUTCMonth() + 1); return d.toISOString(); };
export const daysLeft = (iso: string) => Math.max(0, Math.ceil((Date.parse(iso) - Date.now()) / 86_400_000));

// 「2026年9月20日」：棋聚等公開頁的日期寫法。
export const fmtLongDate = (d: Date | string) =>
  new Intl.DateTimeFormat('zh-TW', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'Asia/Taipei' }).format(typeof d === 'string' ? new Date(d) : d);
