import { getCollection } from 'astro:content';

// 正式站只顯示 status: listed 且有同意日期的老師。
// 沒有取得同意就不公開，這是硬規則，不靠人記得。
const env = (k: string) => globalThis.process?.env?.[k] ?? (import.meta.env as Record<string, unknown>)[k];

export const LEVELS = ['入門', '級位', '段位'] as const;
export const TOPICS = ['規則入門', '佈局', '定石', '死活', '手筋', '收官', '實戰覆盤', '棋理'] as const;

export type Level = (typeof LEVELS)[number];
export type Topic = (typeof TOPICS)[number];

export async function publicTeachers() {
  const showDrafts = env('SITE_ENV') !== 'production';
  const all = await getCollection('teachers');
  return all
    .filter((t) => showDrafts || (t.data.status === 'listed' && t.data.consentDate))
    .sort((a, b) => a.data.city.localeCompare(b.data.city, 'zh-Hant') || a.data.name.localeCompare(b.data.name, 'zh-Hant'));
}

/** 把所有老師的影片攤平成一份清單，帶上老師資訊，供索引頁篩選。 */
export async function allVideos() {
  const teachers = await publicTeachers();
  return teachers.flatMap((t) =>
    t.data.videos.map((v) => ({
      ...v,
      teacherName: t.data.name,
      teacherSlug: t.data.slug,
      rank: t.data.rank,
      city: t.data.city,
      channelUrl: t.data.channelUrl,
    })),
  );
}

export function fmtDuration(sec?: number) {
  if (!sec) return '';
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

/** 依縣市分組，招生用的地區瀏覽。 */
export function byCity<T extends { data: { city: string } }>(teachers: T[]) {
  const map = new Map<string, T[]>();
  for (const t of teachers) {
    const list = map.get(t.data.city) ?? [];
    list.push(t);
    map.set(t.data.city, list);
  }
  return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0], 'zh-Hant'));
}
