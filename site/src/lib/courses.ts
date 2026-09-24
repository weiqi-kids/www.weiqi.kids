import { getCollection } from 'astro:content';

// 正式站只顯示非草稿課程；staging 可用 SHOW_DRAFT_COURSES=1 預覽草稿課程。
//
// SITE_ENV=production 時一律不顯示草稿，不看 SHOW_DRAFT_COURSES。
// 原因：wrangler.jsonc 的 vars 會被 Cloudflare adapter 注入 process.env，
// 只靠 SHOW_DRAFT_COURSES 判斷的話，staging 設定會把正式站的過濾一起蓋掉。
const env = (k: string) => globalThis.process?.env?.[k] ?? (import.meta.env as Record<string, unknown>)[k];

export async function publicCourses() {
  const showDrafts = env('SITE_ENV') !== 'production' && env('SHOW_DRAFT_COURSES') === '1';
  const all = await getCollection('courses');
  return all
    .filter((c) => showDrafts || c.data.status !== 'draft')
    .sort((a, b) => a.data.startDate.getTime() - b.data.startDate.getTime());
}

export const STATUS_LABEL = { draft: '草稿', open: '開放報名', running: '進行中', closed: '已結營' } as const;

export function fmtDate(d: Date) {
  return new Intl.DateTimeFormat('zh-TW', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'Asia/Taipei' }).format(d);
}
