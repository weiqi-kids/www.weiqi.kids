import { getCollection } from 'astro:content';

// 正式站只顯示非草稿課程；staging 可用 SHOW_DRAFT_COURSES=1 預覽草稿課程。
export async function publicCourses() {
  const showDrafts = (globalThis.process?.env?.SHOW_DRAFT_COURSES ?? import.meta.env.SHOW_DRAFT_COURSES) === '1';
  const all = await getCollection('courses');
  return all
    .filter((c) => showDrafts || c.data.status !== 'draft')
    .sort((a, b) => a.data.startDate.getTime() - b.data.startDate.getTime());
}

export const STATUS_LABEL = { draft: '草稿', open: '開放報名', running: '進行中', closed: '已結營' } as const;

export function fmtDate(d: Date) {
  return new Intl.DateTimeFormat('zh-TW', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'Asia/Taipei' }).format(d);
}
