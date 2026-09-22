// 管理區顯示用：把帳號轉成看得懂的描述（名稱、登入方式、已開通課程）。
import { db } from './db';
import { runtimeCourses } from './courses';

export interface AccountSummary { id: string; name: string; email: string | null; line: boolean; courses: string[] }

export async function accountSummaries(ids?: string[]): Promise<Map<string, AccountSummary>> {
  const where = ids ? `WHERE a.id IN (${ids.map(() => '?').join(',') || "''"})` : '';
  const { results } = await db().prepare(
    `SELECT a.id, a.display_name,
       (SELECT subject FROM login_identities i WHERE i.account_id = a.id AND i.provider = 'email') AS email,
       (SELECT COUNT(*) FROM login_identities i WHERE i.account_id = a.id AND i.provider = 'line') AS line,
       (SELECT GROUP_CONCAT(course_slug, '|') FROM course_access c WHERE c.account_id = a.id) AS courses
     FROM accounts a ${where} ORDER BY a.created_at DESC`,
  ).bind(...(ids ?? [])).all<{ id: string; display_name: string; email: string | null; line: number; courses: string | null }>();
  const titles = new Map((await runtimeCourses()).map((c) => [c.data.slug, c.data.title]));
  return new Map(results.map((r) => [r.id, {
    id: r.id, name: r.display_name, email: r.email, line: r.line > 0,
    courses: (r.courses ?? '').split('|').filter(Boolean).map((s) => titles.get(s) ?? s),
  }]));
}

export const loginLabel = (a: AccountSummary) => [a.email, a.line ? 'LINE' : null].filter(Boolean).join('／') || '無登入方式';

export { APPLICATION_STATUS, COURSE_STATUS, label } from '../lib/labels';

// 台北時間 YYYY/MM/DD HH:mm
export const fmtDateTime = (iso: string) =>
  new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(iso));

export const ACTION_LABEL: Record<string, string> = {
  'account.create': '建立帳號', 'account.rename': '修改顯示名稱', 'account.grant_admin': '設為管理員',
  'identity.link': '綁定登入方式', 'identity.unlink': '解除登入方式',
  'login.email': 'Email 登入', 'login.line': 'LINE 登入', 'magic.sent': '寄出登入連結', 'magic.send_failed': '登入信寄送失敗',
  'enrollment.activate': '核款並開通課程', 'enrollment.cancel': '取消報名', 'membership.renew': '續會',
  'camp.join': '加入實作營', 'material.create': '新增技能單張', 'material.edit': '修改技能單張',
  'post.create': '發文', 'post.edit': '修改文章', 'post.withdraw': '撤回文章', 'post.hide': '隱藏內容', 'post.restore': '恢復內容',
  'appeal.create': '提出申訴', 'appeal.upheld': '申訴：維持隱藏', 'appeal.restored': '申訴：恢復內容',
  'instructor_application.submit': '送出講師申請', 'instructor_application.resubmit': '補件後重送講師申請',
  'instructor_application.approved': '核准講師申請', 'instructor_application.changes_requested': '講師申請要求補件', 'instructor_application.rejected': '講師申請不核准',
  'course.add_instructor': '指定課程講師',
};
