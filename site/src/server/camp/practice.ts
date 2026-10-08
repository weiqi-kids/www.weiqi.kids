// 3 週實作：學員自己找題目，每週在課程論壇交一次。系統自動提醒，沒交沒有後果。
import { db } from '../db';
import { notifyMany } from '../notify';
import { fmtDateTime, addDaysIso } from '../../lib/time';
import { PRACTICE_WEEKS } from '../../lib/labels';
import type { Cohort, Showcase } from './types';

const DAY = 86_400_000;
export const WEEK_ASK: Record<number, string> = { 1: '交題目：你要做什麼、給誰用', 2: '交第一版，半成品也可以', 3: '交成果，附連結或截圖' };

export const weekDeadline = (c: Pick<Cohort, 'teaching_start'>, week: number) => addDaysIso(c.teaching_start!, 7 * week);

export async function submissions(cohortId: string) {
  return (await db().prepare('SELECT account_id, week, post_id, submitted_at FROM practice_submissions WHERE cohort_id = ?').bind(cohortId)
    .all<{ account_id: string; week: number; post_id: string; submitted_at: string }>()).results;
}

export async function recordSubmission(cohortId: string, accountId: string, week: number, postId: string) {
  await db().prepare('INSERT OR REPLACE INTO practice_submissions (cohort_id, account_id, week, post_id) VALUES (?, ?, ?, ?)').bind(cohortId, accountId, week, postId).run();
}

// 截止前 2 天提醒還沒交的人；截止隔天再提醒一次。
export async function practiceReminders(c: Cohort, s: Showcase) {
  const now = Date.now();
  const members = (await db().prepare("SELECT account_id FROM enrollments WHERE cohort_id = ? AND status = 'confirmed'").bind(c.id).all<{ account_id: string }>()).results.map((r) => r.account_id);
  if (!members.length) return;
  const done = await submissions(c.id);
  for (const week of [1, 2, 3]) {
    const deadline = Date.parse(weekDeadline(c, week));
    const missing = members.filter((m) => !done.some((d) => d.account_id === m && d.week === week));
    if (!missing.length) continue;
    const phase = now >= deadline - 2 * DAY && now < deadline ? 'pre' : now >= deadline + DAY && now < deadline + 2 * DAY ? 'post' : null;
    if (!phase) continue;
    const key = `practice:${c.id}:${week}:${phase}`;
    const r = await db().prepare('INSERT OR IGNORE INTO reminders_sent (key) VALUES (?)').bind(key).run();
    if (!r.meta.changes) continue;
    const msg = phase === 'pre'
      ? `「${s.title}」${PRACTICE_WEEKS[week]}，${fmtDateTime(weekDeadline(c, week))} 截止：${WEEK_ASK[week]}。`
      : `「${s.title}」${PRACTICE_WEEKS[week]}昨天截止了，你還沒交。現在交也可以：${WEEK_ASK[week]}。`;
    await notifyMany(missing, 'practice.reminder', msg, `/camp/g/${c.id}/forum/`);
  }
}
