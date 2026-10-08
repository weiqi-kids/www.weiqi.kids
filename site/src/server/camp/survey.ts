import { db, audit } from '../db';

export const SURVEY_ITEMS = [
  ['overall', '整體來說，這一團對你有幫助嗎？'],
  ['instructor', '講師的主題教學'],
  ['materials', '技能單張'],
  ['practice', '3 週實作的安排'],
] as const;

export async function surveyOf(cohortId: string, accountId: string) {
  return db().prepare('SELECT * FROM surveys WHERE cohort_id = ? AND account_id = ?').bind(cohortId, accountId).first();
}

export async function submitSurvey(cohortId: string, accountId: string, a: { overall: number; instructor: number; materials: number; practice: number; learned: string; suggestion: string }) {
  if ([a.overall, a.instructor, a.materials, a.practice].some((n) => !(n >= 1 && n <= 5))) return '每一題都要選 1 到 5 分。';
  if (!a.learned.trim()) return '請寫下你學到或做出了什麼。';
  await db().prepare('INSERT OR REPLACE INTO surveys (cohort_id, account_id, overall, instructor, materials, practice, learned, suggestion) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
    .bind(cohortId, accountId, a.overall, a.instructor, a.materials, a.practice, a.learned, a.suggestion || null).run();
  await audit(accountId, 'survey.submit', cohortId);
  return null;
}
