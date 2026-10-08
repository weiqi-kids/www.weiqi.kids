import type { APIRoute } from 'astro';
import { db } from '../../../../server/db';
import { getCohort } from '../../../../server/camp/cohorts';
import { getShowcase } from '../../../../server/camp/showcases';
import { fmtDateTime } from '../../../../lib/time';
import { FORMAT_LABEL } from '../../../../lib/labels';
export const prerender = false;

const cell = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
const row = (...v: unknown[]) => v.map(cell).join(',');

// TTQS 佐證：開課表單＋學員名單（簽到、3 週繳交、問卷）。
export const GET: APIRoute = async ({ params, locals }) => {
  if (locals.account?.is_admin !== 1) return new Response('Forbidden', { status: 403 });
  const c = await getCohort(params.id!);
  const s = c ? await getShowcase(c.showcase_id) : null;
  if (!c || !s) return new Response('Not found', { status: 404 });
  const { results: people } = await db().prepare(
    `SELECT a.display_name, e.code, e.amount, e.status,
       (SELECT checked_in_at FROM attendance t WHERE t.cohort_id = e.cohort_id AND t.account_id = e.account_id) AS checkin,
       (SELECT GROUP_CONCAT(week) FROM practice_submissions p WHERE p.cohort_id = e.cohort_id AND p.account_id = e.account_id) AS weeks,
       sv.overall, sv.instructor, sv.materials, sv.practice, sv.learned, sv.suggestion
     FROM enrollments e JOIN accounts a ON a.id = e.account_id LEFT JOIN surveys sv ON sv.cohort_id = e.cohort_id AND sv.account_id = e.account_id
     WHERE e.cohort_id = ? AND e.status = 'confirmed' ORDER BY a.display_name`,
  ).bind(c.id).all<Record<string, string | number | null>>();
  const lines = [
    row('課程名稱', s.title), row('講師', s.instructor_name), row('團次', `第 ${c.seq} 團`),
    row('主題教學', c.teaching_start ? fmtDateTime(c.teaching_start) : ''), row('方式', FORMAT_LABEL[c.teaching_format ?? s.teaching_format], c.location ?? ''),
    row('需求', s.ttqs_needs), row('目標', s.ttqs_goals), row('大綱', s.ttqs_outline), row('時數', s.ttqs_hours), row('方法', s.ttqs_methods), row('評量', s.ttqs_evaluation), row('預期成果', s.ttqs_expected),
    '',
    row('學員', '報名編號', '團費', '簽到時間', '第1週題目', '第2週第一版', '第3週成果', '整體', '講師', '技能單張', '實作安排', '學到什麼', '建議'),
    ...people.map((p) => {
      const w = String(p.weeks ?? '').split(',');
      return row(p.display_name, p.code, p.amount, p.checkin ? fmtDateTime(String(p.checkin)) : '', w.includes('1') ? '有' : '', w.includes('2') ? '有' : '', w.includes('3') ? '有' : '', p.overall, p.instructor, p.materials, p.practice, p.learned, p.suggestion);
    }),
  ];
  return new Response(`﻿${lines.join('\r\n')}\r\n`, {
    headers: { 'content-type': 'text/csv; charset=utf-8', 'content-disposition': `attachment; filename="ttqs-${s.slug}-${c.seq}.csv"`, 'cache-control': 'private, no-store' },
  });
};
