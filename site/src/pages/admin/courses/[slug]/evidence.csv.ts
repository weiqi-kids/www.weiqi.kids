import type { APIRoute } from 'astro';
import { db } from '../../../../server/db';
export const prerender = false;

// TTQS 佐證索引（第一門課手動整理用；ADR 0010）。
// 欄位依 revamp/4-strategy/2026-09-20-ttqs-export-spec.md §3；PDDRO 指標與場次由管理員在試算表補標。
const HEAD = ['evidence_id', 'pddro', '指標', '文件名稱', '文件版本', '來源資料ID', '課程', '場次', '建立時間', '操作者', '頁碼或段落', '附件連結', '個資遮罩狀態', '備註'];
const cell = (v: unknown) => {
  const s = v == null ? '' : String(v);
  return /[",\n\r]/.test(s) || /^[=+\-@]/.test(s) ? `"${s.replace(/^([=+\-@])/, "'$1").replace(/"/g, '""')}"` : s;
};

export const GET: APIRoute = async ({ params, locals, url }) => {
  const me = locals.account;
  if (!me || me.is_admin !== 1) return new Response('Forbidden', { status: 403 });
  const slug = params.slug!;
  const origin = url.origin;
  const rows: unknown[][] = [];
  const { results: materials } = await db().prepare(
    'SELECT m.id, m.title, m.version, m.updated_at, a.display_name FROM materials m JOIN accounts a ON a.id = m.created_by WHERE m.course_slug = ? ORDER BY m.position',
  ).bind(slug).all<{ id: string; title: string; version: number; updated_at: string; display_name: string }>();
  for (const m of materials) rows.push([`M-${m.id.slice(0, 8)}`, 'Design', '', `技能單張：${m.title}`, m.version, m.id, slug, '第 1 堂', m.updated_at, m.display_name, '', `${origin}/camp/courses/${slug}/materials/`, '不含個資', '']);
  const { results: posts } = await db().prepare(
    `SELECT p.id, p.parent_id, p.title, p.status, p.created_at, a.display_name,
       (SELECT COUNT(*) FROM post_revisions r WHERE r.post_id = p.id) + 1 AS version
     FROM posts p JOIN accounts a ON a.id = p.author_id WHERE p.course_slug = ? ORDER BY p.created_at`,
  ).bind(slug).all<{ id: string; parent_id: string | null; title: string | null; status: string; created_at: string; display_name: string; version: number }>();
  for (const p of posts) {
    const link = `${origin}/camp/courses/${slug}/forum/${p.parent_id ?? p.id}/#p-${p.id}`;
    rows.push([`P-${p.id.slice(0, 8)}`, 'Do', '', p.parent_id ? '論壇回覆' : `論壇文章：${p.title ?? ''}`, p.version, p.id, slug, '', p.created_at, p.display_name, '', link, '未檢查', p.status === 'published' ? '' : `狀態：${p.status}`]);
  }
  const { results: files } = await db().prepare(
    'SELECT f.id, f.post_id, f.mime_type, f.created_at, a.display_name FROM attachments f JOIN accounts a ON a.id = f.owner_id WHERE f.course_slug = ? AND f.deleted_at IS NULL ORDER BY f.created_at',
  ).bind(slug).all<{ id: string; post_id: string; mime_type: string; created_at: string; display_name: string }>();
  for (const f of files) rows.push([`F-${f.id.slice(0, 8)}`, 'Do', '', '課程論壇圖片', 1, f.id, slug, '', f.created_at, f.display_name, '', `${origin}/camp/courses/${slug}/forum/files/${f.id}/`, '未檢查', `附屬於 P-${f.post_id.slice(0, 8)}`]);
  const csv = '﻿' + [HEAD, ...rows].map((r) => r.map(cell).join(',')).join('\r\n') + '\r\n';
  return new Response(csv, { headers: { 'content-type': 'text/csv; charset=utf-8', 'content-disposition': `attachment; filename="ttqs-evidence-${slug}.csv"` } });
};
