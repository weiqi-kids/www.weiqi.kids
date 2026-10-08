// AI 透過好棋寶寶 MCP 存的草稿。本人在網站上按「確定送出」才發出（ADR 0015）。
import { db, audit } from '../db';
import { notify } from '../notify';
import { newId, nowIso } from '../crypto';
import { createPost, getPost } from '../forum';
import { getShowcase } from './showcases';
import { getCohort } from './cohorts';
import { showcaseSpaceRole, cohortSpaceRole } from './access';
import { recordSubmission } from './practice';
import { consentWork } from './works';
import type { Account } from '../types';

export type DraftKind = 'question' | 'practice' | 'work' | 'reply';
export interface Draft {
  id: string; account_id: string; kind: DraftKind; space: string; parent_id: string | null; week: number | null;
  title: string | null; body: string; status: 'pending' | 'sent' | 'discarded'; post_id: string | null; created_at: string;
}

export const draftUrl = (id: string) => `/account/drafts/${id}/`;

// 草稿要發到的論壇，以及這個帳號能不能在那裡發文。
export async function spaceInfo(space: string, me: Account) {
  const [kind, id] = space.split(':');
  if (kind === 'showcase') {
    const s = await getShowcase(id);
    if (!s) return null;
    return { kind, showcase: s, cohort: null, role: showcaseSpaceRole(s, me), base: `/camp/r/${s.slug}/forum/`, title: `${s.title} 討論區` };
  }
  if (kind === 'cohort') {
    const c = await getCohort(id);
    const s = c ? await getShowcase(c.showcase_id) : null;
    if (!c || !s) return null;
    return { kind, showcase: s, cohort: c, role: await cohortSpaceRole(c, s, me), base: `/camp/g/${c.id}/forum/`, title: `${s.title}．第 ${c.seq} 團課程論壇` };
  }
  return null;
}

export async function createDraft(me: Account, d: { kind: DraftKind; space: string; parentId?: string | null; week?: number | null; title?: string | null; body: string }) {
  const info = await spaceInfo(d.space, me);
  if (!info || !info.role.post) return { ok: false as const, error: '你沒有在這個論壇發文的權限。' };
  if (!d.body.trim()) return { ok: false as const, error: '內容是空的。' };
  if (d.kind !== 'reply' && !d.title?.trim()) return { ok: false as const, error: '要有標題。' };
  if (d.kind === 'reply') {
    const parent = d.parentId ? await getPost(d.space, d.parentId) : null;
    if (!parent || parent.parent_id) return { ok: false as const, error: '找不到要回覆的主題文。' };
  }
  if ((d.kind === 'practice' || d.kind === 'work') && info.kind !== 'cohort') return { ok: false as const, error: '實作與作品分享只能發到課程論壇。' };
  const id = newId();
  await db().prepare('INSERT INTO drafts (id, account_id, kind, space, parent_id, week, title, body) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
    .bind(id, me.id, d.kind, d.space, d.parentId ?? null, d.week ?? null, d.title?.slice(0, 200) ?? null, d.body.slice(0, 20_000)).run();
  await audit(me.id, 'draft.create', id, { kind: d.kind });
  return { ok: true as const, id, url: draftUrl(id), where: info.title };
}

export async function getDraft(id: string) {
  return db().prepare('SELECT * FROM drafts WHERE id = ?').bind(id).first<Draft>();
}

export async function pendingDrafts(accountId: string) {
  return (await db().prepare("SELECT * FROM drafts WHERE account_id = ? AND status = 'pending' ORDER BY created_at DESC").bind(accountId).all<Draft>()).results;
}

// 本人按「確定送出」。作品分享可同時勾選「講師認可後同意公開到成果頁」。
export async function sendDraft(d: Draft, me: Account, edits: { title: string | null; body: string; consentPublic: boolean }) {
  if (d.account_id !== me.id || d.status !== 'pending') return { ok: false as const, error: '這份草稿已經送出或不能送出。' };
  const info = await spaceInfo(d.space, me);
  if (!info || !info.role.post) return { ok: false as const, error: '你現在沒有在這個論壇發文的權限。' };
  const isReply = d.kind === 'reply';
  const postId = await createPost(d.space, me, { parentId: isReply ? d.parent_id : null, title: isReply ? null : edits.title ?? d.title, body: edits.body, files: [] });
  await db().prepare("UPDATE drafts SET status = 'sent', post_id = ?, sent_at = ? WHERE id = ?").bind(postId, nowIso(), d.id).run();
  const topicId = isReply ? d.parent_id! : postId;
  const link = `${info.base}${topicId}/`;
  if (info.cohort && d.week && (d.kind === 'practice' || d.kind === 'work')) await recordSubmission(info.cohort.id, me.id, d.week, postId);
  if (d.kind === 'work' && edits.consentPublic) await consentWork(info.showcase.id, postId, me.id, true);
  const owner = info.showcase.owner_id;
  if (d.kind === 'work' && owner !== me.id) await notify(owner, 'work.shared', `${me.display_name} 分享了作品「${edits.title ?? d.title}」，回覆和認可都在這一頁。`, link);
  else if (d.kind === 'question' && owner !== me.id) await notify(owner, 'forum.question', `「${info.showcase.title}」討論區有新提問。`, link);
  else if (d.kind === 'practice' && owner !== me.id) await notify(owner, 'practice.submitted', `${me.display_name} 交了第 ${d.week} 週的實作。`, link);
  else if (isReply) {
    const parent = await getPost(d.space, d.parent_id!);
    if (parent && parent.author_id !== me.id) await notify(parent.author_id, 'forum.reply', `${me.display_name} 回覆了「${parent.title}」。`, link);
  }
  await audit(me.id, 'draft.send', d.id, { kind: d.kind });
  return { ok: true as const, link };
}

export async function discardDraft(d: Draft, me: Account) {
  if (d.account_id !== me.id || d.status !== 'pending') return false;
  await db().prepare("UPDATE drafts SET status = 'discarded' WHERE id = ?").bind(d.id).run();
  return true;
}
