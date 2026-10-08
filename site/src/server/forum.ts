import { env } from 'cloudflare:workers';
import { db, audit } from './db';
import { notify } from './notify';
import { newId, nowIso, sha256hex } from './crypto';
import type { Account } from './types';

// 論壇角色：space 是 'showcase:{id}'（成果的公開討論區）或 'cohort:{id}'（一團的課程論壇）。
export interface SpaceRole { view: boolean; post: boolean; moderate: boolean; admin: boolean }

export const HIDE_REASONS = ['垃圾內容', '個資', '侵權', '錯誤資訊', '違反課程規則', '作者要求', '其他管理原因'] as const;

export interface Post {
  id: string; space: string; author_id: string; author_name: string; parent_id: string | null;
  title: string | null; body: string; status: 'published' | 'withdrawn' | 'hidden'; created_at: string; updated_at: string;
}
export interface Attachment { id: string; post_id: string; mime_type: string; size: number }

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/gif']);
export const MAX_FILE_BYTES = 5 * 1024 * 1024;
export const MAX_FILES = 4;

// R2 尚未啟用時，圖片上傳關閉，其他論壇功能照常。
export const uploadsEnabled = () => !!(env as { UPLOADS?: R2Bucket }).UPLOADS;

// 可見性：一般成員只看得到已發布內容；作者看得到自己被隱藏或撤回的內容；版主與管理員看得到隱藏內容。
export function visibleTo(post: Post, me: Account | null, role: SpaceRole) {
  if (post.status === 'published') return true;
  if (me && post.author_id === me.id) return true;
  return post.status === 'hidden' && role.moderate;
}

const POST_COLS = 'p.id, p.space, p.author_id, a.display_name AS author_name, p.parent_id, p.title, p.body, p.status, p.created_at, p.updated_at';

export async function listTopics(space: string) {
  const { results } = await db().prepare(
    `SELECT ${POST_COLS}, (SELECT COUNT(*) FROM posts r WHERE r.parent_id = p.id AND r.status = 'published') AS replies
     FROM posts p JOIN accounts a ON a.id = p.author_id WHERE p.space = ? AND p.parent_id IS NULL ORDER BY p.created_at DESC`,
  ).bind(space).all<Post & { replies: number }>();
  return results;
}

export async function getPost(space: string, id: string) {
  return db().prepare(`SELECT ${POST_COLS} FROM posts p JOIN accounts a ON a.id = p.author_id WHERE p.space = ? AND p.id = ?`).bind(space, id).first<Post>();
}

export async function listReplies(space: string, parentId: string) {
  const { results } = await db().prepare(`SELECT ${POST_COLS} FROM posts p JOIN accounts a ON a.id = p.author_id WHERE p.space = ? AND p.parent_id = ? ORDER BY p.created_at`).bind(space, parentId).all<Post>();
  return results;
}

export async function attachmentsFor(postIds: string[]) {
  if (!postIds.length) return new Map<string, Attachment[]>();
  const { results } = await db().prepare(`SELECT id, post_id, mime_type, size FROM attachments WHERE deleted_at IS NULL AND post_id IN (${postIds.map(() => '?').join(',')}) ORDER BY created_at`).bind(...postIds).all<Attachment>();
  const map = new Map<string, Attachment[]>();
  for (const a of results) map.set(a.post_id, [...(map.get(a.post_id) ?? []), a]);
  return map;
}

export async function validateFiles(files: File[]): Promise<string | null> {
  const real = files.filter((f) => f.size > 0);
  if (real.length && !uploadsEnabled()) return '圖片上傳尚未開放。';
  if (real.length > MAX_FILES) return `一次最多上傳 ${MAX_FILES} 張圖片。`;
  for (const f of real) {
    if (!ALLOWED_TYPES.has(f.type)) return '只能上傳 JPG、PNG、WebP 或 GIF 圖片。';
    if (f.size > MAX_FILE_BYTES) return '每張圖片不能超過 5 MB。';
  }
  return null;
}

async function storeFiles(space: string, postId: string, ownerId: string, files: File[]) {
  for (const f of files.filter((x) => x.size > 0)) {
    const id = newId();
    const buf = await f.arrayBuffer();
    const key = `${space.replace(':', '/')}/${postId}/${id}`;
    await env.UPLOADS!.put(key, buf, { httpMetadata: { contentType: f.type } });
    await db().prepare('INSERT INTO attachments (id, post_id, space, owner_id, r2_key, mime_type, size, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
      .bind(id, postId, space, ownerId, key, f.type, f.size, await sha256hex(buf)).run();
  }
}

export async function createPost(space: string, me: Account, input: { parentId: string | null; title: string | null; body: string; files: File[] }) {
  const id = newId();
  await db().prepare('INSERT INTO posts (id, space, author_id, parent_id, title, body) VALUES (?, ?, ?, ?, ?, ?)')
    .bind(id, space, me.id, input.parentId, input.title, input.body).run();
  await storeFiles(space, id, me.id, input.files);
  await audit(me.id, 'post.create', id, { space });
  return id;
}

export async function editPost(post: Post, me: Account, title: string | null, body: string) {
  if (post.author_id !== me.id || post.status === 'withdrawn') return false;
  await db().batch([
    db().prepare('INSERT INTO post_revisions (post_id, title, body, revised_by) VALUES (?, ?, ?, ?)').bind(post.id, post.title, post.body, me.id),
    db().prepare('UPDATE posts SET title = ?, body = ?, updated_at = ? WHERE id = ?').bind(title, body, nowIso(), post.id),
  ]);
  await audit(me.id, 'post.edit', post.id);
  return true;
}

// 撤回：作者移除顯示，不刪除紀錄。
export async function withdrawPost(post: Post, me: Account) {
  if (post.author_id !== me.id || post.status !== 'published') return false;
  await db().prepare("UPDATE posts SET status = 'withdrawn', updated_at = ? WHERE id = ?").bind(nowIso(), post.id).run();
  await audit(me.id, 'post.withdraw', post.id);
  return true;
}

export async function moderate(post: Post, me: Account, role: SpaceRole, action: 'hide' | 'restore', reason: string, note: string | null, link: string) {
  if (!role.moderate) return false;
  // 講師隱藏的是其他學員的內容；自己的內容用撤回
  if (post.author_id === me.id) return false;
  if (action === 'hide' && post.status !== 'published') return false;
  if (action === 'restore' && post.status !== 'hidden') return false;
  const actorRole = role.admin ? 'admin' : 'instructor';
  await db().batch([
    db().prepare('UPDATE posts SET status = ?, updated_at = ? WHERE id = ?').bind(action === 'hide' ? 'hidden' : 'published', nowIso(), post.id),
    db().prepare('INSERT INTO moderation_actions (id, post_id, actor_id, actor_role, action, reason, note) VALUES (?, ?, ?, ?, ?, ?, ?)').bind(newId(), post.id, me.id, actorRole, action, reason, note),
  ]);
  const when = new Date().toISOString().slice(0, 10);
  await notify(post.author_id, `post.${action}`,
    action === 'hide' ? `你的內容在 ${when} 被${actorRole === 'admin' ? '協會管理員' : '講師'}隱藏，原因：${reason}。可以在內容頁提出申訴。` : `你被隱藏的內容已在 ${when} 恢復顯示。`, link);
  await audit(me.id, `post.${action}`, post.id, { reason, role: actorRole });
  return true;
}

export async function lastModeration(postId: string) {
  return db().prepare("SELECT actor_role, reason, note, created_at FROM moderation_actions WHERE post_id = ? ORDER BY created_at DESC LIMIT 1").bind(postId)
    .first<{ actor_role: string; reason: string; note: string | null; created_at: string }>();
}

export async function openAppeal(postId: string) {
  return db().prepare("SELECT id, reason, status, created_at FROM appeals WHERE post_id = ? ORDER BY created_at DESC LIMIT 1").bind(postId)
    .first<{ id: string; reason: string; status: string; created_at: string }>();
}

export async function fileAppeal(post: Post, me: Account, reason: string) {
  if (post.author_id !== me.id || post.status !== 'hidden') return false;
  const existing = await openAppeal(post.id);
  if (existing?.status === 'open') return false;
  const id = newId();
  await db().prepare('INSERT INTO appeals (id, post_id, author_id, reason) VALUES (?, ?, ?, ?)').bind(id, post.id, me.id, reason).run();
  await audit(me.id, 'appeal.create', id, { post: post.id });
  return true;
}
