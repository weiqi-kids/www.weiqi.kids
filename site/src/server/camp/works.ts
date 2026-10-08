// 學員作品推薦：講師認可＋作者同意公開，兩者都有才自動出現在成果頁。
import { db, audit } from '../db';
import { notify } from '../notify';
import { nowIso } from '../crypto';

export async function workState(postId: string) {
  return db().prepare('SELECT approved_at, author_consent_at FROM showcase_works WHERE post_id = ?').bind(postId).first<{ approved_at: string | null; author_consent_at: string | null }>();
}

export async function approveWork(showcaseId: string, postId: string, authorId: string, actorId: string, link: string) {
  await db().prepare('INSERT INTO showcase_works (post_id, showcase_id, approved_by, approved_at) VALUES (?, ?, ?, ?) ON CONFLICT (post_id) DO UPDATE SET approved_by = excluded.approved_by, approved_at = excluded.approved_at')
    .bind(postId, showcaseId, actorId, nowIso()).run();
  await audit(actorId, 'work.approve', postId);
  await notify(authorId, 'work.approved', '講師認可了你的作品。你同意的話，作品會公開在成果頁給其他人看。', link);
}

export async function consentWork(showcaseId: string, postId: string, authorId: string, consent: boolean) {
  await db().prepare('INSERT INTO showcase_works (post_id, showcase_id, author_consent_at) VALUES (?, ?, ?) ON CONFLICT (post_id) DO UPDATE SET author_consent_at = excluded.author_consent_at')
    .bind(postId, showcaseId, consent ? nowIso() : null).run();
  await audit(authorId, consent ? 'work.consent' : 'work.withdraw_consent', postId);
}

export async function publicWorks(showcaseId: string) {
  return (await db().prepare(
    `SELECT p.id, p.title, p.body, a.display_name AS author_name, w.approved_at FROM showcase_works w
     JOIN posts p ON p.id = w.post_id JOIN accounts a ON a.id = p.author_id
     WHERE w.showcase_id = ? AND w.approved_at IS NOT NULL AND w.author_consent_at IS NOT NULL AND p.status = 'published' ORDER BY w.approved_at DESC`,
  ).bind(showcaseId).all<{ id: string; title: string | null; body: string; author_name: string; approved_at: string }>()).results;
}
