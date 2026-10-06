import { getCollection, type CollectionEntry } from 'astro:content';
import { db, audit, notify } from './db';
import { runtimeCourses } from './courses';

export type Showcase = CollectionEntry<'topics'>;

export async function findShowcase(slug: string): Promise<Showcase | null> {
  return (await getCollection('topics')).find((t) => t.data.slug === slug) ?? null;
}

export async function interestCount(slug: string) {
  const row = await db().prepare('SELECT COUNT(*) AS n FROM interests WHERE topic_slug = ?').bind(slug).first<{ n: number }>();
  return row?.n ?? 0;
}

export async function hasInterest(slug: string, accountId: string) {
  return !!(await db().prepare('SELECT 1 FROM interests WHERE topic_slug = ? AND account_id = ?').bind(slug, accountId).first());
}

// 已開團：這個成果有開放報名的正式課程。
export async function openGroups(slug: string) {
  return (await runtimeCourses()).filter((c) => c.data.topic === slug && (c.data.status === 'open' || c.data.status === 'draft'));
}

// 登記想學。登記人數第一次到開團人數時通知管理員安排開團。
export async function addInterest(showcase: Showcase, accountId: string) {
  const slug = showcase.data.slug;
  await db().prepare('INSERT OR IGNORE INTO interests (topic_slug, account_id) VALUES (?, ?)').bind(slug, accountId).run();
  await audit(accountId, 'interest.add', slug);
  const n = await interestCount(slug);
  if (n < showcase.data.groupSize) return;
  const first = await db().prepare('INSERT OR IGNORE INTO interest_thresholds (topic_slug) VALUES (?)').bind(slug).run();
  if (!first.meta.changes) return;
  const admins = await db().prepare("SELECT id FROM accounts WHERE is_admin = 1 AND status = 'active'").all<{ id: string }>();
  for (const a of admins.results) await notify(a.id, 'interest.threshold', `「${showcase.data.title}」登記想學已達 ${n} 人，可以安排開團。`, '/admin/interests/');
  await audit(null, 'interest.threshold', slug, { count: n });
}

export async function removeInterest(slug: string, accountId: string) {
  await db().prepare('DELETE FROM interests WHERE topic_slug = ? AND account_id = ?').bind(slug, accountId).run();
  await audit(accountId, 'interest.remove', slug);
}

export async function myInterests(accountId: string) {
  const rows = await db().prepare('SELECT topic_slug, created_at FROM interests WHERE account_id = ? ORDER BY created_at DESC').bind(accountId).all<{ topic_slug: string; created_at: string }>();
  return rows.results;
}

export async function interestTotals() {
  const rows = await db().prepare('SELECT topic_slug, COUNT(*) AS n FROM interests GROUP BY topic_slug').all<{ topic_slug: string; n: number }>();
  return new Map(rows.results.map((r) => [r.topic_slug, r.n]));
}
