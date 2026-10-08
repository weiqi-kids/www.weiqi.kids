// 學員的問題對不到成果時，經本人同意記下來給講師們看（不顯示是誰問的）。
import { db, audit } from '../db';
import { newId } from '../crypto';

export async function recordNeed(accountId: string, problem: string) {
  const text = problem.trim().slice(0, 2000);
  if (!text) return null;
  const id = newId();
  await db().prepare('INSERT INTO learning_needs (id, account_id, problem) VALUES (?, ?, ?)').bind(id, accountId, text).run();
  await audit(accountId, 'need.record', id);
  return id;
}

export async function recentNeeds(limit = 200) {
  return (await db().prepare('SELECT problem, created_at FROM learning_needs ORDER BY created_at DESC LIMIT ?').bind(limit).all<{ problem: string; created_at: string }>()).results;
}
