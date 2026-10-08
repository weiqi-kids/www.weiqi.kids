import { db, audit } from './db';
import { newId } from './crypto';

export async function findIdentity(provider: 'email' | 'line', subject: string) {
  return db().prepare('SELECT account_id FROM login_identities WHERE provider = ? AND subject = ?').bind(provider, subject).first<{ account_id: string }>();
}

// 以登入方式建立新帳號。不依 Email 或名稱去合併既有帳號。
export async function createAccountWithIdentity(provider: 'email' | 'line', subject: string, displayName: string) {
  const id = newId();
  await db().batch([
    db().prepare('INSERT INTO accounts (id, display_name) VALUES (?, ?)').bind(id, displayName.slice(0, 60) || '學員'),
    db().prepare('INSERT INTO login_identities (id, account_id, provider, subject) VALUES (?, ?, ?, ?)').bind(newId(), id, provider, subject),
  ]);
  await audit(id, 'account.create', id, { provider });
  return id;
}

export type LinkResult = 'linked' | 'already' | 'taken';

// 已登入帳號在本人確認後綁定另一種登入方式。
export async function linkIdentity(accountId: string, provider: 'email' | 'line', subject: string): Promise<LinkResult> {
  const existing = await findIdentity(provider, subject);
  if (existing) return existing.account_id === accountId ? 'already' : 'taken';
  const has = await db().prepare('SELECT 1 FROM login_identities WHERE account_id = ? AND provider = ?').bind(accountId, provider).first();
  if (has) return 'taken';
  await db().prepare('INSERT INTO login_identities (id, account_id, provider, subject) VALUES (?, ?, ?, ?)').bind(newId(), accountId, provider, subject).run();
  await audit(accountId, 'identity.link', accountId, { provider });
  return 'linked';
}

export async function identitiesOf(accountId: string) {
  const { results } = await db().prepare('SELECT provider, subject, created_at FROM login_identities WHERE account_id = ? ORDER BY created_at').bind(accountId).all<{ provider: string; subject: string; created_at: string }>();
  return results;
}

export async function unlinkIdentity(accountId: string, provider: 'email' | 'line') {
  const all = await identitiesOf(accountId);
  if (all.length <= 1) return false;
  await db().prepare('DELETE FROM login_identities WHERE account_id = ? AND provider = ?').bind(accountId, provider).run();
  await audit(accountId, 'identity.unlink', accountId, { provider });
  return true;
}

export async function membershipOf(accountId: string) {
  return db().prepare('SELECT started_at, expires_at FROM memberships WHERE account_id = ?').bind(accountId).first<{ started_at: string; expires_at: string }>();
}

export const isMembershipActive = (m: { expires_at: string } | null) => !!m && m.expires_at > new Date().toISOString();
