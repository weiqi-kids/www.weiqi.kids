// 伺服器端 session：cookie 只放隨機值，D1 存 hash，可撤銷。
import type { AstroCookies } from 'astro';
import { db } from './db';
import { randomToken, sha256hex, addDays } from './crypto';
import type { Account } from './types';

export const SESSION_COOKIE = 'wk_session';
const SESSION_DAYS = 30;

export async function createSession(cookies: AstroCookies, accountId: string, secure: boolean) {
  const token = randomToken();
  await db().prepare('INSERT INTO sessions (id_hash, account_id, expires_at) VALUES (?, ?, ?)')
    .bind(await sha256hex(token), accountId, addDays(SESSION_DAYS)).run();
  cookies.set(SESSION_COOKIE, token, { path: '/', httpOnly: true, secure, sameSite: 'lax', maxAge: SESSION_DAYS * 86400 });
}

export async function loadSession(cookies: AstroCookies): Promise<{ account: Account; sessionHash: string } | null> {
  const token = cookies.get(SESSION_COOKIE)?.value;
  if (!token) return null;
  const hash = await sha256hex(token);
  const row = await db().prepare(
    `SELECT a.id, a.display_name, a.is_admin, a.status FROM sessions s JOIN accounts a ON a.id = s.account_id
     WHERE s.id_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ? AND a.status = 'active'`,
  ).bind(hash, new Date().toISOString()).first<Account>();
  return row ? { account: row, sessionHash: hash } : null;
}

export async function destroySession(cookies: AstroCookies) {
  const token = cookies.get(SESSION_COOKIE)?.value;
  if (token) await db().prepare('UPDATE sessions SET revoked_at = ? WHERE id_hash = ?').bind(new Date().toISOString(), await sha256hex(token)).run();
  cookies.delete(SESSION_COOKIE, { path: '/' });
}
