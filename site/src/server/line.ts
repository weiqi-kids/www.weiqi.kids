import { env } from 'cloudflare:workers';
import { db } from './db';
import { randomToken, sha256hex, addMinutes, nowIso } from './crypto';

export const lineEnabled = () => !!(env.LINE_CHANNEL_ID && env.LINE_CHANNEL_SECRET);

const callbackUrl = (origin: string) => `${origin}/auth/callback/line/`;

export async function lineAuthorizeUrl(origin: string, purpose: 'login' | 'link', accountId?: string, redirectTo?: string) {
  const state = randomToken(24);
  await db().prepare('INSERT INTO oauth_states (state_hash, purpose, account_id, redirect_to, expires_at) VALUES (?, ?, ?, ?, ?)')
    .bind(await sha256hex(state), purpose, accountId ?? null, redirectTo ?? null, addMinutes(10)).run();
  const u = new URL('https://access.line.me/oauth2/v2.1/authorize');
  u.searchParams.set('response_type', 'code');
  u.searchParams.set('client_id', env.LINE_CHANNEL_ID!);
  u.searchParams.set('redirect_uri', callbackUrl(origin));
  u.searchParams.set('state', state);
  u.searchParams.set('scope', 'profile openid');
  return u.toString();
}

export async function consumeState(state: string) {
  const hash = await sha256hex(state);
  const now = nowIso();
  const res = await db().prepare('UPDATE oauth_states SET used_at = ? WHERE state_hash = ? AND used_at IS NULL AND expires_at > ?').bind(now, hash, now).run();
  if (res.meta.changes !== 1) return null;
  return db().prepare('SELECT purpose, account_id, redirect_to FROM oauth_states WHERE state_hash = ?').bind(hash)
    .first<{ purpose: 'login' | 'link'; account_id: string | null; redirect_to: string | null }>();
}

// 以授權碼換 id_token，再向 LINE 驗證，取得使用者 userId（sub）與顯示名稱。
export async function exchangeCode(code: string, origin: string): Promise<{ sub: string; name: string } | null> {
  const tokenRes = await fetch('https://api.line.me/oauth2/v2.1/token', {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ grant_type: 'authorization_code', code, redirect_uri: callbackUrl(origin), client_id: env.LINE_CHANNEL_ID!, client_secret: env.LINE_CHANNEL_SECRET! }),
  });
  if (!tokenRes.ok) return null;
  const { id_token } = (await tokenRes.json()) as { id_token?: string };
  if (!id_token) return null;
  const verifyRes = await fetch('https://api.line.me/oauth2/v2.1/verify', {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ id_token, client_id: env.LINE_CHANNEL_ID! }),
  });
  if (!verifyRes.ok) return null;
  const claims = (await verifyRes.json()) as { sub?: string; name?: string };
  return claims.sub ? { sub: claims.sub, name: claims.name ?? 'LINE 使用者' } : null;
}
