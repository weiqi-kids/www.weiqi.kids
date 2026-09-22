import { db, audit } from './db';
import { randomToken, sha256hex, addMinutes, nowIso } from './crypto';
import { sendLoginEmail } from './mail';

const TTL_MINUTES = 15;
const PER_EMAIL_15MIN = 3;
const PER_IP_HOUR = 10;

export type RequestResult = { ok: true; debugLink?: string } | { ok: false; error: string };

export async function requestMagicLink(opts: {
  email: string; purpose: 'login' | 'link'; accountId?: string; redirectTo?: string; ipHash: string; origin: string;
}): Promise<RequestResult> {
  const { email, purpose, ipHash } = opts;
  const since15 = new Date(Date.now() - 15 * 60_000).toISOString();
  const since60 = new Date(Date.now() - 60 * 60_000).toISOString();
  const [byEmail, byIp] = await Promise.all([
    db().prepare('SELECT COUNT(*) AS n FROM magic_tokens WHERE email = ? AND created_at > ?').bind(email, since15).first<{ n: number }>(),
    db().prepare('SELECT COUNT(*) AS n FROM magic_tokens WHERE ip_hash = ? AND created_at > ?').bind(ipHash, since60).first<{ n: number }>(),
  ]);
  if ((byEmail?.n ?? 0) >= PER_EMAIL_15MIN || (byIp?.n ?? 0) >= PER_IP_HOUR)
    return { ok: false, error: '申請次數太多，請稍後再試。' };

  const token = randomToken();
  const hash = await sha256hex(token);
  const now = nowIso();
  await db().batch([
    // 重發使同一 Email、同一用途的舊連結失效
    db().prepare('UPDATE magic_tokens SET revoked_at = ? WHERE email = ? AND purpose = ? AND used_at IS NULL AND revoked_at IS NULL').bind(now, email, purpose),
    db().prepare('INSERT INTO magic_tokens (token_hash, email, purpose, account_id, redirect_to, expires_at, ip_hash) VALUES (?, ?, ?, ?, ?, ?, ?)')
      .bind(hash, email, purpose, opts.accountId ?? null, opts.redirectTo ?? null, addMinutes(TTL_MINUTES), ipHash),
  ]);
  const link = `${opts.origin}/auth/magic-link/?token=${encodeURIComponent(token)}`;
  const sent = await sendLoginEmail(email, link, purpose);
  if (!sent.ok) {
    // 寄不出去就不能留下有效連結
    await db().prepare('UPDATE magic_tokens SET revoked_at = ? WHERE token_hash = ?').bind(nowIso(), hash).run();
    await audit(opts.accountId ?? null, 'magic.send_failed', null, { purpose });
    return { ok: false, error: '寄信服務暫時無法使用，請稍後再試，或改用 LINE 登入。' };
  }
  await audit(opts.accountId ?? null, 'magic.sent', null, { purpose });
  return { ok: true, debugLink: sent.debugLink };
}

export interface ConsumedToken { email: string; purpose: 'login' | 'link'; account_id: string | null; redirect_to: string | null }

// 單次使用：以條件式 UPDATE 確保同一 token 只能成功一次。
export async function consumeMagicLink(token: string): Promise<ConsumedToken | null> {
  const hash = await sha256hex(token);
  const now = nowIso();
  const res = await db().prepare(
    'UPDATE magic_tokens SET used_at = ? WHERE token_hash = ? AND used_at IS NULL AND revoked_at IS NULL AND expires_at > ?',
  ).bind(now, hash, now).run();
  if (res.meta.changes !== 1) return null;
  return db().prepare('SELECT email, purpose, account_id, redirect_to FROM magic_tokens WHERE token_hash = ?').bind(hash).first<ConsumedToken>();
}
