// 通知：一律寫進站內通知（帳號頁），有開推播的裝置另外送網頁推播。
import { db } from './db';
import { sendPush, pushEnabled } from './push';

export async function notify(accountId: string, kind: string, message: string, link?: string) {
  await db().prepare('INSERT INTO notifications (id, account_id, kind, message, link) VALUES (?, ?, ?, ?, ?)')
    .bind(crypto.randomUUID(), accountId, kind, message, link ?? null).run();
  if (!pushEnabled()) return;
  const { results } = await db().prepare('SELECT endpoint, p256dh, auth FROM push_subscriptions WHERE account_id = ?').bind(accountId).all<{ endpoint: string; p256dh: string; auth: string }>();
  for (const sub of results) {
    const r = await sendPush(sub, { title: '台灣好棋寶寶協會', body: message, url: link ?? '/account/' });
    if (r === 'gone') await db().prepare('DELETE FROM push_subscriptions WHERE endpoint = ?').bind(sub.endpoint).run();
  }
}

export async function notifyMany(accountIds: string[], kind: string, message: string, link?: string) {
  for (const id of new Set(accountIds)) await notify(id, kind, message, link);
}

export async function notifyAdmins(kind: string, message: string, link?: string) {
  const { results } = await db().prepare("SELECT id FROM accounts WHERE is_admin = 1 AND status = 'active'").all<{ id: string }>();
  await notifyMany(results.map((r) => r.id), kind, message, link);
}
