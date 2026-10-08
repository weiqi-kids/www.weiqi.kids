import type { APIRoute } from 'astro';
import { db, audit } from '../../../server/db';
export const prerender = false;

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' } });

// 登入者把這台裝置的推播訂閱存起來；DELETE 取消。
export const POST: APIRoute = async ({ request, locals }) => {
  const me = locals.account;
  if (!me) return json({ error: '請先登入。' }, 401);
  const sub = (await request.json().catch(() => null)) as { endpoint?: string; keys?: { p256dh?: string; auth?: string } } | null;
  if (!sub?.endpoint?.startsWith('https://') || !sub.keys?.p256dh || !sub.keys?.auth) return json({ error: '訂閱資料不完整。' }, 400);
  await db().prepare('INSERT INTO push_subscriptions (endpoint, account_id, p256dh, auth, user_agent) VALUES (?, ?, ?, ?, ?) ON CONFLICT (endpoint) DO UPDATE SET account_id = excluded.account_id, p256dh = excluded.p256dh, auth = excluded.auth')
    .bind(sub.endpoint, me.id, sub.keys.p256dh, sub.keys.auth, (request.headers.get('user-agent') ?? '').slice(0, 200)).run();
  await audit(me.id, 'push.subscribe', me.id);
  return json({ ok: true });
};

export const DELETE: APIRoute = async ({ request, locals }) => {
  const me = locals.account;
  if (!me) return json({ error: '請先登入。' }, 401);
  const body = (await request.json().catch(() => null)) as { endpoint?: string } | null;
  if (body?.endpoint) await db().prepare('DELETE FROM push_subscriptions WHERE endpoint = ? AND account_id = ?').bind(body.endpoint, me.id).run();
  return json({ ok: true });
};
