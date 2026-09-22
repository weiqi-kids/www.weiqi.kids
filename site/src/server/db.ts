import { env } from 'cloudflare:workers';

export const db = () => env.DB;

export async function audit(actorId: string | null, action: string, target: string | null, detail?: unknown) {
  await db().prepare('INSERT INTO audit_log (actor_id, action, target, detail) VALUES (?, ?, ?, ?)')
    .bind(actorId, action, target, detail === undefined ? null : JSON.stringify(detail)).run();
}

export async function notify(accountId: string, kind: string, message: string, link?: string) {
  await db().prepare('INSERT INTO notifications (id, account_id, kind, message, link) VALUES (?, ?, ?, ?, ?)')
    .bind(crypto.randomUUID(), accountId, kind, message, link ?? null).run();
}
