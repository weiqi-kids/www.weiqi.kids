import type { APIRoute } from 'astro';
import { env } from 'cloudflare:workers';
import { getCollection } from 'astro:content';
import { parseEnrollment, enrollmentCode, hashIp } from '../../lib/enrollment';

export const prerender = false;

const MAX_PER_HOUR = 5;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' } });

async function verifyTurnstile(token: string, ip: string, secret: string) {
  const body = new FormData();
  body.set('secret', secret);
  body.set('response', token);
  body.set('remoteip', ip);
  const res = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', { method: 'POST', body });
  const data = (await res.json()) as { success: boolean };
  return data.success;
}

export const POST: APIRoute = async ({ request }) => {
  const form = await request.formData();
  const courses = await getCollection('courses');
  const showDrafts = env.SHOW_DRAFT_COURSES === '1';
  const open = new Set(courses.filter((c) => c.data.status === 'open' || (showDrafts && c.data.status === 'draft')).map((c) => c.data.slug));

  const parsed = parseEnrollment(form, open);
  if (!parsed.ok) return json({ error: parsed.error }, parsed.spam ? 400 : 422);

  const ip = request.headers.get('cf-connecting-ip') ?? '0.0.0.0';
  if (env.TURNSTILE_SECRET_KEY) {
    const token = form.get('cf-turnstile-response');
    if (typeof token !== 'string' || !(await verifyTurnstile(token, ip, env.TURNSTILE_SECRET_KEY)))
      return json({ error: '驗證失敗，請重新整理頁面再送出。' }, 400);
  }

  const ipHash = await hashIp(ip, env.IP_HASH_SALT ?? 'weiqi-kids');
  const recent = await env.DB.prepare(
    "SELECT COUNT(*) AS n FROM enrollments WHERE ip_hash = ? AND created_at > strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-1 hour')",
  ).bind(ipHash).first<{ n: number }>();
  if ((recent?.n ?? 0) >= MAX_PER_HOUR) return json({ error: '送出次數太多，請一小時後再試，或直接聯絡協會。' }, 429);

  const v = parsed.value;
  for (let attempt = 0; attempt < 5; attempt++) {
    const code = enrollmentCode(new Date(), crypto.getRandomValues(new Uint8Array(4)));
    try {
      await env.DB.prepare(
        'INSERT INTO enrollments (code, course_slug, name, email, line_name, membership, note, ip_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      ).bind(code, v.course, v.name, v.email, v.lineName, v.membership, v.note, ipHash).run();
      return json({ code, email: v.email }, 201);
    } catch (err) {
      if (!String(err).includes('UNIQUE')) throw err;
    }
  }
  return json({ error: '系統忙碌，請稍後再試。' }, 503);
};

export const ALL: APIRoute = () => json({ error: 'Method not allowed' }, 405);
