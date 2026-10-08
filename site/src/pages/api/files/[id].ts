import type { APIRoute } from 'astro';
import { env } from 'cloudflare:workers';
import { db } from '../../../server/db';
import { getShowcase } from '../../../server/camp/showcases';
import { getCohort } from '../../../server/camp/cohorts';
import { showcaseSpaceRole, cohortSpaceRole } from '../../../server/camp/access';
export const prerender = false;

// 論壇圖片：網址不可猜測，且每次都檢查看得到該論壇的權限。
export const GET: APIRoute = async ({ params, locals }) => {
  const a = await db().prepare('SELECT a.r2_key, a.mime_type, a.space, p.status FROM attachments a JOIN posts p ON p.id = a.post_id WHERE a.id = ? AND a.deleted_at IS NULL')
    .bind(params.id).first<{ r2_key: string; mime_type: string; space: string; status: string }>();
  if (!a || !env.UPLOADS) return new Response('Not found', { status: 404 });
  const [kind, id] = a.space.split(':');
  let view = false;
  if (kind === 'showcase') { const s = await getShowcase(id); view = !!s && showcaseSpaceRole(s, locals.account).view; }
  else if (kind === 'cohort') { const c = await getCohort(id); const s = c && (await getShowcase(c.showcase_id)); view = !!c && !!s && (await cohortSpaceRole(c, s, locals.account)).view; }
  if (!view) return new Response('Not found', { status: 404 });
  const obj = await env.UPLOADS.get(a.r2_key);
  if (!obj) return new Response('Not found', { status: 404 });
  return new Response(obj.body, { headers: { 'content-type': a.mime_type, 'cache-control': 'private, max-age=300', 'x-content-type-options': 'nosniff' } });
};
