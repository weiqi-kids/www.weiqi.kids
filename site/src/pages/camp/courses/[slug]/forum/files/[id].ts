import type { APIRoute } from 'astro';
import { env } from 'cloudflare:workers';
import { db } from '../../../../../../server/db';
import { courseRole, can } from '../../../../../../server/permissions';
import { getPost, visibleTo } from '../../../../../../server/forum';
export const prerender = false;

// 圖片一律經權限檢查才輸出；R2 不開放公開存取。
export const GET: APIRoute = async ({ params, locals }) => {
  const me = locals.account;
  if (!me) return new Response('Unauthorized', { status: 401 });
  const slug = params.slug!;
  const role = await courseRole(me, slug);
  if (!can.viewForum(role)) return new Response('Forbidden', { status: 403 });
  const file = await db().prepare('SELECT post_id, r2_key, mime_type FROM attachments WHERE id = ? AND course_slug = ? AND deleted_at IS NULL')
    .bind(params.id, slug).first<{ post_id: string; r2_key: string; mime_type: string }>();
  if (!file) return new Response('Not found', { status: 404 });
  const post = await getPost(slug, file.post_id);
  if (!post || !visibleTo(post, me, role)) return new Response('Not found', { status: 404 });
  if (!env.UPLOADS) return new Response('Not found', { status: 404 });
  const obj = await env.UPLOADS.get(file.r2_key);
  if (!obj) return new Response('Not found', { status: 404 });
  return new Response(obj.body, { headers: { 'content-type': file.mime_type, 'cache-control': 'private, max-age=300', 'x-content-type-options': 'nosniff', 'content-disposition': 'inline' } });
};
