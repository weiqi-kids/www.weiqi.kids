import type { APIRoute } from 'astro';
import { db } from '../server/db';
import { site } from '../data/site.js';
export const prerender = false;

// 共學營成果頁與團頁在 D1，不在建置時的 sitemap 裡，另外產生。
export const GET: APIRoute = async () => {
  const { results: showcases } = await db().prepare("SELECT slug, updated_at FROM showcases WHERE status = 'published'").all<{ slug: string; updated_at: string }>();
  const { results: cohorts } = await db().prepare("SELECT c.id, c.created_at FROM cohorts c JOIN showcases s ON s.id = c.showcase_id WHERE s.status = 'published' AND c.state IN ('payment', 'confirmed', 'running')").all<{ id: string; created_at: string }>();
  const url = (loc: string, lastmod?: string) => `<url><loc>${site.url}${loc}</loc>${lastmod ? `<lastmod>${lastmod.slice(0, 10)}</lastmod>` : ''}</url>`;
  const body = [url('/camp/'), ...showcases.map((s) => url(`/camp/r/${s.slug}/`, s.updated_at)), ...cohorts.map((c) => url(`/camp/g/${c.id}/`, c.created_at))].join('');
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${body}</urlset>`, {
    headers: { 'content-type': 'application/xml; charset=utf-8', 'cache-control': 'public, max-age=3600' },
  });
};
