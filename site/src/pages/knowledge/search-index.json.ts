import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

export const prerender = true;

// 去掉 Markdown 與 MDX 的標記，只留可搜尋的文字。
function plain(body: string) {
  return body
    .replace(/^---[\s\S]*?\n---\n/, '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[#>*_`|:-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export const GET: APIRoute = async () => {
  const pages = await getCollection('pages');
  const items = pages
    .filter((p) => p.data.kind === 'knowledge')
    .map((p) => ({
      t: p.data.title,
      u: p.data.path,
      d: p.data.description ?? '',
      k: (p.data.keywords ?? []).join(' '),
      b: plain(p.body ?? '').slice(0, 1200),
    }))
    .sort((a, b) => a.u.localeCompare(b.u));

  return new Response(JSON.stringify(items), {
    headers: { 'content-type': 'application/json; charset=utf-8' },
  });
};
