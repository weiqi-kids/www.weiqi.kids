import { defineMiddleware } from 'astro:middleware';

// 動態路由才載入 session；POST 一律檢查 Origin（CSRF）。
export const onRequest = defineMiddleware(async (context, next) => {
  context.locals.account = null;
  if (context.isPrerendered) {
    const res = await next();
    if (context.url.hostname !== 'www.weiqi.kids') res.headers.set('x-robots-tag', 'noindex, nofollow');
    return res;
  }
  if (context.request.method === 'POST') {
    const origin = context.request.headers.get('origin');
    if (!origin || origin !== context.url.origin) return new Response('Forbidden', { status: 403 });
  }
  const { loadSession } = await import('./server/session');
  const s = await loadSession(context.cookies);
  context.locals.account = s?.account ?? null;
  const res = await next();
  res.headers.set('cache-control', 'private, no-store');
  if (context.url.hostname !== 'www.weiqi.kids') res.headers.set('x-robots-tag', 'noindex, nofollow');
  return res;
});
