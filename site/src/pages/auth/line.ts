import type { APIRoute } from 'astro';
import { lineEnabled, lineAuthorizeUrl } from '../../server/line';
import { safeRedirect } from '../../server/http';
export const prerender = false;

// ?link=1：已登入帳號綁定 LINE
export const GET: APIRoute = async ({ url, locals, redirect }) => {
  if (!lineEnabled()) return redirect('/auth/login/');
  const link = url.searchParams.get('link') === '1';
  if (link && !locals.account) return redirect('/auth/login/?next=/account/');
  const target = await lineAuthorizeUrl(url.origin, link ? 'link' : 'login', link ? locals.account!.id : undefined, safeRedirect(url.searchParams.get('next')));
  return redirect(target, 302);
};
