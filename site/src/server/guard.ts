import type { AstroGlobal } from 'astro';
// 管理區：必須是協會管理員。回傳 Response 表示要中止。
export function requireAdmin(Astro: AstroGlobal) {
  const me = Astro.locals.account;
  if (!me) return Astro.redirect(`/auth/login/?next=${encodeURIComponent(Astro.url.pathname)}`);
  if (me.is_admin !== 1) return new Response('Forbidden', { status: 403 });
  return null;
}
