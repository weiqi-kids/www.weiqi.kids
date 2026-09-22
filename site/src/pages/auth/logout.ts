import type { APIRoute } from 'astro';
import { destroySession } from '../../server/session';
export const prerender = false;
export const POST: APIRoute = async ({ cookies, redirect }) => {
  await destroySession(cookies);
  return redirect('/', 303);
};
