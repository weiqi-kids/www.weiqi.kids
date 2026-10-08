// Worker 入口：
// - /mcp 由 OAuth 保護，交給好棋寶寶 MCP（ADR 0015）；/oauth/token 等由 OAuth 套件處理。
// - 其他網頁交給 Astro（包含 /oauth/authorize/ 授權頁）。
// - 每小時排程推進共學營每一團的狀態（ADR 0014）。
import { OAuthProvider } from '@cloudflare/workers-oauth-provider';
import { handle } from '@astrojs/cloudflare/handler';
import { runScheduled } from './server/camp/cohorts';
import { oauthOptions } from './server/oauth';

let provider: OAuthProvider<Cloudflare.Env> | null = null;

export default {
  fetch(request: Request, env: Cloudflare.Env, ctx: ExecutionContext) {
    provider ??= new OAuthProvider<Cloudflare.Env>({ ...oauthOptions(env.SITE_ORIGIN), defaultHandler: { fetch: handle } as never });
    return provider.fetch(request, env, ctx);
  },
  async scheduled(_controller: ScheduledController, _env: Cloudflare.Env, ctx: ExecutionContext) {
    ctx.waitUntil(runScheduled());
  },
};
