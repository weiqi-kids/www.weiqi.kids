// Worker 入口：網頁交給 Astro；每小時排程推進共學營每一團的狀態（ADR 0014）。
import { handle } from '@astrojs/cloudflare/handler';
import { runScheduled } from './server/camp/cohorts';

export default {
  fetch: handle,
  async scheduled(_controller: ScheduledController, _env: Cloudflare.Env, ctx: ExecutionContext) {
    ctx.waitUntil(runScheduled());
  },
} satisfies ExportedHandler<Cloudflare.Env>;
