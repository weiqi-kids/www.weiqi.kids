// 好棋寶寶 MCP 的 OAuth（ADR 0015）：用網站帳號登入。授權頁在 /oauth/authorize/（Astro 頁面）。
import type { OAuthProviderOptions } from '@cloudflare/workers-oauth-provider';
import { db } from './db';
import { handleMcp } from './mcp';
import type { Account } from './types';

export interface McpProps { userId: string }

// origin：這個環境對外的網址（staging 或正式站），用來宣告 MCP 的資源識別。
export const oauthOptions = (origin: string): OAuthProviderOptions<Cloudflare.Env> => ({
  apiRoute: '/mcp',
  apiHandler: {
    async fetch(request: Request, _env: Cloudflare.Env, ctx: ExecutionContext & { props?: McpProps }) {
      const userId = ctx.props?.userId;
      const me = userId ? await db().prepare("SELECT id, display_name, is_admin, status FROM accounts WHERE id = ? AND status = 'active'").bind(userId).first<Account>() : null;
      if (!me) return new Response('帳號不存在或已停用', { status: 403 });
      return handleMcp(request, me);
    },
  } as never,
  defaultHandler: { fetch: () => new Response('Not found', { status: 404 }) } as never, // worker.ts 會換成 Astro
  authorizeEndpoint: '/oauth/authorize/',
  tokenEndpoint: '/oauth/token',
  clientRegistrationEndpoint: '/oauth/register',
  scopesSupported: ['haoqi'],
  clientIdMetadataDocumentEnabled: true,
  refreshTokenIdleTTL: 60 * 60 * 24 * 90,
  resourceMetadata: { resource: `${origin}/mcp`, authorization_servers: [origin], resource_name: '好棋寶寶' },
});
