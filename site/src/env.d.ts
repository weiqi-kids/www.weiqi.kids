/// <reference types="astro/client" />

declare namespace App {
  interface Locals {
    account: import('./server/types').Account | null;
  }
}

declare namespace Cloudflare {
  interface Env {
    DB: D1Database;
    UPLOADS?: R2Bucket;
    TURNSTILE_SECRET_KEY?: string;
    IP_HASH_SALT?: string;
    MAGIC_LINK_DEBUG?: string;
    MAIL_FROM?: string;
    LINE_CHANNEL_ID?: string;
    LINE_CHANNEL_SECRET?: string;
    VAPID_PUBLIC_KEY?: string;
    BANK_ACCOUNT_INFO?: string;
    GITHUB_TOKEN?: string;
    SITE_ORIGIN: string;
    OAUTH_KV: KVNamespace;
    VAPID_PRIVATE_JWK?: string;
    VAPID_SUBJECT?: string;
  }
}

declare module '*.md?raw' {
  const content: string;
  export default content;
}
