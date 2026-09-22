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
    SHOW_DRAFT_COURSES?: string;
    MAGIC_LINK_DEBUG?: string;
    MAIL_FROM?: string;
    LINE_CHANNEL_ID?: string;
    LINE_CHANNEL_SECRET?: string;
  }
}
