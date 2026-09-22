/// <reference types="astro/client" />

declare namespace Cloudflare {
  interface Env {
    DB: D1Database;
    TURNSTILE_SECRET_KEY?: string;
    IP_HASH_SALT?: string;
    SHOW_DRAFT_COURSES?: string;
  }
}
