import { env } from 'cloudflare:workers';
import { sha256hex } from './crypto';

export const clientIpHash = async (request: Request) =>
  sha256hex(`${env.IP_HASH_SALT ?? 'weiqi-kids'}:${request.headers.get('cf-connecting-ip') ?? '0.0.0.0'}`);

// 只允許站內相對路徑，避免開放轉址。
export function safeRedirect(target: string | null | undefined, fallback = '/account/') {
  if (!target || !target.startsWith('/') || target.startsWith('//') || target.includes('\\')) return fallback;
  return target;
}

export const str = (form: FormData, key: string, max = 5000) => {
  const v = form.get(key);
  return typeof v === 'string' ? v.trim().slice(0, max) : '';
};

export const isSecure = (url: URL) => url.protocol === 'https:';
