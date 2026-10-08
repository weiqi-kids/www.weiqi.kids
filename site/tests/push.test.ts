// 網頁推播加密：用 RFC 8291 的解密步驟反解，確認瀏覽器收得到正確內容。
import { describe, it, expect, vi, beforeAll } from 'vitest';

const env: Record<string, string> = {};
vi.mock('cloudflare:workers', () => ({ env }));

const b64u = (b: ArrayBuffer | Uint8Array) => Buffer.from(b instanceof Uint8Array ? b : new Uint8Array(b)).toString('base64url');

async function hkdf(salt: Uint8Array, ikm: Uint8Array, info: Uint8Array, n: number) {
  const k = await crypto.subtle.importKey('raw', ikm, 'HKDF', false, ['deriveBits']);
  return new Uint8Array(await crypto.subtle.deriveBits({ name: 'HKDF', hash: 'SHA-256', salt, info }, k, n * 8));
}

describe('sendPush', () => {
  beforeAll(async () => {
    const vapid = (await crypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, true, ['sign', 'verify'])) as CryptoKeyPair;
    env.VAPID_PUBLIC_KEY = b64u(await crypto.subtle.exportKey('raw', vapid.publicKey));
    env.VAPID_PRIVATE_JWK = JSON.stringify(await crypto.subtle.exportKey('jwk', vapid.privateKey));
  });

  it('產生瀏覽器可以解開的 aes128gcm 內容與 VAPID 標頭', async () => {
    const ua = (await crypto.subtle.generateKey({ name: 'ECDH', namedCurve: 'P-256' }, true, ['deriveBits'])) as CryptoKeyPair;
    const uaPublic = new Uint8Array(await crypto.subtle.exportKey('raw', ua.publicKey));
    const auth = crypto.getRandomValues(new Uint8Array(16));
    let captured: { headers: Record<string, string>; body: Uint8Array } | null = null;
    vi.stubGlobal('fetch', async (_url: string, init: { headers: Record<string, string>; body: Uint8Array }) => { captured = init; return new Response(null, { status: 201 }); });

    const { sendPush } = await import('../src/server/push');
    const r = await sendPush({ endpoint: 'https://push.example.com/abc', p256dh: b64u(uaPublic), auth: b64u(auth) }, { title: 't', body: '開團了', url: '/x/' });
    expect(r).toBe('ok');
    const { headers, body } = captured!;
    expect(headers['Content-Encoding']).toBe('aes128gcm');
    expect(headers.Authorization).toMatch(/^vapid t=[\w-]+\.[\w-]+\.[\w-]+, k=/);

    const salt = body.slice(0, 16);
    const idlen = body[20];
    const asPublic = body.slice(21, 21 + idlen);
    const cipher = body.slice(21 + idlen);
    const asKey = await crypto.subtle.importKey('raw', asPublic, { name: 'ECDH', namedCurve: 'P-256' }, false, []);
    const shared = new Uint8Array(await crypto.subtle.deriveBits({ name: 'ECDH', public: asKey }, ua.privateKey, 256));
    const enc = new TextEncoder();
    const info = new Uint8Array([...enc.encode('WebPush: info\0'), ...uaPublic, ...asPublic]);
    const ikm = await hkdf(auth, shared, info, 32);
    const cek = await hkdf(salt, ikm, enc.encode('Content-Encoding: aes128gcm\0'), 16);
    const nonce = await hkdf(salt, ikm, enc.encode('Content-Encoding: nonce\0'), 12);
    const key = await crypto.subtle.importKey('raw', cek, 'AES-GCM', false, ['decrypt']);
    const plain = new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: nonce }, key, cipher));
    expect(plain[plain.length - 1]).toBe(2);
    expect(JSON.parse(new TextDecoder().decode(plain.slice(0, -1)))).toEqual({ title: 't', body: '開團了', url: '/x/' });

    // VAPID 簽章可用公鑰驗證
    const [h, p, sig] = headers.Authorization.slice(8).split(', k=')[0].split('.');
    const pub = await crypto.subtle.importKey('raw', Buffer.from(env.VAPID_PUBLIC_KEY, 'base64url'), { name: 'ECDSA', namedCurve: 'P-256' }, false, ['verify']);
    expect(await crypto.subtle.verify({ name: 'ECDSA', hash: 'SHA-256' }, pub, Buffer.from(sig, 'base64url'), enc.encode(`${h}.${p}`))).toBe(true);
    expect(JSON.parse(Buffer.from(p, 'base64url').toString()).aud).toBe('https://push.example.com');
  });
});
