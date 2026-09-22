export function randomToken(bytes = 32): string {
  const buf = crypto.getRandomValues(new Uint8Array(bytes));
  return btoa(String.fromCharCode(...buf)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export async function sha256hex(input: string | ArrayBuffer): Promise<string> {
  const data = typeof input === 'string' ? new TextEncoder().encode(input) : input;
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

export const newId = () => crypto.randomUUID();

export const nowIso = () => new Date().toISOString();
export const addMinutes = (m: number) => new Date(Date.now() + m * 60_000).toISOString();
export const addDays = (d: number, from = Date.now()) => new Date(from + d * 86_400_000).toISOString();
