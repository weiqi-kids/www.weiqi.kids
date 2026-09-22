// 報名表的驗證與報名編號產生（純函式，方便測試）。

export interface EnrollmentInput {
  course: string;
  name: string;
  email: string;
  lineName: string | null;
  membership: 'new' | 'member';
  note: string | null;
}

export type ParseResult = { ok: true; value: EnrollmentInput } | { ok: false; error: string; spam?: boolean };

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const clean = (v: FormDataEntryValue | null) => (typeof v === 'string' ? v.trim() : '');

export function parseEnrollment(form: FormData, openCourses: Set<string>): ParseResult {
  if (clean(form.get('website'))) return { ok: false, error: '送出失敗。', spam: true };
  const course = clean(form.get('course'));
  const name = clean(form.get('name'));
  const email = clean(form.get('email')).toLowerCase();
  const lineName = clean(form.get('lineName'));
  const membership = clean(form.get('membership'));
  const note = clean(form.get('note'));
  if (!openCourses.has(course)) return { ok: false, error: '這門課目前沒有開放報名。' };
  if (!name || name.length > 60) return { ok: false, error: '請填寫姓名（60 字以內）。' };
  if (!EMAIL.test(email) || email.length > 120) return { ok: false, error: '請填寫有效的 Email。' };
  if (lineName.length > 60) return { ok: false, error: 'LINE 顯示名稱請在 60 字以內。' };
  if (membership !== 'new' && membership !== 'member') return { ok: false, error: '請選擇會員身分。' };
  if (note.length > 500) return { ok: false, error: '備註請在 500 字以內。' };
  if (clean(form.get('consent')) !== 'yes') return { ok: false, error: '請勾選同意資料使用說明。' };
  return { ok: true, value: { course, name, email, lineName: lineName || null, membership, note: note || null } };
}

// 去掉容易看錯的 0/O、1/I/L。
const ALPHABET = '23456789ABCDEFGHJKMNPQRSTUVWXYZ';

export function enrollmentCode(now: Date, random: Uint8Array): string {
  const taipei = new Date(now.getTime() + 8 * 3600 * 1000);
  const ym = `${taipei.getUTCFullYear()}${String(taipei.getUTCMonth() + 1).padStart(2, '0')}`;
  const suffix = Array.from(random.slice(0, 4), (b) => ALPHABET[b % ALPHABET.length]).join('');
  return `WK-${ym}-${suffix}`;
}

export async function hashIp(ip: string, salt: string): Promise<string> {
  const data = new TextEncoder().encode(`${salt}:${ip}`);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}
