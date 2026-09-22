import { describe, it, expect } from 'vitest';
import { parseEnrollment, enrollmentCode } from '../src/lib/enrollment';

const open = new Set(['pilot']);
const form = (fields: Record<string, string>) => {
  const f = new FormData();
  for (const [k, v] of Object.entries({ course: 'pilot', name: '王小明', email: 'A@Example.com', membership: 'new', consent: 'yes', ...fields })) f.set(k, v);
  return f;
};

describe('parseEnrollment', () => {
  it('接受完整報名並把 Email 轉小寫', () => {
    const r = parseEnrollment(form({}), open);
    expect(r).toEqual({ ok: true, value: { course: 'pilot', name: '王小明', email: 'a@example.com', lineName: null, membership: 'new', note: null } });
  });
  it('拒絕沒有開放的課程', () => {
    expect(parseEnrollment(form({ course: 'other' }), open).ok).toBe(false);
  });
  it('拒絕無效 Email', () => {
    expect(parseEnrollment(form({ email: 'no-at' }), open).ok).toBe(false);
  });
  it('沒勾同意就拒絕', () => {
    expect(parseEnrollment(form({ consent: '' }), open).ok).toBe(false);
  });
  it('填了隱藏欄位視為垃圾送出', () => {
    const r = parseEnrollment(form({ website: 'http://spam' }), open);
    expect(r).toMatchObject({ ok: false, spam: true });
  });
});

describe('enrollmentCode', () => {
  it('用台北時間的年月，並避開易混淆字元', () => {
    const code = enrollmentCode(new Date('2026-09-30T17:00:00Z'), new Uint8Array([0, 1, 2, 3]));
    expect(code).toBe('WK-202610-2345');
  });
  it('格式固定', () => {
    const code = enrollmentCode(new Date(), crypto.getRandomValues(new Uint8Array(4)));
    expect(code).toMatch(/^WK-\d{6}-[2-9A-HJKMNP-Z]{4}$/);
  });
});
