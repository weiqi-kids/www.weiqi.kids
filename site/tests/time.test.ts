import { describe, it, expect } from 'vitest';
import { nextSlotAfter, parseTaipeiLocal, fmtDateTime } from '../src/lib/time';

describe('台北時間', () => {
  it('每週時段：找 notBefore 之後最早的一次', () => {
    // 2026-10-08 是週四；台北週六 10:00 = UTC 02:00
    expect(nextSlotAfter(6, '10:00', Date.parse('2026-10-08T00:00:00Z'))).toBe('2026-10-10T02:00:00.000Z');
    // 同一天但時間已過 → 下週
    expect(nextSlotAfter(6, '10:00', Date.parse('2026-10-10T03:00:00Z'))).toBe('2026-10-17T02:00:00.000Z');
    // 台北週三 19:30 = UTC 11:30
    expect(nextSlotAfter(3, '19:30', Date.parse('2026-10-16T01:30:00Z'))).toBe('2026-10-21T11:30:00.000Z');
  });
  it('datetime-local 視為台北時間', () => {
    expect(parseTaipeiLocal('2026-11-08T14:00')).toBe('2026-11-08T06:00:00.000Z');
    expect(fmtDateTime('2026-11-08T06:00:00.000Z')).toBe('2026/11/08（週日）14:00');
  });
});
