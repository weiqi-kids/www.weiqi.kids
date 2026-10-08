// 報名與匯款。學員填匯款日期與帳號末五碼；一律由管理員核款，講師只看得到狀態。
import { db, audit } from '../db';
import { notify } from '../notify';
import { newId, nowIso } from '../crypto';
import { activeEnrollmentCount, advance } from './cohorts';
import type { Cohort, Enrollment, Showcase } from './types';

export async function enrollmentOf(cohortId: string, accountId: string) {
  return db().prepare('SELECT * FROM enrollments WHERE cohort_id = ? AND account_id = ?').bind(cohortId, accountId).first<Enrollment>();
}
export async function getEnrollment(id: string) {
  return db().prepare('SELECT * FROM enrollments WHERE id = ?').bind(id).first<Enrollment>();
}

const code = () => {
  const d = new Date(Date.now() + 8 * 3_600_000);
  const rand = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'[b % 32]).join('');
  return `WK-${String(d.getUTCFullYear()).slice(2)}${String(d.getUTCMonth() + 1).padStart(2, '0')}-${rand}`;
};

export async function enroll(c: Cohort, s: Showcase, accountId: string) {
  if (c.state !== 'payment') return { ok: false as const, error: '這一團現在不開放報名。' };
  if (accountId === s.owner_id) return { ok: false as const, error: '講師不需要報名自己的團。' };
  const existing = await enrollmentOf(c.id, accountId);
  if (existing && existing.status !== 'cancelled') return { ok: true as const, enrollment: existing };
  if (c.max_size && (await activeEnrollmentCount(c.id)) >= c.max_size) {
    // 額滿：留在登記名單，下一團自動帶過去。
    await db().prepare('INSERT INTO interests (cohort_id, account_id, overflow) VALUES (?, ?, 1) ON CONFLICT (cohort_id, account_id) DO UPDATE SET overflow = 1').bind(c.id, accountId).run();
    return { ok: false as const, error: '這一團已經額滿。你會留在登記名單，下一團開放時自動幫你登記。' };
  }
  if (existing) await db().prepare("DELETE FROM enrollments WHERE id = ? AND status = 'cancelled'").bind(existing.id).run();
  for (let i = 0; i < 5; i++) {
    try {
      const id = newId();
      await db().prepare('INSERT INTO enrollments (id, code, cohort_id, account_id, amount) VALUES (?, ?, ?, ?, ?)').bind(id, code(), c.id, accountId, c.price ?? s.price).run();
      await audit(accountId, 'enrollment.create', id, { cohort: c.id });
      return { ok: true as const, enrollment: (await getEnrollment(id))! };
    } catch (err) {
      if (!String(err).includes('UNIQUE')) throw err;
    }
  }
  return { ok: false as const, error: '系統忙碌，請稍後再試。' };
}

export async function submitPayment(e: Enrollment, c: Cohort, paidOn: string, last5: string) {
  if (!['awaiting_payment', 'submitted'].includes(e.status) || c.state !== 'payment') return '現在不能填匯款資料。';
  if (!/^\d{4}-\d{2}-\d{2}$/.test(paidOn)) return '請填匯款日期。';
  if (!/^\d{5}$/.test(last5)) return '帳號末五碼要是 5 個數字。';
  await db().prepare("UPDATE enrollments SET status = 'submitted', paid_on = ?, account_last5 = ?, submitted_at = ? WHERE id = ?").bind(paidOn, last5, nowIso(), e.id).run();
  await audit(e.account_id, 'enrollment.submit_payment', e.id);
  return null;
}

// 第 1 堂（主題教學）開始前可以全額退款，開課後不退。
export async function requestRefund(e: Enrollment, c: Cohort) {
  if (!c.teaching_start || Date.now() >= Date.parse(c.teaching_start)) return '主題教學開始後就不能退款了。';
  if (e.status === 'awaiting_payment') {
    await db().prepare("UPDATE enrollments SET status = 'cancelled' WHERE id = ?").bind(e.id).run();
  } else if (e.status === 'submitted' || e.status === 'confirmed') {
    await db().prepare("UPDATE enrollments SET status = 'refund_pending', refund_reason = '學員開課前申請退款' WHERE id = ?").bind(e.id).run();
    const { notifyAdmins } = await import('../notify');
    await notifyAdmins('refund.due', `有學員開課前申請退款（報名編號 ${e.code}）。`, '/admin/payments/');
  } else return '這筆報名不能退款。';
  await audit(e.account_id, 'enrollment.refund_request', e.id);
  return null;
}

// ---- 管理員 ----

export async function confirmPayment(e: Enrollment, adminId: string) {
  if (e.status !== 'submitted') return false;
  await db().prepare("UPDATE enrollments SET status = 'confirmed', confirmed_by = ?, confirmed_at = ? WHERE id = ?").bind(adminId, nowIso(), e.id).run();
  await audit(adminId, 'enrollment.confirm', e.id);
  await notify(e.account_id, 'enrollment.confirmed', `協會已確認你的匯款（報名編號 ${e.code}）。`, `/camp/g/${e.cohort_id}/`);
  await advance(e.cohort_id);
  return true;
}

export async function rejectPayment(e: Enrollment, adminId: string, note: string) {
  if (e.status !== 'submitted') return false;
  await db().prepare("UPDATE enrollments SET status = 'awaiting_payment', submitted_at = NULL WHERE id = ?").bind(e.id).run();
  await audit(adminId, 'enrollment.reject', e.id, { note });
  await notify(e.account_id, 'enrollment.rejected', `協會對不到你的匯款（報名編號 ${e.code}）${note ? `：${note}` : ''}。請確認後重新填寫匯款資料。`, `/camp/g/${e.cohort_id}/`);
  return true;
}

export async function markRefunded(e: Enrollment, adminId: string) {
  if (e.status !== 'refund_pending') return false;
  await db().prepare("UPDATE enrollments SET status = 'refunded', refunded_by = ?, refunded_at = ? WHERE id = ?").bind(adminId, nowIso(), e.id).run();
  await audit(adminId, 'enrollment.refunded', e.id);
  await notify(e.account_id, 'enrollment.refunded', `協會已把 ${e.amount.toLocaleString('zh-TW')} 元退回你的帳戶（報名編號 ${e.code}）。`);
  return true;
}
