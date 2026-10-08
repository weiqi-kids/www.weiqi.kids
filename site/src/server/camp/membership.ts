// 常年會費：會員資格是上架成果的條件。會員自行匯款後填資料，由管理員核款。
import { db, audit } from '../db';
import { notify, notifyAdmins } from '../notify';
import { newId, nowIso, addDays } from '../crypto';
import { membershipOf, isMembershipActive } from '../accounts';
import { campRules } from '../../data/site.js';

export async function duesOf(accountId: string) {
  return (await db().prepare('SELECT id, amount, paid_on, status, review_note, created_at FROM membership_dues WHERE account_id = ? ORDER BY created_at DESC').bind(accountId)
    .all<{ id: string; amount: number; paid_on: string; status: string; review_note: string | null; created_at: string }>()).results;
}

export async function submitDues(accountId: string, paidOn: string, last5: string) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(paidOn)) return '請填匯款日期。';
  if (!/^\d{5}$/.test(last5)) return '帳號末五碼要是 5 個數字。';
  const pending = await db().prepare("SELECT 1 FROM membership_dues WHERE account_id = ? AND status = 'submitted'").bind(accountId).first();
  if (pending) return '你已經填過一筆，正在等管理員核對。';
  const id = newId();
  await db().prepare('INSERT INTO membership_dues (id, account_id, amount, paid_on, account_last5) VALUES (?, ?, ?, ?, ?)').bind(id, accountId, campRules.membershipFee, paidOn, last5).run();
  await audit(accountId, 'dues.submit', id);
  await notifyAdmins('dues.submitted', '有一筆常年會費等待核對。', '/admin/payments/');
  return null;
}

// 核款：會員有效就從到期日延一年，否則從今天起算一年。
export async function confirmDues(id: string, adminId: string) {
  const d = await db().prepare("SELECT account_id FROM membership_dues WHERE id = ? AND status = 'submitted'").bind(id).first<{ account_id: string }>();
  if (!d) return false;
  const m = await membershipOf(d.account_id);
  const from = m && isMembershipActive(m) ? Date.parse(m.expires_at) : Date.now();
  const expires = addDays(365, from);
  await db().batch([
    db().prepare("UPDATE membership_dues SET status = 'confirmed', reviewed_by = ?, reviewed_at = ? WHERE id = ?").bind(adminId, nowIso(), id),
    m ? db().prepare('UPDATE memberships SET expires_at = ?, updated_by = ?, updated_at = ? WHERE account_id = ?').bind(expires, adminId, nowIso(), d.account_id)
      : db().prepare('INSERT INTO memberships (account_id, started_at, expires_at, updated_by) VALUES (?, ?, ?, ?)').bind(d.account_id, nowIso(), expires, adminId),
  ]);
  await audit(adminId, 'dues.confirm', id, { expires });
  await notify(d.account_id, 'dues.confirmed', '協會已確認你的常年會費，會員資格有效一年。現在可以建立成果了。', '/account/teach/');
  return true;
}

export async function rejectDues(id: string, adminId: string, note: string) {
  const d = await db().prepare("SELECT account_id FROM membership_dues WHERE id = ? AND status = 'submitted'").bind(id).first<{ account_id: string }>();
  if (!d) return false;
  await db().prepare("UPDATE membership_dues SET status = 'rejected', reviewed_by = ?, reviewed_at = ?, review_note = ? WHERE id = ?").bind(adminId, nowIso(), note, id).run();
  await audit(adminId, 'dues.reject', id, { note });
  await notify(d.account_id, 'dues.rejected', `協會對不到你的常年會費匯款${note ? `：${note}` : ''}。請確認後重新填寫。`, '/account/membership/');
  return true;
}
