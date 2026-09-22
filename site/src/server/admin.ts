import { db, audit, notify } from './db';
import { newId, nowIso, addDays } from './crypto';
import { findIdentity, createAccountWithIdentity, membershipOf, isMembershipActive } from './accounts';

export interface EnrollmentRow {
  code: string; course_slug: string; name: string; email: string; line_name: string | null; membership: string;
  note: string | null; status: string; account_id: string | null; created_at: string; review_note: string | null;
}

// 核款確認並開通課程（ADR 0004、0005）：
// - 報名已連到帳號就用該帳號；否則用報名 Email 的登入方式找帳號，沒有就預先建立 Email 帳號（不合併其他帳號）。
// - 沒有有效會員資格時建立一年期會員；第一次報名即入會。會員已到期時需管理員勾選「同時續會」。
export async function confirmEnrollment(code: string, adminId: string, opts: { renew: boolean; note: string | null }) {
  const e = await db().prepare('SELECT * FROM enrollments WHERE code = ?').bind(code).first<EnrollmentRow>();
  if (!e || e.status === 'activated' || e.status === 'cancelled') return { ok: false as const, error: '這筆報名無法開通。' };
  let accountId = e.account_id;
  if (!accountId) accountId = (await findIdentity('email', e.email))?.account_id ?? (await createAccountWithIdentity('email', e.email, e.name));

  const m = await membershipOf(accountId);
  const active = isMembershipActive(m);
  if (!active && m && !opts.renew) return { ok: false as const, error: '這位學員的會員已到期，請確認已收到續會款項後勾選「同時續會一年」。' };

  const now = nowIso();
  const stmts = [
    db().prepare("UPDATE enrollments SET status = 'activated', account_id = ?, reviewed_by = ?, reviewed_at = ?, review_note = ?, updated_at = ? WHERE code = ?").bind(accountId, adminId, now, opts.note, now, code),
    db().prepare('INSERT OR IGNORE INTO course_access (account_id, course_slug, source_code, opened_by) VALUES (?, ?, ?, ?)').bind(accountId, e.course_slug, code, adminId),
  ];
  if (!m) stmts.push(db().prepare('INSERT INTO memberships (account_id, started_at, expires_at, updated_by) VALUES (?, ?, ?, ?)').bind(accountId, now, addDays(365), adminId));
  else if (!active) stmts.push(db().prepare('UPDATE memberships SET started_at = ?, expires_at = ?, updated_by = ?, updated_at = ? WHERE account_id = ?').bind(now, addDays(365), adminId, now, accountId));
  await db().batch(stmts);
  await audit(adminId, 'enrollment.activate', code, { account: accountId, membership: !m ? 'created' : active ? 'unchanged' : 'renewed' });
  await notify(accountId, 'enrollment.activated', '協會已確認款項並開通課程，可以加入實作營了。', `/camp/courses/${e.course_slug}/forum/`);
  return { ok: true as const, accountId };
}

export async function cancelEnrollment(code: string, adminId: string, note: string | null) {
  await db().prepare("UPDATE enrollments SET status = 'cancelled', reviewed_by = ?, reviewed_at = ?, review_note = ?, updated_at = ? WHERE code = ? AND status = 'submitted'")
    .bind(adminId, nowIso(), note, nowIso(), code).run();
  await audit(adminId, 'enrollment.cancel', code, { note });
}

export async function extendMembership(accountId: string, adminId: string) {
  const m = await membershipOf(accountId);
  const from = m && isMembershipActive(m) ? Date.parse(m.expires_at) : Date.now();
  const expires = addDays(365, from);
  if (m) await db().prepare('UPDATE memberships SET expires_at = ?, updated_by = ?, updated_at = ? WHERE account_id = ?').bind(expires, adminId, nowIso(), accountId).run();
  else await db().prepare('INSERT INTO memberships (account_id, started_at, expires_at, updated_by) VALUES (?, ?, ?, ?)').bind(accountId, nowIso(), expires, adminId).run();
  await audit(adminId, 'membership.renew', accountId, { expires });
}

export async function resolveAppeal(appealId: string, adminId: string, decision: 'upheld' | 'restored', note: string | null) {
  const a = await db().prepare("SELECT id, post_id, author_id FROM appeals WHERE id = ? AND status = 'open'").bind(appealId).first<{ id: string; post_id: string; author_id: string }>();
  if (!a) return false;
  const now = nowIso();
  const stmts = [db().prepare('UPDATE appeals SET status = ?, resolved_by = ?, resolved_at = ?, resolution_note = ? WHERE id = ?').bind(decision, adminId, now, note, a.id)];
  if (decision === 'restored') {
    stmts.push(db().prepare("UPDATE posts SET status = 'published', updated_at = ? WHERE id = ? AND status = 'hidden'").bind(now, a.post_id));
    stmts.push(db().prepare("INSERT INTO moderation_actions (id, post_id, actor_id, actor_role, action, reason, note) VALUES (?, ?, ?, 'admin', 'restore', '申訴成立', ?)").bind(newId(), a.post_id, adminId, note));
  }
  await db().batch(stmts);
  const post = await db().prepare('SELECT course_slug, parent_id FROM posts WHERE id = ?').bind(a.post_id).first<{ course_slug: string; parent_id: string | null }>();
  const link = post ? `/camp/courses/${post.course_slug}/forum/${post.parent_id ?? a.post_id}/` : undefined;
  await notify(a.author_id, 'appeal.resolved', decision === 'restored' ? '你的申訴成立，內容已恢復顯示。' : `你的申訴經協會審查後維持隱藏。${note ? `說明：${note}` : ''}`, link);
  await audit(adminId, `appeal.${decision}`, a.id, { note });
  return true;
}

export async function reviewApplication(id: string, adminId: string, status: 'approved' | 'changes_requested' | 'rejected', note: string | null) {
  const app = await db().prepare('SELECT account_id FROM instructor_applications WHERE id = ?').bind(id).first<{ account_id: string }>();
  if (!app) return false;
  await db().prepare('UPDATE instructor_applications SET status = ?, reviewer_id = ?, reviewed_at = ?, review_note = ?, updated_at = ? WHERE id = ?').bind(status, adminId, nowIso(), note, nowIso(), id).run();
  const msg = { approved: '你的講師申請已核准。', changes_requested: '你的講師申請需要補件。', rejected: '你的講師申請未通過。' }[status];
  await notify(app.account_id, 'instructor_application', `${msg}${note ? `協會意見：${note}` : ''}`, '/account/teaching/');
  await audit(adminId, `instructor_application.${status}`, id, { note });
  return true;
}

// 講師須有已核准的講師申請，才能加入課程講師。
// 例外：第一門課開課前沒有任何人上過課，管理員可填寫理由直接指定講師，理由寫入稽核紀錄。
export async function addInstructor(slug: string, accountId: string, adminId: string, overrideReason: string | null) {
  const approved = await db().prepare("SELECT 1 FROM instructor_applications WHERE account_id = ? AND status = 'approved'").bind(accountId).first();
  if (!approved && !overrideReason) return { ok: false as const, error: '這個帳號沒有已核准的講師申請；若為試點講師，請填寫例外理由。' };
  const account = await db().prepare('SELECT 1 FROM accounts WHERE id = ?').bind(accountId).first();
  if (!account) return { ok: false as const, error: '找不到這個帳號。' };
  await db().prepare('INSERT OR IGNORE INTO course_staff (course_slug, account_id, added_by) VALUES (?, ?, ?)').bind(slug, accountId, adminId).run();
  await audit(adminId, 'course.add_instructor', slug, { account: accountId, override: overrideReason });
  return { ok: true as const };
}
