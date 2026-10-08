import { db, audit } from '../db';
import { notify, notifyAdmins } from '../notify';
import { newId, nowIso } from '../crypto';
import { membershipOf, isMembershipActive } from '../accounts';
import { campRules } from '../../data/site.js';
import { demoLinks, type Showcase, type Slot } from './types';

export async function getShowcase(id: string) {
  return db().prepare('SELECT * FROM showcases WHERE id = ?').bind(id).first<Showcase>();
}
export async function getShowcaseBySlug(slug: string) {
  return db().prepare('SELECT * FROM showcases WHERE slug = ?').bind(slug).first<Showcase>();
}
export async function listPublished() {
  return (await db().prepare("SELECT * FROM showcases WHERE status = 'published' ORDER BY published_at DESC").all<Showcase>()).results;
}
export async function listByOwner(ownerId: string) {
  return (await db().prepare('SELECT * FROM showcases WHERE owner_id = ? ORDER BY updated_at DESC').bind(ownerId).all<Showcase>()).results;
}
export async function listForReview() {
  return (await db().prepare("SELECT * FROM showcases WHERE status = 'submitted' ORDER BY updated_at").all<Showcase>()).results;
}
export async function slotsOf(showcaseId: string, activeOnly = true) {
  return (await db().prepare(`SELECT * FROM showcase_slots WHERE showcase_id = ?${activeOnly ? ' AND active = 1' : ''} ORDER BY weekday, start_time`).bind(showcaseId).all<Slot>()).results;
}

export async function canCreateShowcase(accountId: string) {
  return isMembershipActive(await membershipOf(accountId));
}

const randomSlug = () => Array.from(crypto.getRandomValues(new Uint8Array(6)), (b) => 'abcdefghjkmnpqrstuvwxyz23456789'[b % 31]).join('');

export async function createDraft(ownerId: string, instructorName: string) {
  const id = newId();
  await db().prepare('INSERT INTO showcases (id, slug, owner_id, instructor_name, price, pay_days, vote_days) VALUES (?, ?, ?, ?, ?, ?, ?)')
    .bind(id, randomSlug(), ownerId, instructorName, campRules.minFee, campRules.defaultPayDays, campRules.defaultVoteDays).run();
  await audit(ownerId, 'showcase.create', id);
  return id;
}

// 送審前必須齊備：7 項誘因資料、開團設定、TTQS 開課表單。回傳缺少的項目（中文）。
export function missingItems(s: Showcase, slots: Slot[]) {
  const miss: string[] = [];
  const need = (ok: boolean, name: string) => { if (!ok) miss.push(name); };
  need(!!s.title.trim(), '成果名稱');
  need(!!s.summary.trim(), '一句話介紹');
  need(demoLinks(s).length > 0, '1. 成果展示（至少一個連結）');
  need(!!s.before_after.trim(), '2. 用 AI 之前和之後差在哪裡');
  need(!!s.learner_outcome.trim(), '3. 學完能做出什麼');
  need(!!s.audience.trim() && !!s.prerequisites.trim(), '4. 適合誰、需要什麼基礎');
  need(!!s.instructor_bio.trim(), '5. 講師本業與為什麼做這個');
  need(!!s.method.trim(), '6. 做法說明');
  need(!!(s.sample_problem.trim() && s.sample_cause.trim() && s.sample_solutions.trim() && s.sample_flow.trim()), '7. 試閱技能單張（四個部分都要填）');
  need(s.price >= campRules.minFee, `團費（最低 ${campRules.minFee.toLocaleString('zh-TW')} 元）`);
  need(s.teaching_format === 'online' || !!s.location.trim(), '實體上課地點');
  need(s.schedule_mode !== 'slots' || slots.length > 0, '每週時段（至少一個）');
  need(s.schedule_mode !== 'fixed' || !!s.fixed_start, '預訂的主題教學日期');
  need(!s.max_size || s.max_size >= s.min_size, '人數上限不能小於開團人數');
  for (const [k, n] of [['ttqs_needs', '需求'], ['ttqs_goals', '目標'], ['ttqs_outline', '大綱'], ['ttqs_hours', '時數'], ['ttqs_methods', '方法'], ['ttqs_evaluation', '評量'], ['ttqs_expected', '預期成果']] as const)
    need(!!(s[k] as string).trim(), `開課表單：${n}`);
  return miss;
}

export const editable = (s: Showcase) => s.status === 'draft' || s.status === 'changes_requested' || s.status === 'published';

export async function saveFields(s: Showcase, actorId: string, fields: Partial<Record<keyof Showcase, string | number | null>>) {
  const keys = Object.keys(fields);
  if (!keys.length) return;
  await db().prepare(`UPDATE showcases SET ${keys.map((k) => `${k} = ?`).join(', ')}, updated_at = ? WHERE id = ?`)
    .bind(...keys.map((k) => fields[k as keyof Showcase] ?? null), nowIso(), s.id).run();
  await audit(actorId, 'showcase.edit', s.id, { fields: keys });
}

export async function setSlots(showcaseId: string, actorId: string, wanted: { weekday: number; start_time: string }[]) {
  const current = await slotsOf(showcaseId, false);
  const key = (x: { weekday: number; start_time: string }) => `${x.weekday}-${x.start_time}`;
  const want = new Set(wanted.map(key));
  const stmts = [];
  for (const c of current) stmts.push(db().prepare('UPDATE showcase_slots SET active = ? WHERE id = ?').bind(want.has(key(c)) ? 1 : 0, c.id));
  const have = new Set(current.map(key));
  for (const w of wanted) if (!have.has(key(w))) stmts.push(db().prepare('INSERT INTO showcase_slots (id, showcase_id, weekday, start_time) VALUES (?, ?, ?, ?)').bind(newId(), showcaseId, w.weekday, w.start_time));
  if (stmts.length) await db().batch(stmts);
  await audit(actorId, 'showcase.slots', showcaseId, { slots: wanted });
}

export async function submitForReview(s: Showcase, actorId: string) {
  const miss = missingItems(s, await slotsOf(s.id));
  if (miss.length) return { ok: false as const, missing: miss };
  if (!(await canCreateShowcase(s.owner_id))) return { ok: false as const, missing: ['有效的協會會員資格'] };
  await db().prepare("UPDATE showcases SET status = 'submitted', updated_at = ? WHERE id = ?").bind(nowIso(), s.id).run();
  await audit(actorId, 'showcase.submit', s.id);
  await notifyAdmins('showcase.submitted', `講師送審成果「${s.title}」。`, `/admin/showcases/${s.id}/`);
  return { ok: true as const };
}

export async function reviewShowcase(s: Showcase, adminId: string, decision: 'publish' | 'changes', note: string | null) {
  const now = nowIso();
  if (decision === 'publish') {
    await db().prepare("UPDATE showcases SET status = 'published', review_note = ?, reviewed_by = ?, reviewed_at = ?, published_at = COALESCE(published_at, ?), updated_at = ? WHERE id = ?")
      .bind(note, adminId, now, now, now, s.id).run();
    const { ensureGathering } = await import('./cohorts');
    await ensureGathering(s.id);
    await notify(s.owner_id, 'showcase.published', `你的成果「${s.title}」已公開，開始接受登記想學。`, `/camp/r/${s.slug}/`);
  } else {
    await db().prepare("UPDATE showcases SET status = 'changes_requested', review_note = ?, reviewed_by = ?, reviewed_at = ?, updated_at = ? WHERE id = ?")
      .bind(note, adminId, now, now, s.id).run();
    await notify(s.owner_id, 'showcase.changes', `你的成果「${s.title}」需要補件。${note ? `協會意見：${note}` : ''}`, `/account/teach/showcases/${s.id}/`);
  }
  await audit(adminId, `showcase.${decision}`, s.id, { note });
}

export async function archiveShowcase(s: Showcase, actorId: string) {
  await db().prepare("UPDATE showcases SET status = 'archived', updated_at = ? WHERE id = ?").bind(nowIso(), s.id).run();
  // 只關掉還在登記中的團；已進入排時間以後的團照常進行。
  await db().prepare("UPDATE cohorts SET state = 'cancelled', closed_at = ?, close_reason = '成果下架' WHERE showcase_id = ? AND state = 'gathering'").bind(nowIso(), s.id).run();
  await audit(actorId, 'showcase.archive', s.id);
}
