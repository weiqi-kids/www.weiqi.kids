// 一團的狀態機（ADR 0014）：登記中 → 排時間 → 等匯款 → 已成團 → 上課中 → 已結營；另有沒有成團、講師取消。
// advance() 可重複呼叫：排程每小時對所有進行中的團呼叫一次，使用者操作後也會立刻呼叫。
import { db, audit } from '../db';
import { notify, notifyMany, notifyAdmins } from '../notify';
import { newId, nowIso } from '../crypto';
import { campRules } from '../../data/site.js';
import { fmtDateTime, nextSlotAfter, addDaysIso, addMonthIso } from '../../lib/time';
import { OPEN_STATES, type Cohort, type Showcase, type Slot } from './types';

const DAY = 86_400_000;
const PROPOSE_DAYS = 7;          // 投票方式：講師提出候選日期的期限
const PROPOSE_REMIND_DAYS = 3;

export async function getCohort(id: string) {
  return db().prepare('SELECT * FROM cohorts WHERE id = ?').bind(id).first<Cohort>();
}
export async function cohortsOf(showcaseId: string) {
  return (await db().prepare('SELECT * FROM cohorts WHERE showcase_id = ? ORDER BY seq DESC').bind(showcaseId).all<Cohort>()).results;
}
export async function interestCount(cohortId: string) {
  return (await db().prepare('SELECT COUNT(*) AS n FROM interests WHERE cohort_id = ?').bind(cohortId).first<{ n: number }>())?.n ?? 0;
}
export async function interestedIds(cohortId: string) {
  return (await db().prepare('SELECT account_id FROM interests WHERE cohort_id = ?').bind(cohortId).all<{ account_id: string }>()).results.map((r) => r.account_id);
}
export async function paidCount(cohortId: string, includeSubmitted = false) {
  const states = includeSubmitted ? "('confirmed', 'submitted')" : "('confirmed')";
  return (await db().prepare(`SELECT COUNT(*) AS n FROM enrollments WHERE cohort_id = ? AND status IN ${states}`).bind(cohortId).first<{ n: number }>())?.n ?? 0;
}
export async function activeEnrollmentCount(cohortId: string) {
  return (await db().prepare("SELECT COUNT(*) AS n FROM enrollments WHERE cohort_id = ? AND status IN ('awaiting_payment', 'submitted', 'confirmed')").bind(cohortId).first<{ n: number }>())?.n ?? 0;
}
async function enrolledIds(cohortId: string, states = "('confirmed')") {
  return (await db().prepare(`SELECT account_id FROM enrollments WHERE cohort_id = ? AND status IN ${states}`).bind(cohortId).all<{ account_id: string }>()).results.map((r) => r.account_id);
}
const showcaseOf = (c: Cohort) => db().prepare('SELECT * FROM showcases WHERE id = ?').bind(c.showcase_id).first<Showcase>();

// 每個公開的成果都要有登記中的團：講師填時段的話每個時段一個，否則一個。
// 某時段已有進行中的團（排時間以後）就不另開，等它結束後才重新開放登記。
export async function ensureGathering(showcaseId: string) {
  const s = await db().prepare('SELECT * FROM showcases WHERE id = ?').bind(showcaseId).first<Showcase>();
  if (!s || s.status !== 'published') return;
  const cohorts = await cohortsOf(showcaseId);
  const open = cohorts.filter((c) => (OPEN_STATES as string[]).includes(c.state));
  let seq = cohorts.reduce((m, c) => Math.max(m, c.seq), 0);
  const create = async (slotId: string | null) => {
    const id = newId();
    seq += 1;
    await db().prepare('INSERT INTO cohorts (id, showcase_id, slot_id, seq) VALUES (?, ?, ?, ?)').bind(id, showcaseId, slotId, seq).run();
    // 上一團想報名時已額滿的人，自動進這一團的登記名單。
    const prev = cohorts.find((c) => c.slot_id === slotId && !(OPEN_STATES as string[]).includes(c.state));
    if (prev) await db().prepare('INSERT OR IGNORE INTO interests (cohort_id, account_id, carried) SELECT ?, account_id, 1 FROM interests WHERE cohort_id = ? AND overflow = 1').bind(id, prev.id).run();
    return id;
  };
  if (s.schedule_mode === 'slots') {
    const slots = (await db().prepare('SELECT * FROM showcase_slots WHERE showcase_id = ? AND active = 1').bind(showcaseId).all<Slot>()).results;
    for (const slot of slots) if (!open.some((c) => c.slot_id === slot.id)) { const id = await create(slot.id); await advance(id); }
  } else if (!open.length) {
    const id = await create(null);
    await advance(id);
  }
}

export async function addInterest(cohortId: string, accountId: string) {
  const c = await getCohort(cohortId);
  if (!c || c.state !== 'gathering') return false;
  await db().prepare('INSERT OR IGNORE INTO interests (cohort_id, account_id) VALUES (?, ?)').bind(cohortId, accountId).run();
  await audit(accountId, 'interest.add', cohortId);
  await advance(cohortId);
  return true;
}

export async function removeInterest(cohortId: string, accountId: string) {
  await db().prepare("DELETE FROM interests WHERE cohort_id = ? AND account_id = ? AND cohort_id IN (SELECT id FROM cohorts WHERE state = 'gathering')").bind(cohortId, accountId).run();
  await audit(accountId, 'interest.remove', cohortId);
}

async function update(c: Cohort, fields: Partial<Cohort>) {
  const keys = Object.keys(fields);
  await db().prepare(`UPDATE cohorts SET ${keys.map((k) => `${k} = ?`).join(', ')} WHERE id = ?`).bind(...keys.map((k) => (fields as Record<string, unknown>)[k] ?? null), c.id).run();
  Object.assign(c, fields);
}

const link = (c: Cohort) => `/camp/g/${c.id}/`;

// 滿人數：把成果當下的設定複製到這一團，之後講師改設定不影響這一團。
async function snapshot(c: Cohort, s: Showcase) {
  await update(c, {
    price: s.price, teaching_format: s.teaching_format, location: s.location, schedule_mode: s.schedule_mode, group_rule: s.group_rule,
    min_size: s.min_size, max_size: s.max_size, pay_days: s.pay_days, vote_days: s.vote_days, teaching_minutes: s.teaching_minutes, threshold_at: nowIso(),
  });
}

// 主題教學時間確定 → 開放報名匯款。匯款期限不晚於上課前一天。
async function openPayment(c: Cohort, s: Showcase, teachingStart: string) {
  const payDeadline = new Date(Math.min(Date.now() + (c.pay_days ?? campRules.defaultPayDays) * DAY, Date.parse(teachingStart) - DAY)).toISOString();
  await update(c, {
    state: 'payment', teaching_start: teachingStart,
    teaching_end: new Date(Date.parse(teachingStart) + (c.teaching_minutes ?? 120) * 60_000).toISOString(),
    ends_at: addMonthIso(teachingStart), pay_deadline: payDeadline,
  });
  const ids = await interestedIds(c.id);
  await notifyMany(ids, 'cohort.payment', `「${s.title}」開團了！主題教學 ${fmtDateTime(teachingStart)}。請在 ${fmtDateTime(payDeadline)} 前報名並匯款 ${(c.price ?? s.price).toLocaleString('zh-TW')} 元。`, link(c));
  await notify(s.owner_id, 'cohort.payment', `你的成果「${s.title}」滿 ${ids.length} 人，開團了。主題教學 ${fmtDateTime(teachingStart)}，學員匯款期限 ${fmtDateTime(payDeadline)}。`, `/account/teach/cohorts/${c.id}/`);
  await audit(null, 'cohort.payment', c.id, { teachingStart, payDeadline });
}

async function closeCohort(c: Cohort, state: 'ended' | 'unfilled' | 'cancelled', reason: string) {
  await update(c, { state, closed_at: nowIso(), close_reason: reason });
  if (state !== 'ended') {
    // 已匯款的人列入待退款；還沒匯款的報名取消。
    await db().batch([
      db().prepare("UPDATE enrollments SET status = 'refund_pending', refund_reason = ? WHERE cohort_id = ? AND status IN ('submitted', 'confirmed')").bind(reason, c.id),
      db().prepare("UPDATE enrollments SET status = 'cancelled' WHERE cohort_id = ? AND status = 'awaiting_payment'").bind(c.id),
    ]);
  }
  await audit(null, `cohort.${state}`, c.id, { reason });
  await ensureGathering(c.showcase_id);
}

async function remindOnce(key: string, fn: () => Promise<void>) {
  const r = await db().prepare('INSERT OR IGNORE INTO reminders_sent (key) VALUES (?)').bind(key).run();
  if (r.meta.changes) await fn();
}

export async function advance(cohortId: string): Promise<void> {
  const c = await getCohort(cohortId);
  if (!c) return;
  const s = await showcaseOf(c);
  if (!s) return;
  const now = Date.now();

  if (c.state === 'gathering') {
    if (s.status !== 'published') return;
    const n = await interestCount(c.id);
    if (n < s.min_size) return;
    await snapshot(c, s);
    if (s.schedule_mode === 'slots' && c.slot_id) {
      const slot = await db().prepare('SELECT * FROM showcase_slots WHERE id = ?').bind(c.slot_id).first<Slot>();
      if (!slot) return;
      // 匯款期間結束後最近的一次該時段。
      const start = nextSlotAfter(slot.weekday, slot.start_time, now + (c.pay_days ?? s.pay_days) * DAY + DAY);
      return openPayment(c, s, start);
    }
    if (s.schedule_mode === 'fixed' && s.fixed_start && Date.parse(s.fixed_start) > now + 2 * DAY) return openPayment(c, s, s.fixed_start);
    await update(c, { state: 'scheduling', propose_by: addDaysIso(nowIso(), PROPOSE_DAYS) });
    await notify(s.owner_id, 'cohort.scheduling',
      s.schedule_mode === 'vote' ? `「${s.title}」登記滿 ${n} 人了，請在 7 天內提出主題教學的候選日期，讓登記的人投票。` : `「${s.title}」登記滿 ${n} 人了，原本預訂的日期已經太近或已過，請在 7 天內訂新的主題教學日期。`,
      `/account/teach/cohorts/${c.id}/`);
    return;
  }

  if (c.state === 'scheduling') {
    const options = (await db().prepare('SELECT COUNT(*) AS n FROM vote_options WHERE cohort_id = ?').bind(c.id).first<{ n: number }>())?.n ?? 0;
    if (!options && c.propose_by && now >= Date.parse(c.propose_by)) {
      await notifyMany(await interestedIds(c.id), 'cohort.unfilled', `「${s.title}」講師沒有在期限內排出日期，這團不開。想學可以再登記。`, `/camp/r/${s.slug}/`);
      return closeCohort(c, 'unfilled', '講師沒有在期限內提出日期');
    }
    if (!options && c.threshold_at && now >= Date.parse(c.threshold_at) + PROPOSE_REMIND_DAYS * DAY)
      await remindOnce(`propose:${c.id}`, () => notify(s.owner_id, 'cohort.propose_reminder', `「${s.title}」還沒排主題教學日期，${fmtDateTime(c.propose_by!)} 前沒排出來，這團就不開。`, `/account/teach/cohorts/${c.id}/`));
    if (c.vote_closes_at && now >= Date.parse(c.vote_closes_at))
      await remindOnce(`pick:${c.id}`, () => notify(s.owner_id, 'cohort.vote_closed', `「${s.title}」投票結束了，請看結果決定主題教學日期。`, `/account/teach/cohorts/${c.id}/`));
    return;
  }

  if (c.state === 'payment') {
    const deadline = Date.parse(c.pay_deadline!);
    if (now >= deadline - DAY && now < deadline) {
      const pending = (await db().prepare("SELECT COUNT(*) AS n FROM enrollments WHERE cohort_id = ? AND status = 'submitted'").bind(c.id).first<{ n: number }>())?.n ?? 0;
      if (pending) await remindOnce(`admin-pay:${c.id}`, () => notifyAdmins('payment.deadline', `「${s.title}」第 ${c.seq} 團明天截止匯款，還有 ${pending} 筆待核款。`, '/admin/payments/'));
      await remindOnce(`pay-last:${c.id}`, async () => notifyMany(await enrolledIds(c.id, "('awaiting_payment')"), 'payment.reminder', `「${s.title}」匯款明天截止，匯款後記得在網站填匯款資料。`, link(c)));
    }
    if (now < deadline) return;
    // 已填匯款資料、管理員還沒核對的也先算進來；之後核對不符由管理員處理。
    const paid = await paidCount(c.id, true);
    if (c.group_rule === 'paid' && paid < (c.min_size ?? s.min_size)) {
      await notifyMany([...(await interestedIds(c.id)), ...(await enrolledIds(c.id, "('awaiting_payment', 'submitted', 'confirmed')"))], 'cohort.unfilled',
        `「${s.title}」匯款期限內付款只有 ${paid} 人，沒有成團。已匯款的人會全額退款。想學可以再登記。`, `/camp/r/${s.slug}/`);
      await notify(s.owner_id, 'cohort.unfilled', `「${s.title}」第 ${c.seq} 團付款只有 ${paid} 人，沒有成團。`, `/account/teach/cohorts/${c.id}/`);
      return closeCohort(c, 'unfilled', '付款人數不足');
    }
    await db().prepare("UPDATE enrollments SET status = 'cancelled' WHERE cohort_id = ? AND status = 'awaiting_payment'").bind(c.id).run();
    await update(c, { state: 'confirmed', confirmed_at: nowIso() });
    await notifyMany(await enrolledIds(c.id, "('submitted', 'confirmed')"), 'cohort.confirmed', `「${s.title}」成團了！主題教學 ${fmtDateTime(c.teaching_start!)}。`, `/camp/g/${c.id}/room/`);
    await notify(s.owner_id, 'cohort.confirmed', `「${s.title}」第 ${c.seq} 團成團，${paid} 人。主題教學 ${fmtDateTime(c.teaching_start!)}。`, `/account/teach/cohorts/${c.id}/`);
    return;
  }

  if (c.state === 'confirmed') {
    const start = Date.parse(c.teaching_start!);
    if (now >= start - DAY && now < start) {
      const where = c.teaching_format === 'onsite' ? `地點：${c.location}` : c.meeting_url ? '直播連結在上課區' : '直播連結講師會貼在上課區';
      await remindOnce(`teach-eve:${c.id}`, async () => notifyMany([...(await enrolledIds(c.id)), s.owner_id], 'teaching.reminder', `明天 ${fmtDateTime(c.teaching_start!)} 是「${s.title}」的主題教學。${where}。`, `/camp/g/${c.id}/room/`));
      if (c.teaching_format === 'online' && !c.meeting_url) await remindOnce(`url:${c.id}`, () => notify(s.owner_id, 'teaching.url', `「${s.title}」明天上課，還沒貼直播連結。`, `/account/teach/cohorts/${c.id}/`));
    }
    if (now >= start) await update(c, { state: 'running' });
    else return;
  }

  if (c.state === 'running') {
    const { practiceReminders } = await import('./practice');
    await practiceReminders(c, s);
    if (now < Date.parse(c.ends_at!)) return;
    const gross = (await db().prepare("SELECT COALESCE(SUM(amount), 0) AS n FROM enrollments WHERE cohort_id = ? AND status = 'confirmed'").bind(c.id).first<{ n: number }>())?.n ?? 0;
    await db().prepare('INSERT OR IGNORE INTO payouts (cohort_id, instructor_id, gross, amount) VALUES (?, ?, ?, ?)').bind(c.id, s.owner_id, gross, Math.round(gross * campRules.instructorShare)).run();
    await notifyMany(await enrolledIds(c.id), 'cohort.ended', `「${s.title}」結營了，請花 2 分鐘填問卷。課程論壇會保留，可以繼續交流。`, `/camp/g/${c.id}/room/survey/`);
    if (gross) await notifyAdmins('payout.due', `「${s.title}」第 ${c.seq} 團結營，講師分潤 ${Math.round(gross * campRules.instructorShare).toLocaleString('zh-TW')} 元待匯。`, '/admin/payouts/');
    return closeCohort(c, 'ended', '結營');
  }
}

// ---- 講師操作 ----

export async function proposeOptions(c: Cohort, s: Showcase, actorId: string, startsAt: string[]) {
  if (c.state !== 'scheduling' || s.schedule_mode !== 'vote') return '這一團現在不能提出候選日期。';
  const future = startsAt.filter((t) => Date.parse(t) > Date.now() + 2 * DAY);
  if (future.length < 1) return '請至少提出一個兩天以後的日期。';
  const voteDays = c.vote_days ?? s.vote_days;
  await db().batch([
    db().prepare('DELETE FROM votes WHERE option_id IN (SELECT id FROM vote_options WHERE cohort_id = ?)').bind(c.id),
    db().prepare('DELETE FROM vote_options WHERE cohort_id = ?').bind(c.id),
    ...future.map((t) => db().prepare('INSERT INTO vote_options (id, cohort_id, starts_at) VALUES (?, ?, ?)').bind(newId(), c.id, t)),
    db().prepare('UPDATE cohorts SET vote_closes_at = ? WHERE id = ?').bind(addDaysIso(nowIso(), voteDays), c.id),
  ]);
  await notifyMany(await interestedIds(c.id), 'cohort.vote', `「${s.title}」要排主題教學的日期了，請在 ${voteDays} 天內投票（結果供講師參考）。`, link(c));
  await audit(actorId, 'cohort.propose', c.id, { options: future });
  return null;
}

export async function vote(c: Cohort, accountId: string, optionIds: string[]) {
  if (c.state !== 'scheduling' || !c.vote_closes_at || Date.now() >= Date.parse(c.vote_closes_at)) return false;
  const valid = new Set((await db().prepare('SELECT id FROM vote_options WHERE cohort_id = ?').bind(c.id).all<{ id: string }>()).results.map((r) => r.id));
  await db().batch([
    db().prepare('DELETE FROM votes WHERE account_id = ? AND option_id IN (SELECT id FROM vote_options WHERE cohort_id = ?)').bind(accountId, c.id),
    ...optionIds.filter((o) => valid.has(o)).map((o) => db().prepare('INSERT INTO votes (option_id, account_id) VALUES (?, ?)').bind(o, accountId)),
  ]);
  return true;
}

export async function voteResults(cohortId: string) {
  return (await db().prepare('SELECT o.id, o.starts_at, COUNT(v.account_id) AS votes FROM vote_options o LEFT JOIN votes v ON v.option_id = o.id WHERE o.cohort_id = ? GROUP BY o.id ORDER BY o.starts_at').bind(cohortId)
    .all<{ id: string; starts_at: string; votes: number }>()).results;
}

// 講師決定主題教學日期（投票只供參考；預訂日期太近時也用這個）。
export async function setTeachingDate(c: Cohort, s: Showcase, actorId: string, startsAt: string) {
  if (c.state !== 'scheduling') return '這一團現在不能改日期。';
  if (Date.parse(startsAt) <= Date.now() + 2 * DAY) return '日期要在兩天以後，學員才有時間匯款。';
  await audit(actorId, 'cohort.set_date', c.id, { startsAt });
  await openPayment(c, s, startsAt);
  return null;
}

export async function setMeetingUrl(c: Cohort, actorId: string, url: string) {
  if (url && !/^https:\/\//.test(url)) return '直播連結要以 https:// 開頭。';
  await update(c, { meeting_url: url || null });
  await audit(actorId, 'cohort.meeting_url', c.id);
  return null;
}

export async function cancelCohort(c: Cohort, s: Showcase, actorId: string, reason: string) {
  if (!['scheduling', 'payment', 'confirmed', 'running'].includes(c.state)) return '這一團現在不能取消。';
  const affected = [...(await interestedIds(c.id)), ...(await enrolledIds(c.id, "('awaiting_payment', 'submitted', 'confirmed')"))];
  await notifyMany(affected, 'cohort.cancelled', `講師取消了「${s.title}」第 ${c.seq} 團${reason ? `：${reason}` : ''}。已匯款的人會全額退款。`, `/camp/r/${s.slug}/`);
  await notifyAdmins('refund.due', `講師取消「${s.title}」第 ${c.seq} 團，請處理退款。`, '/admin/payments/');
  await audit(actorId, 'cohort.cancel', c.id, { reason });
  await closeCohort(c, 'cancelled', reason || '講師取消');
  return null;
}

// 排程入口：每小時推進所有進行中的團。
export async function runScheduled() {
  const { results } = await db().prepare(`SELECT id FROM cohorts WHERE state IN ('gathering', 'scheduling', 'payment', 'confirmed', 'running')`).all<{ id: string }>();
  for (const r of results) {
    try { await advance(r.id); } catch (err) { console.error('advance failed', r.id, err); }
  }
}
