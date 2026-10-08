// 頁面共用：成果目前的開團狀態摘要。
import { db } from '../db';
import { daysLeft } from '../../lib/time';
import { weekdayLabel } from '../../lib/time';
import { OPEN_STATES, type Cohort, type Showcase, type Slot } from './types';

export interface CohortView extends Cohort { interests: number; paid: number; slot: Slot | null; slotLabel: string | null }

export async function cohortViews(s: Showcase): Promise<CohortView[]> {
  const { results } = await db().prepare(
    `SELECT c.*, (SELECT COUNT(*) FROM interests i WHERE i.cohort_id = c.id) AS interests,
       (SELECT COUNT(*) FROM enrollments e WHERE e.cohort_id = c.id AND e.status IN ('submitted', 'confirmed')) AS paid
     FROM cohorts c WHERE c.showcase_id = ? ORDER BY c.seq DESC`,
  ).bind(s.id).all<Cohort & { interests: number; paid: number }>();
  const slots = (await db().prepare('SELECT * FROM showcase_slots WHERE showcase_id = ?').bind(s.id).all<Slot>()).results;
  return results.map((c) => {
    const slot = slots.find((x) => x.id === c.slot_id) ?? null;
    return { ...c, slot, slotLabel: slot ? `${weekdayLabel(slot.weekday)} ${slot.start_time}` : null };
  });
}

export const openViews = (v: CohortView[]) => v.filter((c) => (OPEN_STATES as string[]).includes(c.state));

// 成果牆卡片上的一行狀態。
export function headline(s: Showcase, views: CohortView[]) {
  const open = openViews(views);
  const pay = open.find((c) => c.state === 'payment');
  if (pay) return { tone: 'hot', text: `開團中，匯款剩 ${daysLeft(pay.pay_deadline!)} 天`, href: `/camp/g/${pay.id}/` };
  const sched = open.find((c) => c.state === 'scheduling');
  if (sched) return { tone: 'hot', text: '已滿人數，正在排時間', href: `/camp/g/${sched.id}/` };
  const running = open.find((c) => c.state === 'confirmed' || c.state === 'running');
  const gathering = open.filter((c) => c.state === 'gathering').sort((a, b) => b.interests - a.interests)[0];
  if (gathering) return { tone: 'normal', text: `已登記 ${gathering.interests}／${s.min_size} 人開團`, ratio: Math.min(1, gathering.interests / s.min_size) };
  if (running) return { tone: 'normal', text: '這一團上課中，結營後重新開放登記' };
  return { tone: 'normal', text: '還沒開放登記' };
}
