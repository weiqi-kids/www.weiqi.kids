import { db } from '../db';
import type { Account } from '../types';
import type { SpaceRole } from '../forum';
import type { Cohort, Showcase } from './types';

export const isAdmin = (me: Account | null) => me?.is_admin === 1;
export const isOwner = (s: Pick<Showcase, 'owner_id'>, me: Account | null) => !!me && s.owner_id === me.id;

// 已核款的學員才進得了上課區與課程論壇。
export async function isCohortMember(c: Cohort, me: Account | null) {
  if (!me) return false;
  return !!(await db().prepare("SELECT 1 FROM enrollments WHERE cohort_id = ? AND account_id = ? AND status = 'confirmed'").bind(c.id, me.id).first());
}

export async function canEnterRoom(c: Cohort, s: Showcase, me: Account | null) {
  return isAdmin(me) || isOwner(s, me) || (await isCohortMember(c, me));
}

export function showcaseSpaceRole(s: Showcase, me: Account | null): SpaceRole {
  const open = s.status === 'published' && s.forum_open === 1;
  const mod = isOwner(s, me) || isAdmin(me);
  return { view: open || mod, post: !!me && (open || mod), moderate: mod, admin: isAdmin(me) };
}

export async function cohortSpaceRole(c: Cohort, s: Showcase, me: Account | null): Promise<SpaceRole> {
  const mod = isOwner(s, me) || isAdmin(me);
  const member = mod || (await isCohortMember(c, me));
  return { view: member, post: member, moderate: mod, admin: isAdmin(me) };
}

// 技能單張：講師、管理員，以及這個成果任何一團已核款的學員看得到全部。
export async function canReadMaterials(s: Showcase, me: Account | null) {
  if (!me) return false;
  if (isAdmin(me) || isOwner(s, me)) return true;
  return !!(await db().prepare("SELECT 1 FROM enrollments e JOIN cohorts c ON c.id = e.cohort_id WHERE c.showcase_id = ? AND e.account_id = ? AND e.status = 'confirmed'").bind(s.id, me.id).first());
}
