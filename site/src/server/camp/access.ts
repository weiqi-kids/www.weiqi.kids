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
