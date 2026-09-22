import { db } from './db';
import type { Account } from './types';

export interface CourseRole {
  admin: boolean;
  instructor: boolean;
  access: boolean;     // 已核款開通
  campMember: boolean; // 已自行加入實作營
}

export async function courseRole(account: Account | null, slug: string): Promise<CourseRole> {
  if (!account) return { admin: false, instructor: false, access: false, campMember: false };
  const [staff, access, camp] = await Promise.all([
    db().prepare('SELECT 1 FROM course_staff WHERE course_slug = ? AND account_id = ?').bind(slug, account.id).first(),
    db().prepare('SELECT 1 FROM course_access WHERE course_slug = ? AND account_id = ?').bind(slug, account.id).first(),
    db().prepare('SELECT 1 FROM camp_members WHERE course_slug = ? AND account_id = ?').bind(slug, account.id).first(),
  ]);
  return { admin: account.is_admin === 1, instructor: !!staff, access: !!access, campMember: !!camp };
}

// 權限矩陣（revamp/4-strategy/2026-09-20-data-model-and-permissions.md §3）
export const can = {
  viewMaterials: (r: CourseRole) => r.admin || r.instructor || r.access,
  editMaterials: (r: CourseRole) => r.instructor,
  joinCamp: (r: CourseRole) => r.access && !r.campMember && !r.instructor,
  viewForum: (r: CourseRole) => r.admin || r.instructor || r.campMember,
  post: (r: CourseRole) => r.instructor || r.campMember,
  moderate: (r: CourseRole) => r.admin || r.instructor,
};

export const HIDE_REASONS = ['垃圾內容', '個資', '侵權', '錯誤資訊', '違反課程規則', '作者要求', '其他管理原因'] as const;
