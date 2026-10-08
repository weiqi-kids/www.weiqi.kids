// 技能單張：問題、發生的原因、解決方法有哪些、我建議的流程。跟著成果，每團沿用；修改留版本。
import { db, audit } from '../db';
import { newId, nowIso } from '../crypto';

export interface Material { id: string; showcase_id: string; title: string; problem: string; cause: string; solutions: string; flow: string; position: number; version: number; updated_at: string }

export async function materialsOf(showcaseId: string) {
  return (await db().prepare('SELECT * FROM materials WHERE showcase_id = ? AND archived = 0 ORDER BY position, updated_at').bind(showcaseId).all<Material>()).results;
}

type Body = { title: string; problem: string; cause: string; solutions: string; flow: string };

export async function saveMaterial(showcaseId: string, actorId: string, id: string | null, b: Body) {
  if (Object.values(b).some((v) => !v.trim())) return '五個欄位都要填。';
  if (id) {
    const cur = await db().prepare('SELECT version FROM materials WHERE id = ? AND showcase_id = ?').bind(id, showcaseId).first<{ version: number }>();
    if (!cur) return '找不到這張技能單張。';
    const v = cur.version + 1;
    await db().batch([
      db().prepare('UPDATE materials SET title = ?, problem = ?, cause = ?, solutions = ?, flow = ?, version = ?, updated_by = ?, updated_at = ? WHERE id = ?').bind(b.title, b.problem, b.cause, b.solutions, b.flow, v, actorId, nowIso(), id),
      db().prepare('INSERT INTO material_versions (material_id, version, body, edited_by) VALUES (?, ?, ?, ?)').bind(id, v, JSON.stringify(b), actorId),
    ]);
  } else {
    const nid = newId();
    const pos = ((await db().prepare('SELECT COALESCE(MAX(position), 0) AS p FROM materials WHERE showcase_id = ?').bind(showcaseId).first<{ p: number }>())?.p ?? 0) + 1;
    await db().batch([
      db().prepare('INSERT INTO materials (id, showcase_id, title, problem, cause, solutions, flow, position, updated_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)').bind(nid, showcaseId, b.title, b.problem, b.cause, b.solutions, b.flow, pos, actorId),
      db().prepare('INSERT INTO material_versions (material_id, version, body, edited_by) VALUES (?, 1, ?, ?)').bind(nid, JSON.stringify(b), actorId),
    ]);
  }
  await audit(actorId, 'material.save', id ?? showcaseId);
  return null;
}

export async function archiveMaterial(showcaseId: string, actorId: string, id: string) {
  await db().prepare('UPDATE materials SET archived = 1, updated_at = ? WHERE id = ? AND showcase_id = ?').bind(nowIso(), id, showcaseId).run();
  await audit(actorId, 'material.archive', id);
}
