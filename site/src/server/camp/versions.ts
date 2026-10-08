// 講師 repo 的版本：講師登記 → 本人在網站送審 → 管理員審過才給學員（ADR 0015）。
import { env } from 'cloudflare:workers';
import { db, audit } from '../db';
import { notify, notifyAdmins } from '../notify';
import { newId, nowIso } from '../crypto';
import type { Showcase } from './types';

export interface SkillVersion {
  id: string; showcase_id: string; repo_url: string; commit_sha: string; note: string | null;
  status: 'draft' | 'submitted' | 'approved' | 'rejected'; review_summary: string | null; review_note: string | null; reviewed_at: string | null; created_at: string;
}
export interface ReviewSummary {
  compared_to: string | null;
  changed_files: { path: string; status: string }[];
  schedules: { file: string; crons: string[] }[];
  hosts: string[];
  secret_warnings: string[];
  error?: string;
}

export const parseRepo = (url: string) => {
  const m = /^https:\/\/github\.com\/([\w.-]+)\/([\w.-]+?)(?:\.git)?\/?$/.exec(url.trim());
  return m ? { owner: m[1], repo: m[2] } : null;
};

async function gh(path: string) {
  const headers: Record<string, string> = { 'User-Agent': 'weiqi-kids-site', Accept: 'application/vnd.github+json' };
  if (env.GITHUB_TOKEN) headers.Authorization = `Bearer ${env.GITHUB_TOKEN}`;
  const res = await fetch(`https://api.github.com${path}`, { headers });
  if (!res.ok) throw new Error(`GitHub ${res.status}`);
  return res.json() as Promise<any>;
}

export async function versionsOf(showcaseId: string) {
  return (await db().prepare('SELECT * FROM skill_versions WHERE showcase_id = ? ORDER BY created_at DESC').bind(showcaseId).all<SkillVersion>()).results;
}
export async function approvedVersion(showcaseId: string) {
  return db().prepare("SELECT * FROM skill_versions WHERE showcase_id = ? AND status = 'approved' ORDER BY reviewed_at DESC LIMIT 1").bind(showcaseId).first<SkillVersion>();
}
export async function getVersion(id: string) {
  return db().prepare('SELECT * FROM skill_versions WHERE id = ?').bind(id).first<SkillVersion>();
}
export const versionNumber = async (v: SkillVersion) =>
  ((await db().prepare("SELECT COUNT(*) AS n FROM skill_versions WHERE showcase_id = ? AND status = 'approved' AND reviewed_at <= ?").bind(v.showcase_id, v.reviewed_at ?? nowIso()).first<{ n: number }>())?.n ?? 0);

// 講師（或講師的 AI）登記一個版本：網址＋commit（可給分支名，會換成完整 commit）。
export async function registerVersion(s: Showcase, actorId: string, repoUrl: string, ref: string, note: string | null) {
  const repo = parseRepo(repoUrl);
  if (!repo) return { ok: false as const, error: 'GitHub 網址要像 https://github.com/帳號/repo。' };
  let sha: string;
  try {
    const info = await gh(`/repos/${repo.owner}/${repo.repo}`);
    if (info.private) return { ok: false as const, error: '這個 repo 是私人的，要先改成公開。' };
    sha = (await gh(`/repos/${repo.owner}/${repo.repo}/commits/${encodeURIComponent(ref || info.default_branch)}`)).sha;
  } catch {
    return { ok: false as const, error: '讀不到這個 repo 或版本，確認網址正確、repo 是公開的，而且已經 push。' };
  }
  const id = newId();
  await db().prepare('INSERT INTO skill_versions (id, showcase_id, repo_url, commit_sha, note, created_by) VALUES (?, ?, ?, ?, ?, ?)')
    .bind(id, s.id, `https://github.com/${repo.owner}/${repo.repo}`, sha, note, actorId).run();
  await audit(actorId, 'version.register', s.id, { sha });
  return { ok: true as const, id, sha };
}

// 自動整理給管理員看：跟上一個審過的版本差在哪、定時更新設定、程式會連到哪些網站、疑似機密。
async function summarize(v: SkillVersion): Promise<ReviewSummary> {
  const repo = parseRepo(v.repo_url)!;
  const prev = await approvedVersion(v.showcase_id);
  const out: ReviewSummary = { compared_to: prev && prev.repo_url === v.repo_url ? prev.commit_sha : null, changed_files: [], schedules: [], hosts: [], secret_warnings: [] };
  try {
    let paths: { path: string; status: string }[];
    if (out.compared_to) {
      const cmp = await gh(`/repos/${repo.owner}/${repo.repo}/compare/${out.compared_to}...${v.commit_sha}`);
      paths = (cmp.files ?? []).map((f: any) => ({ path: f.filename, status: f.status }));
    } else {
      const tree = await gh(`/repos/${repo.owner}/${repo.repo}/git/trees/${v.commit_sha}?recursive=1`);
      paths = (tree.tree ?? []).filter((t: any) => t.type === 'blob').map((t: any) => ({ path: t.path, status: 'added' }));
    }
    out.changed_files = paths.slice(0, 500);
    const hosts = new Set<string>();
    const textLike = /\.(md|txt|ya?ml|json|js|mjs|cjs|ts|py|sh|toml|html|css|astro|env|example|ini|cfg)$/i;
    for (const f of paths.filter((p) => p.status !== 'removed' && (textLike.test(p.path) || !p.path.includes('.'))).slice(0, 60)) {
      const res = await fetch(`https://raw.githubusercontent.com/${repo.owner}/${repo.repo}/${v.commit_sha}/${f.path}`);
      if (!res.ok) continue;
      const text = (await res.text()).slice(0, 200_000);
      for (const m of text.matchAll(/https?:\/\/([a-z0-9.-]+\.[a-z]{2,})/gi)) hosts.add(m[1].toLowerCase());
      if (/^\.github\/workflows\//.test(f.path)) out.schedules.push({ file: f.path, crons: [...text.matchAll(/cron:\s*['"]([^'"]+)['"]/g)].map((m) => m[1]) });
      if (/(sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)/.test(text)) out.secret_warnings.push(f.path);
    }
    out.hosts = [...hosts].filter((h) => !/(^|\.)github(usercontent)?\.com$|(^|\.)example\.(com|org)$/.test(h)).sort();
  } catch (err) {
    out.error = '自動整理失敗，請直接到 GitHub 看這個版本。';
    console.error('summarize failed', err);
  }
  return out;
}

// 本人在網站上按「送審」。
export async function submitVersion(v: SkillVersion, s: Showcase, actorId: string) {
  if (v.status !== 'draft' && v.status !== 'rejected') return '這個版本已經送審或審過了。';
  const summary = await summarize(v);
  await db().prepare("UPDATE skill_versions SET status = 'submitted', review_summary = ? WHERE id = ?").bind(JSON.stringify(summary), v.id).run();
  await audit(actorId, 'version.submit', v.id);
  await notifyAdmins('version.submitted', `${s.instructor_name}送審「${s.title}」的新版本。`, `/admin/versions/${v.id}/`);
  return null;
}

export async function reviewVersion(v: SkillVersion, s: Showcase, adminId: string, approve: boolean, note: string | null) {
  if (v.status !== 'submitted') return false;
  await db().prepare('UPDATE skill_versions SET status = ?, review_note = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?').bind(approve ? 'approved' : 'rejected', note, adminId, nowIso(), v.id).run();
  if (approve) await db().prepare('UPDATE showcases SET repo_url = ?, updated_at = ? WHERE id = ?').bind(v.repo_url, nowIso(), s.id).run();
  await audit(adminId, approve ? 'version.approve' : 'version.reject', v.id, { note });
  await notify(s.owner_id, 'version.reviewed', approve ? `「${s.title}」的新版本審核通過，學員現在拿到的是這一版。` : `「${s.title}」的新版本沒有通過。${note ? `協會意見：${note}` : ''}`, `/account/teach/showcases/${s.id}/?step=5`);
  return true;
}
