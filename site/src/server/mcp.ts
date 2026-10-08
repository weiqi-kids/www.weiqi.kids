// 好棋寶寶 MCP（ADR 0015）：學員與講師的 AI 透過它對網站做事。寫入一律是草稿，本人在網站上確認。
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { WebStandardStreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js';
import { z } from 'zod';
import { db } from './db';
import { membershipOf, isMembershipActive } from './accounts';
import { listPublished, getShowcaseBySlug, getShowcase, listByOwner, slotsOf, missingItems, createDraft as createShowcaseDraft, saveFields, setSlots, editable, canCreateShowcase } from './camp/showcases';
import { cohortViews, openViews } from './camp/view';
import { getCohort, addInterest, removeInterest } from './camp/cohorts';
import { materialsOf } from './camp/materials';
import { approvedVersion, registerVersion } from './camp/versions';
import { createDraft, pendingDrafts, draftUrl } from './camp/drafts';
import { recordNeed, recentNeeds } from './camp/needs';
import { submissions, weekDeadline, WEEK_ASK } from './camp/practice';
import { isOwner, isAdmin, cohortSpaceRole, showcaseSpaceRole } from './camp/access';
import { listTopics, getPost, listReplies } from './forum';
import { GUIDES } from './camp/guides';
import { demoLinks, type Showcase } from './camp/types';
import { fmtDateTime, weekdayLabel, parseTaipeiLocal } from '../lib/time';
import { COHORT_STATE, ENROLLMENT_STATUS, FORMAT_LABEL, SCHEDULE_LABEL, RULE_LABEL, SHOWCASE_STATUS, PRACTICE_WEEKS, label } from '../lib/labels';
import { site, campRules } from '../data/site.js';
import type { Account } from './types';

// 管理員審核時自動整理的事實：執行前告訴學員會連到哪些網站、有哪些定時更新。
const reviewFacts = (raw: string | null) => {
  if (!raw) return {};
  try {
    const r = JSON.parse(raw) as { hosts?: string[]; schedules?: { file: string; crons: string[] }[] };
    return { connects_to: r.hosts ?? [], scheduled_updates: r.schedules ?? [] };
  } catch { return {}; }
};
const ok = (data: unknown) => ({ content: [{ type: 'text' as const, text: typeof data === 'string' ? data : JSON.stringify(data, null, 2) }] });
const fail = (message: string) => ({ content: [{ type: 'text' as const, text: message }], isError: true });

export function createServer(me: Account, origin: string) {
  const server = new McpServer({ name: '好棋寶寶', version: '1.0.0' }, {
    instructions: '台灣好棋寶寶協會 AI 共學營。學員用 haoqi-learner skill，講師用 haoqi-teacher skill；沒有 skill 的 AI 工具，先呼叫 get_guide 取得引導（topic: learner 或 teacher）。寫入的東西都是草稿，要把回傳的網址給使用者，在網站上按確定才送出。',
  });
  const abs = (path: string) => `${origin}${path}`;

  const showcaseSummary = async (s: Showcase) => {
    const views = openViews(await cohortViews(s));
    const v = await approvedVersion(s.id);
    return {
      slug: s.slug, title: s.title, instructor: s.instructor_name, summary: s.summary,
      before_after: s.before_after, learner_outcome: s.learner_outcome, audience: s.audience, prerequisites: s.prerequisites,
      setup_requirements: s.setup_requirements, page: abs(`/camp/r/${s.slug}/`),
      approved_version: v ? { repo: v.repo_url, commit: v.commit_sha, ...reviewFacts(v.review_summary) } : null,
      license: s.license, students_may_publish_modified_version: s.allow_derivative === 1,
      price: s.price, teaching: FORMAT_LABEL[s.teaching_format],
      registration: views.map((c) => ({ state: label(COHORT_STATE, c.state), slot: c.slotLabel, registered: c.interests, opens_at: s.min_size, cohort_page: c.state === 'gathering' ? null : abs(`/camp/g/${c.id}/`) })),
    };
  };

  server.registerTool('whoami', { description: '目前登入的好棋寶寶帳號：名稱、是否為有效會員（會員才能上架成果）、是否為管理員。', inputSchema: {} }, async () => {
    const m = await membershipOf(me.id);
    return ok({ name: me.display_name, member: isMembershipActive(m), member_until: m ? fmtDateTime(m.expires_at) : null, admin: me.is_admin === 1, account_page: abs('/account/') });
  });

  server.registerTool('get_guide', {
    description: '取得好棋寶寶的引導文件。learner：陪學員試做的步驟；teacher：協助講師的步驟；repo_structure：講師 repo 的規定結構與協會範例；writing：寫作規則。',
    inputSchema: { topic: z.enum(['learner', 'teacher', 'repo_structure', 'writing']) },
  }, async ({ topic }) => ok(GUIDES[topic]));

  server.registerTool('find_showcases', {
    description: '列出講師公開的成果，可用關鍵字篩選（成果名稱、介紹、適合誰、能做出什麼）。用來把學員的工作問題對到講師做過的成果。',
    inputSchema: { query: z.string().optional().describe('關鍵字，例如「供應鏈」「新聞」「價格」；不填列出全部') },
  }, async ({ query }) => {
    const words = (query ?? '').split(/[\s,，、]+/).filter(Boolean);
    const all = await listPublished();
    const hit = words.length ? all.filter((s) => words.some((w) => [s.title, s.summary, s.audience, s.learner_outcome, s.before_after].join(' ').includes(w))) : all;
    return ok({ total_published: all.length, matched: await Promise.all((hit.length ? hit : all).map(showcaseSummary)), note: hit.length || !words.length ? undefined : '關鍵字沒有直接對到，列出全部讓你判斷；真的都不像，問學員要不要用 record_learning_need 記下問題。' });
  });

  server.registerTool('get_showcase', {
    description: '讀一項成果的完整內容：8 項資料、做法說明、全部技能單張、管理員審過的 GitHub 版本、授權、開團設定與登記狀況、討論區最新提問。',
    inputSchema: { slug: z.string().describe('成果代碼，find_showcases 回傳的 slug') },
  }, async ({ slug }) => {
    const s = await getShowcaseBySlug(slug);
    if (!s || (s.status !== 'published' && !isOwner(s, me) && !isAdmin(me))) return fail('找不到這項成果。');
    const slots = await slotsOf(s.id);
    const topics = (await listTopics(`showcase:${s.id}`)).filter((t) => t.status === 'published').slice(0, 10);
    return ok({
      ...(await showcaseSummary(s)),
      demos: demoLinks(s), instructor_bio: s.instructor_bio, method: s.method,
      sample_sheet: { problem: s.sample_problem, cause: s.sample_cause, solutions: s.sample_solutions, flow: s.sample_flow },
      skill_sheets: (await materialsOf(s.id)).map((m) => ({ title: m.title, problem: m.problem, cause: m.cause, solutions: m.solutions, flow: m.flow })),
      schedule: SCHEDULE_LABEL[s.schedule_mode], weekly_slots: slots.map((x) => `${weekdayLabel(x.weekday)} ${x.start_time}`),
      group_rule: RULE_LABEL[s.group_rule], max_size: s.max_size, forum_open: s.forum_open === 1,
      recent_questions: topics.map((t) => ({ post_id: t.id, title: t.title, replies: t.replies, url: abs(`/camp/r/${s.slug}/forum/${t.id}/`) })),
    });
  });

  server.registerTool('record_learning_need', {
    description: '學員的問題對不到任何成果時，經學員同意，把問題記下來給講師們看（不會顯示是誰）。一定要先問學員同不同意。',
    inputSchema: { problem: z.string().min(5).describe('學員的工作問題，用學員的話寫清楚') },
  }, async ({ problem }) => (await recordNeed(me.id, problem)) ? ok('已記下，講師們看得到。之後有講師上架相關成果，會出現在好棋寶寶。') : fail('問題是空的。'));

  server.registerTool('register_interest', {
    description: '學員做不來，登記想學某項成果（免費、不用入會、可取消）。講師有每週時段時要指定時段。一定要學員說好才登記。',
    inputSchema: { slug: z.string(), slot: z.string().optional().describe('時段，例如「週六 10:00」；成果有多個時段時必填') },
  }, async ({ slug, slot }) => {
    const s = await getShowcaseBySlug(slug);
    if (!s || s.status !== 'published') return fail('找不到這項成果。');
    if (isOwner(s, me)) return fail('講師不需要登記自己的成果。');
    const gathering = openViews(await cohortViews(s)).filter((c) => c.state === 'gathering');
    if (!gathering.length) return fail('這項成果目前有一團進行中，結營後才重新開放登記。');
    const target = gathering.length === 1 ? gathering[0] : gathering.find((c) => slot && c.slotLabel?.replace(/\s/g, '') === slot.replace(/\s/g, ''));
    if (!target) return fail(`請指定時段：${gathering.map((c) => c.slotLabel).join('、')}`);
    await addInterest(target.id, me.id);
    return ok({ registered: true, slot: target.slotLabel, page: abs(`/camp/r/${s.slug}/`), note: `滿 ${s.min_size} 人自動開團，開團時網站會通知你報名匯款。` });
  });

  server.registerTool('cancel_interest', { description: '取消登記想學。', inputSchema: { slug: z.string() } }, async ({ slug }) => {
    const s = await getShowcaseBySlug(slug);
    if (!s) return fail('找不到這項成果。');
    for (const c of openViews(await cohortViews(s)).filter((x) => x.state === 'gathering')) await removeInterest(c.id, me.id);
    return ok('已取消登記。');
  });

  server.registerTool('my_learning', {
    description: '學員自己的狀態：登記想學的成果、報名的團與核款狀態、主題教學時間、3 週實作每週截止與是否已交、還沒送出的草稿。',
    inputSchema: {},
  }, async () => {
    const { results: wants } = await db().prepare(
      `SELECT s.title, s.slug, s.min_size, c.state, (SELECT COUNT(*) FROM interests x WHERE x.cohort_id = c.id) AS n FROM interests i
       JOIN cohorts c ON c.id = i.cohort_id JOIN showcases s ON s.id = c.showcase_id WHERE i.account_id = ? AND c.state IN ('gathering', 'scheduling', 'payment')`,
    ).bind(me.id).all<{ title: string; slug: string; min_size: number; state: string; n: number }>();
    const { results: mine } = await db().prepare(
      `SELECT e.status, c.id, c.seq, c.state, c.teaching_start, s.title FROM enrollments e JOIN cohorts c ON c.id = e.cohort_id JOIN showcases s ON s.id = c.showcase_id
       WHERE e.account_id = ? AND e.status != 'cancelled' ORDER BY e.created_at DESC`,
    ).bind(me.id).all<{ status: string; id: string; seq: number; state: string; teaching_start: string | null; title: string }>();
    const cohorts = await Promise.all(mine.map(async (e) => {
      const subs = (await submissions(e.id)).filter((x) => x.account_id === me.id);
      return {
        cohort_id: e.id, title: `${e.title}．第 ${e.seq} 團`, state: label(COHORT_STATE, e.state), payment: label(ENROLLMENT_STATUS, e.status),
        teaching: e.teaching_start ? fmtDateTime(e.teaching_start) : null, page: abs(`/camp/g/${e.id}/`),
        practice: e.teaching_start && e.status === 'confirmed' ? [1, 2, 3].map((w) => ({ week: w, ask: WEEK_ASK[w], due: fmtDateTime(weekDeadline({ teaching_start: e.teaching_start }, w)), submitted: subs.some((x) => x.week === w) })) : null,
      };
    }));
    const drafts = (await pendingDrafts(me.id)).map((d) => ({ kind: d.kind, title: d.title, url: abs(draftUrl(d.id)) }));
    return ok({ interests: wants.map((w) => ({ title: w.title, slug: w.slug, state: label(COHORT_STATE, w.state), registered: w.n, opens_at: w.min_size })), cohorts, pending_drafts: drafts });
  });

  const spaceFrom = async (where: { slug?: string; cohort_id?: string }) => {
    if (where.cohort_id) return `cohort:${where.cohort_id}`;
    if (where.slug) { const s = await getShowcaseBySlug(where.slug); return s ? `showcase:${s.id}` : null; }
    return null;
  };
  const draftResult = (r: Awaited<ReturnType<typeof createDraft>>) => r.ok
    ? ok({ draft_saved: true, where: r.where, confirm_url: abs(r.url), note: '把這個網址給使用者，在網站上看過、按「確定送出」才會發出。' })
    : fail(r.error);

  server.registerTool('draft_question', {
    description: '學員卡住時，把整理好的問題（做到哪、哪裡出錯、試過什麼）存成成果討論區的提問草稿。學員同意才存；回傳網址給學員去按確定。',
    inputSchema: { slug: z.string(), title: z.string(), body: z.string() },
  }, async ({ slug, title, body }) => {
    const space = await spaceFrom({ slug });
    return space ? draftResult(await createDraft(me, { kind: 'question', space, title, body })) : fail('找不到這項成果。');
  });

  server.registerTool('draft_practice', {
    description: '開團後的 3 週實作：把學員這週的進度存成課程論壇的繳交草稿（第 1 週題目、第 2 週第一版、第 3 週成果）。',
    inputSchema: { cohort_id: z.string(), week: z.number().int().min(1).max(3), title: z.string(), body: z.string() },
  }, async ({ cohort_id, week, title, body }) => draftResult(await createDraft(me, { kind: 'practice', space: `cohort:${cohort_id}`, week, title, body })));

  server.registerTool('draft_work_share', {
    description: '學員做出成果時，主動整理作品與心得（做了什麼、作品連結或截圖、怎麼做的、心得）存成課程論壇的分享草稿。學員在網站按確定，同一頁可勾選同意講師認可後公開到成果頁。',
    inputSchema: { cohort_id: z.string(), title: z.string(), body: z.string(), week: z.number().int().min(1).max(3).optional().describe('同時當作第幾週的繳交，通常是 3') },
  }, async ({ cohort_id, title, body, week }) => draftResult(await createDraft(me, { kind: 'work', space: `cohort:${cohort_id}`, week: week ?? null, title, body })));

  server.registerTool('list_posts', {
    description: '列出一個論壇的主題文：成果的公開討論區（給 slug）或一團的課程論壇（給 cohort_id）。',
    inputSchema: { slug: z.string().optional(), cohort_id: z.string().optional() },
  }, async (where) => {
    const space = await spaceFrom(where);
    if (!space) return fail('要給 slug 或 cohort_id。');
    if (space.startsWith('cohort:')) {
      const c = await getCohort(space.slice(7)); const s = c && (await getShowcase(c.showcase_id));
      if (!c || !s || !(await cohortSpaceRole(c, s, me)).view) return fail('你看不到這個課程論壇。');
    }
    const topics = (await listTopics(space)).filter((t) => t.status === 'published').slice(0, 30);
    return ok(topics.map((t) => ({ post_id: t.id, title: t.title, author: t.author_name, at: fmtDateTime(t.created_at), replies: t.replies })));
  });

  server.registerTool('read_post', {
    description: '讀一則主題文和所有回覆。',
    inputSchema: { post_id: z.string(), slug: z.string().optional(), cohort_id: z.string().optional() },
  }, async ({ post_id, ...where }) => {
    const space = await spaceFrom(where);
    if (!space) return fail('要給 slug 或 cohort_id。');
    const [kind, id] = space.split(':');
    const s = kind === 'showcase' ? await getShowcase(id) : await (async () => { const c = await getCohort(id); return c ? getShowcase(c.showcase_id) : null; })();
    const role = !s ? null : kind === 'showcase' ? showcaseSpaceRole(s, me) : await cohortSpaceRole((await getCohort(id))!, s, me);
    if (!role?.view) return fail('你看不到這個論壇。');
    const topic = await getPost(space, post_id);
    if (!topic || topic.status !== 'published') return fail('找不到這則文章。');
    const replies = (await listReplies(space, post_id)).filter((r) => r.status === 'published');
    return ok({ title: topic.title, author: topic.author_name, body: topic.body, replies: replies.map((r) => ({ author: r.author_name, at: fmtDateTime(r.created_at), body: r.body })) });
  });

  server.registerTool('draft_reply', {
    description: '把回覆存成草稿（講師回覆學員、學員互相回覆都可以）。回傳網址給本人去按確定。',
    inputSchema: { post_id: z.string().describe('要回覆的主題文'), body: z.string(), slug: z.string().optional(), cohort_id: z.string().optional() },
  }, async ({ post_id, body, ...where }) => {
    const space = await spaceFrom(where);
    return space ? draftResult(await createDraft(me, { kind: 'reply', space, parentId: post_id, body })) : fail('要給 slug 或 cohort_id。');
  });

  // ---- 講師 ----

  server.registerTool('my_showcases', { description: '講師自己的成果：狀態、還缺什麼、編輯網址，以及各團狀態。', inputSchema: {} }, async () => {
    const list = await listByOwner(me.id);
    return ok(await Promise.all(list.map(async (s) => ({
      showcase_id: s.id, slug: s.slug, title: s.title, status: label(SHOWCASE_STATUS, s.status), missing: missingItems(s, await slotsOf(s.id)),
      edit_url: abs(`/account/teach/showcases/${s.id}/`),
      cohorts: (await cohortViews(s)).filter((c) => c.state !== 'gathering').slice(0, 5).map((c) => ({ cohort_id: c.id, seq: c.seq, state: label(COHORT_STATE, c.state), teaching: c.teaching_start ? fmtDateTime(c.teaching_start) : null })),
    }))));
  });

  const showcaseFields = {
    showcase_id: z.string().optional().describe('要修改的成果；不給就新建一份草稿'),
    title: z.string().optional(), summary: z.string().optional(), instructor_name: z.string().optional(),
    demo_links: z.array(z.object({ title: z.string(), url: z.string().url(), note: z.string().optional() })).optional().describe('1. 成果展示'),
    before_after: z.string().optional().describe('2. 用 AI 之前和之後差在哪裡'), learner_outcome: z.string().optional().describe('3. 學完能做出什麼'),
    audience: z.string().optional().describe('4. 適合誰'), prerequisites: z.string().optional().describe('4. 需要什麼基礎'),
    instructor_bio: z.string().optional().describe('5. 講師本業與為什麼做這個'), method: z.string().optional().describe('6. 做法說明（Markdown）'),
    sample_problem: z.string().optional(), sample_cause: z.string().optional(), sample_solutions: z.string().optional(), sample_flow: z.string().optional(),
    setup_requirements: z.string().optional().describe('8. 自己跑起來要準備什麼'),
    repo_url: z.string().optional(), license: z.string().optional(), allow_derivative: z.boolean().optional(),
    price: z.number().int().min(campRules.minFee).optional(), teaching_format: z.enum(['online', 'onsite']).optional(), location: z.string().optional(),
    teaching_minutes: z.number().int().min(30).max(480).optional(),
    schedule_mode: z.enum(['slots', 'vote', 'fixed']).optional(), weekly_slots: z.array(z.object({ weekday: z.number().int().min(0).max(6).describe('0 = 週日'), start_time: z.string().regex(/^\d{2}:\d{2}$/) })).optional(),
    fixed_start: z.string().optional().describe('預訂日期，台北時間 YYYY-MM-DDTHH:MM'),
    group_rule: z.enum(['paid', 'signup']).optional(), min_size: z.number().int().min(1).optional(), max_size: z.number().int().min(1).nullable().optional(),
    pay_days: z.number().int().min(1).max(60).optional(), vote_days: z.number().int().min(1).max(30).optional(), forum_open: z.boolean().optional(),
    ttqs_needs: z.string().optional(), ttqs_goals: z.string().optional(), ttqs_outline: z.string().optional(), ttqs_hours: z.string().optional(),
    ttqs_methods: z.string().optional(), ttqs_evaluation: z.string().optional(), ttqs_expected: z.string().optional(),
  };

  server.registerTool('save_showcase_draft', {
    description: '講師新建或修改成果草稿（8 項資料、repo、授權、開團設定、TTQS 開課表單），只給要改的欄位。回傳還缺什麼和網站網址；送審要講師本人在網站上按。',
    inputSchema: showcaseFields,
  }, async (input) => {
    if (!(await canCreateShowcase(me.id))) return fail('要先是有效的協會會員才能上架成果，請到網站帳號頁繳常年會費。');
    let s = input.showcase_id ? await getShowcase(input.showcase_id) : null;
    if (input.showcase_id && (!s || s.owner_id !== me.id)) return fail('找不到你的這項成果。');
    if (!s) s = (await getShowcase(await createShowcaseDraft(me.id, input.instructor_name ?? me.display_name)))!;
    if (!editable(s)) return fail('這項成果審核中或已下架，現在不能改。');
    const { showcase_id: _id, demo_links, weekly_slots, fixed_start, allow_derivative, forum_open, max_size, ...rest } = input;
    const fields: Record<string, string | number | null> = {};
    for (const [k, v] of Object.entries(rest)) if (v !== undefined) fields[k] = v as string | number;
    if (demo_links) fields.demo_links = JSON.stringify(demo_links.map((d) => ({ title: d.title, url: d.url, note: d.note ?? '' })));
    if (fixed_start !== undefined) fields.fixed_start = parseTaipeiLocal(fixed_start);
    if (allow_derivative !== undefined) fields.allow_derivative = allow_derivative ? 1 : 0;
    if (forum_open !== undefined) fields.forum_open = forum_open ? 1 : 0;
    if (max_size !== undefined) fields.max_size = max_size;
    await saveFields(s, me.id, fields as never);
    if (weekly_slots) await setSlots(s.id, me.id, weekly_slots);
    const now = (await getShowcase(s.id))!;
    return ok({ showcase_id: now.id, status: label(SHOWCASE_STATUS, now.status), missing: missingItems(now, await slotsOf(now.id)), edit_url: abs(`/account/teach/showcases/${now.id}/`), submit_url: abs(`/account/teach/showcases/${now.id}/?step=4`), preview_url: abs(`/camp/r/${now.slug}/`) });
  });

  server.registerTool('register_skill_version', {
    description: '講師 push 之後，登記 repo 的這個版本（commit 或分支名；不給就用預設分支最新版）。講師本人要在網站上按「送審這個版本」，管理員審過學員才拿得到。',
    inputSchema: { showcase_id: z.string(), repo_url: z.string().optional(), ref: z.string().optional(), note: z.string().optional().describe('這一版改了什麼') },
  }, async ({ showcase_id, repo_url, ref, note }) => {
    const s = await getShowcase(showcase_id);
    if (!s || s.owner_id !== me.id) return fail('找不到你的這項成果。');
    const r = await registerVersion(s, me.id, repo_url ?? s.repo_url, ref ?? '', note ?? null);
    return r.ok ? ok({ commit: r.sha, submit_url: abs(`/account/teach/showcases/${s.id}/?step=5`), note: '請講師到這個網址按「送審這個版本」。' }) : fail(r.error);
  });

  server.registerTool('cohort_progress', {
    description: '講師看自己某一團的進度：每位學員的核款狀態（不含匯款資料）、主題教學簽到、3 週實作每週有沒有交，以及還沒有講師回覆的主題文。',
    inputSchema: { cohort_id: z.string() },
  }, async ({ cohort_id }) => {
    const c = await getCohort(cohort_id);
    const s = c ? await getShowcase(c.showcase_id) : null;
    if (!c || !s || (!isOwner(s, me) && !isAdmin(me))) return fail('找不到你的這一團。');
    const { results: roster } = await db().prepare(
      `SELECT e.account_id, e.status, a.display_name, (SELECT 1 FROM attendance t WHERE t.cohort_id = e.cohort_id AND t.account_id = e.account_id) AS present
       FROM enrollments e JOIN accounts a ON a.id = e.account_id WHERE e.cohort_id = ? AND e.status != 'cancelled'`,
    ).bind(c.id).all<{ account_id: string; status: string; display_name: string; present: number | null }>();
    const subs = await submissions(c.id);
    const topics = (await listTopics(`cohort:${c.id}`)).filter((t) => t.status === 'published');
    const unanswered = [];
    for (const t of topics.slice(0, 40)) {
      if (t.author_id === s.owner_id) continue;
      const replies = await listReplies(`cohort:${c.id}`, t.id);
      if (!replies.some((r) => r.author_id === s.owner_id && r.status === 'published')) unanswered.push({ post_id: t.id, title: t.title, author: t.author_name, at: fmtDateTime(t.created_at), url: abs(`/camp/g/${c.id}/forum/${t.id}/`) });
    }
    return ok({
      title: `${s.title}．第 ${c.seq} 團`, state: label(COHORT_STATE, c.state), teaching: c.teaching_start ? fmtDateTime(c.teaching_start) : null,
      students: roster.map((r) => ({ name: r.display_name, payment: label(ENROLLMENT_STATUS, r.status), checked_in: !!r.present, practice: [1, 2, 3].map((w) => ({ week: PRACTICE_WEEKS[w], submitted: subs.some((x) => x.account_id === r.account_id && x.week === w) })) })),
      waiting_for_your_reply: unanswered,
    });
  });

  server.registerTool('list_learning_needs', { description: '學員記下來、還沒有成果對應的工作問題（不含是誰），講師想開新課時參考。', inputSchema: {} }, async () => {
    if (!(await canCreateShowcase(me.id)) && !isAdmin(me)) return fail('講師（協會會員）才看得到。');
    return ok((await recentNeeds(100)).map((n) => ({ problem: n.problem, at: fmtDateTime(n.created_at) })));
  });

  return server;
}

// 每個請求建立一個無狀態的 MCP 伺服器。
export async function handleMcp(request: Request, me: Account) {
  const server = createServer(me, new URL(request.url).origin);
  const transport = new WebStandardStreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
  await server.connect(transport);
  return transport.handleRequest(request);
}

export const MCP_SITE = site.url;
