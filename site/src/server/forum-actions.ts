// 公開討論區與課程論壇共用的表單處理。
import type { Account } from './types';
import { getPost, createPost, editPost, withdrawPost, moderate, fileAppeal, validateFiles, listReplies, attachmentsFor, lastModeration, openAppeal, visibleTo, HIDE_REASONS, type Post, type SpaceRole } from './forum';
import { str } from './http';

type Result = { redirect?: string; error?: string; message?: string; postId?: string };

export async function handleNewTopic(form: FormData, space: string, me: Account, role: SpaceRole): Promise<Result> {
  if (!role.post) return { error: '你沒有在這裡發文的權限。' };
  const title = str(form, 'title', 200);
  const body = str(form, 'body', 20_000);
  const files = form.getAll('images').filter((f): f is File => f instanceof File);
  const error = !title || !body ? '標題和內容都要填。' : await validateFiles(files);
  if (error) return { error };
  const postId = await createPost(space, me, { parentId: null, title, body, files });
  return { postId };
}

export async function handleThreadAction(form: FormData, space: string, topic: Post, me: Account, role: SpaceRole, here: string): Promise<Result> {
  const action = str(form, 'action', 20);
  const targetId = str(form, 'post', 60) || topic.id;
  const target = targetId === topic.id ? topic : await getPost(space, targetId);
  if (!target || (target.parent_id && target.parent_id !== topic.id)) return { error: '找不到這則內容。' };
  if (action === 'reply' && role.post) {
    const body = str(form, 'body', 20_000);
    const files = form.getAll('images').filter((f): f is File => f instanceof File);
    const error = !body ? '請填寫回覆內容。' : await validateFiles(files);
    if (error) return { error };
    await createPost(space, me, { parentId: topic.id, title: null, body, files });
    return { redirect: here };
  }
  if (action === 'edit') {
    const body = str(form, 'body', 20_000);
    const title = target.parent_id ? null : str(form, 'title', 200) || target.title;
    return body && (await editPost(target, me, title, body)) ? { redirect: here } : { error: '無法修改這則內容。' };
  }
  if (action === 'withdraw') return (await withdrawPost(target, me)) ? { redirect: here } : { error: '無法撤回這則內容。' };
  if (action === 'hide' || action === 'restore') {
    const reason = action === 'hide' ? str(form, 'reason', 40) : '恢復顯示';
    if (action === 'hide' && !(HIDE_REASONS as readonly string[]).includes(reason)) return { error: '請選擇隱藏原因。' };
    return (await moderate(target, me, role, action, reason, str(form, 'note', 500) || null, here)) ? { redirect: here } : { error: '無法執行這個操作。' };
  }
  if (action === 'appeal') {
    const reason = str(form, 'reason', 2000);
    return reason && (await fileAppeal(target, me, reason)) ? { message: '已送出申訴，協會會重新審查。' } : { error: '無法送出申訴（可能已有處理中的申訴）。' };
  }
  return {};
}

export async function loadThread(space: string, topic: Post, me: Account | null, role: SpaceRole) {
  const replies = (await listReplies(space, topic.id)).filter((r) => visibleTo(r, me, role));
  const all = [topic, ...replies];
  const files = await attachmentsFor(all.map((p) => p.id));
  const extra = new Map<string, { mod: Awaited<ReturnType<typeof lastModeration>>; appeal: Awaited<ReturnType<typeof openAppeal>> }>();
  for (const p of all.filter((x) => x.status === 'hidden')) extra.set(p.id, { mod: await lastModeration(p.id), appeal: await openAppeal(p.id) });
  return { all, files, extra };
}
