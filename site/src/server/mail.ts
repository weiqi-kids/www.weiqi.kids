import { env } from 'cloudflare:workers';

export type SendResult = { ok: true; debugLink?: string } | { ok: false };

// 以 Cloudflare Email Service 寄出登入信。
// 寄件網域尚未啟用時：staging（MAGIC_LINK_DEBUG=1）改為在頁面上顯示連結供測試；正式環境回報失敗。
export async function sendLoginEmail(to: string, link: string, purpose: 'login' | 'link'): Promise<SendResult> {
  const subject = purpose === 'login' ? '台灣好棋寶寶協會網站登入連結' : '確認綁定 Email 到你的網站帳號';
  const text = `${purpose === 'login' ? '點下面的連結登入' : '點下面的連結，確認把這個 Email 綁定到你的帳號'}（15 分鐘內有效，只能用一次）：\n\n${link}\n\n如果不是你本人申請，請忽略這封信。\n\n台灣好棋寶寶協會`;
  const html = `<p>${purpose === 'login' ? '點下面的連結登入' : '點下面的連結，確認把這個 Email 綁定到你的帳號'}（15 分鐘內有效，只能用一次）：</p><p><a href="${link}">${link}</a></p><p>如果不是你本人申請，請忽略這封信。</p><p>台灣好棋寶寶協會</p>`;
  const binding = (env as unknown as { EMAIL?: { send: (m: unknown) => Promise<unknown> } }).EMAIL;
  if (binding && env.MAIL_FROM) {
    try {
      await binding.send({ to, from: { email: env.MAIL_FROM, name: '台灣好棋寶寶協會' }, subject, text, html });
      return { ok: true };
    } catch (err) {
      console.error('send_email failed', err);
      return { ok: false };
    }
  }
  if (env.MAGIC_LINK_DEBUG === '1') return { ok: true, debugLink: link };
  return { ok: false };
}
