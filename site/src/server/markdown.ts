// 使用者內容（論壇、技能單張）轉 HTML：不允許原始 HTML 與危險連結協定。
import { micromark } from 'micromark';
import { gfm, gfmHtml } from 'micromark-extension-gfm';

export function renderUserMarkdown(src: string): string {
  return micromark(src, { extensions: [gfm()], htmlExtensions: [gfmHtml()], allowDangerousHtml: false, allowDangerousProtocol: false });
}
