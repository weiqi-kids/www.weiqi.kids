/**
 * 將「內容語言」（<html lang>）以事件參數 content_language 傳給 GA4。
 * 搭配 GA4 後台註冊「事件範圍」自訂維度（參數名 content_language）即可在報表依語系拆分。
 *
 * gtag('set') 設定的參數會附加到後續送出的事件（含 page_view）。
 * 於 app 啟動時與每次 SPA 換頁時各設定一次，確保跨語系切換也正確。
 */
function setContentLanguage() {
  if (typeof window === 'undefined' || typeof window.gtag !== 'function') return;
  const lang =
    (typeof document !== 'undefined' && document.documentElement.lang) || 'zh-TW';
  window.gtag('set', { content_language: lang });
}

// 啟動時先設定一次（早於首次 page_view）
setContentLanguage();

// 每次換頁（SPA 路由變更）再設定，涵蓋跨語系導覽
export function onRouteDidUpdate() {
  setContentLanguage();
}
