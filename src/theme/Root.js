import React, { useEffect } from 'react';
import GlobalSchema from '@site/src/components/SEO/GlobalSchema';

// 支援的語系列表
const SUPPORTED_LOCALES = [
  'zh-tw', 'zh-cn', 'zh-hk', 'en', 'ja', 'ko', 'es', 'pt', 'hi', 'id', 'ar'
];
const DEFAULT_LOCALE = 'zh-tw';
const LOCALE_STORAGE_KEY = 'preferred-locale';

// 將瀏覽器語言對應到支援的 locale
function mapBrowserLanguageToLocale(browserLang) {
  if (!browserLang) return null;

  const lang = browserLang.toLowerCase();

  // 精確匹配
  if (lang === 'zh-tw' || lang === 'zh-hant-tw') return 'zh-tw';
  if (lang === 'zh-hk' || lang === 'zh-hant-hk') return 'zh-hk';
  if (lang === 'zh-cn' || lang === 'zh-hans-cn') return 'zh-cn';

  // 繁體中文變體
  if (lang === 'zh-hant' || lang === 'zh-cht') return 'zh-tw';

  // 簡體中文變體
  if (lang === 'zh-hans' || lang === 'zh-chs' || lang === 'zh') return 'zh-cn';

  // 其他語言 - 取前綴匹配
  const prefix = lang.split('-')[0];

  const prefixMap = {
    'en': 'en',
    'ja': 'ja',
    'ko': 'ko',
    'es': 'es',
    'pt': 'pt',
    'hi': 'hi',
    'id': 'id',
    'ar': 'ar',
  };

  return prefixMap[prefix] || null;
}

// 從 URL 取得目前的 locale
function getCurrentLocaleFromURL() {
  if (typeof window === 'undefined') return DEFAULT_LOCALE;

  const pathname = window.location.pathname;

  // 檢查路徑是否以支援的 locale 開頭
  for (const locale of SUPPORTED_LOCALES) {
    if (locale === DEFAULT_LOCALE) continue; // 預設語系在根路徑
    if (pathname.startsWith(`/${locale}/`) || pathname === `/${locale}`) {
      return locale;
    }
  }

  return DEFAULT_LOCALE;
}

// 建構目標 URL
function buildTargetURL(targetLocale) {
  const currentLocale = getCurrentLocaleFromURL();
  const pathname = window.location.pathname;
  const search = window.location.search;
  const hash = window.location.hash;

  let newPathname;

  if (currentLocale === DEFAULT_LOCALE) {
    // 目前在預設語系（根路徑）
    if (targetLocale === DEFAULT_LOCALE) {
      newPathname = pathname;
    } else {
      newPathname = `/${targetLocale}${pathname}`;
    }
  } else {
    // 目前在其他語系
    const pathWithoutLocale = pathname.replace(new RegExp(`^/${currentLocale}/?`), '/');

    if (targetLocale === DEFAULT_LOCALE) {
      newPathname = pathWithoutLocale || '/';
    } else {
      newPathname = `/${targetLocale}${pathWithoutLocale}`;
    }
  }

  return newPathname + search + hash;
}

// 語言偵測和重定向 Hook
function useLocaleRedirect() {
  useEffect(() => {
    // 只在客戶端執行
    if (typeof window === 'undefined') return;

    const currentLocale = getCurrentLocaleFromURL();

    // 檢查是否有儲存的偏好語言
    const savedLocale = localStorage.getItem(LOCALE_STORAGE_KEY);

    if (savedLocale && SUPPORTED_LOCALES.includes(savedLocale)) {
      // 如果有儲存的偏好，且與目前不同，則重定向
      if (savedLocale !== currentLocale) {
        const targetURL = buildTargetURL(savedLocale);
        window.location.replace(targetURL);
        return;
      }
    } else {
      // 沒有儲存的偏好，偵測瀏覽器語言
      const browserLanguages = navigator.languages || [navigator.language];

      for (const browserLang of browserLanguages) {
        const matchedLocale = mapBrowserLanguageToLocale(browserLang);
        if (matchedLocale && matchedLocale !== currentLocale) {
          // 找到匹配的語言，儲存並重定向
          localStorage.setItem(LOCALE_STORAGE_KEY, matchedLocale);
          const targetURL = buildTargetURL(matchedLocale);
          window.location.replace(targetURL);
          return;
        }
        if (matchedLocale === currentLocale) {
          // 已經在正確的語系，儲存偏好
          localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale);
          return;
        }
      }

      // 沒有匹配到任何支援的語言，使用預設並儲存
      localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale);
    }
  }, []);
}

// 監聽語言切換並儲存偏好
function useLocaleSaveOnChange() {
  useEffect(() => {
    if (typeof window === 'undefined') return;

    // 使用 MutationObserver 監聽 html lang 屬性變化
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'lang') {
          const newLocale = getCurrentLocaleFromURL();
          localStorage.setItem(LOCALE_STORAGE_KEY, newLocale);
        }
      }
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['lang']
    });

    // 也監聽 URL 變化（用於 SPA 導航）
    const handlePopState = () => {
      const newLocale = getCurrentLocaleFromURL();
      localStorage.setItem(LOCALE_STORAGE_KEY, newLocale);
    };

    window.addEventListener('popstate', handlePopState);

    // 監聽 Docusaurus 的路由變化
    const handleClick = (e) => {
      // 檢查是否點擊了語言切換連結
      const link = e.target.closest('a[href]');
      if (link) {
        const href = link.getAttribute('href');
        for (const locale of SUPPORTED_LOCALES) {
          if (href === `/${locale}/` || href === `/${locale}` ||
              (locale === DEFAULT_LOCALE && href === '/')) {
            // 立即儲存目標語系，避免跳轉後被 useLocaleRedirect 覆蓋
            localStorage.setItem(LOCALE_STORAGE_KEY, locale);
            break;
          }
          // 檢查路徑是否包含語系前綴
          if (href.startsWith(`/${locale}/`)) {
            localStorage.setItem(LOCALE_STORAGE_KEY, locale);
            break;
          }
        }
      }
    };

    document.addEventListener('click', handleClick);

    return () => {
      observer.disconnect();
      window.removeEventListener('popstate', handlePopState);
      document.removeEventListener('click', handleClick);
    };
  }, []);
}

// 自訂 Root 組件 - 注入全域 SEO Schema 和語言偵測
export default function Root({ children }) {
  useLocaleRedirect();
  useLocaleSaveOnChange();

  return (
    <>
      <GlobalSchema />
      {children}
    </>
  );
}
