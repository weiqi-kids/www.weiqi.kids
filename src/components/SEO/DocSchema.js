import React from 'react';
import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import PersonSchema from '@site/src/components/SEO/PersonSchema';
import { founders } from '@site/src/data/members';

/**
 * DocSchema — 對所有 docs 頁面自動注入頁面層 JSON-LD（語系感知）。
 * 透過 swizzle 的 @theme/DocItem/Layout 包裝，在每個 doc 渲染時呼叫。
 *
 * 規則：
 *  - 創始夥伴細頁 → PersonSchema（資料取自 members.js + frontmatter industry_tags）
 *  - 其餘 docs → Article + WebPage(speakable) + og:type=article
 *  - alphago 內容頁、lightman-chang：已有「手動」Schema，略過以免重複
 *
 * 註：BreadcrumbList 由 Docusaurus 主題自動產生，這裡不重複輸出。
 */
const SPEAKABLE = [
  '.article-summary', '.speakable-content', '.key-takeaway',
  '.key-answer', '.expert-quote', '.actionable-steps li', '.faq-answer-content',
];

function toISO(epochMillis) {
  if (!epochMillis) return undefined;
  // 不可用 new Date() 取現在時間，但用既有 timestamp 轉 ISO 是安全的
  // metadata.lastUpdatedAt 與 frontMatter date 的 getTime() 皆已是毫秒，不可再乘 1000
  return new Date(epochMillis).toISOString();
}

const BCP47 = { 'zh-tw': 'zh-TW', 'zh-cn': 'zh-CN', 'zh-hk': 'zh-HK' };

export default function DocSchema() {
  const { siteConfig, i18n } = useDocusaurusContext();
  const siteUrl = siteConfig.url;
  const locale = (i18n && i18n.currentLocale) || 'zh-tw';
  const inLang = BCP47[locale] || locale;
  let doc;
  try {
    doc = useDoc();
  } catch (e) {
    return null; // 非 doc 情境（保險）
  }
  const { metadata, frontMatter } = doc;
  const permalink = metadata.permalink || '';
  const pathNoSlash = permalink.replace(/\/$/, '');
  const docsIdx = pathNoSlash.indexOf('/docs/');
  if (docsIdx < 0) return null;
  const slug = pathNoSlash.slice(docsIdx + '/docs/'.length); // 不含語系、不含 /docs/

  // 已有手動 Schema 的頁面 → 略過
  if (slug.startsWith('alphago/') && slug !== 'alphago') return null;
  if (slug === 'about/members/founding/lightman-chang') return null;

  const title = metadata.title || frontMatter.title || '';
  const description = metadata.description || frontMatter.description || title;
  const dateModified = toISO(metadata.lastUpdatedAt);
  const datePublished = frontMatter.date || dateModified || '2024-01-01';

  // ── 創始夥伴細頁 → PersonSchema ──
  const memberMatch = slug.match(/^about\/members\/founding\/([^/]+)$/);
  if (memberMatch && memberMatch[1] !== 'index') {
    const m = founders.find((f) => f.slug === memberMatch[1]);
    if (m) {
      const orgParts = (m.org || '').split('／').map((s) => s.trim()).filter(Boolean);
      const tags = Array.isArray(frontMatter.industry_tags) ? frontMatter.industry_tags : [];
      const knows = (tags.length ? tags : [m.key5, m.title, ...orgParts])
        .filter(Boolean);
      const knowsAbout = knows.length >= 2 ? knows : [...knows, '圍棋', '好棋寶寶協會'];
      const jobTitle = [m.title, m.org].filter(Boolean).join('｜') || '創始夥伴';
      return (
        <PersonSchema
          name={m.name}
          alternateName={m.enName}
          jobTitle={jobTitle}
          description={description || `${m.name}｜台灣好棋寶寶協會創始夥伴`}
          slug={slug}
          inLanguage={inLang}
          knowsAbout={knowsAbout}
          worksFor={orgParts.map((name) => ({ name }))}
        />
      );
    }
  }

  // ── 其餘 docs → Article + WebPage(speakable) ──
  const canonicalUrl = `${siteUrl}${permalink}`;
  const ogImage = `${siteUrl}/img/social-card.png`;
  const schema = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'WebPage',
        '@id': `${canonicalUrl}#webpage`,
        url: canonicalUrl,
        name: title,
        description,
        isPartOf: { '@id': `${siteUrl}#website` },
        primaryImageOfPage: { '@type': 'ImageObject', url: ogImage },
        ...(datePublished ? { datePublished } : {}),
        ...(dateModified ? { dateModified } : {}),
        inLanguage: inLang,
        speakable: { '@type': 'SpeakableSpecification', cssSelector: SPEAKABLE },
      },
      {
        '@type': 'Article',
        '@id': `${canonicalUrl}#article`,
        mainEntityOfPage: { '@id': `${canonicalUrl}#webpage` },
        headline: title,
        description,
        image: { '@type': 'ImageObject', url: ogImage, width: 1200, height: 630 },
        author: { '@id': `${siteUrl}#organization` },
        publisher: { '@id': `${siteUrl}#organization` },
        ...(datePublished ? { datePublished } : {}),
        ...(dateModified ? { dateModified } : {}),
        isAccessibleForFree: true,
        inLanguage: inLang,
        isPartOf: { '@type': 'WebSite', '@id': `${siteUrl}#website` },
      },
    ],
  };

  return (
    <Head>
      <meta property="og:type" content="article" />
      {datePublished && <meta property="article:published_time" content={String(datePublished)} />}
      {dateModified && <meta property="article:modified_time" content={dateModified} />}
      <script type="application/ld+json">{JSON.stringify(schema)}</script>
    </Head>
  );
}
