import React from 'react';
import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * ArticleSchema 組件 - 為文章頁面注入完整 JSON-LD Schema
 * 包含 WebPage、Article、BreadcrumbList、Person、ImageObject
 *
 * @param {Object} props
 * @param {string} props.title - 文章標題
 * @param {string} props.description - 文章描述
 * @param {string} props.slug - 文章路徑 (不含 /docs/)
 * @param {string} props.datePublished - 發布日期 (ISO 8601)
 * @param {string} props.dateModified - 修改日期 (ISO 8601)
 * @param {string} props.section - 文章分類
 * @param {string[]} props.keywords - 關鍵字陣列
 * @param {number} props.wordCount - 字數
 * @param {string[]} props.relatedLinks - 相關文章連結
 * @param {Object[]} props.breadcrumbs - 麵包屑 [{name, path}]
 */
export default function ArticleSchema({
  title,
  description,
  slug,
  datePublished = '2024-01-01',
  dateModified,
  section = '技術文章',
  keywords = [],
  wordCount = 5000,
  relatedLinks = [],
  breadcrumbs = [],
}) {
  const { siteConfig } = useDocusaurusContext();
  const siteUrl = siteConfig.url;
  const canonicalUrl = `${siteUrl}/docs/${slug}/`;
  const ogImage = `${siteUrl}/img/social-card.png`;

  const schema = {
    "@context": "https://schema.org",
    "@graph": [
      // WebPage with Speakable
      {
        "@type": "WebPage",
        "@id": `${canonicalUrl}#webpage`,
        "url": canonicalUrl,
        "name": title,
        "description": description,
        "inLanguage": "zh-TW",
        "isPartOf": { "@id": `${siteUrl}#website` },
        "primaryImageOfPage": { "@type": "ImageObject", "url": ogImage },
        "datePublished": datePublished,
        "dateModified": dateModified || datePublished,
        "speakable": {
          "@type": "SpeakableSpecification",
          "cssSelector": [
            ".article-summary",
            ".speakable-content",
            ".key-takeaway",
            ".key-answer",
            ".expert-quote",
            ".actionable-steps li",
            ".faq-answer-content"
          ]
        }
      },
      // Article
      {
        "@type": "Article",
        "@id": `${canonicalUrl}#article`,
        "mainEntityOfPage": {
          "@id": `${canonicalUrl}#webpage`,
          "significantLink": relatedLinks.length > 0
            ? relatedLinks.map(link => `${siteUrl}${link}`)
            : [`${siteUrl}/docs/for-engineers/how-it-works/alphago-explained/`]
        },
        "headline": title,
        "description": description,
        "image": { "@type": "ImageObject", "url": ogImage, "width": 1200, "height": 630 },
        "author": { "@id": `${siteUrl}/docs/about/#person` },
        "publisher": { "@id": `${siteUrl}#organization` },
        "datePublished": datePublished,
        "dateModified": dateModified || datePublished,
        "articleSection": section,
        "keywords": keywords.join(', '),
        "wordCount": wordCount,
        "inLanguage": "zh-TW",
        "isAccessibleForFree": true,
        "isPartOf": {
          "@type": "WebSite",
          "@id": `${siteUrl}#website`
        }
      },
      // BreadcrumbList
      {
        "@type": "BreadcrumbList",
        "itemListElement": [
          { "@type": "ListItem", "position": 1, "name": "首頁", "item": siteUrl },
          { "@type": "ListItem", "position": 2, "name": "給工程師", "item": `${siteUrl}/docs/for-engineers/` },
          { "@type": "ListItem", "position": 3, "name": "技術原理", "item": `${siteUrl}/docs/for-engineers/how-it-works/` },
          { "@type": "ListItem", "position": 4, "name": "AlphaGo 完整解析", "item": `${siteUrl}/docs/for-engineers/how-it-works/alphago-explained/` },
          ...breadcrumbs.map((crumb, index) => ({
            "@type": "ListItem",
            "position": 5 + index,
            "name": crumb.name,
            "item": `${siteUrl}${crumb.path}`
          }))
        ]
      },
      // Person (Author)
      {
        "@type": "Person",
        "@id": `${siteUrl}/docs/about/#person`,
        "name": "好棋寶寶協會編輯團隊",
        "url": `${siteUrl}/docs/about/`,
        "worksFor": { "@id": `${siteUrl}#organization` },
        "description": "專注於圍棋 AI 研究與教育推廣的技術團隊",
        "knowsAbout": ["圍棋 AI", "AlphaGo", "KataGo", "機器學習", "深度學習"],
        "hasCredential": [
          {
            "@type": "EducationalOccupationalCredential",
            "name": "AI 圍棋研究專家",
            "credentialCategory": "技術認證"
          }
        ]
      },
      // ImageObject
      {
        "@type": "ImageObject",
        "@id": `${canonicalUrl}#primaryimage`,
        "url": ogImage,
        "width": 1200,
        "height": 630,
        "caption": `${title} - 好棋寶寶協會`,
        "representativeOfPage": true,
        "license": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "creditText": "台灣好棋寶寶協會"
      }
    ]
  };

  return (
    <Head>
      <script type="application/ld+json">
        {JSON.stringify(schema)}
      </script>
    </Head>
  );
}
