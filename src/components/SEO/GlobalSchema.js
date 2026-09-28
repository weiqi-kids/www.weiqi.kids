import React from 'react';
import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * GlobalSchema 組件 - 注入全域 JSON-LD Schema
 * 包含 Organization 和 WebSite Schema
 */
export default function GlobalSchema() {
  const { siteConfig } = useDocusaurusContext();
  const siteUrl = siteConfig.url;

  const schema = {
    "@context": "https://schema.org",
    "@graph": [
      // Organization Schema
      {
        "@type": "Organization",
        "@id": `${siteUrl}#organization`,
        "name": "台灣好棋寶寶協會",
        "alternateName": "Taiwan Good Go Baby Association",
        "url": siteUrl,
        "logo": {
          "@type": "ImageObject",
          "url": `${siteUrl}/img/logo.svg`,
          "width": 200,
          "height": 200
        },
        "sameAs": [
          "https://github.com/weiqi-kids"
        ],
        "contactPoint": {
          "@type": "ContactPoint",
          "contactType": "customer service",
          "url": `${siteUrl}/docs/about`
        },
        "description": "台灣的非營利組織，跨域夥伴因圍棋結緣，合作開發開源 AI 工具"
      },
      // WebSite Schema
      // 🔴 2026-09-27 移除 potentialAction/SearchAction：sitelinks 搜尋框這個功能 Google
      //    已於 2024-11 停止使用，標記留著沒有任何效果。站內搜尋本身不受影響，那是給人用的。
      //    判準來源：seo-ops rules/jsonld-rules.json（structuredData.deprecated）。
      {
        "@type": "WebSite",
        "@id": `${siteUrl}#website`,
        "name": "好棋寶寶協會官網",
        "url": siteUrl,
        "publisher": {
          "@id": `${siteUrl}#organization`
        },
        "inLanguage": ["zh-TW", "zh-CN", "zh-HK", "en", "ja", "ko", "es", "pt", "hi", "id", "ar"]
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
