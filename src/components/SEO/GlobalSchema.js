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
          "https://mastodon.weiqi.kids/",
          "https://peertube.weiqi.kids/"
        ],
        "contactPoint": {
          "@type": "ContactPoint",
          "contactType": "customer service",
          "url": `${siteUrl}/docs/aboutus`
        },
        "description": "推動圍棋文化與 AI 研究的台灣非營利組織"
      },
      // WebSite Schema with SearchAction
      {
        "@type": "WebSite",
        "@id": `${siteUrl}#website`,
        "name": "好棋寶寶協會官網",
        "url": siteUrl,
        "publisher": {
          "@id": `${siteUrl}#organization`
        },
        "inLanguage": ["zh-TW", "zh-CN", "zh-HK", "en", "ja", "ko", "es", "pt", "hi", "id", "ar"],
        "potentialAction": {
          "@type": "SearchAction",
          "target": {
            "@type": "EntryPoint",
            "urlTemplate": `${siteUrl}/search?q={search_term_string}`
          },
          "query-input": "required name=search_term_string"
        }
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
