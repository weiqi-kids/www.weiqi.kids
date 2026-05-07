import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * CollectionPageSchema — 用於 /research/、/intel/、/apps/ 等列表頁
 * 注入 BreadcrumbList + ItemList JSON-LD Schema
 *
 * Props:
 *   pagePath: 該頁路徑（如 "/intel/"）
 *   pageName: 該頁標題（如 "產業供應鏈情報"）
 *   description: 該頁描述
 *   items: [{ name, url, description? }] 該頁列出的項目
 *   itemSchemaType: 每個項目的 Schema 類型（預設 "WebSite"）
 */
export default function CollectionPageSchema({
  pagePath,
  pageName,
  description,
  items = [],
  itemSchemaType = 'WebSite',
}) {
  const { siteConfig } = useDocusaurusContext();
  const siteUrl = siteConfig.url;
  const pageUrl = `${siteUrl}${pagePath}`;

  const schema = {
    '@context': 'https://schema.org',
    '@graph': [
      // BreadcrumbList
      {
        '@type': 'BreadcrumbList',
        '@id': `${pageUrl}#breadcrumb`,
        itemListElement: [
          {
            '@type': 'ListItem',
            position: 1,
            name: '首頁',
            item: siteUrl,
          },
          {
            '@type': 'ListItem',
            position: 2,
            name: pageName,
            item: pageUrl,
          },
        ],
      },
      // CollectionPage
      {
        '@type': 'CollectionPage',
        '@id': `${pageUrl}#collection`,
        url: pageUrl,
        name: pageName,
        description,
        isPartOf: { '@id': `${siteUrl}#website` },
        breadcrumb: { '@id': `${pageUrl}#breadcrumb` },
        publisher: { '@id': `${siteUrl}#organization` },
      },
      // ItemList — 列出該頁所有條目
      {
        '@type': 'ItemList',
        '@id': `${pageUrl}#itemlist`,
        name: pageName,
        numberOfItems: items.length,
        itemListElement: items.map((item, idx) => ({
          '@type': 'ListItem',
          position: idx + 1,
          item: {
            '@type': itemSchemaType,
            name: item.name,
            url: item.url,
            ...(item.description ? { description: item.description } : {}),
          },
        })),
      },
    ],
  };

  return (
    <Head>
      <script type="application/ld+json">
        {JSON.stringify(schema)}
      </script>
    </Head>
  );
}
