import React from 'react';
import Head from '@docusaurus/Head';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * PersonSchema 組件 - 為夥伴／人物頁面注入完整 JSON-LD Schema
 * 包含 ProfilePage(WebPage)、Person、BreadcrumbList
 *
 * 設計為可重用：所有創始夥伴頁面共用同一元件，只需傳入對應 props。
 * 符合協會 SEO 規範的 Person 必填欄位：knowsAbout(≥2)、hasCredential(≥1)、sameAs(≥1)。
 *
 * @param {Object} props
 * @param {string} props.name - 姓名（對外正式名稱）
 * @param {string} [props.alternateName] - 別名／英文名／署名
 * @param {string} props.jobTitle - 職稱／定位
 * @param {string} props.description - 一句話簡介（用於 meta 與 Person.description）
 * @param {string} props.slug - 頁面路徑（不含 /docs/ 與前後斜線），例如 about/members/founding/lightman-chang
 * @param {string} [props.image] - 大頭照絕對或相對路徑；省略則用站台社交卡
 * @param {string} [props.email] - Email
 * @param {string} [props.telephone] - 電話
 * @param {string[]} [props.sameAs] - 外部權威連結（個人網站、GitHub、社群、代表作品）
 * @param {string[]} props.knowsAbout - 專業領域（≥2）
 * @param {Array<string|Object>} props.hasCredential - 認證；字串或 {name, category}（≥1）
 * @param {Array<{name:string, url?:string}>} [props.worksFor] - 任職組織（協會以外）
 * @param {Array<{name:string, position?:number, path?:string}>} [props.breadcrumbs] - 額外麵包屑
 * @param {string} [props.inLanguage] - 頁面語言（BCP 47），預設 zh-TW；翻譯頁應傳入對應語言
 */
export default function PersonSchema({
  name,
  alternateName,
  jobTitle,
  description,
  slug,
  image,
  email,
  telephone,
  sameAs = [],
  knowsAbout = [],
  hasCredential = [],
  worksFor = [],
  breadcrumbs = [],
  inLanguage = 'zh-TW',
}) {
  const { siteConfig, i18n } = useDocusaurusContext();
  const siteUrl = siteConfig.url;
  // 翻譯頁網址含語系前綴（預設語系除外），確保 @id／url 指向正確的本地化頁面
  const localePrefix =
    i18n && i18n.currentLocale && i18n.currentLocale !== i18n.defaultLocale
      ? `/${i18n.currentLocale}`
      : '';
  const canonicalUrl = `${siteUrl}${localePrefix}/docs/${slug}/`;
  const personId = `${canonicalUrl}#person`;
  const imageUrl = image
    ? (image.startsWith('http') ? image : `${siteUrl}${image}`)
    : `${siteUrl}/img/social-card.png`;

  // 認證：允許傳字串或物件
  const credentials = hasCredential.map((c) =>
    typeof c === 'string'
      ? {
          '@type': 'EducationalOccupationalCredential',
          name: c,
          credentialCategory: '專業認證',
        }
      : {
          '@type': 'EducationalOccupationalCredential',
          name: c.name,
          credentialCategory: c.category || '專業認證',
        }
  );

  // 任職組織：協會永遠在列，其餘附加
  const organizations = [
    { '@id': `${siteUrl}#organization` },
    ...worksFor.map((o) =>
      o.url
        ? { '@type': 'Organization', name: o.name, url: o.url }
        : { '@type': 'Organization', name: o.name }
    ),
  ];

  const localeBase = `${siteUrl}${localePrefix}`;
  const defaultCrumbs = [
    { '@type': 'ListItem', position: 1, name: '首頁', item: `${localeBase}/` },
    { '@type': 'ListItem', position: 2, name: '關於協會', item: `${localeBase}/docs/about/` },
    { '@type': 'ListItem', position: 3, name: '認識夥伴', item: `${localeBase}/docs/about/members/founding/` },
    { '@type': 'ListItem', position: 4, name: name, item: canonicalUrl },
  ];

  const schema = {
    '@context': 'https://schema.org',
    '@graph': [
      // ProfilePage（人物檔案頁）
      {
        '@type': 'ProfilePage',
        '@id': `${canonicalUrl}#webpage`,
        url: canonicalUrl,
        name: `${name}｜台灣好棋寶寶協會`,
        description: description,
        inLanguage: inLanguage,
        isPartOf: { '@id': `${siteUrl}#website` },
        primaryImageOfPage: { '@type': 'ImageObject', url: imageUrl },
        mainEntity: { '@id': personId },
        speakable: {
          '@type': 'SpeakableSpecification',
          cssSelector: [
            '.article-summary',
            '.speakable-content',
            '.key-takeaway',
            '.key-answer',
            '.expert-quote',
            '.actionable-steps li',
            '.faq-answer-content',
          ],
        },
      },
      // Person
      {
        '@type': 'Person',
        '@id': personId,
        name: name,
        ...(alternateName ? { alternateName } : {}),
        url: canonicalUrl,
        ...(image || true ? { image: imageUrl } : {}),
        jobTitle: jobTitle,
        description: description,
        ...(email ? { email: `mailto:${email}` } : {}),
        ...(telephone ? { telephone } : {}),
        worksFor: organizations,
        memberOf: { '@id': `${siteUrl}#organization` },
        knowsAbout: knowsAbout,
        hasCredential: credentials,
        ...(sameAs.length > 0 ? { sameAs } : {}),
      },
      // BreadcrumbList
      {
        '@type': 'BreadcrumbList',
        '@id': `${canonicalUrl}#breadcrumb`,
        itemListElement: [
          ...defaultCrumbs,
          ...breadcrumbs.map((crumb, index) => ({
            '@type': 'ListItem',
            position: crumb.position || 5 + index,
            name: crumb.name,
            item: crumb.path ? `${siteUrl}${crumb.path}` : canonicalUrl,
          })),
        ],
      },
    ],
  };

  return (
    <Head>
      <script type="application/ld+json">{JSON.stringify(schema)}</script>
    </Head>
  );
}
