// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: '好棋寶寶協會 | Weiqi.Kids',
  tagline: '台灣好棋寶寶協會｜致力於圍棋文化前進的推手',
  favicon: 'img/favicon.ico',

  // SEO 全域設定
  headTags: [
    // Open Graph 基本標籤
    {
      tagName: 'meta',
      attributes: {
        property: 'og:type',
        content: 'website',
      },
    },
    {
      tagName: 'meta',
      attributes: {
        property: 'og:site_name',
        content: '好棋寶寶協會',
      },
    },
    // Twitter Card
    {
      tagName: 'meta',
      attributes: {
        name: 'twitter:card',
        content: 'summary_large_image',
      },
    },
    // 額外 SEO 標籤
    {
      tagName: 'meta',
      attributes: {
        name: 'robots',
        content: 'index, follow',
      },
    },
    {
      tagName: 'meta',
      attributes: {
        name: 'author',
        content: '台灣好棋寶寶協會',
      },
    },
  ],

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://www.weiqi.kids',
  baseUrl: '/',
  trailingSlash: true,

  // GitHub pages deployment config.
  organizationName: 'weiqi-kids', // Usually your GitHub org/user name.
  projectName: 'www.weiqi.kids', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'zh-tw',
    locales: [
      'zh-tw',
      'zh-cn',
      'zh-hk',
      'en',
      'ja',
      'ko',
      'es',
      'pt',
      'hi',
      'id',
      'ar',
    ],
    localeConfigs: {
      "zh-tw": {
        label: '繁體中文',
      },
      "zh-cn": {
        label: '简体中文',
      },
      "zh-hk": {
        label: '粵語（香港）',
      },
      "en": {
        label: 'English',
      },
      "ja": {
        label: '日本語',
      },
      "ko": {
        label: '한국어',
      },
      "es": {
        label: 'Español',
      },
      "pt": {
        label: 'Português',
      },
      "hi": {
        label: 'हिन्दी',
      },
      "id": {
        label: 'Bahasa Indonesia',
      },
      "ar": {
        label: 'العربية',
        direction: 'rtl',
      }
    }
  },

  // Mermaid 支援
  markdown: {
    mermaid: true,
  },
  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['zh', 'en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        indexDocs: true,
        indexBlog: false,
        indexPages: true,
        docsRouteBasePath: '/docs',
      },
    ],
  ],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // 社交分享卡片圖片
      image: 'img/social-card.png',
      metadata: [
        { name: 'keywords', content: '圍棋, Go, 好棋寶寶, AI, KataGo, AlphaGo, 圍棋教學, 圍棋入門' },
        { name: 'description', content: '台灣好棋寶寶協會官網 - 提供圍棋教學、AI 研究資源，推動圍棋文化發展' },
      ],
      navbar: {
        title: 'Weiqi.Kids',
        logo: {
          alt: 'My Site Logo',
          src: 'img/logo.svg',
        },
        items: [
          { to: '/docs/for-players', label: '圍棋棋友', position: 'left' },
          { to: '/docs/for-engineers', label: 'AI工程師', position: 'left' },
          { to: '/docs/evolution', label: '圍棋 AI 演進整理', position: 'left' },
          { to: '/docs/aboutus', label: '協會介紹', position: 'right' },
          {
            type: 'localeDropdown',
            position: 'right',
          },
          {
            href: 'https://github.com/weiqi-kids/www.weiqi.kids',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        copyright: `Copyright © ${new Date().getFullYear()} 台灣好棋寶寶協會. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
