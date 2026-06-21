// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: '好棋寶寶協會 | Weiqi.Kids',
  tagline: '台灣好棋寶寶協會｜致力於圍棋文化前進的推手',
  favicon: 'img/favicon.ico',

  // Umami Analytics（共用 weiqikids Umami 實例）
  // Website ID 對應 www.weiqi.kids，見 /root/CLAUDE.md「Umami Website IDs」
  scripts: [
    {
      src: 'https://analytics.weiqi.kids/script.js',
      defer: true,
      'data-website-id': 'a4b64b22-906d-4918-934a-7da72b5aced9',
    },
  ],

  // KaTeX 數學公式樣式
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  // SEO 全域設定
  headTags: [
    // Open Graph 基本標籤
    // 註：og:type 不在此全域寫死；文章頁由 ArticleSchema 設為 article，
    // 其餘頁面預設（不輸出 og:type，等同 website），避免重複標籤。
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
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
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
          editUrl: 'https://github.com/weiqi-kids/www.weiqi.kids/tree/main/',
          editLocalizedFiles: true,
          showLastUpdateTime: true,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        // Google Analytics 4（官方 gtag 外掛，自動處理 SPA 換頁追蹤）
        gtag: {
          trackingID: 'G-16V1KSEH6W',
          anonymizeIP: true,
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
        { name: 'description', content: '11 位來自律師、ISO、技術、醫療等領域的夥伴，把產業 know-how 變成 41+ 個全部免費開源的 AI 工具。歡迎跨域合作。' },
      ],
      navbar: {
        title: 'Weiqi.Kids',
        logo: {
          alt: 'My Site Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'dropdown',
            label: '夥伴',
            position: 'left',
            items: [
              { to: '/docs/about/members/founding/', label: '認識夥伴' },
              { to: '/impact-report/2026/', label: '年度成果報告' },
              { to: '/docs/about', label: '關於協會' },
              { href: 'mailto:lightman.chang@gmail.com?subject=%E3%80%90%E5%90%88%E4%BD%9C%E6%8F%90%E6%A1%88%E3%80%91', label: '合作提案' },
            ],
          },
          { to: '/apps/', label: 'AI 工具', position: 'left' },
          { to: '/research/', label: '論文', position: 'left' },
          { to: '/intel/', label: '產業情報', position: 'left' },
          {
            type: 'dropdown',
            label: '圍棋資源',
            position: 'left',
            items: [
              { to: '/docs/learn', label: '學圍棋' },
              { to: '/docs/alphago', label: 'AlphaGo' },
              { to: '/docs/animations', label: '動畫教室' },
              { to: '/docs/tech', label: '技術文件' },
            ],
          },
          {
            type: 'localeDropdown',
            position: 'right',
            className: 'navbar__item--compact',
          },
          {
            href: 'https://github.com/weiqi-kids/www.weiqi.kids',
            'aria-label': 'GitHub',
            position: 'right',
            className: 'header-github-link navbar__item--compact',
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
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
        respectPrefersColorScheme: false,
      },
    }),
};

export default config;
