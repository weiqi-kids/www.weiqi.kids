// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Weiqi.Kids',
  tagline: '台灣好棋寶寶協會｜致力於圍棋文化前進的推手',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://www.weiqi.kids',
  baseUrl: '/',

  // GitHub pages deployment config.
  organizationName: 'weiqi.kids', // Usually your GitHub org/user name.
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
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
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
        links: [
          {
            title: '關於協會',
            items: [
              {
                label: '快速了解',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: '圍棋資源',
            items: [
            ],
          },
          {
            title: '友商介紹',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/facebook/docusaurus',
              },
            ],
          },
          {
            title: '特別感謝',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/facebook/docusaurus',
              },
            ],
          }
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Weiqi.Kids. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
