import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import cloudflare from '@astrojs/cloudflare';
import remarkMath from 'remark-math';
import remarkDirective from 'remark-directive';
import rehypeKatex from 'rehype-katex';
import remarkCallouts from './src/lib/remark-callouts.mjs';

const PRIVATE = /\/(account|auth|admin)\/|\/forum\/|\/enroll\/|\/materials\//;

export default defineConfig({
  site: 'https://www.weiqi.kids',
  trailingSlash: 'always',
  build: { format: 'directory' },
  adapter: cloudflare({ imageService: 'passthrough', prerenderEnvironment: 'node' }),
  integrations: [
    mdx(),
    react(),
    sitemap({ filter: (page) => page.endsWith('/') && !PRIVATE.test(page) }),
  ],
  markdown: {
    remarkPlugins: [remarkMath, remarkDirective, remarkCallouts],
    rehypePlugins: [rehypeKatex],
    syntaxHighlight: { type: 'shiki', excludeLangs: ['mermaid', 'math'] },
    shikiConfig: { theme: 'github-light' },
  },
});
