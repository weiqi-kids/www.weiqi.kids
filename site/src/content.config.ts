import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// 所有由舊站轉寫或新寫的內容頁。網址由 `path` 決定（不依檔名）。
const pages = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/pages' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    path: z.string().regex(/^\/.*\/$/),
    kind: z.enum(['knowledge', 'association', 'member', 'partner', 'gathering']),
    order: z.number().optional(),
    keywords: z.array(z.string()).optional(),
    datePublished: z.coerce.date().optional(),
    dateModified: z.coerce.date().optional(),
    legacySource: z.string().optional(),
    sourceVerbatim: z.boolean().optional(),
    image: z.string().optional(),
    // 頁面首圖（吉祥物插畫，放在 public/media/illustrations/）
    hero: z.string().optional(),
    heroAlt: z.string().optional(),
    // 圖說：這張圖在說明的那件事（alt 負責描述畫面，兩者不重複）
    heroCaption: z.string().optional(),
  }),
});

// AI 共學營課程主題（不是正式課程）。
const topics = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/topics' }),
  schema: z.object({
    title: z.string(),
    slug: z.string(),
    summary: z.string(),
    order: z.number(),
    ttqsName: z.string(),
    audience: z.string(),
    caseSet: z.enum(['apps', 'monitoring', 'intel', 'research', 'none']).default('none'),
  }),
});

// 正式課程：講師申請、管理員審核並填寫 TTQS 表單後才建立。
const courses = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/courses' }),
  schema: z.object({
    title: z.string(),
    slug: z.string(),
    topic: z.string(),
    status: z.enum(['draft', 'open', 'running', 'closed']),
    instructors: z.array(z.string()),
    startDate: z.coerce.date(),
    endDate: z.coerce.date(),
    fee: z.number().default(6000),
    location: z.string(),
    sessions: z.array(z.object({ label: z.string(), date: z.coerce.date(), kind: z.enum(['teaching', 'practice']) })).length(4),
    summary: z.string(),
  }),
});

// 好棋寶寶棋聚單場活動。
const gatherings = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/gatherings' }),
  schema: z.object({
    title: z.string(),
    slug: z.string(),
    startDate: z.coerce.date(),
    endDate: z.coerce.date().optional(),
    location: z.string(),
    format: z.string(),
    signup: z.string(),
    summary: z.string(),
  }),
});

export const collections = { pages, topics, courses, gatherings };
