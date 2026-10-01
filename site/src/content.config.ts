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
    hero: z.string().optional(),
    heroAlt: z.string().optional(),
    heroCaption: z.string().optional(),
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

// 最新消息：協會公告與紀事，日期新的排前面。
const news = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/news' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    summary: z.string(),
    // 這則消息的依據頁面或外部連結，讓讀者可以查證
    source: z.string().optional(),
    sourceLabel: z.string().optional(),
  }),
});


// 圍棋教學影片索引：收錄其他老師的教學影片，導流到他們的頻道協助招生。
// 一位老師一個檔案，影片放在 videos 陣列裡，不另開集合，維護才輕。
const teachers = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/teachers' }),
  schema: z.object({
    name: z.string(),
    slug: z.string(),
    // 棋力：照老師自己公開的寫法，例如「職業五段」「業餘六段」
    rank: z.string(),
    // 所在地：招生用，縣市必填、行政區選填
    city: z.string(),
    district: z.string().optional(),
    channelUrl: z.string().url(),
    channelName: z.string().optional(),
    summary: z.string(),
    // 授課方式與是否招生中，決定老師頁要不要顯示招生區塊
    teaching: z.array(z.enum(['實體', '線上'])).default([]),
    recruiting: z.boolean().default(false),
    contact: z.string().optional(),
    // 收錄同意：沒有同意日期就不會顯示在正式站
    consentDate: z.coerce.date().optional(),
    status: z.enum(['draft', 'listed']).default('draft'),
    videos: z.array(z.object({
      id: z.string(),
      title: z.string(),
      level: z.enum(['入門', '級位', '段位']),
      topics: z.array(z.enum(['規則入門', '佈局', '定石', '死活', '手筋', '收官', '實戰覆盤', '棋理'])).min(1),
      duration: z.number().optional(),
    })).default([]),
  }),
});

export const collections = { pages, topics, courses, gatherings, news, teachers };
