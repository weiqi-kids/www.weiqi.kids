import { getCollection, type CollectionEntry } from 'astro:content';

export type Page = CollectionEntry<'pages'>;

// 沒有對應內容頁的上層網址，用這些名稱組麵包屑。
export const SECTION_LABELS: Record<string, string> = {
  '/association/': '協會',
  '/association/members/': '成員',
  '/association/partners/': '合作夥伴',
  '/association/activities/': '協會活動',
  '/gatherings/': '好棋寶寶棋聚',
  '/gatherings/archive/': '歷史棋聚',
  '/camp/': 'AI 共學營',
  '/knowledge/': '公開知識',
};

let cache: Page[] | null = null;
export async function allPages(): Promise<Page[]> {
  cache ??= await getCollection('pages');
  return cache;
}

export function ancestors(path: string): string[] {
  const parts = path.split('/').filter(Boolean);
  const out: string[] = [];
  for (let i = 1; i < parts.length; i++) out.push(`/${parts.slice(0, i).join('/')}/`);
  return out;
}

export function parentPath(path: string, pages: Page[]): string | null {
  const known = new Set(pages.map((p) => p.data.path));
  const anc = ancestors(path).reverse();
  return anc.find((a) => known.has(a) || SECTION_LABELS[a]) ?? null;
}

export function breadcrumbsFor(path: string, title: string, pages: Page[]) {
  const byPath = new Map(pages.map((p) => [p.data.path, p.data.title]));
  const crumbs = ancestors(path)
    .filter((a) => byPath.has(a) || SECTION_LABELS[a])
    .map((a) => ({ name: SECTION_LABELS[a] ?? byPath.get(a)!, href: a }));
  return [...crumbs, { name: title, href: path }];
}

export function sortPages(list: Page[]) {
  return [...list].sort((a, b) => (a.data.order ?? 999) - (b.data.order ?? 999) || a.data.title.localeCompare(b.data.title, 'zh-Hant'));
}

export function childrenOf(path: string, pages: Page[]) {
  return sortPages(pages.filter((p) => p.data.path !== path && parentPath(p.data.path, pages) === path));
}
