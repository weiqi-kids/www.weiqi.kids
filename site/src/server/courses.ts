import { getCollection, type CollectionEntry } from 'astro:content';
import { env } from 'cloudflare:workers';

export type Course = CollectionEntry<'courses'>;

// 執行期可見的課程：staging 設 SHOW_DRAFT_COURSES=1 時包含草稿課程。
export async function runtimeCourses(): Promise<Course[]> {
  const all = await getCollection('courses');
  return all.filter((c) => c.data.status !== 'draft' || env.SHOW_DRAFT_COURSES === '1');
}

export async function findCourse(slug: string) {
  return (await runtimeCourses()).find((c) => c.data.slug === slug) ?? null;
}
