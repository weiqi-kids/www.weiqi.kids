// 一次性轉寫：把舊 Docusaurus 站（repo 根目錄 docs/）的文章轉成新站 src/content/pages/。
// 依 data/legacy-map.json 決定哪些檔案建立新頁、放在哪個網址。
// 只做語法與連結轉換，不改寫正文；轉出的檔案標 sourceVerbatim: true。
// 用法：node scripts/migrate-legacy.mjs（在 site/ 目錄執行；會覆寫 src/content/pages/ 下的轉寫檔）
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join, posix } from 'node:path';
import { execSync } from 'node:child_process';

const REPO = join(process.cwd(), '..');
const OUT = join(process.cwd(), 'src/content/pages');
const map = JSON.parse(readFileSync('data/legacy-map.json', 'utf8'));
const LEGACY_TAG = 'legacy-docusaurus-final';
const PDF_BASE = `https://raw.githubusercontent.com/weiqi-kids/www.weiqi.kids/${LEGACY_TAG}/static`;

// 舊檔案路徑 → 新網址；舊 route → 新網址
const bySrc = new Map(map.filter((e) => e.src).map((e) => [e.src, e.redirect]));
const byRoute = new Map(map.map((e) => [e.old, e.redirect]));
const unresolved = [];

const CHARTS = ['EloChart', 'GoBoard', 'MCTSTree', 'PolicyHeatmap'];
const MDX_COMPONENTS = { KeyTakeaway: 1, FAQ: 1, ExpertQuote: 1, ProfileHero: 1, ProfileWorks: 1 };

function resolveDocTarget(fromSrc, target) {
  // target：去掉 #anchor 與 query 後的連結路徑
  let file;
  if (target.startsWith('/docs/')) {
    const route = target.endsWith('/') ? target : `${target}/`;
    if (byRoute.has(route)) return byRoute.get(route);
    file = `docs/${target.slice(6)}`;
  } else {
    file = posix.normalize(posix.join(posix.dirname(fromSrc), target));
  }
  file = file.replace(/\/$/, '');
  const candidates = [
    file, `${file}.md`, `${file}.mdx`, `${file}/index.md`, `${file}/index.mdx`,
  ];
  // Docusaurus 會去掉檔名的數字前綴；連結可能寫有或沒寫前綴
  for (const c of candidates) if (bySrc.has(c)) return bySrc.get(c);
  const dir = posix.dirname(file);
  const base = posix.basename(file).replace(/\.mdx?$/, '');
  for (const src of bySrc.keys()) {
    if (posix.dirname(src) === dir && posix.basename(src).replace(/\.mdx?$/, '').replace(/^\d+-/, '') === base) return bySrc.get(src);
  }
  const route = `/docs/${file.replace(/^docs\//, '').replace(/\.mdx?$/, '').replace(/\/index$/, '')}/`;
  if (byRoute.has(route)) return byRoute.get(route);
  return null;
}

function rewriteLinks(src, body) {
  return body.replace(/\]\(([^)\s]+)(\s+"[^"]*")?\)/g, (all, url, title = '') => {
    if (/^(https?:|mailto:|tel:|#)/.test(url)) return all;
    if (url.startsWith('/pdf/')) return `](${PDF_BASE}${url}${title})`;
    if (url.startsWith('/img/')) return all;
    const [, path, hash = ''] = url.match(/^([^#?]*)(.*)$/);
    if (!path) return all;
    if (/\.(png|jpe?g|gif|svg|webp)$/i.test(path)) return all;
    // 舊站連結有的以檔案目錄為基準，有的以頁面網址為基準（網址結尾有 /），兩種都試
    const oldRoute = map.find((m) => m.src === src)?.old;
    const target = resolveDocTarget(src, path)
      ?? (oldRoute && !path.startsWith('/') ? resolveDocTarget(src, `/docs/${posix.join(oldRoute.slice(6), path).replace(/^\/+/, '')}`) : null);
    if (!target) { unresolved.push(`${src}: ${url}`); return all; }
    return `](${target}${hash}${title})`;
  });
}

function stripArticleSchema(body, fm) {
  return body.replace(/<ArticleSchema[\s\S]*?\/>\s*/g, (block) => {
    const pub = block.match(/datePublished="([^"]+)"/);
    const mod = block.match(/dateModified="([^"]+)"/);
    const kw = block.match(/keywords=\{\[([^\]]*)\]\}/);
    if (pub) fm.datePublished = pub[1];
    if (mod) fm.dateModified = mod[1];
    if (kw) fm.keywords = [...kw[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
    return '';
  });
}

function gitDates(src) {
  try {
    const out = execSync(`git log --follow --format=%as -- "${src}"`, { cwd: REPO }).toString().trim().split('\n').filter(Boolean);
    return { first: out.at(-1), last: out[0] };
  } catch { return {}; }
}

function parseFrontmatter(raw) {
  const m = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!m) return [{}, raw];
  const fm = {};
  for (const line of m[1].split('\n')) {
    const kv = line.match(/^([A-Za-z_]+):\s*(.*)$/);
    if (!kv) continue;
    let v = kv[2].trim();
    if (/^".*"$|^'.*'$/.test(v)) v = v.slice(1, -1);
    fm[kv[1]] = v;
  }
  return [fm, raw.slice(m[0].length)];
}

function yaml(v) {
  if (Array.isArray(v)) return `[${v.map((x) => JSON.stringify(x)).join(', ')}]`;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return JSON.stringify(v);
}

let written = 0;
for (const e of map) {
  if (!e.create) continue;
  const raw = readFileSync(join(REPO, e.src), 'utf8');
  const isMdx = e.src.endsWith('.mdx') || /<(KeyTakeaway|FAQ|ExpertQuote|ProfileHero|ProfileWorks|EloChart|GoBoard|MCTSTree|PolicyHeatmap)\b/.test(raw);
  let [legacyFm, body] = parseFrontmatter(raw);
  const fm = {
    title: legacyFm.title || e.title,
    ...(legacyFm.description ? { description: legacyFm.description } : {}),
    path: e.redirect,
    kind: e.kind,
    ...(legacyFm.sidebar_position ? { order: Number(legacyFm.sidebar_position) } : {}),
  };

  // 舊站 SEO 元件與 import 由新站 layout 處理
  body = body.replace(/^import .* from ['"]@site\/.*['"];?\s*$/gm, '');
  body = stripArticleSchema(body, fm);
  body = body.replace(/<PersonSchema[\s\S]*?\/>\s*/g, '');
  // 第一個 H1 由 layout 輸出
  body = body.replace(/^\s*# .+\n/, '');
  // Docusaurus 的 :::info 標題（空白分隔的舊寫法）→ :::info[標題]
  body = body.replace(/^:::(note|tip|info|caution|warning|danger)[ \t]+(.+)$/gm, ':::$1[$2]');
  if (!isMdx) body = body.replace(/className=/g, 'class=');
  body = rewriteLinks(e.src, body);

  if (isMdx) {
    const used = [...new Set([...body.matchAll(/<([A-Z][A-Za-z]+)\b/g)].map((m) => m[1]))];
    const imports = [];
    for (const name of used) {
      if (CHARTS.includes(name)) {
        imports.push(`import ${name} from '~/components/charts/${name}.jsx';`);
        body = body.replace(new RegExp(`<${name}\\b(?![^>]*client:)`, 'g'), `<${name} client:visible`);
      } else if (MDX_COMPONENTS[name]) {
        imports.push(`import ${name} from '~/components/mdx/${name}.astro';`);
      }
    }
    if (imports.length) body = `${imports.join('\n')}\n\n${body.trimStart()}`;
  }

  const dates = gitDates(e.src);
  fm.datePublished ??= dates.first;
  fm.dateModified ??= dates.last;
  if (!fm.datePublished) delete fm.datePublished;
  if (!fm.dateModified) delete fm.dateModified;
  fm.legacySource = e.src;
  fm.sourceVerbatim = true;

  const rel = e.redirect.replace(/^\/|\/$/g, '') || 'index';
  const file = join(OUT, `${rel}.${isMdx ? 'mdx' : 'md'}`);
  mkdirSync(dirname(file), { recursive: true });
  const head = Object.entries(fm).map(([k, v]) => `${k}: ${yaml(v)}`).join('\n');
  writeFileSync(file, `---\n${head}\n---\n\n${body.trim()}\n`);
  written++;
}

console.log(`轉寫 ${written} 篇。`);
if (unresolved.length) {
  console.log(`無法對應的站內連結 ${unresolved.length} 個：`);
  for (const u of unresolved) console.log(`  ${u}`);
}
