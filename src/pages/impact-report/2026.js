import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import Translate, {translate} from '@docusaurus/Translate';
import {industries} from '@site/src/data/links/industries';
import {researchPapers} from '@site/src/data/links/research';
import {appLinks} from '@site/src/data/links/apps';
import {intelLinks} from '@site/src/data/links/intel';
import styles from './2026.module.css';

// 把全部 41 個 AI 工具/論文/供應鏈合併，按 industry_tag 分組
function allProjects() {
  return [
    ...researchPapers.map((p) => ({...p, source: 'research'})),
    ...appLinks.map((p) => ({...p, source: 'apps'})),
    ...intelLinks.map((p) => ({...p, source: 'intel'})),
  ];
}

// 活動成果（來自 ref/作品集，聚合數字，不引用個別檔案）
const activities = [
  {
    id: 'carnival',
    icon: '🎪',
    titleKey: 'impact.act.carnival.title',
    titleDefault: '圍棋嘉年華',
    descKey: 'impact.act.carnival.desc',
    descDefault: '大型線下圍棋活動，於和平高中舉辦，含場刊製作、場地租借、活動照片紀錄。',
    metric: '200+',
    metricLabelKey: 'impact.act.photos',
    metricLabelDefault: '影像紀錄',
  },
  {
    id: 'city-match',
    icon: '🏆',
    titleKey: 'impact.act.citymatch.title',
    titleDefault: '城市網路對抗賽',
    descKey: 'impact.act.citymatch.desc',
    descDefault: '2024 春季場：跨 10 縣市的線上圍棋對抗賽，含對弈成績與參賽隊伍介紹。',
    metric: '10',
    metricLabelKey: 'impact.act.cities',
    metricLabelDefault: '縣市參賽',
  },
  {
    id: 'tournament',
    icon: '🥇',
    titleKey: 'impact.act.tournament.title',
    titleDefault: '113 年全國圍棋公開賽',
    descKey: 'impact.act.tournament.desc',
    descDefault: '與臺北市體育總會合辦全國圍棋公開賽，含側拍花絮、活動照片、清水場次紀錄。',
    metric: '303',
    metricLabelKey: 'impact.act.records',
    metricLabelDefault: '紀錄檔案',
  },
  {
    id: 'fun-club',
    icon: '☕',
    titleKey: 'impact.act.funclub.title',
    titleDefault: '好棋同樂會',
    descKey: 'impact.act.funclub.desc',
    descDefault: '不定期主題講座與棋聚：蔡丞韋職業五段、金工微距、健康領域熊赫囑廚，以及隨食旅人棋友會。',
    metric: '4+',
    metricLabelKey: 'impact.act.sessions',
    metricLabelDefault: '主題講座',
  },
  {
    id: 'promo-video',
    icon: '🎬',
    titleKey: 'impact.act.promo.title',
    titleDefault: '圍棋推廣影片',
    descKey: 'impact.act.promo.desc',
    descDefault: '完整影片製作：音樂、成品分享、製作素材、側拍花絮。',
    metric: '21',
    metricLabelKey: 'impact.act.assets',
    metricLabelDefault: '影片素材',
  },
  {
    id: 'shorts',
    icon: '📱',
    titleKey: 'impact.act.shorts.title',
    titleDefault: '圍棋教育短影音',
    descKey: 'impact.act.shorts.desc',
    descDefault: '12 部短影音系列：圍棋教我的事、定不下來的孩子、坐姿不良輸家無常、棋如人生等。',
    metric: '12',
    metricLabelKey: 'impact.act.shortcount',
    metricLabelDefault: '部短影音',
  },
];

function StatCard({number, labelKey, labelDefault}) {
  return (
    <div className={styles.statCard}>
      <div className={styles.statNumber}>{number}</div>
      <div className={styles.statLabel}>
        <Translate id={labelKey}>{labelDefault}</Translate>
      </div>
    </div>
  );
}

function IndustryColumn({industry, projects}) {
  return (
    <div className={styles.industryColumn}>
      <div className={styles.industryHeader}>
        <span className={styles.industryIcon}>{industry.icon}</span>
        <Heading as="h3" className={styles.industryTitle}>
          <Translate id={industry.titleKey}>{industry.titleDefault}</Translate>
        </Heading>
        <span className={styles.industryCount}>{projects.length}</span>
      </div>
      <p className={styles.industryDesc}>
        <Translate id={industry.descKey}>{industry.descDefault}</Translate>
      </p>
      <ul className={styles.industryList}>
        {projects.slice(0, 5).map((p) => (
          <li key={p.id}>
            <a href={p.href} target="_blank" rel="noopener noreferrer">
              {p.titleDefault}
            </a>
          </li>
        ))}
        {projects.length > 5 && (
          <li className={styles.industryMore}>
            <Translate
              id="impact.industry.more"
              values={{count: projects.length - 5}}>
              {'… 及其餘 {count} 個'}
            </Translate>
          </li>
        )}
      </ul>
    </div>
  );
}

function ActivityCard({a}) {
  return (
    <div className={styles.actCard}>
      <div className={styles.actHeader}>
        <span className={styles.actIcon}>{a.icon}</span>
        <Heading as="h3" className={styles.actTitle}>
          <Translate id={a.titleKey}>{a.titleDefault}</Translate>
        </Heading>
      </div>
      <p className={styles.actDesc}>
        <Translate id={a.descKey}>{a.descDefault}</Translate>
      </p>
      <div className={styles.actMetric}>
        <span className={styles.actMetricNumber}>{a.metric}</span>
        <span className={styles.actMetricLabel}>
          <Translate id={a.metricLabelKey}>{a.metricLabelDefault}</Translate>
        </span>
      </div>
    </div>
  );
}

export default function ImpactReport2026() {
  const all = allProjects();
  const byIndustry = industries
    .map((ind) => ({
      industry: ind,
      projects: all.filter((p) => p.industry_tag === ind.id),
    }))
    .filter((g) => g.projects.length > 0);

  return (
    <Layout
      title={translate({
        id: 'impact.meta.title',
        message: '2026 年度開源成果報告 — 好棋寶寶協會',
      })}
      description={translate({
        id: 'impact.meta.description',
        message:
          '41+ 開源公益專案、67 位跨域夥伴、6 大線下活動成果、11 語系覆蓋。2026 年度完整影響力報告，CC-BY 4.0 全部公開。',
      })}>
      <header className={styles.hero}>
        <div className="container">
          <div className={styles.heroBadge}>
            <Translate id="impact.hero.badge">2026 年度報告</Translate>
          </div>
          <Heading as="h1" className={styles.heroTitle}>
            <Translate id="impact.hero.title">開源成果年度報告</Translate>
          </Heading>
          <p className={styles.heroSubtitle}>
            <Translate id="impact.hero.subtitle">
              台灣好棋寶寶協會 ／ 67 位跨域夥伴 × 41+ 開源公益專案 × 6 大線下活動
            </Translate>
          </p>
          <p className={styles.heroQuote}>
            <Translate id="impact.hero.quote">
              「夥伴提供領域知識、理事長負責 AI 整合，產出 CC-BY 4.0 公益專案，永久公開。」
            </Translate>
          </p>
        </div>
      </header>

      <main>
        {/* 總體影響數字 */}
        <section className={styles.section}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <Translate id="impact.numbers.title">一年總體影響</Translate>
            </Heading>
            <div className={styles.statGrid}>
              <StatCard
                number="67"
                labelKey="impact.numbers.partners"
                labelDefault="位跨域夥伴"
              />
              <StatCard
                number={researchPapers.length}
                labelKey="impact.numbers.papers"
                labelDefault="篇學術論文"
              />
              <StatCard
                number={appLinks.length}
                labelKey="impact.numbers.apps"
                labelDefault="個 AI 工具"
              />
              <StatCard
                number={intelLinks.length}
                labelKey="impact.numbers.intel"
                labelDefault="條供應鏈情報"
              />
              <StatCard
                number={industries.length}
                labelKey="impact.numbers.industries"
                labelDefault="個產業分類"
              />
              <StatCard
                number="11"
                labelKey="impact.numbers.locales"
                labelDefault="語系覆蓋"
              />
              <StatCard
                number="6"
                labelKey="impact.numbers.activities"
                labelDefault="大線下活動成果"
              />
              <StatCard
                number="CC-BY 4.0"
                labelKey="impact.numbers.license"
                labelDefault="完全免費開源"
              />
            </div>
          </div>
        </section>

        {/* 6 個產業分類 + 旗艦案例 */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <Translate id="impact.industries.title">7 大產業分類</Translate>
            </Heading>
            <p className={styles.sectionDesc}>
              <Translate id="impact.industries.desc">
                每個專案都標註對應產業，讓商界夥伴可以快速找到「跟我同產業」的工具與情報。
              </Translate>
            </p>
            <div className={styles.industryGrid}>
              {byIndustry.map(({industry, projects}) => (
                <IndustryColumn
                  key={industry.id}
                  industry={industry}
                  projects={projects}
                />
              ))}
            </div>
          </div>
        </section>

        {/* 6 大線下活動成果 */}
        <section className={styles.section}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <Translate id="impact.activities.title">6 大活動成果</Translate>
            </Heading>
            <p className={styles.sectionDesc}>
              <Translate id="impact.activities.desc">
                除了開源 AI 工具，協會也持續舉辦圍棋線下活動與推廣影片製作。以下是 2023-2024 的活動成果。
              </Translate>
            </p>
            <div className={styles.actGrid}>
              {activities.map((a) => (
                <ActivityCard key={a.id} a={a} />
              ))}
            </div>
          </div>
        </section>

        {/* 67 位夥伴 */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <Translate id="impact.partners.title">67 位夥伴的跨域組合</Translate>
            </Heading>
            <p className={styles.sectionDesc}>
              <Translate id="impact.partners.desc">
                30 位協會召集人 + 2 位後續加入會員 + 5 位獨立合作夥伴 + 30 位 BNI 大台中榮耀分會夥伴。
                跨產業組合涵蓋律師、會計師、ISO 顧問、中醫、整合醫學、整復、行銷、活動企劃、音樂、IP 設計、保險、財富管理、不動產、教育、職業棋士等。
              </Translate>
            </p>
            <div className={styles.partnersCta}>
              <Link to="/docs/about/members/founding/" className={styles.partnersLink}>
                <Translate id="impact.partners.cta">查看完整 67 位夥伴 →</Translate>
              </Link>
            </div>
          </div>
        </section>

        {/* 引用格式 */}
        <section className={styles.section}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              <Translate id="impact.cite.title">媒體與學術引用</Translate>
            </Heading>
            <p className={styles.sectionDesc}>
              <Translate id="impact.cite.desc">
                本報告所有專案均採 CC-BY 4.0 授權，歡迎媒體報導、學術論文引用、教育教材使用。
              </Translate>
            </p>
            <div className={styles.citeBox}>
              <div className={styles.citeLabel}>
                <Translate id="impact.cite.format">建議引用格式：</Translate>
              </div>
              <code className={styles.citeText}>
                台灣好棋寶寶協會 (2026). 開源成果年度報告. weiqi.kids.
                Retrieved from https://www.weiqi.kids/impact-report/2026/
              </code>
            </div>
            <div className={styles.citeBox}>
              <div className={styles.citeLabel}>
                BibTeX：
              </div>
              <pre className={styles.citeText}>{`@misc{weiqikids2026,
  author = {Taiwan Good Go Baby Association},
  title  = {2026 Open-Source Impact Report},
  year   = {2026},
  url    = {https://www.weiqi.kids/impact-report/2026/},
  note   = {CC-BY 4.0}
}`}</pre>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className={styles.sectionAlt}>
          <div className="container">
            <div className={styles.ctaWrap}>
              <Heading as="h2" className={styles.ctaTitle}>
                <Translate id="impact.cta.title">想成為下一個案例？</Translate>
              </Heading>
              <p className={styles.ctaDesc}>
                <Translate id="impact.cta.desc">
                  你貢獻領域知識、理事長負責 AI 整合，把你的專業變成可被社會使用的開源工具。完全免費、無會員費。
                </Translate>
              </p>
              <div className={styles.ctaButtons}>
                <a
                  href="mailto:lightman.chang@gmail.com?subject=%E3%80%90%E5%90%88%E4%BD%9C%E6%8F%90%E6%A1%88%E3%80%91"
                  className={styles.ctaPrimary}>
                  <Translate id="impact.cta.proposal">合作提案</Translate>
                </a>
                <a
                  href="mailto:lightman.chang@gmail.com?subject=%E3%80%90%E5%85%88%E8%81%8A%E8%81%8A%E3%80%91"
                  className={styles.ctaSecondary}>
                  <Translate id="impact.cta.meet">先見面聊聊</Translate>
                </a>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
