import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import Translate, {translate} from '@docusaurus/Translate';

import IntelCard from '@site/src/components/IntelCard';
import PaperCard from '@site/src/components/PaperCard';
import AppCard from '@site/src/components/AppCard';

import {officialLinks} from '@site/src/data/links/official';
import {researchPapers} from '@site/src/data/links/research';
import {intelLinks} from '@site/src/data/links/intel';
import {appLinks} from '@site/src/data/links/apps';
import {friendSites} from '@site/src/data/links/friends';
import {founders} from '@site/src/data/members';
import {activities} from '@site/src/data/activities';

import styles from './styles.module.css';

const appIcons = {
  ecommerce: '🛒', supplement: '💊', education: '📚', job: '💼',
  security: '🛡️', health: '🦠', law: '⚖️', trade: '🌐',
  listening: '📊', policy: '📜',
};

const officialIcons = { status: '🟢', social: '💬', video: '🎬' };

// 從 intelLinks 挑選首頁精選 6 條（橫跨大類別）
const intelHighlightIds = ['memory', 'auto', 'solar', 'housing', 'pharma', 'defense'];
const intelHighlights = intelHighlightIds
  .map((id) => intelLinks.find((s) => s.id === id))
  .filter(Boolean);

// 論文全部 5 篇（量少全顯）
const paperHighlights = researchPapers;

// AI 工具精選 6 個
const appHighlightIds = ['ecommerce', 'education', 'security', 'health', 'listening', 'policy'];
const appHighlights = appHighlightIds
  .map((id) => appLinks.find((a) => a.id === id))
  .filter(Boolean);

// ─── USP 區塊：協會三軸並立（圍棋公益／跨域開源／學術原創）───
function UspAxisCard({icon, titleKey, titleDefault, metric, metricLabelKey, metricLabelDefault, bodyKey, bodyDefault, audienceKey, audienceDefault, ctaKey, ctaDefault, ctaHref}) {
  return (
    <Link to={ctaHref} className={styles.uspAxis}>
      <div className={styles.uspAxisIcon}>{icon}</div>
      <Heading as="h3" className={styles.uspAxisTitle}>
        <Translate id={titleKey}>{titleDefault}</Translate>
      </Heading>
      <div className={styles.uspAxisMetric}>
        <span className={styles.uspAxisMetricNumber}>{metric}</span>
        <span className={styles.uspAxisMetricLabel}>
          <Translate id={metricLabelKey}>{metricLabelDefault}</Translate>
        </span>
      </div>
      <p className={styles.uspAxisBody}>
        <Translate id={bodyKey}>{bodyDefault}</Translate>
      </p>
      <div className={styles.uspAxisAudience}>
        <Translate id={audienceKey}>{audienceDefault}</Translate>
      </div>
      <div className={styles.uspAxisCta}>
        <Translate id={ctaKey}>{ctaDefault}</Translate>
      </div>
    </Link>
  );
}

function USPSection() {
  return (
    <section className={clsx(styles.section, styles.uspSection)}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">
            <Translate id="homepage.usp.title">圍棋文化前進的推手</Translate>
          </Heading>
          <p className={styles.sectionDescription}>
            <Translate id="homepage.usp.lead">
              不只推圍棋。把圍棋人從 AlphaGo 學到的 AI 適應力，帶到每一個跨域場域。
            </Translate>
          </p>
        </div>
        <div className={styles.uspGrid}>
          <UspAxisCard
            icon="⚫"
            titleKey="homepage.usp.axis1.title"
            titleDefault="圍棋公益"
            metric="10"
            metricLabelKey="homepage.usp.axis1.metricLabel"
            metricLabelDefault="縣市・187 張紀錄"
            bodyKey="homepage.usp.axis1.body"
            bodyDefault="城市對抗賽、嘉年華、好棋同樂會主題講座、全國公開賽"
            audienceKey="homepage.usp.axis1.audience"
            audienceDefault="樂齡・聽障・親子・學齡兒童"
            ctaKey="homepage.usp.axis1.cta"
            ctaDefault="看實體活動 →"
            ctaHref="/docs/about/activities/"
          />
          <UspAxisCard
            icon="🤖"
            titleKey="homepage.usp.axis2.title"
            titleDefault="跨域開源"
            metric="41+"
            metricLabelKey="homepage.usp.axis2.metricLabel"
            metricLabelDefault="AI 工具・11 大產業"
            bodyKey="homepage.usp.axis2.body"
            bodyDefault="夥伴提供領域 know-how，理事長負責 AI 整合，產出全部 CC-BY 4.0"
            audienceKey="homepage.usp.axis2.audience"
            audienceDefault="消費・健康・資安・貿易・政策"
            ctaKey="homepage.usp.axis2.cta"
            ctaDefault="看開源工具 →"
            ctaHref="/apps/"
          />
          <UspAxisCard
            icon="📐"
            titleKey="homepage.usp.axis3.title"
            titleDefault="學術原創"
            metric="8"
            metricLabelKey="homepage.usp.axis3.metricLabel"
            metricLabelDefault="篇論文・含完整證明"
            bodyKey="homepage.usp.axis3.body"
            bodyDefault="圍棋次佳手公式、官子計算、SGD 收斂、深度泛化、AI 對齊、Collatz、四色定理、Erdős-Sidon"
            audienceKey="homepage.usp.axis3.audience"
            audienceDefault="圍棋・機器學習・AI 安全・數學"
            ctaKey="homepage.usp.axis3.cta"
            ctaDefault="看論文 →"
            ctaHref="/research/"
          />
        </div>
        <p className={styles.uspFuture}>
          <Translate id="homepage.usp.future">
            下一站：前進社區，邀請長輩帶孫子下圍棋（2026 啟動）
          </Translate>
        </p>
      </div>
    </section>
  );
}

// ─── 實體活動成果展示（5 個代表活動）───
function ActivityCard({activity}) {
  return (
    <div className={styles.activityCard}>
      {activity.image ? (
        <img src={activity.image} alt={activity.titleDefault} className={styles.activityImage} />
      ) : (
        <div className={styles.activityIcon}>{activity.icon || '📌'}</div>
      )}
      <div className={styles.activityBody}>
        <Heading as="h3" className={styles.activityTitle}>
          <Translate id={activity.titleKey}>{activity.titleDefault}</Translate>
        </Heading>
        <div className={styles.activityMetricRow}>
          <span className={styles.activityMetric}>{activity.metric}</span>
          <span className={styles.activityMetricLabel}>
            <Translate id={activity.metricLabelKey}>{activity.metricLabelDefault}</Translate>
          </span>
        </div>
        <p className={styles.activityDesc}>
          <Translate id={activity.descKey}>{activity.descDefault}</Translate>
        </p>
        <div className={styles.activityAudience}>
          <Translate id={activity.audienceKey}>{activity.audienceDefault}</Translate>
        </div>
      </div>
    </div>
  );
}

function ActivitiesShowcase() {
  return (
    <section className={clsx(styles.section, styles.sectionAlt)}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">
            <Translate id="homepage.act.title">實體活動成果</Translate>
          </Heading>
          <p className={styles.sectionDescription}>
            <Translate id="homepage.act.desc">
              協會 2023-2024 已辦理大量圍棋公益活動，從樂齡到親子、從台北到金門，
              累積完整的活動紀錄、影片資產與媒體素材。
            </Translate>
          </p>
        </div>
        <div className={styles.activitiesGrid}>
          {activities.map((a) => (
            <ActivityCard key={a.id} activity={a} />
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── 三大支柱入口卡 ───
function PillarCard({to, icon, titleKey, titleDefault, descKey, descDefault, count, countLabel}) {
  return (
    <Link to={to} className={styles.pillarCard}>
      <div className={styles.pillarIcon}>{icon}</div>
      <Heading as="h3" className={styles.pillarTitle}>
        <Translate id={titleKey}>{titleDefault}</Translate>
      </Heading>
      <p className={styles.pillarDesc}>
        <Translate id={descKey}>{descDefault}</Translate>
      </p>
      <div className={styles.pillarStat}>
        <span className={styles.pillarCount}>{count}</span>
        <span className={styles.pillarCountLabel}>{countLabel}</span>
      </div>
      <div className={styles.pillarCta}>
        <Translate id="homepage.pillar.cta">查看全部 →</Translate>
      </div>
    </Link>
  );
}

function PillarSection() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">
            <Translate id="homepage.pillars.title">深入三大產出</Translate>
          </Heading>
          <p className={styles.sectionDescription}>
            <Translate id="homepage.pillars.desc">
              學術論文・產業情報・AI 工具——每篇都做完整根因分析，每個都全部開源 CC-BY 4.0。
            </Translate>
          </p>
        </div>
        <div className={styles.pillarGrid}>
          <PillarCard
            to="/research/"
            icon="📐"
            titleKey="homepage.pillar.research.title"
            titleDefault="學術研究"
            descKey="homepage.pillar.research.desc"
            descDefault="圍棋理論、機器學習、數學研究，每篇含互動視覺化與完整證明。"
            count={researchPapers.length}
            countLabel={translate({id: 'homepage.pillar.research.count', message: '篇論文'})}
          />
          <PillarCard
            to="/intel/"
            icon="📊"
            titleKey="homepage.pillar.intel.title"
            titleDefault="產業供應鏈情報"
            descKey="homepage.pillar.intel.desc"
            descDefault="23 條供應鏈、上中下游公司、每日自動更新，對標 ETF 基線。"
            count={intelLinks.length}
            countLabel={translate({id: 'homepage.pillar.intel.count', message: '條供應鏈'})}
          />
          <PillarCard
            to="/apps/"
            icon="🤖"
            titleKey="homepage.pillar.apps.title"
            titleDefault="AI 應用工具"
            descKey="homepage.pillar.apps.desc"
            descDefault="消費、健康、教育、資安、貿易等 AI 工具，免費公開供大眾使用。"
            count={appLinks.length}
            countLabel={translate({id: 'homepage.pillar.apps.count', message: '個工具'})}
          />
        </div>
      </div>
    </section>
  );
}

// ─── 通用區塊 ───
function HighlightSection({titleKey, titleDefault, descKey, descDefault, allHref, allLabel, children, columns = 3, alt = false}) {
  return (
    <section className={clsx(styles.section, alt && styles.sectionAlt)}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">
            <Translate id={titleKey}>{titleDefault}</Translate>
          </Heading>
          <p className={styles.sectionDescription}>
            <Translate id={descKey}>{descDefault}</Translate>
          </p>
        </div>
        <div className={clsx(styles.cardGrid, columns === 2 && styles.cardGridTwo)}>
          {children}
        </div>
        {allHref && (
          <div className={styles.viewAll}>
            <Link to={allHref} className={styles.viewAllLink}>
              {allLabel} →
            </Link>
          </div>
        )}
      </div>
    </section>
  );
}

// ─── 圍棋源起紀念碑 ───
function WeiqiOriginSection() {
  return (
    <section className={clsx(styles.section, styles.originSection)}>
      <div className="container">
        <div className={styles.originContent}>
          <span className={styles.originBadge}>
            <Translate id="homepage.origin.badge">源起紀念碑</Translate>
          </span>
          <Heading as="h2" className={styles.originTitle}>
            <Translate id="homepage.origin.title">為什麼是圍棋？</Translate>
          </Heading>
          <p className={styles.originLead}>
            <Translate id="homepage.origin.lead">
              我們是「圍棋文化前進的推手」——因為 AI 浪潮第一個席捲的是圍棋界。
            </Translate>
          </p>
          <p className={styles.originBody}>
            <Translate id="homepage.origin.body">
              2016 年 AlphaGo 擊敗李世乭，圍棋人是最早被 AI 洗禮、也最早擁抱 AI 的一群人。今天，我們把這份經驗帶到健康、產業、教育、公共政策等各領域，讓更多人能夠善用這股技術浪潮。
            </Translate>
          </p>
          <div className={styles.originLinks}>
            <Link to="/docs/learn/" className={styles.originLink}>
              <Translate id="homepage.origin.cta.learn">圍棋學習資源 →</Translate>
            </Link>
            <Link to="/docs/alphago/" className={styles.originLink}>
              <Translate id="homepage.origin.cta.alphago">AlphaGo 演進整理 →</Translate>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

// ─── 創始會員 ───
function MembersSection() {
  return (
    <section className={clsx(styles.section, styles.membersSection)}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">
            <Translate id="homepage.members.title">37 位來自不同領域的夥伴</Translate>
          </Heading>
          <p className={styles.sectionDescription}>
            <Translate id="homepage.members.desc">
              律師、會計、ISO 顧問、中醫、整合醫學、健康教育、整復、行銷、活動企劃、音樂創作、AI 工程。每位都是「AI 整合者 × 領域專家」模式的活範例。
            </Translate>
          </p>
        </div>
        <div className={styles.membersGrid}>
          {founders.map((m) => (
            <Link
              key={m.slug}
              to={`/docs/about/members/founding/${m.slug}`}
              className={styles.memberCard}
              title={m.name}>
              <div
                className={styles.memberAvatar}
                style={{background: m.avatarColor}}
                aria-hidden="true">
                {m.name.slice(-1)}
              </div>
              <div className={styles.memberName}>{m.name}</div>
              <div className={styles.memberTitle}>{m.title}</div>
            </Link>
          ))}
        </div>
        <div className={styles.disclosureBox}>
          <p className={styles.disclosureText}>
            <Translate id="homepage.members.disclosure">
              目前所有研究與工具專案由理事長 CΛ / Lightman 主導執行，創始會員提供跨領域諮詢與審閱，誠邀更多會員與外部合作者深度參與下一階段。
            </Translate>
          </p>
        </div>
        <div className={styles.viewAll}>
          <Link to="/docs/about/members/" className={styles.viewAllLink}>
            <Translate id="homepage.members.cta">認識所有會員</Translate> →
          </Link>
        </div>
      </div>
    </section>
  );
}

// ─── 友站區塊（簡潔小卡） ───
function FriendsSection() {
  return (
    <section className={clsx(styles.section, styles.friendsSection)}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">
            <Translate id="homepage.friends.title">媒體聯播網</Translate>
          </Heading>
          <p className={styles.sectionDescription}>
            <Translate id="homepage.friends.desc">合作夥伴媒體，提供醫療健康領域的專業內容。</Translate>
          </p>
        </div>
        <div className={styles.friendsGrid}>
          {friendSites.map((f) => (
            <div key={f.id} className={styles.friendCard}>
              <div className={styles.friendName}>{f.name}</div>
              <div className={styles.friendLinks}>
                <a href={f.href} target="_blank" rel="noopener noreferrer" className={styles.friendLink}>
                  <Translate id="homepage.friends.link.web">官網</Translate>
                </a>
                <span className={styles.friendDot}>·</span>
                <a href={f.socialHref} target="_blank" rel="noopener noreferrer" className={styles.friendLink}>
                  <Translate id="homepage.friends.link.social">社群</Translate>
                </a>
                <span className={styles.friendDot}>·</span>
                <a href={f.videoHref} target="_blank" rel="noopener noreferrer" className={styles.friendLink}>
                  <Translate id="homepage.friends.link.video">影音</Translate>
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function HomepageLinks() {
  return (
    <>
      <USPSection />
      <ActivitiesShowcase />
      <PillarSection />

      {/* 學術研究精選 */}
      <HighlightSection
        titleKey="homepage.research.title"
        titleDefault="學術研究精選"
        descKey="homepage.research.desc"
        descDefault="從圍棋次佳手公式到深度網路泛化,每篇論文都有完整證明 + 互動視覺化。"
        allHref="/research/"
        allLabel={translate({id: 'homepage.research.all', message: '查看全部論文'})}
        columns={3}
        alt>
        {paperHighlights.slice(0, 3).map((p) => (
          <PaperCard key={p.id} {...p} />
        ))}
      </HighlightSection>

      {/* 產業情報精選 */}
      <HighlightSection
        titleKey="homepage.intel.title"
        titleDefault="產業情報精選"
        descKey="homepage.intel.desc"
        descDefault="精選 6 條主要產業供應鏈,每日掌握上中下游動態與基線 ETF 表現。"
        allHref="/intel/"
        allLabel={translate({id: 'homepage.intel.all', message: '查看全部 23 條供應鏈'})}
        columns={3}>
        {intelHighlights.map((s) => (
          <IntelCard key={s.id} {...s} />
        ))}
      </HighlightSection>

      {/* AI 工具精選 */}
      <HighlightSection
        titleKey="homepage.apps.title"
        titleDefault="AI 工具精選"
        descKey="homepage.apps.desc"
        descDefault="消費、健康、教育、資安、貿易等領域的 AI 工具,免費公開使用。"
        allHref="/apps/"
        allLabel={translate({id: 'homepage.apps.all', message: '查看全部 10 個工具'})}
        columns={3}
        alt>
        {appHighlights.map((a) => (
          <AppCard key={a.id} {...a} icon={appIcons[a.icon] || '🔧'} />
        ))}
      </HighlightSection>

      {/* 圍棋源起紀念碑 */}
      <WeiqiOriginSection />

      {/* 創始會員 */}
      <MembersSection />

      {/* 官方平台 */}
      <HighlightSection
        titleKey="homepage.official.title"
        titleDefault="官方平台"
        descKey="homepage.official.desc"
        descDefault="好棋寶寶協會官方服務 — 狀態監控、Mastodon 社群、PeerTube 影音。"
        columns={3}
        alt>
        {officialLinks.map((o) => (
          <AppCard key={o.id} {...o} icon={officialIcons[o.icon] || '🔗'} />
        ))}
      </HighlightSection>

      {/* 友站聯播 */}
      <FriendsSection />
    </>
  );
}
