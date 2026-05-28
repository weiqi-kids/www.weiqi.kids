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

// ─── USP 宣告區塊（不收費 + 三條「不是什麼」）───
function USPSection() {
  return (
    <section className={clsx(styles.section, styles.uspSection)}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">
            <Translate id="homepage.usp.title">我們不收費，全部開源</Translate>
          </Heading>
          <p className={styles.sectionDescription}>
            <Translate id="homepage.usp.lead">
              完全免費、無會員費、全部開源（CC-BY 4.0）。我們不做付費培訓、不收引薦費、不賣會員制。
            </Translate>
          </p>
        </div>
        <div className={styles.uspGrid}>
          <div className={styles.uspItem}>
            <div className={styles.uspNot}>
              <Translate id="homepage.usp.not1.label">不是純圍棋協會</Translate>
            </div>
            <p className={styles.uspExplain}>
              <Translate id="homepage.usp.not1.desc">
                我們把圍棋當紐帶不當目的
              </Translate>
            </p>
          </div>
          <div className={styles.uspItem}>
            <div className={styles.uspNot}>
              <Translate id="homepage.usp.not2.label">不是純 AI 推廣</Translate>
            </div>
            <p className={styles.uspExplain}>
              <Translate id="homepage.usp.not2.desc">
                我們有真正自產的開源工具
              </Translate>
            </p>
          </div>
          <div className={styles.uspItem}>
            <div className={styles.uspNot}>
              <Translate id="homepage.usp.not3.label">不是純商務人脈</Translate>
            </div>
            <p className={styles.uspExplain}>
              <Translate id="homepage.usp.not3.desc">
                我們有共同開源產出當證據
              </Translate>
            </p>
          </div>
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
            <Translate id="homepage.pillars.title">三大合作面向</Translate>
          </Heading>
          <p className={styles.sectionDescription}>
            <Translate id="homepage.pillars.desc">
              夥伴提供領域知識，理事長負責 AI 整合。產出三類成果：AI 工具、產業情報、學術論文，全部開源。
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
              AI 浪潮第一個席捲的是圍棋界。
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
