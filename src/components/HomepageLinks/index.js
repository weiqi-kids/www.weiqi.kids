import clsx from 'clsx';
import Heading from '@theme/Heading';
import Translate, {translate} from '@docusaurus/Translate';
import styles from './styles.module.css';

// SVG imports
import StatusSvg from '@site/static/img/links/status.svg';
import SocialSvg from '@site/static/img/links/social.svg';
import VideoSvg from '@site/static/img/links/video.svg';
import EcommerceSvg from '@site/static/img/links/ecommerce.svg';
import SecuritySvg from '@site/static/img/links/security.svg';
import HealthSvg from '@site/static/img/links/health.svg';
import SupplementSvg from '@site/static/img/links/supplement.svg';
import LawSvg from '@site/static/img/links/law.svg';
import TradeSvg from '@site/static/img/links/trade.svg';
import JobSvg from '@site/static/img/links/job.svg';
import EducationSvg from '@site/static/img/links/education.svg';
import PolicySvg from '@site/static/img/links/policy.svg';
import NewsSvg from '@site/static/img/links/news.svg';
import TlsrNewsSvg from '@site/static/img/links/tlsr-news.svg';
import TlsrSocialSvg from '@site/static/img/links/tlsr-social.svg';
import TlsrVideoSvg from '@site/static/img/links/tlsr-video.svg';

// 官方 - 3 個
function getOfficialLinks() {
  return [
    {
      title: translate({id: 'homepage.links.official.status.title', message: '網站狀態列'}),
      href: 'https://status.weiqi.kids/',
      Svg: StatusSvg,
      description: translate({id: 'homepage.links.official.status.desc', message: '即時監控所有好棋寶寶服務的運作狀態，確保服務品質與穩定性。'}),
    },
    {
      title: translate({id: 'homepage.links.official.social.title', message: '社群'}),
      href: 'https://mastodon.weiqi.kids/',
      Svg: SocialSvg,
      description: translate({id: 'homepage.links.official.social.desc', message: '加入好棋寶寶 Mastodon 社群，與圍棋愛好者交流互動。'}),
    },
    {
      title: translate({id: 'homepage.links.official.video.title', message: '影音'}),
      href: 'https://peertube.weiqi.kids/',
      Svg: VideoSvg,
      description: translate({id: 'homepage.links.official.video.desc', message: '觀看好棋寶寶 PeerTube 影音頻道，學習圍棋知識與技巧。'}),
    },
  ];
}

// AI 研究 - 9 個
function getAIResearchLinks() {
  return [
    {
      title: translate({id: 'homepage.links.ai.ecommerce.title', message: '電商產品研究'}),
      href: 'https://ecommerce.weiqi.kids/',
      Svg: EcommerceSvg,
      description: translate({id: 'homepage.links.ai.ecommerce.desc', message: '買前必看！AI 驅動的電商產品深度研究與分析。'}),
    },
    {
      title: translate({id: 'homepage.links.ai.security.title', message: '資安威脅情報中心'}),
      href: 'https://security.weiqi.kids/',
      Svg: SecuritySvg,
      description: translate({id: 'homepage.links.ai.security.desc', message: '即時掌握全球資安威脅動態，保護數位資產安全。'}),
    },
    {
      title: translate({id: 'homepage.links.ai.health.title', message: 'EpiAlert 疫情快訊'}),
      href: 'https://epialert.weiqi.kids/',
      Svg: HealthSvg,
      description: translate({id: 'homepage.links.ai.health.desc', message: '全球疫情即時監測與預警，守護公共衛生安全。'}),
    },
    {
      title: translate({id: 'homepage.links.ai.supplement.title', message: '保健食品產品情報'}),
      href: 'https://supplement.weiqi.kids/',
      Svg: SupplementSvg,
      description: translate({id: 'homepage.links.ai.supplement.desc', message: '保健食品深度分析，提供科學化的產品評估資訊。'}),
    },
    {
      title: translate({id: 'homepage.links.ai.law.title', message: '全球框架法規變動'}),
      href: 'https://risk.weiqi.kids/',
      Svg: LawSvg,
      description: translate({id: 'homepage.links.ai.law.desc', message: '追蹤全球框架、法規與產業規則的最新變動。'}),
    },
    {
      title: translate({id: 'homepage.links.ai.trade.title', message: '全球貿易情報分析'}),
      href: 'https://trade.weiqi.kids/',
      Svg: TradeSvg,
      description: translate({id: 'homepage.links.ai.trade.desc', message: '深入分析全球貿易動態，掌握國際市場脈動。'}),
    },
    {
      title: translate({id: 'homepage.links.ai.job.title', message: '求職技能需求觀測站'}),
      href: 'https://skills.weiqi.kids/',
      Svg: JobSvg,
      description: translate({id: 'homepage.links.ai.job.desc', message: '觀測就業市場技能需求變化，規劃職涯發展方向。'}),
    },
    {
      title: translate({id: 'homepage.links.ai.education.title', message: '學生學習地圖'}),
      href: 'https://learn.weiqi.kids/',
      Svg: EducationSvg,
      description: translate({id: 'homepage.links.ai.education.desc', message: '為學生規劃最適合的學習路徑，提升學習效率。'}),
    },
    {
      title: translate({id: 'homepage.links.ai.policy.title', message: '政策承諾追蹤'}),
      href: 'https://lightchang.github.io/brighterarc/',
      Svg: PolicySvg,
      description: translate({id: 'homepage.links.ai.policy.desc', message: '追蹤政府與政治人物的政策承諾，促進政治透明與責任。'}),
    },
  ];
}

// 友站 - 亞太醫頭條 3 個
function getAPPILinks() {
  return [
    {
      title: translate({id: 'homepage.links.appi.news.title', message: '亞太醫頭條官網'}),
      href: 'https://appi.news/',
      Svg: NewsSvg,
      description: translate({id: 'homepage.links.appi.news.desc', message: '亞太地區醫療健康新聞的專業媒體平台。'}),
    },
    {
      title: translate({id: 'homepage.links.appi.social.title', message: '亞太醫頭條社群'}),
      href: 'https://mastodon.appi.news/',
      Svg: SocialSvg,
      description: translate({id: 'homepage.links.appi.social.desc', message: '加入亞太醫頭條 Mastodon 社群，討論醫療健康議題。'}),
    },
    {
      title: translate({id: 'homepage.links.appi.video.title', message: '亞太醫頭條影音'}),
      href: 'https://peertube.appi.news/',
      Svg: VideoSvg,
      description: translate({id: 'homepage.links.appi.video.desc', message: '觀看亞太醫頭條影音內容，掌握醫療健康資訊。'}),
    },
  ];
}

// 友站 - 銅蛇醫報 3 個
function getTLSRLinks() {
  return [
    {
      title: translate({id: 'homepage.links.tlsr.news.title', message: '銅蛇醫報官網'}),
      href: 'https://tlsr.news/',
      Svg: TlsrNewsSvg,
      description: translate({id: 'homepage.links.tlsr.news.desc', message: '銅蛇醫報官方網站，提供專業醫療資訊與報導。'}),
    },
    {
      title: translate({id: 'homepage.links.tlsr.social.title', message: '銅蛇醫報社群'}),
      href: 'https://mastodon.tlsr.news/',
      Svg: TlsrSocialSvg,
      description: translate({id: 'homepage.links.tlsr.social.desc', message: '加入銅蛇醫報 Mastodon 社群，交流醫療見解。'}),
    },
    {
      title: translate({id: 'homepage.links.tlsr.video.title', message: '銅蛇醫報影音'}),
      href: 'https://peertube.tlsr.news/',
      Svg: TlsrVideoSvg,
      description: translate({id: 'homepage.links.tlsr.video.desc', message: '觀看銅蛇醫報影音頻道，學習醫療保健知識。'}),
    },
  ];
}

function LinkCard({ Svg, title, description, href }) {
  return (
    <div className={clsx('col col--4')}>
      <a href={href} target="_blank" rel="noopener noreferrer" className={styles.cardLink}>
        <div className={styles.card}>
          <div className="text--center">
            <Svg className={styles.featureSvg} role="img" />
          </div>
          <div className="text--center padding-horiz--md">
            <Heading as="h3">{title}</Heading>
            <p>{description}</p>
          </div>
        </div>
      </a>
    </div>
  );
}

function LinkSection({ title, description, links }) {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">{title}</Heading>
          {description && <p className={styles.sectionDescription}>{description}</p>}
        </div>
        <div className="row">
          {links.map((props, idx) => (
            <LinkCard key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

// 檢查是否已到顯示日期 (2026/3/1)
function shouldShowFriendSites() {
  const now = new Date();
  const showDate = new Date(2026, 2, 1); // 2026年3月1日 (月份從0開始)
  return now >= showDate;
}

export default function HomepageLinks() {
  const showFriendSites = shouldShowFriendSites();

  return (
    <>
      <LinkSection
        title={<Translate id="homepage.links.official.section.title">官方</Translate>}
        description={<Translate id="homepage.links.official.section.desc">好棋寶寶協會官方平台，提供服務狀態監控、社群互動與影音內容。</Translate>}
        links={getOfficialLinks()}
      />
      <LinkSection
        title={<Translate id="homepage.links.ai.section.title">AI 研究</Translate>}
        description={<Translate id="homepage.links.ai.section.desc">運用人工智慧技術，深入分析各領域情報，提供即時洞察與決策支援。</Translate>}
        links={getAIResearchLinks()}
      />
      {showFriendSites && (
        <>
          <LinkSection
            title={<Translate id="homepage.links.appi.section.title">友站 - 亞太醫頭條</Translate>}
            description={<Translate id="homepage.links.appi.section.desc">亞太地區醫療健康領域的專業媒體夥伴。</Translate>}
            links={getAPPILinks()}
          />
          <LinkSection
            title={<Translate id="homepage.links.tlsr.section.title">友站 - 銅蛇醫報</Translate>}
            description={<Translate id="homepage.links.tlsr.section.desc">專注於醫療保健資訊的合作媒體。</Translate>}
            links={getTLSRLinks()}
          />
        </>
      )}
    </>
  );
}
