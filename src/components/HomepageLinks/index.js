import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// 官方 - 3 個
const OfficialLinks = [
  {
    title: '網站狀態列',
    href: 'https://status.weiqi.kids/',
    Svg: require('@site/static/img/links/status.svg').default,
    description: '即時監控所有好棋寶寶服務的運作狀態，確保服務品質與穩定性。',
  },
  {
    title: '社群',
    href: 'https://mastodon.weiqi.kids/',
    Svg: require('@site/static/img/links/social.svg').default,
    description: '加入好棋寶寶 Mastodon 社群，與圍棋愛好者交流互動。',
  },
  {
    title: '影音',
    href: 'https://peertube.weiqi.kids/',
    Svg: require('@site/static/img/links/video.svg').default,
    description: '觀看好棋寶寶 PeerTube 影音頻道，學習圍棋知識與技巧。',
  },
];

// AI 研究 - 8 個
const AIResearchLinks = [
  {
    title: '電商產品研究',
    href: 'https://ecommerce.weiqi.kids/',
    Svg: require('@site/static/img/links/ecommerce.svg').default,
    description: '買前必看！AI 驅動的電商產品深度研究與分析。',
  },
  {
    title: '資安威脅情報中心',
    href: 'https://security.weiqi.kids/',
    Svg: require('@site/static/img/links/security.svg').default,
    description: '即時掌握全球資安威脅動態，保護數位資產安全。',
  },
  {
    title: 'EpiAlert 疫情快訊',
    href: 'https://epialert.weiqi.kids/',
    Svg: require('@site/static/img/links/health.svg').default,
    description: '全球疫情即時監測與預警，守護公共衛生安全。',
  },
  {
    title: '保健食品產品情報',
    href: 'https://supplement.weiqi.kids/',
    Svg: require('@site/static/img/links/supplement.svg').default,
    description: '保健食品深度分析，提供科學化的產品評估資訊。',
  },
  {
    title: '全球框架法規變動',
    href: 'https://risk.weiqi.kids/',
    Svg: require('@site/static/img/links/law.svg').default,
    description: '追蹤全球框架、法規與產業規則的最新變動。',
  },
  {
    title: '全球貿易情報分析',
    href: 'https://trade.weiqi.kids/',
    Svg: require('@site/static/img/links/trade.svg').default,
    description: '深入分析全球貿易動態，掌握國際市場脈動。',
  },
  {
    title: '求職技能需求觀測站',
    href: 'https://skills.weiqi.kids/',
    Svg: require('@site/static/img/links/job.svg').default,
    description: '觀測就業市場技能需求變化，規劃職涯發展方向。',
  },
  {
    title: '學生學習地圖',
    href: 'https://learn.weiqi.kids/',
    Svg: require('@site/static/img/links/education.svg').default,
    description: '為學生規劃最適合的學習路徑，提升學習效率。',
  },
];

// 友站 - 亞太醫頭條 3 個
const APPILinks = [
  {
    title: '亞太醫頭條官網',
    href: 'https://appi.news/',
    Svg: require('@site/static/img/links/news.svg').default,
    description: '亞太地區醫療健康新聞的專業媒體平台。',
  },
  {
    title: '亞太醫頭條社群',
    href: 'https://mastodon.appi.news/',
    Svg: require('@site/static/img/links/social.svg').default,
    description: '加入亞太醫頭條 Mastodon 社群，討論醫療健康議題。',
  },
  {
    title: '亞太醫頭條影音',
    href: 'https://peertube.appi.news/',
    Svg: require('@site/static/img/links/video.svg').default,
    description: '觀看亞太醫頭條影音內容，掌握醫療健康資訊。',
  },
];

// 友站 - 銅蛇醫報 3 個（銅蛇/基督教主題）
const TLSRLinks = [
  {
    title: '銅蛇醫報官網',
    href: 'https://tlsr.news/',
    Svg: require('@site/static/img/links/tlsr-news.svg').default,
    description: '銅蛇醫報官方網站，提供專業醫療資訊與報導。',
  },
  {
    title: '銅蛇醫報社群',
    href: 'https://mastodon.tlsr.news/',
    Svg: require('@site/static/img/links/tlsr-social.svg').default,
    description: '加入銅蛇醫報 Mastodon 社群，交流醫療見解。',
  },
  {
    title: '銅蛇醫報影音',
    href: 'https://peertube.tlsr.news/',
    Svg: require('@site/static/img/links/tlsr-video.svg').default,
    description: '觀看銅蛇醫報影音頻道，學習醫療保健知識。',
  },
];

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

export default function HomepageLinks() {
  return (
    <>
      <LinkSection
        title="官方"
        description="好棋寶寶協會官方平台，提供服務狀態監控、社群互動與影音內容。"
        links={OfficialLinks}
      />
      <LinkSection
        title="AI 研究"
        description="運用人工智慧技術，深入分析各領域情報，提供即時洞察與決策支援。"
        links={AIResearchLinks}
      />
      <LinkSection
        title="友站 - 亞太醫頭條"
        description="亞太地區醫療健康領域的專業媒體夥伴。"
        links={APPILinks}
      />
      <LinkSection
        title="友站 - 銅蛇醫報"
        description="專注於醫療保健資訊的合作媒體。"
        links={TLSRLinks}
      />
    </>
  );
}
