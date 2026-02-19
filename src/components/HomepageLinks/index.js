import clsx from 'clsx';
import Heading from '@theme/Heading';
import Translate, {translate} from '@docusaurus/Translate';
import styles from './styles.module.css';

// SVG imports
import StatusSvg from '@site/static/img/links/status.svg';
import SocialSvg from '@site/static/img/links/social.svg';
import VideoSvg from '@site/static/img/links/video.svg';

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
    <LinkSection
      title={<Translate id="homepage.links.official.section.title">官方</Translate>}
      description={<Translate id="homepage.links.official.section.desc">好棋寶寶協會官方平台，提供服務狀態監控、社群互動與影音內容。</Translate>}
      links={getOfficialLinks()}
    />
  );
}
