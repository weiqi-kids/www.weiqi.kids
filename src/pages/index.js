import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Translate, {translate} from '@docusaurus/Translate';
import HomepageLinks from '@site/src/components/HomepageLinks';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          <Translate id="homepage.hero.title" description="Hero H1 title">
            因為圍棋認識彼此，因為 AI 一起做事
          </Translate>
        </Heading>
        <p className="hero__subtitle">
          <Translate id="homepage.tagline" description="Hero subtitle below H1">
            商界夥伴 × 跨域 AI × 公益圍棋
          </Translate>
        </p>
        <p className={styles.heroStats}>
          <Translate id="homepage.hero.stats" description="Hero stats line">
            37 位跨域夥伴 · 41+ AI 工具 · 9 篇學術論文 · 每日自動更新
          </Translate>
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="mailto:lightman.chang@gmail.com?subject=%E3%80%90%E5%90%88%E4%BD%9C%E6%8F%90%E6%A1%88%E3%80%91">
            <Translate id="homepage.cta.proposal">合作提案</Translate>
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="mailto:lightman.chang@gmail.com?subject=%E3%80%90%E5%85%88%E8%81%8A%E8%81%8A%E3%80%91">
            <Translate id="homepage.cta.meet">先見面聊聊</Translate>
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/docs/about/members/founding/">
            <Translate id="homepage.cta.members">認識夥伴</Translate>
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={translate({id: 'homepage.meta.title', message: '首頁'})}
      description={translate({id: 'homepage.meta.description', message: '台灣好棋寶寶協會——跨域夥伴因圍棋結緣，一起把各自的專業做成開源 AI 工具與學術研究。'})}>
      <HomepageHeader />
      <main>
        <HomepageLinks />
      </main>
    </Layout>
  );
}
