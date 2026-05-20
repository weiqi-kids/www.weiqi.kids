import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Translate, {translate} from '@docusaurus/Translate';
import HomepageLinks from '@site/src/components/HomepageLinks';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">
          <Translate id="homepage.tagline" description="Site subtitle">
            開源研究社群．從圍棋出發
          </Translate>
        </p>
        <p className={styles.heroStats}>
          <Translate id="homepage.hero.stats" description="Hero stats line">
            11 位創始會員 · 41 個公益專案 · 全部開源 · 每日自動更新
          </Translate>
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/about/">
            <Translate id="homepage.cta.about">認識協會</Translate>
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/research/">
            <Translate id="homepage.cta.research">研究成果</Translate>
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
      description={translate({id: 'homepage.meta.description', message: '台灣好棋寶寶協會官網 - 提供圍棋教學、AI 研究資源，推動圍棋文化發展'})}>
      <HomepageHeader />
      <main>
        <HomepageLinks />
      </main>
    </Layout>
  );
}
