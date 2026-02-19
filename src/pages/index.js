import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Translate, {translate} from '@docusaurus/Translate';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
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
            台灣好棋寶寶協會｜致力於圍棋文化前進的推手
          </Translate>
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/for-players/">
            <Translate id="homepage.forPlayers">我是棋友</Translate>
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/for-engineers/">
            <Translate id="homepage.forEngineers">我是工程師</Translate>
          </Link>
        </div>
        <div className={styles.buttons} style={{marginTop: '1rem'}}>
          <Link
            className="button button--outline button--secondary"
            to="/docs/intro">
            <Translate id="homepage.quickStart">1分鐘快速了解</Translate>
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
        <HomepageFeatures />
        <HomepageLinks />
      </main>
    </Layout>
  );
}
