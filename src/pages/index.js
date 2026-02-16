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
            to="/docs/intro">
            <Translate id="homepage.quickStart">1分鐘快速了解 ⏱️</Translate>
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
      title={translate({id: 'homepage.meta.title', message: 'Hello from {title}'}, {title: siteConfig.title})}
      description={translate({id: 'homepage.meta.description', message: 'Description will go into a meta tag in <head />'})}>
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <HomepageLinks />
      </main>
    </Layout>
  );
}
