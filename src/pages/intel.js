import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Translate, {translate} from '@docusaurus/Translate';
import CategorySection from '@site/src/components/CategorySection';
import IntelCard from '@site/src/components/IntelCard';
import CollectionPageSchema from '@site/src/components/SEO/CollectionPageSchema';
import {intelCategories, intelLinks} from '@site/src/data/links/intel';
import styles from './landing.module.css';

export default function IntelPage() {
  const totalCompanies = intelLinks.reduce((sum, s) => sum + s.companies.total, 0);
  const totalSites = intelLinks.length;
  const description = translate({
    id: 'intel.meta.description',
    message: `${totalSites} 條供應鏈、${totalCompanies}+ 家公司、每日自動追蹤上中下游新聞與財報。對標主要 ETF 基線，掌握全球產業動態。`,
  });
  const items = intelLinks.map((s) => ({
    name: `${s.repo} | ${s.titleDefault}`,
    url: s.href,
    description: s.descDefault,
  }));

  return (
    <Layout
      title={translate({id: 'intel.meta.title', message: '產業供應鏈情報 — 好棋寶寶協會'})}
      description={description}>
      <CollectionPageSchema
        pagePath="/intel/"
        pageName="產業供應鏈情報"
        description={description}
        items={items}
        itemSchemaType="DataCatalog"
      />
      <header className={styles.heroSlim}>
        <div className="container">
          <Heading as="h1">
            <Translate id="intel.hero.title">產業供應鏈情報</Translate>
          </Heading>
          <p className={styles.heroDesc}>
            <Translate id="intel.hero.desc" values={{sites: totalSites, companies: totalCompanies}}>
              {'{sites} 條供應鏈，{companies}+ 家公司，每日自動抓取新聞、財報、產業數據。每條鏈都對標代表性 ETF 作為市場基線。'}
            </Translate>
          </p>
        </div>
      </header>

      <main>
        {intelCategories.map((cat) => {
          const sites = intelLinks.filter((s) => s.category === cat.id);
          if (sites.length === 0) return null;
          return (
            <CategorySection
              key={cat.id}
              id={cat.id}
              titleKey={cat.titleKey}
              titleDefault={cat.titleDefault}
              descKey={cat.descKey}
              descDefault={cat.descDefault}
              columns={sites.length === 1 ? 3 : sites.length === 2 ? 2 : 3}>
              {sites.map((s) => (
                <IntelCard key={s.id} {...s} />
              ))}
            </CategorySection>
          );
        })}
      </main>
    </Layout>
  );
}
