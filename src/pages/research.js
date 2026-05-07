import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Translate, {translate} from '@docusaurus/Translate';
import CategorySection from '@site/src/components/CategorySection';
import PaperCard from '@site/src/components/PaperCard';
import CollectionPageSchema from '@site/src/components/SEO/CollectionPageSchema';
import {researchCategories, researchPapers} from '@site/src/data/links/research';
import styles from './landing.module.css';

export default function ResearchPage() {
  const description = translate({
    id: 'research.meta.description',
    message: '好棋寶寶協會發表的學術論文：圍棋理論、機器學習理論、數學研究。每篇皆含互動視覺化網頁與完整數學證明。',
  });
  const items = researchPapers.map((p) => ({
    name: p.titleDefault,
    url: p.href,
    description: p.descDefault,
  }));

  return (
    <Layout
      title={translate({id: 'research.meta.title', message: '學術研究 — 好棋寶寶協會'})}
      description={description}>
      <CollectionPageSchema
        pagePath="/research/"
        pageName="學術研究"
        description={description}
        items={items}
        itemSchemaType="ScholarlyArticle"
      />
      <header className={styles.heroSlim}>
        <div className="container">
          <Heading as="h1">
            <Translate id="research.hero.title">學術研究發表</Translate>
          </Heading>
          <p className={styles.heroDesc}>
            <Translate id="research.hero.desc">
              我們發表的論文，每篇都是完整數學證明 + 互動視覺化網頁 + GitHub 開源。涵蓋圍棋理論、機器學習，以及經典數學問題。
            </Translate>
          </p>
        </div>
      </header>

      <main>
        {researchCategories.map((cat) => {
          const papers = researchPapers.filter((p) => p.category === cat.id);
          if (papers.length === 0) return null;
          return (
            <CategorySection
              key={cat.id}
              id={cat.id}
              titleKey={cat.titleKey}
              titleDefault={cat.titleDefault}
              descKey={cat.descKey}
              descDefault={cat.descDefault}
              columns={3}>
              {papers.map((p) => (
                <PaperCard key={p.id} {...p} />
              ))}
            </CategorySection>
          );
        })}
      </main>
    </Layout>
  );
}
