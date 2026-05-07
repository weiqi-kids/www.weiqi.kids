import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Translate, {translate} from '@docusaurus/Translate';
import CategorySection from '@site/src/components/CategorySection';
import AppCard from '@site/src/components/AppCard';
import CollectionPageSchema from '@site/src/components/SEO/CollectionPageSchema';
import {appCategories, appLinks} from '@site/src/data/links/apps';
import styles from './landing.module.css';

const appIcons = {
  ecommerce: '🛒',
  supplement: '💊',
  education: '📚',
  job: '💼',
  security: '🛡️',
  health: '🦠',
  law: '⚖️',
  trade: '🌐',
  listening: '📊',
  policy: '📜',
};

export default function AppsPage() {
  const description = translate({
    id: 'apps.meta.description',
    message: '好棋寶寶協會的 10 個 AI 應用工具：消費購物、健康教育、資安疫情、貿易聲量、政策追蹤。免費公開供大眾使用。',
  });
  const items = appLinks.map((a) => ({
    name: a.titleDefault,
    url: a.href,
    description: a.descDefault,
  }));

  return (
    <Layout
      title={translate({id: 'apps.meta.title', message: 'AI 應用工具 — 好棋寶寶協會'})}
      description={description}>
      <CollectionPageSchema
        pagePath="/apps/"
        pageName="AI 應用工具"
        description={description}
        items={items}
        itemSchemaType="SoftwareApplication"
      />
      <header className={styles.heroSlim}>
        <div className="container">
          <Heading as="h1">
            <Translate id="apps.hero.title">AI 應用工具</Translate>
          </Heading>
          <p className={styles.heroDesc}>
            <Translate id="apps.hero.desc">
              運用人工智慧技術，深入分析各領域情報，提供即時洞察與決策支援。所有工具免費公開。
            </Translate>
          </p>
        </div>
      </header>

      <main>
        {appCategories.map((cat) => {
          const apps = appLinks.filter((a) => a.category === cat.id);
          if (apps.length === 0) return null;
          return (
            <CategorySection
              key={cat.id}
              id={cat.id}
              titleKey={cat.titleKey}
              titleDefault={cat.titleDefault}
              descKey={cat.descKey}
              descDefault={cat.descDefault}
              columns={3}>
              {apps.map((a) => (
                <AppCard key={a.id} {...a} icon={appIcons[a.icon] || '🔧'} />
              ))}
            </CategorySection>
          );
        })}
      </main>
    </Layout>
  );
}
