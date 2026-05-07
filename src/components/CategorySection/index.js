import clsx from 'clsx';
import Heading from '@theme/Heading';
import Translate from '@docusaurus/Translate';
import styles from './styles.module.css';

// 分類區塊：標題 + 描述 + 卡片網格
// children 由父層放入 IntelCard / PaperCard / AppCard
export default function CategorySection({
  id,
  titleKey,
  titleDefault,
  descKey,
  descDefault,
  columns = 3,
  children,
}) {
  return (
    <section id={id} className={styles.section}>
      <div className="container">
        <div className={styles.header}>
          <Heading as="h2" className={styles.title}>
            <Translate id={titleKey}>{titleDefault}</Translate>
          </Heading>
          {descKey && (
            <p className={styles.description}>
              <Translate id={descKey}>{descDefault}</Translate>
            </p>
          )}
        </div>
        <div
          className={clsx(styles.grid, columns === 2 && styles.gridTwo, columns === 4 && styles.gridFour)}>
          {children}
        </div>
      </div>
    </section>
  );
}
