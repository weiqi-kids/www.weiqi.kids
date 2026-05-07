import Translate from '@docusaurus/Translate';
import styles from './styles.module.css';

// AI 工具卡片（簡潔版）：用於 /apps/ 與首頁工具精選
export default function AppCard({
  href,
  icon,
  titleKey,
  titleDefault,
  descKey,
  descDefault,
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={styles.card}>
      <div className={styles.iconWrapper}>
        <span className={styles.icon} aria-hidden="true">{icon}</span>
      </div>
      <div className={styles.body}>
        <h3 className={styles.title}>
          <Translate id={titleKey}>{titleDefault}</Translate>
        </h3>
        <p className={styles.description}>
          <Translate id={descKey}>{descDefault}</Translate>
        </p>
      </div>
    </a>
  );
}
