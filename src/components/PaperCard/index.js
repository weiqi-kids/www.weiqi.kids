import Translate from '@docusaurus/Translate';
import styles from './styles.module.css';

// 論文卡片：用於 /research/ 與首頁學術精選
export default function PaperCard({
  id,
  repo,
  href,
  repoHref,
  icon,
  titleKey,
  titleDefault,
  descKey,
  descDefault,
}) {
  return (
    <div className={styles.card}>
      <a href={href} target="_blank" rel="noopener noreferrer" className={styles.mainLink}>
        <div className={styles.header}>
          <span className={styles.icon} aria-hidden="true">{icon}</span>
          <div className={styles.titleBlock}>
            <span className={styles.repoName}>{repo}</span>
            <span className={styles.divider} aria-hidden="true">｜</span>
            <span className={styles.title}>
              <Translate id={titleKey}>{titleDefault}</Translate>
            </span>
          </div>
        </div>
        <p className={styles.description}>
          <Translate id={descKey}>{descDefault}</Translate>
        </p>
      </a>
      <div className={styles.actions}>
        <a href={href} target="_blank" rel="noopener noreferrer" className={styles.actionPrimary}>
          <Translate id="research.action.read">閱讀互動論文 →</Translate>
        </a>
        <a href={repoHref} target="_blank" rel="noopener noreferrer" className={styles.actionSecondary}>
          GitHub
        </a>
      </div>
    </div>
  );
}
