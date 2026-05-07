import clsx from 'clsx';
import Translate, {translate} from '@docusaurus/Translate';
import styles from './styles.module.css';

// 版型 B — 數據儀表板風
// 上：emoji + repo 名 + 中文產業名（雙語並列）
// 中：3 個大數字塊（公司數 / 上中下 / 基線 ETF），無基線時改 2 塊置中
// 下：追蹤主題 + 前往儀表板按鈕
export default function IntelCard({
  id,
  repo,
  href,
  icon,
  titleKey,
  titleDefault,
  descKey,
  descDefault,
  companies,
  topicsHighlight,
  baseline,
}) {
  const hasBaseline = baseline && baseline.length > 0;
  const baselineTickers = hasBaseline
    ? baseline.map((b) => b.ticker).join(' · ')
    : null;
  const baselineTooltip = hasBaseline
    ? baseline.map((b) => `${b.ticker} — ${b.name}`).join('\n')
    : null;

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={styles.card}
      aria-label={`${repo} - ${titleDefault}`}>
      <div className={styles.header}>
        <span className={styles.icon} aria-hidden="true">{icon}</span>
        <div className={styles.titleBlock}>
          <span className={styles.repoName}>{repo}</span>
          <span className={styles.divider} aria-hidden="true">｜</span>
          <span className={styles.industryName}>
            <Translate id={titleKey}>{titleDefault}</Translate>
          </span>
        </div>
      </div>

      <div className={clsx(styles.metrics, !hasBaseline && styles.metricsTwoCol)}>
        <div className={styles.metric}>
          <div className={styles.metricValue}>{companies.total}</div>
          <div className={styles.metricLabel}>
            <Translate id="intel.metric.companies">公司</Translate>
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricValue}>
            {companies.upstream} · {companies.midstream} · {companies.downstream}
          </div>
          <div className={styles.metricLabel}>
            <Translate id="intel.metric.structure">上 · 中 · 下</Translate>
          </div>
        </div>
        {hasBaseline && (
          <div className={styles.metric} title={baselineTooltip}>
            <div className={styles.metricValue}>{baselineTickers}</div>
            <div className={styles.metricLabel}>
              <Translate id="intel.metric.baseline">基線 ETF</Translate>
            </div>
          </div>
        )}
      </div>

      <div className={styles.topics}>
        <span className={styles.topicsLabel}>
          <Translate id="intel.topics.label">追蹤主題</Translate>
        </span>
        <span className={styles.topicsValue}>
          {topicsHighlight.join(' · ')}
        </span>
      </div>

      <div className={styles.cta}>
        <Translate id="intel.cta">前往儀表板 →</Translate>
      </div>
    </a>
  );
}
