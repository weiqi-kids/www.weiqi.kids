import Link from '@docusaurus/Link';
import {founders} from '@site/src/data/members';
import styles from './styles.module.css';

function MemberCard({m}) {
  const photoSrc = m.photo ? `/img/members/${m.slug}.${m.photo}` : null;
  const initial = m.name.slice(-1); // 取末字（中文姓名 last char = given name 首字）

  return (
    <Link
      to={`/docs/about/members/founding/${m.slug}/`}
      className={styles.card}>
      <div
        className={styles.avatar}
        style={photoSrc ? undefined : {background: m.avatarColor}}>
        {photoSrc ? (
          <img src={photoSrc} alt={m.name} className={styles.avatarImg} loading="lazy" decoding="async" />
        ) : (
          <span className={styles.avatarInitial}>{initial}</span>
        )}
      </div>
      <div className={styles.nameRow}>
        <span className={styles.name}>{m.name}</span>
        {m.enName && <span className={styles.enName}>{m.enName}</span>}
      </div>
      {m.key5 && (
        <div className={styles.key5}>「{m.key5}」</div>
      )}
      {m.title && (
        <div className={styles.title}>{m.title}</div>
      )}
      <div className={styles.org}>{m.org}</div>
    </Link>
  );
}

export default function MembersDirectory() {
  return (
    <div className={styles.directory}>
      <div className={styles.grid}>
        {founders.map((m) => (
          <MemberCard key={m.slug} m={m} />
        ))}
      </div>
    </div>
  );
}
