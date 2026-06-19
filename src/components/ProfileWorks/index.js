import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

/**
 * ProfileWorks — 代表作品卡片 grid（可重用於夥伴頁）
 * 以「分組 + 卡片」取代密集連結行。全用 design-tokens.css 變數。
 *
 * @param {Object} props
 * @param {Array<{title:string, items:Array<{title:string, desc?:string, href:string, icon?:string, external?:boolean}>}>} props.groups
 */
export default function ProfileWorks({ groups = [] }) {
  return (
    <div className={styles.works}>
      {groups.map((group, gi) => (
        <div key={gi} className={styles.group}>
          {group.title && <h3 className={styles.groupTitle}>{group.title}</h3>}
          <div className={styles.grid}>
            {group.items.map((item, ii) => {
              const inner = (
                <>
                  {item.icon && <span className={styles.icon}>{item.icon}</span>}
                  <span className={styles.text}>
                    <span className={styles.title}>{item.title}</span>
                    {item.desc && <span className={styles.desc}>{item.desc}</span>}
                  </span>
                </>
              );
              return item.external ? (
                <a
                  key={ii}
                  href={item.href}
                  className={styles.card}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {inner}
                </a>
              ) : (
                <Link key={ii} href={item.href} className={styles.card}>
                  {inner}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
