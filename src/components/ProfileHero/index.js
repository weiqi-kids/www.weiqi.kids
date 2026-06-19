import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

/**
 * ProfileHero — 名片式人物 hero 卡
 * 可重用於所有夥伴頁：左頭像（照片或色塊），右姓名 + slogan + 職務 pill + 連結按鈕。
 * 全部使用 design-tokens.css 的 CSS 變數，禁止 hardcode。
 *
 * @param {Object} props
 * @param {string} props.name - 姓名
 * @param {string} [props.enName] - 英文名／署名
 * @param {string} [props.slogan] - key5 一句話定位（暖橙）
 * @param {string[]} [props.titles] - 職務（pill badges）
 * @param {Object} [props.avatar] - 頭像設定
 * @param {string} [props.avatar.photo] - 照片路徑（相對 baseUrl，如 /img/members/xxx.jpg）；無則用色塊
 * @param {string} [props.avatar.color] - 色塊底色（無照片時）
 * @param {string} [props.avatar.initial] - 色塊顯示字（無照片時）
 * @param {number} [props.avatar.zoom] - 照片縮放倍率（聚焦臉部用，預設 1）
 * @param {string} [props.avatar.fx] - 臉部水平焦點百分比字串（預設 '50%'）
 * @param {string} [props.avatar.fy] - 臉部垂直焦點百分比字串（預設 '50%'）
 * @param {string} [props.avatar.alt] - 照片 alt
 * @param {Array<{label:string, href:string, icon?:string, external?:boolean}>} [props.links] - 連結按鈕
 * @param {string} [props.location] - 地點（選填）
 */
export default function ProfileHero({
  name,
  enName,
  slogan,
  titles = [],
  avatar = {},
  links = [],
  location,
}) {
  const { photo, color, initial, zoom = 1, fx = '50%', fy = '50%', alt } = avatar;

  return (
    <div className={styles.hero}>
      <div
        className={styles.avatar}
        style={photo ? undefined : { background: color || 'var(--color-brand)' }}
      >
        {photo ? (
          <img
            src={photo}
            alt={alt || name}
            className={styles.avatarImg}
            style={{
              width: `calc(${zoom} * 100%)`,
              height: `calc(${zoom} * 100%)`,
              transform: `translate(calc(-1 * ${fx}), calc(-1 * ${fy}))`,
            }}
            loading="eager"
          />
        ) : (
          <span className={styles.avatarInitial}>{initial || name.slice(-1)}</span>
        )}
      </div>

      <div className={styles.body}>
        <div className={styles.nameRow}>
          <span className={styles.name}>{name}</span>
          {enName && <span className={styles.enName}>{enName}</span>}
        </div>

        {slogan && <div className={styles.slogan}>「{slogan}」</div>}

        {titles.length > 0 && (
          <ul className={styles.titles}>
            {titles.map((t, i) => (
              <li key={i} className={styles.titlePill}>
                {t}
              </li>
            ))}
          </ul>
        )}

        {location && <div className={styles.location}>📍 {location}</div>}

        {links.length > 0 && (
          <div className={styles.links}>
            {links.map((l, i) =>
              l.external ? (
                <a
                  key={i}
                  href={l.href}
                  className={styles.linkBtn}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {l.icon && <span className={styles.linkIcon}>{l.icon}</span>}
                  {l.label}
                </a>
              ) : (
                <Link key={i} href={l.href} className={styles.linkBtn}>
                  {l.icon && <span className={styles.linkIcon}>{l.icon}</span>}
                  {l.label}
                </Link>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}
