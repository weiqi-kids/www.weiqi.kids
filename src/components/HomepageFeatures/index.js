import clsx from 'clsx';
import Heading from '@theme/Heading';
import Translate from '@docusaurus/Translate';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: <Translate id="homepage.feature.multilingual.title">11 種語言</Translate>,
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <Translate id="homepage.feature.multilingual.desc">
        全球唯一支援 11 種語言的圍棋學習平台，包含繁中、簡中、粵語、英、日、韓、西、葡、印地、印尼、阿拉伯語，讓圍棋文化無國界傳播。
      </Translate>
    ),
  },
  {
    title: <Translate id="homepage.feature.katago.title">繁中 KataGo 文件</Translate>,
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <Translate id="homepage.feature.katago.desc">
        全網唯一繁體中文 KataGo 技術文件，為台灣工程師提供完整的圍棋 AI 開發資源，降低入門門檻。
      </Translate>
    ),
  },
  {
    title: <Translate id="homepage.feature.free.title">免費公開</Translate>,
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <Translate id="homepage.feature.free.desc">
        所有內容免費公開，無需註冊即可閱讀。我們相信知識應該無障礙分享，讓每個人都能享受圍棋帶來的智慧與樂趣。
      </Translate>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
