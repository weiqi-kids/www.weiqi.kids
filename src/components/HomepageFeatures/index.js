import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '圍棋文化',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        圍棋文化蘊含深厚哲理，融合東方智慧與對弈禮儀。透過棋局交鋒、心法傳承，啟發思維、陶冶性情，推動世代共學與文化交流。
      </>
    ),
  },
  {
    title: '工程技術',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        軟體工程結合邏輯思維與團隊協作，透過系統開發流程打造穩定、可擴充的應用程式，成為推動圍棋教育與科技融合的關鍵力量。
      </>
    ),
  },
  {
    title: '多語共學',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        多語共學促進跨文化理解與全球視野。透過多語環境學習圍棋，孩子能在語言與邏輯的交錯中，拓展認知並深化世界連結。
      </>
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
