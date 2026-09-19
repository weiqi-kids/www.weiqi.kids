import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Translate, {translate} from '@docusaurus/Translate';
import FAQ from '@site/src/components/SEO/FAQ';
import styles from './landing.module.css';

/**
 * 全站常見問題頁（/faq/）
 *
 * 立案理由：反思層自 2026-09-02 起連續多日列為「需要你決定」，2026-09-19 站主拍板要做。
 * 它同時補上 GEO／AI 引用就緒度稽核唯一還沒亮綠的選配項——該稽核的判準是
 * 「sitemap 裡有 /faq/ 型網址，且該頁帶 FAQPage 結構化資料」（見 seo-ops lib/geo.mjs
 * 的 discoverFaqUrl 與 checks.faq），所以問答一律走既有的 SEO/FAQ 元件，由它產 JSON-LD。
 *
 * 答案內容全部取自首頁已核准上線的文案與數字（i18n/zh-tw/code.json 的 homepage.* 鍵），
 * 沒有新造事實。要改數字請連同首頁一起改，不要只改這裡。
 */
export default function FaqPage() {
  const description = translate({
    id: 'faq.meta.description',
    message: '好棋寶寶協會常見問題：協會在做什麼、工具與論文怎麼取用、小孩想學圍棋有哪些活動、想合作從哪裡開始。',
  });

  const items = [
    {
      question: translate({id: 'faq.q.about', message: '這個協會到底在做什麼？'}),
      answer: translate({
        id: 'faq.a.about',
        message: '三件事：辦圍棋活動、把夥伴各自的專業做成開源 AI 工具、寫論文。活動辦過 10 個縣市，留下 187 張現場紀錄；工具目前 41 個以上，橫跨 11 大產業，全部 CC-BY 4.0 開源；論文 9 篇，每篇都附完整數學證明和互動視覺化網頁。37 位夥伴裡有律師、會計、ISO 顧問、中醫、健康教育、行銷、活動，本來各做各的行業。',
      }),
    },
    {
      question: translate({id: 'faq.q.whyAi', message: '一個圍棋協會，為什麼在做 AI 工具？'}),
      answer: translate({
        id: 'faq.a.whyAi',
        message: '2016 年 AlphaGo 贏了李世乭，圍棋界是最早被 AI 掃過一輪、也最早開始用它的地方。那份經驗後來被帶到健康、產業、教育和公共政策上。圍棋是這群人認識彼此的起點，後面做什麼就跟著各自的專業走。',
      }),
    },
    {
      question: translate({id: 'faq.q.free', message: '網站上的工具和文件要付費嗎？可以商用嗎？'}),
      answer: translate({
        id: 'faq.a.free',
        message: '免費，不用註冊就能讀。AI 工具是 CC-BY 4.0，標明來源就能自由使用，商用也可以。',
      }),
    },
    {
      question: translate({id: 'faq.q.kids', message: '我家小孩想學圍棋，協會有活動可以參加嗎？'}),
      answer: translate({
        id: 'faq.a.kids',
        message: '有。嘉年華、城市對抗賽、全國公開賽、主題講座都辦過，對象以親子、學齡兒童和社區為主，合作單位包括台北市體育總會和弈客圍棋平台。活動頁放了歷年現場紀錄，看照片比看文字準。2026 年會開始往社區走，想邀長輩帶孫子一起下。',
      }),
    },
    {
      question: translate({id: 'faq.q.papers', message: '學術論文在哪裡看？'}),
      answer: translate({
        id: 'faq.a.papers',
        message: '網站的「學術研究」頁列了 9 篇，主題是圍棋理論、機器學習，還有幾個經典數學問題。每篇除了論文本身，都有互動視覺化網頁和 GitHub 原始碼。',
      }),
    },
    {
      question: translate({id: 'faq.q.collab', message: '想合作的話，從哪裡開始？'}),
      answer: translate({
        id: 'faq.a.collab',
        message: '首頁有「合作提案」和「先見面聊聊」兩個入口，挑一個就好。現在的跨域夥伴大多是這樣進來的。',
      }),
    },
  ];

  return (
    <Layout
      title={translate({id: 'faq.meta.title', message: '常見問題 — 好棋寶寶協會'})}
      description={description}>
      <header className={styles.heroSlim}>
        <div className="container">
          <Heading as="h1">
            <Translate id="faq.hero.title">常見問題</Translate>
          </Heading>
          <p className={styles.heroDesc}>
            <Translate id="faq.hero.desc">
              協會、網站上的工具、活動與合作，這裡放常被問到的幾個問題。
            </Translate>
          </p>
        </div>
      </header>
      <main className="container margin-vert--lg">
        <FAQ items={items} title={translate({id: 'faq.section.title', message: '問與答'})} />
      </main>
    </Layout>
  );
}
