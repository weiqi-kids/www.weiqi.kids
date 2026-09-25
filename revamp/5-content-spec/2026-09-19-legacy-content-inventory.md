# 舊站來源盤點清單（第一版）

版本：0.1（2026-09-19）  
狀態：來源盤點；尚未核准任何逐頁轉移、公開或刪除。

本清單把目前 Git repository 中的舊站內容分組，作為逐頁判定的入口。分組不是最終 disposition；只有完成逐頁確認、權利確認與 SEO 判斷後，才可標記為 A（轉寫）、B（歷史）、C（不公開）或 D（淘汰）。ADR、agents 文件、分析報告與新規格文件不屬於舊站內容來源。

第一輪逐頁來源表見 [`2026-09-20-legacy-page-inventory.md`](2026-09-20-legacy-page-inventory.md)。

## 1. 來源摘要

| 來源群組 | 目前檔案範圍 | 初步內容類型 | 初步候選去處 | 必須確認 |
| --- | --- | --- | --- | --- |
| 協會介紹與制度 | `docs/about/index.md`、`intro.md`、`cooperation.md`、`meetings.md` | 組織介紹、合作、會議 | 改寫為主辦者介紹，部分歷史保存 | 現行主張、合作關係、對外承諾 |
| 成員與創會人物 | `docs/about/members/` | 人物、職務、創會歷史 | 歷史索引或經本人同意後的公開人物資料 | 是否仍有效、姓名與職稱、公開同意 |
| 活動與合作 | `docs/about/activities/` | 活動、影片、夥伴、公益、社群 | 歷史資料；仍有價值者改寫為案例 | 影像／姓名／商標權利、日期、是否仍合作 |
| 地點 | `docs/about/locations/` | 區域與活動地點 | 歷史保存或併入協會歷史 | 是否仍提供服務，避免造成現行據點誤解 |
| 內部行政 | `docs/about/internal/` | SOP、財務、收據、內部雜項 | C：管理員限定保存 | 個資、付款、內部流程與留存期限 |
| 圍棋入門與學習 | `docs/learn/` | 教學、歷史、AI 時代 | A 候選：公開知識內容或課程背景；其餘 B | 是否服務共學營定位、內容更新與 SEO |
| AlphaGo 技術教材 | `docs/alphago/` | 技術教材、研究脈絡 | A 候選：教材／知識庫；不等於新課程 | 版權、技術版本、是否需重新編排 |
| AI 與產業技術 | `docs/tech/` | 產業、實作、技術與研究 | A 候選：技能單張原始素材或公開文章 | 是否能轉為「問題／原因／方案／流程」格式 |
| 動畫與互動 | `docs/animations/`、`src/components/D3Charts/` | 視覺化與互動內容 | A、B 或 D 待判定 | 執行環境、維護成本、實際用途 |
| 自訂入口頁 | `src/pages/`、`src/data/` | 首頁、FAQ、研究、應用、人物／活動資料 | 逐頁重寫、併入新資料模型或歷史保存 | SEO 流量、資料來源、是否含未確認人物資料 |

## 2. 逐頁盤點的來源範圍

以下是第一輪需要逐頁建立 inventory 的路徑。`index.md` 與 `_category_.json` 也要檢查，因為它們可能同時是內容與導覽來源。

### 協會、人物、活動與歷史

- `docs/about/index.md`
- `docs/about/intro.md`
- `docs/about/cooperation.md`
- `docs/about/meetings.md`
- `docs/about/members/`
- `docs/about/activities/`
- `docs/about/locations/`

### 內部資料

- `docs/about/internal/`

這一群預設先進入 C 候選，不會因為檔案位於公開 repository 就當成可公開內容。

### 學習與技術內容

- `docs/learn/`
- `docs/alphago/`
- `docs/tech/`
- `docs/animations/`

這一群不能直接等同於共學營教材。要逐篇判斷是否能支持「跨產業理解 AI 應用」或轉寫成某門課的技能單張。

### 自訂頁面與資料檔

- `src/pages/index.js`
- `src/pages/faq.js`
- `src/pages/apps.js`
- `src/pages/intel.js`
- `src/pages/research.js`
- `src/pages/impact-report/2026.js`
- `src/data/activities.js`
- `src/data/members.js`
- `src/data/links/`

自訂頁面的公開文案與資料檔要分開判斷；資料檔中的人物或機構資料不自動建立新帳號、會員或學員資料。

## 3. 逐筆 inventory 欄位

下一版每一筆來源都要填滿下表，不用「整個資料夾保留」代替逐筆判斷：

| 欄位 | 判斷方式 |
| --- | --- |
| source_path、現有 URL | 以 git 檔案與 Docusaurus route 為準 |
| title、content_type | 讀取 front matter 與正文確認，不依檔名猜測 |
| current_claim | 摘錄目前仍對外作出的主張，例如免費、會員、合作或課程承諾 |
| new_role | 共學營價值、課程、技能單張、案例、協會介紹、歷史或無 |
| disposition | A、B、C、D 或待判定 |
| destination、public_level | 新 URL、歷史索引、登入後、實作營內、管理員限定或不公開 |
| owner、consent_status | 誰能確認內容與公開權利；未知就標記未確認 |
| seo_action | 保留、改寫、301、canonical、noindex 或不適用 |
| privacy_risk、reason | 個資／醫療／法律／付款／影像等風險與決策理由 |

## 4. 優先逐頁判定順序

1. `src/pages/index.js`、`src/pages/faq.js`、`docs/about/index.md`、`docs/about/intro.md`：先解除協會舊定位、免費或無會員費等衝突。
2. `docs/about/cooperation.md`、`docs/about/activities/`、`src/pages/impact-report/`：確認哪些是歷史資料，哪些仍可作為共學營案例。
3. `docs/tech/`、`docs/learn/ai-era/`、`docs/tech/hands-on/`：評估能否轉為課程技能單張或公開知識內容。
4. `docs/about/members/`、`src/data/members.js`：先做身份、更新日期與公開同意檢查，再決定公開、歷史或不公開。
5. `docs/about/internal/`、付款與行政資料：確認管理員保存範圍與刪除／留存政策。
6. 翻譯來源、重複頁、空頁與互動元件：最後處理，不在繁中主流程穩定前建立多語系頁面。

## 5. 第一版盤點結論

- 舊站資料不是整批搬遷，也不是全部丟棄；新站只接收有明確新角色的資料。
- 協會的歷史、人物與活動資料應與現行共學營課程資料分開，避免訪客把歷史紀錄誤解成目前的招生承諾。
- 舊站內容中的免費、無會員費或舊合作主張必須逐頁處理，不能沿用到新站。
- `members.js` 與人物頁面先視為內容來源，不視為已同意加入新站、已成為會員或可登入的帳號資料。
- 真正開始轉移前，必須先完成本清單的逐頁版本，並確認歷史資料的保存位置與公開程度。
