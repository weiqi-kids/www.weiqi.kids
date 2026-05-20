---
title: 改版策略計劃書
project: weiqi.kids
phase: 4-strategy
author: Writer
date: 2026-05-20
status: draft
based_on:
  - revamp/0-positioning/positioning.md (v2)
  - revamp/1-discovery/discovery.md (2026-05-14)
  - revamp/2-competitive/competitive.md (2026-05-20)
  - revamp/3-analysis/analysis.md (2026-05-20)
---

# 改版策略計劃書

## 執行摘要

| 項目 | 內容 |
|------|------|
| 改版目標 | 將首頁主敘事從「開源研究社群」轉為「商界夥伴 × AI × 圍棋」，補上會員實績、合作 CTA、量測機制 |
| 改版範圍 | **新增 2** 個頁面（合作邀請、年度成果報告）/ **大改 1** 個頁面（首頁）/ **修改 11** 個會員檔案 / **優化 4** 個既有頁面（research/intel/apps/about） |
| 分期數量 | **3 個 Phase**（基礎修復 → 核心改版 → 深化護城河） |
| 預計 KPI | 90 天內：mailto 點擊 ≥ 30、About 頁停留中位數 ≥ 90s、會員實績頁 PV ≥ 50、GitHub 外連 ≥ 100、實際 inbound 商務洽談 ≥ 5 件 |
| 涵蓋 gap | G1-G14（含 1 個 P2 後補項目「repo README 對齊」於 5-content-spec 補登 G15）|

---

## 1. 改版範圍

### 1.1 範圍總覽

| 類型 | 數量 | 頁面／檔案列表 |
|------|------|--------------|
| **新增** | 2 | (a) `/about/cooperation`（合作邀請＋棋會邀請流程說明）；(b) `/impact-report/2026/`（年度開源成果報告 / 一頁式 + PDF） |
| **大改** | 1 | `src/pages/index.js` 首頁（Hero / Stats / 三大軸線 / 案例敘事 / USP 宣告） |
| **修改** | 11 | `docs/about/members/*.md` 11 位會員實績卡（本業＋協會貢獻＋對應 AI 工具雙向連結） |
| **優化** | 4 | (a) `src/pages/research.js`、(b) `apps.js`、(c) `intel.js`（加產業標籤 + 「主要貢獻會員」欄）；(d) `docs/about/intro.md`（補使命／願景／歷程） |
| **修改設定** | 3 | `docusaurus.config.js`（Umami 注入＋Meta description）/ `i18n/{11 語系}/code.json`（hero stats / tagline）/ `i18n/{11 語系}/docusaurus-theme-classic/navbar.json`（navbar 結構） |

### 1.2 排除範圍（本次改版不做）

| 項目 | 排除原因 |
|------|----------|
| 視覺設計大改（重新設計 logo / 配色） | positioning §4 已定調簡約克制；當前視覺與定位無衝突，不需要重做 |
| 4 個 SOP 內部文件填充（G14） | 內部運營，不影響對外受眾；延後到 5-content-spec 或之後 |
| 4 個 locations 檔案（高雄／台中／台北／雲林）| L3 次要受眾、非本次主敘事範圍 |
| 11 語系 L1 主敘事的「獨立撰寫英文版」（G13） | 規模太大，本次採機械翻譯後人工抽查；獨立撰寫延後到 Phase 4（本計劃外） |
| 付費會員制 / 認證體系 | positioning §3 明確「不做付費會員制」 |

---

## 2. 優先級排序

### 2.1 Impact-Effort 矩陣

| ID | 項目 | Impact | Effort | 象限 | 優先級 |
|----|------|--------|--------|------|--------|
| G6 | Umami 追蹤碼注入 | 高（KPI 量測前提）| 低（10 行 config）| **Quick Win** | P0a-pre |
| G5 | Hero 重寫 | 高 | 低（文案＋11 語系翻譯）| **Quick Win** | P0a |
| G1 | mailto CTA | 高 | 低 | **Quick Win** | P0a |
| G4 | 「不收費」USP 宣告 | 高 | 低 | **Quick Win** | P0a |
| Meta description 更新 | 高 | 低 | **Quick Win** | P0a |
| navbar 重設計（加「合作」「夥伴」入口）| 高 | 中（11 語系翻譯）| **Quick Win** | P0a |
| G7 | 實體棋會公開（含 G2 棋會邀請 CTA）| 高（Persona B 決策依賴）| 中（需用戶確認實際狀況）| **Major** | P0a |
| G2 | 11 位會員實績卡 | 高 | **高**（11 × 內容生產）| **Major** | P0b |
| G3 | 會員 ↔ 工具雙向連結 | 高（網狀結構效益）| 中（每個專案頁加 1 欄位）| **Major** | P0b |
| G8 | 首頁合作案例敘事區（3-5 個案例卡）| 高 | 中（理事長主筆）| **Major** | P0b |
| G11 | 會員 testimonial 引言 | 中 | 低（每人 1-2 句）| **Quick Win** | P0b |
| G9 | 議題化分類（產業標籤）| 中 | 中（盤點 + tag 設計）| **Major** | P1 |
| G10 | 年度開源成果報告 | 高（對外可引用資產）| 高 | **Major** | P1 |
| G12 | L3 入口統一（圍棋資源）| 低（次要受眾）| 中（navbar 子選單）| **Fill In** | P2 |
| G13 | 英文版深度內容 | 低（待國際流量驗證）| 高 | **Consider Later** | P2 |
| G14 | 4 個 SOP 文件 | 低（內部）| 中 | **Consider Later** | P2 |
| G15 | 40+ repo README 對齊新定位（3-analysis Reviewer 補列）| 中（次要受眾 C 入口）| 低（每 repo 1-2 行）| **Quick Win** | P2 |

### 2.2 優先級清單

#### P0a — 敘事骨幹（不可拆，Phase 1 一次到位）

| ID | 項目 | 原因 | 來源 |
|----|------|------|------|
| **G6** | Umami 追蹤碼注入（先做）| 不做則本計劃所有 KPI 量測失效 | 1-discovery §6 第 1 項；3-analysis G6 |
| **G5** | Hero 重寫為「商界夥伴 × AI × 圍棋」 | 主敘事與新定位對齊；意識階段第一印象 | 3-analysis G5 |
| **G1** | 首頁主 CTA「合作提案」mailto + 次級 CTA「實體棋會邀請」| Persona A/B 雙路徑入口 | 2-competitive 可借鏡 P0a；3-analysis 三大關鍵發現 #3 |
| **G4** | 「完全免費、無會員費、全部開源」USP 宣告（首頁第二屏）| 反向護城河 / 對 BNI/TAIA/YPO 差異化 | 2-competitive USP；3-analysis 4.2 |
| **Meta-desc** | Meta description 全站更新 | SEO 對齊新定位 | positioning §6 P0a |
| **navbar** | navbar 加「合作」「夥伴／實績」入口；移除 L3 平行擠占 | 主受眾入口缺失 | positioning §6 P1；3-analysis G12 |

#### P0b — 11 人實績與雙向連結（Phase 2）

| ID | 項目 | 原因 | 來源 |
|----|------|------|------|
| **G2** | 11 位會員實績卡（本業＋協會貢獻＋對應 AI 工具）| 「考慮」階段判斷依據 | 3-analysis G2 |
| **G3** | 會員頁 ↔ AI 工具頁雙向連結 | 結構性高槓桿點 | 3-analysis Reviewer 觀察 #1 |
| **G8** | 首頁合作案例敘事區（3-5 個案例卡）| 「夥伴提供問題 → AI 整合 → 開源產出」具體故事 | positioning §6 P0b；3-analysis G8 |
| **G11** | 11 位會員 testimonial 引言（每人 1-2 句）| 人格化敘事 | 2-competitive 可借鏡；3-analysis G11 |
| **G7** | 實體棋會時間／地點／頻率公開 | Persona B 決策依賴 | positioning 待補；3-analysis G7 |
| **about/intro** | `docs/about/intro.md` 補使命／願景／歷程 | 「考慮」階段內容缺失 | 1-discovery §2.2 |

#### P1 — 差異化護城河（Phase 3）

| ID | 項目 | 原因 | 來源 |
|----|------|------|------|
| **G9** | 議題化產業分類（research/apps/intel 加 industry tag）| 主受眾「考慮」階段篩選需求 | 2-competitive 可借鏡；3-analysis G9 |
| **G10** | 年度開源成果報告 2026（PDF + 一頁式網頁） | 對外可被引用的內容資產 | 2-competitive YPO 借鏡；3-analysis G10 |

#### P2 — 後補項目（不阻擋本次改版，可滾動排程）

| ID | 項目 | 原因 | 來源 |
|----|------|------|------|
| **G12** | L3「圍棋資源」單一入口收斂 | Persona D 入口優化 | positioning §6 P2；3-analysis G12 |
| **G13** | 英文版主敘事獨立撰寫 | 國際 SEO；待 Umami 數據驗證需求後啟動 | positioning §6 P2；3-analysis G13 |
| **G14** | 4 個 SOP 文件填充 | 內部運營 | 1-discovery §2.2 |
| **G15** | 40+ repo README 對齊新定位 | Persona C 次要受眾入口 | 3-analysis Reviewer 補列 |

---

## 3. 分期規劃

### Phase 1：敘事骨幹（P0a，目標 ≤ 5 個工作日完成）

| 項目 | 內容 |
|------|------|
| 目標 | 把首頁敘事與量測機制一次到位，使 Phase 2 內容生產可被 KPI 反饋 |
| 範圍 | docusaurus.config.js / src/pages/index.js / navbar.json / 11 語系 code.json |
| 成功指標 | (a) Umami 訪客數 ≥ 0 連續 3 天上報；(b) 首頁 hero 文案上線 + 11 語系；(c) mailto 點擊事件可在 Umami 後台看見 |

#### 包含項目

| ID | 項目 | 類型 | 順序 |
|----|------|------|------|
| G6 | Umami 注入（先做、單獨 commit）| 修改設定 | 1 |
| G5 | Hero 重寫 | 大改首頁 | 2 |
| G4 | 「不收費」USP 宣告 | 大改首頁 | 2（與 G5 同 commit）|
| G1 | mailto + 棋會邀請雙 CTA | 大改首頁 | 3 |
| navbar | 加「合作」「夥伴」入口（先英中兩語）| 修改 navbar | 4 |
| Meta-desc | 全站 Meta description 更新 | 修改設定 | 5 |
| **驗收** | 11 語系翻譯 + 線上部署 + Umami 後台確認 | — | 6 |

> **依賴關係**：G6 必須最先（否則 G5/G1 上線後就丟失「改版前後對照」基線）；其餘 G5/G4/G1 可同一個 commit 上線；navbar 與 Meta-desc 可作為第二個 commit。

### Phase 2：11 人實績與雙向連結（P0b，目標 ≤ 3 週完成）

| 項目 | 內容 |
|------|------|
| 目標 | 把 11 位會員從「簡介型」升級為「實績卡」，並把 40+ 專案與會員雙向掛鉤 |
| 範圍 | 11 個會員 .md / 約 20 個專案頁的「主要貢獻會員」欄位 / 首頁合作案例區塊 / about/intro.md |
| 成功指標 | (a) 11 位會員實績卡上線；(b) 至少 5 個旗艦專案有「貢獻會員」連結；(c) 首頁 3-5 個案例卡上線；(d) about/intro.md 完成 |

#### 包含項目

| ID | 項目 | 類型 | 預計工時 |
|----|------|------|---------|
| G2 + G11 | 11 位會員實績卡（含 testimonial）| 修改 × 11 | 11 × 2h = 22h（含 11 位會員回饋等待）|
| G3 | 5 個旗艦專案頁加「貢獻會員」+ 11 位會員頁加「對應 AI 工具」| 修改 × 16 | 8h |
| G8 | 首頁 3-5 個案例卡（理事長主筆）| 新區塊 | 6h |
| G7 | 棋會頻率／時間公開 | 修改 activities/ | 2h（前提：用戶確認實際情況）|
| about/intro | 使命／願景／歷程補齊 | 修改 1 篇 | 4h |

#### 5 個旗艦專案選擇（按已知會員-專案 mapping 推測，需用戶確認）

| 旗艦專案 | 推測主要貢獻會員 | 驗證方式 |
|---------|---------------|---------|
| huatuo-ai | 鄭骨館（整復／中醫）| 5-content-spec 階段問用戶確認 |
| tax-ai | 會計師會員 | 同上 |
| six-hats | （目前不明）| 同上 |
| tapwater-ai | （目前不明）| 同上 |
| bless.link | （目前不明）| 同上 |

> 若 5 位 mapping 不完整，**降為 3 位確認 + 2 位 TBD**；不阻擋 Phase 2 推進，TBD 兩位先掛「尋找夥伴中」。

### Phase 3：差異化護城河（P1，可滾動 4-8 週）

| 項目 | 內容 |
|------|------|
| 目標 | 把 40+ 專案聚合成「年度成果報告」對外資產；加產業標籤讓主受眾篩選 |
| 範圍 | 新增 `/impact-report/2026/` 頁面 + PDF 產生器 / 修改 research.js / apps.js / intel.js 加 industry tag |
| 成功指標 | (a) 年度報告上線 + PDF 可下載 + 媒體可引用版本（OG 圖、引用格式）；(b) industry tag 3-4 個分類確定 + 至少 80% 專案有 tag |

#### G9 議題化分類定案（依現有專案分布）

**先盤點現有 40+ 專案的產業分布**（4-strategy 階段執行）：

| 推測產業 | 已知專案 | 數量 |
|---------|---------|------|
| 醫療健康 | huatuo-ai（中醫）/ tapwater-ai（淨水）/ EpiAlert（疫情）| ~3-5 |
| 財稅法規 | tax-ai / 全球框架法規變動 / 政策承諾追蹤 | ~3-4 |
| 教育學習 | bless.link / 學生學習地圖 / 圍棋 docs | ~3-5 |
| 圍棋研究 | go-second-best-move-formula / go-yose / animations | ~3-5 |
| 商業情報 | 電商產品研究 / 保健食品產品情報 / 23 條供應鏈 intel | ~5-10 |
| 資安監測 | 資安威脅情報中心 / 聲量監測 | ~2-3 |

**決議規則**：密度 ≥ 5 個的獨立成分類，其餘合併到「其他」或「跨領域」。**待 4-strategy 完成後實際數一次** —— 推估會有 4 個產業分類（醫療／財稅法規／教育圍棋／商業情報），其餘合併。

### Phase 4（本次改版範圍外，列入後續路線圖）

| ID | 項目 | 觸發條件 |
|----|------|---------|
| G12 | L3 圍棋資源入口收斂 | Phase 1-3 上線後 30 天觀察 Persona D 流量 |
| G13 | 英文版主敘事獨立撰寫 | Umami 顯示英文版流量 ≥ 中文版 10% |
| G14 | 4 個 SOP 內部文件 | 協會內部規模成長到需要時 |
| G15 | repo README 對齊 | Phase 3 完成後（年度報告可作為 README 統一引用源）|

---

## 4. 頁面規劃（重點頁面）

### 4.1 首頁（src/pages/index.js）大改

| 項目 | 內容 |
|------|------|
| 頁面 URL | `/` |
| 類型 | 大改 |
| 所屬階段 | Phase 1（Hero / CTA / USP）+ Phase 2（案例敘事區）|
| 優先級 | P0a + P0b |

#### 頁面策略

| 項目 | 說明 |
|------|------|
| 頁面目標 | 5 秒讓主受眾判斷「值得花時間了解」；30 秒讓他們點 CTA |
| 目標受眾 | Persona A（二代接班型）為主、Persona B（退休專業型）次之 |
| 關鍵訊息 | 1. 11 位跨域夥伴 × 40+ 開源 AI 工具<br>2. 完全免費、無會員費、全部開源<br>3. 商界 × AI × 圍棋三位一體 |
| 主要 CTA | 「合作提案」`mailto:lightman.chang@gmail.com?subject=...` |
| 次要 CTA | 「實體棋會邀請」（連結到 `/about/cooperation` 或 `/about/activities/`）|

#### 內容大綱

```
1. Hero
   - 標題：商界夥伴 × AI × 圍棋（一句話定位待定稿）
   - Subtitle：把產業 know-how 變成開源 AI 工具
   - Stats：「11 位跨域夥伴 · 41+ 開源 AI 工具 · 全部免費」
   - 主 CTA：合作提案 / 次 CTA：棋會邀請、認識夥伴

2. USP 宣告區（第二屏）
   - 「完全免費、無會員費、全部開源（CC-BY 4.0）」一行明確宣告
   - 對照：「不是純圍棋協會、不是純 AI 推廣、不是純商務人脈」

3. 三大合作面向（重命名 / 重排）
   - 案例 → 工具 → 論文（從具體到抽象遞進）
   - 對應 positioning §6 P1

4. 合作案例敘事區（Phase 2 加入）
   - 3-5 個敘事卡：「[會員 X] 提出 [問題] → 理事長 AI 整合 → 開源產出 [工具]」
   - 每卡可點進對應的會員頁＋工具頁（雙向連結 G3）

5. 11 位會員預覽
   - 跨產業組合可視化（律師＋ISO＋技術＋行銷＋醫療＋整復…）
   - CTA：認識所有會員 → 列表頁

6. 媒體聯播網 / 友站
   - 保留現有區塊
```

#### SEO 規劃

| 項目 | 內容 |
|------|------|
| 目標關鍵字 | 主：「中小企業 AI 導入」「開源 AI 工具」「跨域合作」；長尾：「圍棋 AI 商業」「公益開源專案」 |
| Title | 好棋寶寶協會 \| 跨域 × 開源 × AI 公益專案 |
| Meta Description | 11 位來自律師、ISO、技術、醫療等領域的夥伴，把產業 know-how 變成 41+ 個全部免費開源的 AI 工具。歡迎跨域合作。 |

---

### 4.2 合作邀請頁（`/about/cooperation`）新增

| 項目 | 內容 |
|------|------|
| 頁面 URL | `/docs/about/cooperation` 或 `/about/cooperation/` |
| 類型 | 新增 |
| 所屬階段 | Phase 1 |
| 優先級 | P0a |

#### 頁面策略

| 項目 | 說明 |
|------|------|
| 頁面目標 | 把首頁 CTA 點擊接住，明確「下一步」 |
| 目標受眾 | Persona A/B |
| 關鍵訊息 | 1. 兩條合作路徑（mailto / 棋會）<br>2. 期待回信時程（3 個工作日內）<br>3. 「我們會這樣分工」（理事長 AI 整合 × 夥伴領域知識） |
| 主要 CTA | mailto 直連 |
| 次要 CTA | 看會員（如果還沒看）/ 看案例 |

#### 內容大綱

```
1. 「兩條路徑」說明
   - A 路徑：mailto 寫信給理事長（適合主動類型）
   - B 路徑：實體棋會見面（適合先看人再說）

2. 期待回信時程：3 個工作日內

3. 分工模型說明
   - 理事長 = AI 整合工程師
   - 夥伴 = 領域專家
   - 對外產出 = CC-BY 4.0 公益專案

4. 已合作會員的 1-2 句感受（從 G11 testimonial 抽出 2-3 則）

5. mailto / 棋會時間表（再次強調 CTA）
```

---

### 4.3 11 位會員實績卡（`docs/about/members/*.md`）修改

| 項目 | 內容 |
|------|------|
| 類型 | 修改 × 11 |
| 所屬階段 | Phase 2 |
| 優先級 | P0b |

#### 頁面策略

| 項目 | 說明 |
|------|------|
| 頁面目標 | 把每位會員從「介紹型」變成「實績型」 |
| 目標受眾 | Persona A（找同產業）/ Persona B（看同儕）|
| 關鍵訊息 | 1. 本業 + 產業標籤<br>2. 協會內貢獻（具體案例＋與哪些 AI 工具相關）<br>3. 1-2 句 testimonial 引言（本人撰寫）|
| 主要 CTA | 連到對應 AI 工具頁 / 連到合作邀請頁 |

#### 統一模板

```yaml
---
title: [姓名]（[本業簡稱]）
industry_tags: [醫療 / 法律 / 財稅 / 教育 / 圍棋 / 商業情報 / 資安]
ai_tools: [專案 slug 列表]
---

## 本業
[1-2 段，背景＋累積年資＋專業領域]

## 協會內貢獻
[具體列出對哪些 AI 工具有貢獻、提供什麼知識／案例／審閱]

## 對應 AI 工具
- [工具 1]（連結到工具頁，工具頁也反向連回此會員頁）
- [工具 2]

## 引言（本人撰寫）
> [1-2 句個人理念或合作感受]

## 聯繫方式
- 圍棋棋會見面（推薦）
- 透過理事長轉介：[mailto]
```

---

### 4.4 年度成果報告（`/impact-report/2026/`）新增

| 項目 | 內容 |
|------|------|
| 頁面 URL | `/impact-report/2026/`（中英對照 + PDF 下載）|
| 類型 | 新增 |
| 所屬階段 | Phase 3 |
| 優先級 | P1 |

#### 頁面策略

| 項目 | 說明 |
|------|------|
| 頁面目標 | 提供對外可被引用的數據資產，同時是給創始會員的禮物 |
| 目標受眾 | 媒體 / 學術引用 / 未來潛在合作對象 / 11 位會員本人 |
| 關鍵訊息 | 1. 11 位夥伴 × 41+ 專案 × 11 語系覆蓋的總體影響<br>2. 6 個產業分類的代表案例<br>3. CC-BY 4.0 全部公開 |

#### 內容大綱

```
1. 總體影響數字
   - 41+ 開源專案、X 個 GitHub stars、Y 國語系覆蓋
   - 累計 commits / contributors / 媒體引用數

2. 6 個產業分類各 1-2 個旗艦案例

3. 11 位夥伴的跨域組合圖

4. 引用格式（媒體 / 論文）

5. PDF 下載按鈕
```

---

## 5. 成功指標

### 整體 KPI（90 天內）

| 指標 | 當前基準 | Phase 1 後 | Phase 2 後 | 最終目標（Phase 3 後）| 來源 |
|------|----------|-----------|-----------|-------------------|------|
| Umami 連續上報天數 | 0 | ≥ 3 天 | ≥ 30 天 | ≥ 90 天 | positioning §5 前提 |
| mailto CTA 點擊（90 天累計）| 不適用 | ≥ 0（基線建立）| ≥ 10 | **≥ 30** | positioning §5 Leading |
| About 頁停留時間中位數 | 無數據 | 無數據 | ≥ 60s | **≥ 90s** | positioning §5 Leading |
| 會員實績頁 PV（90 天累計）| ≈ 0 | 無數據 | ≥ 20 | **≥ 50** | 3-analysis 校準 |
| GitHub 外連事件（90 天）| 無事件 | ≥ 0 | ≥ 30 | **≥ 100** | positioning §5 Leading |
| 品牌字 + AI 聯合查詢 GSC impression | 未驗證 | 出現於 GSC | ≥ 50 | **≥ 200** | positioning §5 Leading |
| 實際 inbound 商務洽談 | 0 | 0-1 | 1-3 | **≥ 5 件** | positioning §5 **Hard** |
| 由 inbound 轉合作案例 | 0 | 0 | 0-1 | **≥ 1 件** / 半年 | positioning §5 **Hard** |

### 各階段驗收標準

#### Phase 1 驗收（敘事骨幹）

- [ ] Umami 後台連續 3 天有訪客數據上報
- [ ] 首頁 Hero / Stats / USP / CTA 與本計劃 §4.1 對齊
- [ ] 11 語系 hero stats / tagline 全部更新（41+ 數字以實際為準）
- [ ] navbar 含「合作提案」入口
- [ ] Meta description 全站更新
- [ ] mailto CTA 在 Umami 後台可看見點擊事件
- [ ] 線上 `curl -s https://www.weiqi.kids/ | grep "a4b64b22"` 命中 1 筆（Umami 注入驗證）
- [ ] 通過 CLAUDE.md 任務完成品質關卡（連結 / SEO / Schema / Git）

#### Phase 2 驗收（11 人實績）

- [ ] 11 位會員 .md 全部依模板補齊（產業標籤＋AI 工具＋testimonial 引言）
- [ ] 至少 5 個旗艦 AI 工具頁有「主要貢獻會員」連結
- [ ] 首頁案例敘事區有 ≥ 3 個案例卡
- [ ] about/intro.md 不再有「待補充」
- [ ] 棋會頻率／時間／地點明確公開於 activities/index.md

#### Phase 3 驗收（差異化護城河）

- [ ] `/impact-report/2026/` 頁面上線（中文版優先）
- [ ] PDF 可下載（建議 ≤ 2 MB）
- [ ] 6 個產業分類在 research/apps/intel 任一處可被使用者篩選
- [ ] ≥ 80% 專案掛上 industry tag

---

## 6. 風險與依賴

### 風險

| 風險 | 影響 | 緩解措施 |
|------|------|----------|
| Phase 2 內容生產仰賴 11 位會員回饋，可能延遲 | Phase 2 拖延 | 預設「先做 3-5 位高貢獻會員 + 對應 5 個旗艦專案」作為最小可上線範圍；其餘等回饋 |
| 「棋會」實際頻率與定位文件假設不符 | Persona B 雙 CTA 失效 | 4-strategy 完成前必須由用戶確認；若無實體棋會，棋會邀請 CTA 改為「未來活動 - 訂閱通知」 |
| Persona A/B 假設可能在 Umami 啟用後被推翻 | 內容方向需大調整 | Phase 1 上線後 30 天先觀察流量結構再開始 Phase 2 大幅內容生產 |
| 41 個專案的具體 mapping 不完整（誰主筆誰提供知識）| G3 雙向連結缺資料 | 5-content-spec 階段做 1 次 mapping 表 review；缺資料的工具標「尋找夥伴中」 |
| Meta description 變更影響既有 SEO 排名 | 短期搜尋流量波動 | 用 GSC 監測；保留舊關鍵字字串作為 ranking 緩衝（如「圍棋」「AI」） |

### 依賴

| 依賴項目 | 影響範圍 | 處理方式 |
|----------|----------|----------|
| Umami Website ID（CLAUDE.md 已記載）| Phase 1 G6 | 直接複製 ID `a4b64b22-906d-4918-934a-7da72b5aced9` |
| `lightman.chang@gmail.com` 信箱 SPAM 過濾規則 | Phase 1 G1 mailto 可用性 | 改版上線前 1 週用戶手動驗證 inbox 規則 |
| 11 位會員的 testimonial 引言 | Phase 2 G11 | 改版前 2 週由理事長個別徵詢 |
| GSC 已驗證網站所有權 | KPI 量測 | 改版前 1 週確認；若未驗證，加 verification meta tag |

---

## 7. 追溯表

| 計劃項目 | 來源階段 | 來源 ID / 章節 | 說明 |
|----------|----------|---------------|------|
| G6 Umami 注入 | 1-discovery | §6 第 1 項 P0 | discovery 已給 docusaurus.config.js patch |
| G5 Hero 重寫 | 0-positioning + 3-analysis | positioning §6 P0a / analysis G5 | 主敘事與新定位對齊 |
| G1 mailto + 棋會雙 CTA | 0-positioning + 2-competitive + 3-analysis | positioning §6 P0a / competitive 可借鏡 / analysis #3 | 雙 Persona 雙路徑 |
| G4 「不收費」USP | 2-competitive + 3-analysis | competitive USP / analysis 4.2 | 反向護城河對 BNI/TAIA/YPO |
| navbar 重設計 | 0-positioning | §6 P1 | 從「給棋友／工程師」改為「夥伴／實績／工具／圍棋」 |
| G2 + G11 11 位會員實績卡 | 0-positioning + 3-analysis | positioning §6 P0b / analysis G2 G11 | 從簡介型升級 |
| G3 會員 ↔ 工具雙向連結 | 3-analysis Reviewer | 3-analysis review 觀察 #1 | 結構性高槓桿點 |
| G8 案例敘事區 | 0-positioning + 3-analysis | positioning §6 P0b / analysis G8 | 由理事長主筆 |
| G7 棋會公開 | 1-discovery + 3-analysis | discovery §6 第 4 項 / analysis G7 | Persona B 決策依賴 |
| G9 議題化分類 | 2-competitive + 3-analysis | competitive 可借鏡 / analysis G9 | YPO 5 大主題範式 |
| G10 年度成果報告 | 2-competitive Reviewer + 3-analysis | competitive review 觀察 #2 / analysis G10 | YPO Global Impact Report |

---

## 8. 待用戶確認事項（4-strategy 完成前必答）

| # | 事項 | 影響 | 急迫性 |
|---|------|------|--------|
| 1 | 協會目前是否有定期實體棋會？頻率／地點？ | 棋會邀請 CTA 是否可用（G1 次級 CTA + G7） | 🔴 高（Phase 1 範圍）|
| 2 | 11 位會員中，5 位旗艦專案的明確 mapping | Phase 2 §3.5 旗艦專案-會員對照 | 🟡 中（Phase 2 開始前）|
| 3 | 是否要把「合作提案」信箱從個人 gmail 升級為協會專屬信箱（如 hi@weiqi.kids）？| 信箱品牌一致性 | 🟢 低（Phase 1 可先用 gmail，後續升級）|
| 4 | Phase 2 是否先做 3-5 位高貢獻會員，其餘隨後？ | Phase 2 範圍縮減 | 🟡 中 |
| 5 | 年度成果報告（G10）是否在 Phase 3 啟動，或延到本次改版後？ | Phase 3 範圍 | 🟢 低（可彈性）|
