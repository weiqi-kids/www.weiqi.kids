---
title: 5-Content-Spec 內容規格審查結果
project: weiqi.kids
phase: 5-content-spec/review
author: Reviewer
date: 2026-05-20
status: approved-with-conditions
based_on: revamp/5-content-spec/content-spec.md (2026-05-20)
---

# 內容規格審查結果

## 審查結果：✅ 通過（附條件）

> 規格書涵蓋 strategy.md 全部 4 個重點頁面 + navbar + Meta description（共 6 個規格節 §A-§F）；§G 共用檢查清單 + §H 待用戶素材清單 + §I 追溯表構成完整工作交接。
> 進入執行階段前，§H 第 1、7 項（廖宜鋒資料 + 三位實際合作會員的 testimonial 引言）為 Phase 2 必須取得；Phase 1 不阻擋。

---

## 檢查摘要

| 類別 | 結果 | 說明 |
|------|------|------|
| 涵蓋率 | ✅ | strategy.md §4.1-§4.4 全部 4 個頁面 + navbar + Meta-desc 共 6 個規格節 |
| 完整性 | ✅ | 每個頁面有目標／受眾／訊息／結構／SEO／視覺／檢查清單；§A 與 §C 還含 11 語系策略 |
| 具體性 | ✅ | 每個區塊有完整文案稿（A1 中英文 / A2 USP / B1 雙路徑 / B2 分工模型）；不是只給「方向」 |
| 可執行性 | ✅ | §C 12 位會員變異點表可直接套用到 frontmatter；§F Meta description 11 個關鍵頁字數已限縮在 155 字內 |
| 一致性 | ✅ | 全程套用「無棋會分支」（§B / §C 引言類型分流 / §D6 商業情報案例對應）；品牌調性禁用詞清單對齊 positioning §4 |
| 追溯性 | ✅ | §I 追溯表 6 條皆指明對應 strategy.md 章節與 gap ID |

---

## 涵蓋率檢查

| strategy.md 頁面 | 規格書節 | 狀態 |
|-----------------|---------|------|
| §4.1 首頁大改 | §A | ✅ |
| §4.2 合作邀請頁 | §B | ✅（含 §9.3 無棋會分支文案）|
| §4.3 會員實績卡 | §C | ✅（含 12 位變異點 + frontmatter 模板）|
| §4.4 年度成果報告 | §D | ✅ |
| §3 Phase 1 navbar | §E | ✅ |
| §3 Phase 1 Meta-desc | §F | ✅（11 個關鍵頁全列出）|

---

## 抽樣驗證結果

### 首頁規格（§A）

| 測試 | 結果 | 說明 |
|------|------|------|
| 可執行測試 | ✅ | 直接可進入 `src/pages/index.js` 修改階段，文案稿（中／英）齊備 |
| 可驗收測試 | ✅ | 「Hero stats 文字」「mailto subject 字元」「H2 文字」皆有明確規格 |
| Schema 完整性 | ✅ | 7 種必填 Schema 中含 WebPage / Organization / BreadcrumbList / FAQPage（4 種）；剩 3 種（Article / Person / ImageObject）依 CLAUDE.md §2.2 為「條件式」或在會員頁／案例頁出現 — **可接受** |

### 會員實績卡規格（§C）

| 測試 | 結果 | 說明 |
|------|------|------|
| 12 位變異點完整 | ✅ | 11 位現有 + 廖宜鋒（待補）= 12 列；slug / industry_tags / ai_tools / 引言類型全有 |
| AI 工具掛鉤誠實度 | ✅ | 只有 3 位（張饒輝 / 黃子彥 / 廖宜鋒）有實際工具掛鉤；其餘 9 位明確標「未來合作意向」+「加入協會的期望」引言類型 — 與 member-project-mapping memory 一致 |
| 內容模板可執行 | ✅ | 6 區塊（本業 / 協會內貢獻 / 對應 AI 工具 / 引言 / 跨域連結 / 聯繫方式）含字數規格 |

### 「無棋會分支」套用一致性

| 檢查點 | 結果 |
|------|------|
| §A 首頁無「實體棋會」按鈕 | ✅（雙 mailto subject）|
| §B 合作頁「兩種合作目的」非「兩條路徑」 | ✅ |
| §C 引言類型分流（黃子彥/廖宜鋒/張饒輝 vs 其餘 9 位）| ✅ |
| §F Meta-desc 不暗示「已有定期棋會」 | ✅ |
| §B5 「未來棋會意向徵詢」弱訊息 | ✅ |

---

## 問題清單

### 必須修改（Blockers）

**無**。規格書可直接進入 Phase 1 / Phase 2 執行。

### 建議改善（Suggestions，可後補不阻擋）

1. **§A4 第 2 卡（廖宜鋒 × tax-ai）標「廖宜鋒資料補登後啟用」** — 用戶會上傳後一次補齊。**建議**：在執行 Phase 2 時，若用戶資料未到，第 2 卡用「**[會員 Y]（待補）× tax-ai**」placeholder，避免空卡造成視覺破洞；若到上線日仍未到，第 2 卡可暫換為「**張饒輝 × six-hats**」「**張饒輝 × bless.link**」等理事長獨力的案例（敘事仍可成立但說服力較弱）。
2. **§C「廖宜鋒 slug 待用戶決定，建議 james-liao」** — 建議 5-content-spec Reviewer 階段直接定 slug，避免之後追補時兩個位置（content-spec / 實際檔案）對不上。**建議**：採用 `james-liao`（與既有 `dawn-lee`、`ian-kuo` 一致風格）作為預設值；用戶若不同意可在資料上傳時改 1 行。
3. **§E navbar 8 個非英文語系（zh-cn / zh-hk / ja / ko / es / pt / hi / id / ar）翻譯空白** — 不阻擋 Phase 1 zh-tw / en 上線，但 §H 第 6 項應抬升急迫性為 🔴 高（與 Phase 1 同步完成）。
4. **§F Meta description 全站 11 個關鍵頁** — 規格書未列 `/docs/about/intro.md` 與 `/docs/about/activities/index.md` 的 Meta-desc。**建議**：補上這兩個（Phase 2 內容填充時順手做）。
5. **§A6 媒體聯播網「維持現有」** — 規格書沒檢視現有 friends.js 的合作媒體是否還對齊新定位（vs 原「開源研究社群」定位）。**建議**：Phase 1 前抽查一次，若有不對齊的媒體合作可暫時隱藏到 Phase 3 重新評估。

### 遺漏項目

- [ ] 無重大遺漏（6 個規格節覆蓋 strategy 全部 4 頁面 + navbar + Meta-desc）。
- [ ] 一個次要觀察：**§A2 USP 宣告區塊的「三條不是什麼」對照可在 H3 標題之上補一個「比較」表格**（vs BNI / TAIA / YPO），但這是「nice-to-have」性質，不阻擋。

---

## 特別關注

- [x] **涵蓋率是否完整？** → 策略計劃的 4 個重點頁面全部都有規格書，加 navbar 與 Meta-desc 共 6 節。
- [x] **「無棋會分支」是否一致套用？** → §A / §B / §C / §F 四節皆有對應處理，無遺漏。
- [x] **「member-project-mapping」的誠實敘事是否套用？** → §A4 案例敘事卡 3 卡中 2 卡為「實際合作」（黃子彥 × 2 案例 + 廖宜鋒 × 1 案例）；§C 12 位變異點表明確區分「實際合作」與「未來合作意向」兩類；§A5 11 位夥伴預覽用「持續擴充」彈性表述 — **敘事誠實度通過**。

### 審查者額外觀察（不阻擋）

1. **§A1 Hero stats「41+ 開源 AI 工具 · 全部免費」的「41+」字串中的「+」很關鍵** — 它把「精確數字」轉成「持續增加」的暗示，避免下次新增 1 個專案時 hero 又要動。**建議在 5-content-spec v1.1 明確標註「stats 數字維護 SOP：每加 1 個 AI 工具，hero 數字僅當突破整十整百時手動更新（41→50→100）」**，否則「每加一個就改 hero」會變成永久流量負擔（11 語系都要動）。
2. **§B1 「兩種合作目的」設計的 mailto subject 用全角中括號【】是個高品質細節** — Gmail / Outlook 對 subject line 的全形字符顯示一致，且【】在中文書信中標籤化效果強，比英文 `[Proposal]` 更符合 Persona A/B 的閱讀習慣。
3. **§C contributing_role 四選一（領域知識主筆 / 跨域審閱 / 案例提供 / 未來合作意向）的設計可作為協會內部「貢獻層級」的基礎概念** — 未來可發展為「貢獻者地圖」或「協會內部 reputation system」。但這超出本次改版範圍，列為 Phase 4 路線圖即可。
4. **§D 年度成果報告的內容結構 D1-D6 可作為「年度開源生態系報告」的範本** — 即使 weiqi.kids 不啟動，這個結構對任何協會的「年度公開課績」都是高品質模板。可考慮把 §D 抽出來作為 revamp/ 流程的可重複資產（不阻擋本次改版）。

---

## 下一步

- [x] Writer 修改（無 blocker，5 項建議可在執行 Phase 1 時一併處理）
- [ ] **用戶素材徵詢**（§H 7 項）：
  - 🔴 急：廖宜鋒完整資料 + 三位 testimonial 引言 + navbar 11 語系翻譯
  - 🟡 中：12 位會員頭像 + 新 OG 圖 + hero stats 數字呈現
- [ ] 進入執行階段：Phase 1.1 = G6 Umami 注入單獨 commit + 部署 → Phase 1.2 = 其他 P0a 合併 commit
- [ ] Phase 2 開始前必確認：5 個旗艦專案的最終 mapping（content-spec §A4 三卡敘事）

---

## 審查者備註

5-content-spec 的最高價值是把 4-strategy 的「該做什麼」轉成「**這樣寫**」—— 規格書中文案完整稿（§A1 Hero / §A2 USP / §B1 雙路徑 / §B5 棋會意向）可以幾乎直接複製貼上到 src/pages/index.js 與 docusaurus.config.js，執行階段不需要再「想文案」。

特別欣賞兩個結構性設計：

1. **§C frontmatter 統一模板**：把 G3 雙向連結（`ai_tools` 欄）、G9 議題化分類（`industry_tags` 欄）、G11 testimonial（引言類型分流）三個 P0b/P1 gap **在 frontmatter 層級一次解決**。Phase 2 完成的同時，等於把 Phase 3 G9 80% 的結構基礎也完成了。

2. **§C 12 位變異點表的「contributing_role 四選一」**：把「實際合作」vs「未來合作意向」兩種狀態正面對待，不假裝 11 位會員都對 41 個工具有同等貢獻 — 這對 Persona A 的「值得花時間了解嗎」判斷是關鍵的誠實度。如果文案虛報「11 位都共同創造」，Persona A 點進 9 個沒實際工具掛鉤的會員頁就會發現名實不符，反而傷信任。

整份規格書最大的「省力效益」在 §G 共用檢查清單與 §H 待用戶素材清單：執行階段把這兩個清單當 checklist 用，可以避免漏項；用戶提供素材時對照 §H 即可，不需要每次來信都重新解釋「我還需要什麼」。
