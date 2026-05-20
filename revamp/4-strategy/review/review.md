---
title: 4-Strategy 改版策略計劃審查結果
project: weiqi.kids
phase: 4-strategy/review
author: Reviewer
date: 2026-05-20
status: approved-with-conditions
based_on: revamp/4-strategy/strategy.md (2026-05-20)
---

# 策略計劃審查結果

## 審查結果：✅ 通過（附條件）

> 計劃書涵蓋 14 個 gap（G1-G14）+ 1 個 Reviewer 補列（G15），分為 P0a/P0b/P1/P2 共 4 層；3 個 Phase 範圍清楚、依賴關係明確；§8 列出 5 個必答用戶事項使後續執行有 checkpoint。
> 進入 `5-content-spec` 階段不需要再回頭補資料；但 §8 第 1 項（實體棋會狀況）必須在 Phase 1 開始前由用戶確認，否則 G1 次級 CTA 與 G7 都會失效。

---

## 檢查摘要

| 類別 | 結果 | 說明 |
|------|------|------|
| 範圍完整性 | ✅ | 新增 2 / 大改 1 / 修改 11 / 優化 4 / 設定 3 — 數字與後續頁面規劃一致；§1.2 明確列出 5 條排除範圍 |
| 優先級 | ✅ | P0a/P0b/P1/P2 四層；Impact-Effort 矩陣標清楚；每個 ID 都有來源追溯 |
| 分期規劃 | ✅ | 3 個 Phase 各有目標／範圍／成功指標／驗收清單；Phase 1 內部有「6 個項目的執行順序」（G6→G5/G4/G1→navbar→Meta-desc→驗收），避免依賴錯亂 |
| 頁面規劃 | ✅ | 4 個重點頁面（首頁／合作邀請／會員實績卡／年度報告）皆有完整模板：目標／受眾／訊息／CTA／大綱／SEO |
| 可測量性 | ✅ | KPI 表把 Phase 1/2/3 三個檢查點 + 最終目標分開列，每項都有當前基準（含「無數據」「不適用」的誠實標註）|
| 追溯性 | ✅ | §7 追溯表 11 項皆指明來源階段＋章節 ID；G15 補列也明確標示由 3-analysis Reviewer 提出 |

---

## 追溯驗證結果（抽樣）

| 計劃項目 | 差距來源 | Discovery 來源 | 鏈條完整 |
|----------|----------|----------------|----------|
| G6 Umami 注入 | analysis G6 | discovery §6 第 1 項 P0（含 patch）| ✅ |
| G3 雙向連結 | analysis Reviewer 觀察 #1 | discovery §2.3（11 位會員檔案行數）| ✅ |
| G7 棋會公開 | analysis G7 | discovery §6 第 4 項（待用戶確認） | ✅（風險已標記）|
| G10 年度成果報告 | competitive Reviewer 觀察 #2 | YPO Global Impact Report 範式（competitive §競品 D）| ✅ |
| navbar 重設計 | positioning §6 P1 | discovery §2.4 + §5.P0 第 3 項 | ✅ |
| G4「不收費」USP | competitive §三條不是什麼 | — | ✅ 反向定位無 Discovery 對應，但 USP 邏輯成立 |

---

## 遺漏檢查結果

| 差距 ID | 優先級 | 計劃中對應 | 狀態 |
|---------|--------|------------|------|
| G1 mailto CTA | P0a | ✅ Phase 1 / §4.1 §4.2 | |
| G2 11 位會員實績卡 | P0b | ✅ Phase 2 / §4.3 | |
| G3 會員 ↔ 工具雙向連結 | P0b | ✅ Phase 2（5 個旗艦）| |
| G4 「不收費」USP | P0a | ✅ Phase 1 / §4.1 第二屏 | |
| G5 Hero 重寫 | P0a | ✅ Phase 1 / §4.1 | |
| G6 Umami 注入 | P0a-pre | ✅ Phase 1 第 1 件事 | |
| G7 實體棋會公開 | P0a + P1 | ✅ Phase 1 G1 次級 CTA + Phase 2 G7 補資料 | |
| G8 首頁案例敘事 | P0b | ✅ Phase 2 / §4.1 大綱第 4 區塊 | |
| G9 議題化分類 | P1 | ✅ Phase 3（含產業分布預估表）| |
| G10 年度報告 | P1 | ✅ Phase 3 / §4.4 | |
| G11 testimonial 引言 | P0b | ✅ Phase 2（與 G2 合併）| |
| G12 L3 入口收斂 | P2 | ✅ Phase 4（後續路線圖）| |
| G13 英文版獨立撰寫 | P2 | ✅ Phase 4（觸發條件清楚）| |
| G14 4 SOP 文件 | P2 | ✅ Phase 4 | |
| G15 repo README 對齊 | P2 | ✅ Phase 4 | |

**結論**：15 個 gap 全部對應到計劃，無遺漏。

---

## 問題清單

### 必須修改（Blockers）

**無**。本計劃可直接進入 `5-content-spec` 階段；但 §8 第 1 項（實體棋會狀況）是 Phase 1 執行的 gating question，需在執行 Phase 1 前由用戶回答。

### 建議改善（Suggestions，可後補不阻擋）

1. **Phase 1 內部「navbar 與 Meta-desc 為第二個 commit」的描述可更精確** — 計劃書 §3 Phase 1 包含項目中說「順序 4 / 5」但又補充「可作為第二個 commit」。**建議**：明確改為「**Phase 1.1 = G6 單獨 commit（部署 + 驗證 Umami）**；**Phase 1.2 = G5/G4/G1/navbar/Meta-desc 合併 commit**」兩個 sub-step，使部署檢核點更清楚。理由：Umami 注入後最好留 24-48h 觀察基線數據才動其他內容，否則「改版前後對照」基線不準。
2. **G10 年度成果報告的時程「Phase 3 可彈性」與「P1 優先級」有輕微張力** — 計劃書 §2.2 P1 與 §8 第 5 項「年度報告可延到改版後」並存。**建議**：若用戶選擇延後，明確把 G10 移到 Phase 4，不要留在 Phase 3 範圍模糊；否則 Phase 3 「差異化護城河」少了一個主件會顯得單薄。
3. **「41+ 開源 AI 工具」數字一致性** — 計劃書 §4.1 §4.4 用「41+」，但 §1 改版範圍說「新增 11 位會員實績卡 + 約 20 個專案頁的『主要貢獻會員』欄位」。**建議**：在 Phase 2 範圍內明確 5-content-spec 階段要盤點完整的「41 個專案清單與會員 mapping 表」，避免「20 個」與「41 個」之間的差距變成黑洞。
4. **KPI「Phase 1 後 mailto 點擊 ≥ 0」描述弱** — 看似是「下限 0 = 一定達標」的廢話。**建議**：改為「Phase 1 後 30 天內 mailto 點擊有任何非 0 數字進入 Umami 後台」，避免驗收項目空洞。
5. **§8 待用戶確認事項應有明確截止點** — 計劃書未說「這 5 個問題什麼時候必須有答案」。**建議**：明確標出「第 1 項 = Phase 1 開始前；第 2, 4 項 = Phase 2 開始前；第 3, 5 項 = Phase 3 開始前」，使 `5-content-spec` 階段排程有依據。

### 遺漏項目

- [ ] 無重大遺漏（15 個 gap 全對應）。
- [ ] 一個次要觀察：**positioning §3 提到「目前所有研究與工具專案由理事長 CΛ / Lightman 主導執行」這個「11 位會員深度參與尚淺」的事實**，計劃中（特別是 §6 風險表）可加一條：「**會員實際貢獻深度可能不足以填滿 G2/G3/G11 內容**，需 Phase 2 開始前確認；若不足，G11 testimonial 可先收『加入協會之初的期望』而非『實際合作感受』」。不阻擋。

---

## 特別關注

- [x] **是否有優先級需要調整？** → 不需要。P0a 五件不可拆 + P0b 雙向連結這兩個從 3-analysis 帶來的洞察被正確保留在 Phase 1/2 結構中。
- [x] **分期是否合理？** → 三 Phase 結構合理。Phase 1（敘事骨幹）→ Phase 2（內容填充）→ Phase 3（差異化護城河）符合「先打地基再蓋房」邏輯。
- [x] **KPI 是否可達成？**
  - Leading KPI（mailto 30 / About 90s / 會員頁 PV 50 / GitHub 外連 100）→ 在 Phase 1 量測啟動後可達；
  - Hard KPI（inbound 商務洽談 5 件 / 半年合作 1 件）→ **較緊**，取決於 11 位會員的網絡曝光與媒體溝通力道。**建議**：Phase 3 完成後與用戶討論是否需要主動推廣（媒體投稿、LinkedIn 寫文）來補足 inbound 流量。

### 審查者額外觀察（不阻擋）

1. **計劃書 §4.1 首頁內容大綱第 3 區塊「三大合作面向」順序「案例→工具→論文」是高質量設計**：從具體到抽象的訊息層次，比現在的「論文→情報→工具」更符合 Persona A 的認知（先看「跟我類似的人做了什麼」，再看「他們做出什麼工具」，最後才會看「他們發了什麼學術論文」）。
2. **Phase 1 完成後預留「24-48h 基線觀察」是個被忽略的價值**：如果 Phase 1 上線後 1 週才開始 Phase 2，可以多獲得「改版前→改版後」對照數據，這對 Hard KPI 的可信度極為重要。建議在計劃書 §3 Phase 1 驗收清單中加一項「**Phase 1 上線後 ≥ 5 天觀察期才開始 Phase 2 內容生產**」。
3. **§4.3 會員實績卡統一模板的 yaml frontmatter 設計（industry_tags / ai_tools）已經為 G9 議題化分類預留結構**：Phase 2 完成時，G9 在 Phase 3 的實作成本會自然降低。這個前置設計是 Writer 的隱性高分。

---

## 下一步

- [x] Writer 修改（無 blocker，5 項建議可在 strategy.md v2 補上或直接於 5-content-spec 處理）
- [ ] **用戶必答 §8 第 1 項（實體棋會狀況）** — 進入 Phase 1 執行前的 gating question
- [ ] 進入 `5-content-spec`（每頁內容規格）
- [ ] `5-content-spec` 階段優先處理：
  - (a) 41 個專案 × 11 位會員的完整 mapping 表
  - (b) 11 位會員的 testimonial 引言徵詢草稿
  - (c) 4 個重點頁面（首頁 / 合作邀請 / 會員實績卡 / 年度報告）的逐字稿與微復本

---

## 審查者備註

Writer 的「**P0a 不可拆五件事 → 一次性 Phase 1 上線**」與「**Phase 1 上線 → 24-48h 觀察 → Phase 2 開始**」隱含設計，是把 3-analysis Reviewer 提的「Persona 為假設」風險轉化為「實際數據驗證再大投入」的工作流程。這比把所有 P0 一起做的「all-or-nothing」更具韌性 — 即使 Phase 1 上線後流量數據顯示 Persona 假設錯誤，Phase 2 還來得及調整內容方向，不會浪費 11 × 2h 的內容生產成本。

整份計劃最強的設計是 **§4.3 yaml frontmatter 統一模板**：把 G9 議題化分類所需的 `industry_tags` 與 G3 雙向連結所需的 `ai_tools` 在 Phase 2 階段就以結構化方式埋入 — 等於 Phase 2 順手完成 80% 的 Phase 3 結構基礎。`5-content-spec` 階段如果能保留並擴充這個 frontmatter 設計，整個改版的可維護性會大幅提升。
