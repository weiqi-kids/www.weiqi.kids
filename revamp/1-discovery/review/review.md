---
title: 1-Discovery 盤點報告審查結果
project: weiqi.kids
phase: 1-discovery/review
author: Reviewer
date: 2026-05-20
status: approved-with-conditions
based_on: revamp/1-discovery/discovery.md (2026-05-14)
---

# 盤點報告審查結果

## 審查結果：✅ 通過（附條件）

> Writer 已明確標註未跑項目與不阻擋判斷依據；P0 行動清單具體、可驗證。
> 進入 `2-competitive` 階段前，§6 第 1 項（Umami 追蹤碼注入）必須先完成。

---

## 檢查摘要

| 類別 | 結果 | 說明 |
|------|------|------|
| 技術數據 | ⚠️ 部分 | Lighthouse / Mozilla Observatory / SSL Labs / W3C Validator 未跑，但 Writer 已自證「不涉及定位文件 Hard KPI 門檻、不阻擋下一階段」，可後補 |
| 內容盤點 | ✅ | 一級目錄 12 項、待補充清單 19 檔、會員檔案 11 位逐人列出行數，涵蓋完整且狀態明確 |
| 數據來源 | ✅ | 每節皆有「數據來源」段落（GitHub Traffic API、curl、本機 docs/、navbar.json），日期 2026-05-14 標註清楚 |
| KPI | ✅ | 7 個 KPI 對照「定位文件原目標 vs 現況基準 vs 建議調整」三欄；其中 1 個從相對比例改為絕對數（會員實績頁 PV ≥ 50 / 90 天）合理 |

---

## 驗證結果（抽樣重跑）

| 驗證項目 | 報告數據 | 重新驗證（2026-05-20） | 一致性 |
|----------|----------|----------------------|--------|
| Umami 追蹤碼是否注入 | 🔴 未注入 | `curl -s https://www.weiqi.kids/ \| grep -c a4b64b22` → 0 | ✅ 一致（P0 仍存在）|
| HSTS Header | `max-age=31556952` | `curl -sI` → `strict-transport-security: max-age=31556952` | ✅ 一致 |
| 首頁回應碼 | 200 | `curl -sI -o /dev/null -w "%{http_code}"` → 200 | ✅ 一致 |
| 首頁 hero 文案 | 「40 個公益專案」（2026-05-14）| **已於 2026-05-20 更新為「41 個公益專案」**（新增 Erdős–Sidon 論文）| ⚠️ 報告數據已過時 1 處 |

> **註**：本次審查日（2026-05-20）首頁 hero 已更新為 41 個公益專案、新增 optimal-golomb-ruler 論文卡。Discovery 報告寫於 2026-05-14，§2.4 引用的 hero stats 與 §1.3 Meta description 為當時快照，未影響本階段判斷。

---

## 問題清單

### 必須處理（Blockers，進入 2-competitive 前完成）

1. **🔴 Umami 追蹤碼注入** → §6 第 1 項已給出具體 docusaurus.config.js patch 與部署後驗證指令，照辦即可。**理由**：定位文件 §5 中 5/6 個 Leading KPI + 1/2 個 Hard KPI 全部依賴流量分析，無此則改版無法驗收。

### 建議改善（Suggestions，不阻擋）

1. **後補 Lighthouse + Observatory + SSL Labs + W3C** — 建議用 `revamp/tools/site-audit.sh https://www.weiqi.kids/ --output revamp/1-discovery/site-audit.json` 一次跑完，補進 `discovery.md` 的 §1.1 與 §1.2 表格。
2. **GSC 連通狀態確認** — Writer 已標為 ❓，需用戶確認屬於 owner 驗證後即可用；非阻擋。
3. **協會實體棋會活動現況確認** — §6 第 4 項，需用戶端確認是否有定期活動，影響後續 `5-content-spec` 的 activities/ 內容填充策略。

### 遺漏項目

- [ ] 無重大遺漏。報告結構完整、KPI 校準到位、行動清單具體。

---

## 下一步

- [x] Writer 補充遺漏項目（無重大遺漏，可不修）
- [x] 修正錯誤數據（hero stats 數值已自然修正，非報告錯誤而是時序差異）
- [ ] **進入 2-competitive 階段前先完成**：Umami 追蹤碼注入（§6 第 1 項 P0）
- [ ] 通過後進入 2-competitive 階段

---

## 審查者備註

Writer 的 §4「KPI 基準校準」與 §6「進入 2-competitive 前的必做事項」是本報告最高價值的兩節：
- §4 把定位文件的相對目標（如「會員實績頁 PV 佔比 ≥ 8%」）改為絕對數（「90 天累計 PV ≥ 50」），這在「站台月流量為 0」的現況下更可量測。
- §6 第 1 項直接給出可貼上的 docusaurus.config.js patch，免去 2-competitive 階段反覆 context-switch。

這份報告可作為後續 `2-competitive` 階段的基線（特別是與競品流量分析對比時）。
