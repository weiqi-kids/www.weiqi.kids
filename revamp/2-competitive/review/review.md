---
title: 2-Competitive 競品分析審查結果
project: weiqi.kids
phase: 2-competitive/review
author: Reviewer
date: 2026-05-20
status: approved-with-conditions
based_on: revamp/2-competitive/competitive.md (2026-05-20)
---

# 競品分析審查結果

## 審查結果：✅ 通過（附條件）

> 競品選擇對齊定位文件 §3「不是純圍棋協會／純 AI 社群／純商業社團」三條差異化軸線，逐條有對應對象；標竿（YPO）切入「內容資產化」面向，與差距分析直接掛鉤。
> 進入 `3-analysis` 階段不需要再回頭補資料；但「未跑 Lighthouse / 未取 SimilarWeb 流量數據」是已知缺項，Writer 已自行標註不阻擋判斷。

---

## 檢查摘要

| 類別 | 結果 | 說明 |
|------|------|------|
| 競品選擇 | ✅ | 4 個（3 直接 + 1 標竿），數量在 3–5 區間。每個競品的選擇理由與 positioning §3 「三條不是什麼」軸線一一對應 |
| 數據品質 | ⚠️ | HTTP headers / 回應時間 / robots / sitemap 有實測；Lighthouse 與 SimilarWeb 未取，Writer 已自承並列為「後補不阻擋」 |
| 分析品質 | ✅ | 比較矩陣兩個（功能/內容 + 技術）；差距、優勢、機會分別有獨立節；YPO 帶出「內容資產護城河」洞察不流於表面 |
| 可執行性 | ✅ | 每個發現有對應建議，P0a/P0b/P1/P2 優先級分明；7 項可借鏡做法皆對齊既有 P0a-P2 軸線 |
| 與前階段一致性 | ✅ | 與 0-positioning §3、§6 完全對齊；風險指認段直接回應 §3 圍棋／AI 關鍵字競爭問題；P0 流量分析啟用呼應 1-discovery §6 第 1 項 |

---

## 驗證結果（抽樣重跑）

| 驗證項目 | 報告數據 | 重新驗證（2026-05-20 二次測） | 一致性 |
|----------|----------|---------------------------|--------|
| TAIA 首頁回應時間 | 0.06s | 0.39s（二測） | ⚠️ 差距大但兩次均屬「快」；首測命中 CDN cache、二測 miss，**結論「TAIA 技術最快」不變** |
| BNI 首頁回應時間 | 1.03s | 1.03s | ✅ 完全一致 |
| 圍棋協會 robots.txt 配錯 | 🔴 回傳 HTML | （Writer 報告已驗）| ✅ 符合 |
| 競品 D YPO Apply CTA | "Apply Today" / "Apply For Membership"（首頁多處）| 抽樣 WebFetch 確認 | ✅ 符合 |

---

## 問題清單

### 必須修改（Blockers）

**無**。報告已通過 5/5 類別審查，無阻擋進入 `3-analysis`。

### 建議改善（Suggestions，可後補不阻擋）

1. **後補 Lighthouse 比較表**（5 站一起跑）— 建議 `revamp/tools/competitive-audit.sh https://www.weiqi.kids/ https://www.weiqi.org.tw/ https://www.aiatw.org/ https://bni.com.tw/zh-TW/index https://www.ypo.org/`；補回 competitive.md §「技術比較」表格。**理由**：weiqi.kids 自家定位文件 §5 Hard KPI 不涉及效能，但 BNI 用 PHP 7.2 EOL、weiqi.org.tw 用 ASP.NET 的技術陳舊度，若用 Lighthouse 性能分數佐證會更有說服力。
2. **次要受眾的競品補做**（若 `3-analysis` 確認次要受眾權重高）— Writer 已在報告末段標明：若 AI 工程師／圍棋棋友權重高，需補純 AI 開源社群（Hugging Face、中華民國人工智慧學會 taai.org.tw）與純圍棋學習平台（gotw.tw 等）。**理由**：避免 L2／L3 受眾路徑誤判。
3. **「圍棋對局信任」差異化的可信度補強** — Writer 已標「前提是有定期實體棋會」（呼應 1-discovery §6 第 4 項），這條軸線在 `3-analysis` 訪談或問卷裡需要實際驗證；否則「圍棋作為信任場域」會被定位為純行銷敘事。
4. **議題化關鍵字策略**先收斂 — §機會表提到「醫療／法律／財稅／教育／圍棋／資安」6 個產業分類；建議在 `4-strategy` 階段先收斂到 3–4 個（依現有 40+ 專案分佈密度），避免分類過細稀釋每個的內容厚度。

### 遺漏項目

- [ ] 無重大遺漏；報告自承的兩個缺口（Lighthouse、SimilarWeb）已列為「不阻擋」並標明後補方法。

---

## 特別注意

- [x] **競品選擇是否需要調整？** → 不需要。三條差異化軸線各有對應，加 1 個標竿，比例合理。
- [x] **是否遺漏重要競品？** → 主要受眾（B2B）無重大遺漏；次要受眾競品（HuggingFace / TAAI / gotw.tw）建議延到 `3-analysis` 後決定是否補做。
- [x] **差異化方向是否可行？** → 三條「不是什麼」+ 一條 USP 結構清楚，且每條都有具體文案／頁面落點建議；可行性高。

### 審查者額外觀察（不阻擋）

1. **「議題化分類頁」是高槓桿做法**：YPO 5 大主題、TAIA AI Award 分類都印證這條路。weiqi.kids 已經有 40+ 專案的素材，但目前 `/research/` 用「mathematics/ml-theory/ai-safety/go」分類，**對主要受眾（商界）不直觀**；建議 `4-strategy` 時把「分類維度」從技術領域改為產業領域（醫療／法律／財稅／教育／圍棋）。
2. **「年度開源成果報告」可作為改版第一個對外里程碑**：當 P0a/P0b 完成後，把報告作為「對外宣告改版完成」的內容資產，同時也是給創始會員的禮物（每人實績被聚合成可分享 PDF）— 一石多鳥。
3. **「不收費」作為 USP 的反向定位** — 是 4 個競品都無法跟進的護城河（BNI/TAIA/YPO 商業模式都建立在會費上）。Writer 已點出，建議 `5-content-spec` 階段把「完全免費、無會員費、全部開源」三句話固定為首頁第二屏的明確訊息。

---

## 下一步

- [x] Writer 修改（無 blocker）
- [ ] 進入 `3-analysis`（受眾分析 + 內容差距）
- [ ] `3-analysis` 階段優先處理：(a) 訪談／問卷驗證 Persona A/B 假設；(b) 確認次要受眾權重是否需補做競品

---

## 審查者備註

Writer 的「**三條不是什麼 + 一條 USP**」結構是本份報告最高價值的設計：把抽象的差異化定位變成可被反覆引用的 4 句話模組，後續 `4-strategy` 與 `5-content-spec` 的首頁文案、SEO description、會員實績卡引言都可以直接套用，避免每階段重新發明說法。

此外，「**完全免費＋全部開源**」是 4 個競品商業模式上的盲點（他們都靠會費／培訓費／企業會員費），這條反向定位在 `4-strategy` 階段值得獨立一節討論「不收費的可持續性」（理事長個人投入時間 / 開源生態貢獻者激勵）— 否則 USP 雖然清楚，但會被質疑「為什麼可以持續」。
