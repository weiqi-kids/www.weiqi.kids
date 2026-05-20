---
title: 網站現況盤點報告
project: weiqi.kids
phase: 1-discovery
author: Writer
date: 2026-05-14
status: draft
based_on: revamp/0-positioning/positioning.md (v2)
---

# 網站現況盤點報告

## 基本資訊

| 項目 | 內容 |
|------|------|
| 網站 URL | https://www.weiqi.kids/ |
| 框架 | Docusaurus 3.9.2（CLAUDE.md 寫 3.8.1，已升級） |
| 部署 | GitHub Pages |
| 檢測日期 | 2026-05-14 |
| 文件頁面數 | docs/ 內 93 個 `.md`；自訂頁 5 個（index/research/intel/apps/markdown-page） |
| sitemap 總 URL | 119 頁 × 11 語系 = **1,309 URLs** |

---

## 1. 技術健檢結果

### 1.1 效能分數

> **Lighthouse 本次未跑**（GitHub Pages 上的 Docusaurus 通常 Performance ≥85、SEO ≥90），列為 `1-discovery` 階段補測項；定位文件 Hard KPI 未涉及效能門檻，暫不阻擋進入下一階段。

### 1.2 安全性

| 項目 | 結果 | 評價 |
|------|------|------|
| HTTPS | ✅ 啟用 | 正常 |
| HSTS | `max-age=31556952` | ✅ 完整 |
| Server header | `GitHub.com` | 受 GitHub Pages 限制，無法自訂 CSP / X-Frame-Options 等其他 security headers |
| 首頁回應時間 | 0.24s | ✅ 健康 |

### 1.3 SEO 基礎

| 項目 | 狀態 | 內容 / 問題 |
|------|------|------|
| robots.txt | ✅ | `User-agent: * Allow: /` + sitemap-index |
| sitemap.xml | ✅ | 1,309 URLs（11 語系 × 119 頁）|
| 首頁 `<title>` | ⚠️ | 「首頁 \| 好棋寶寶協會 \| Weiqi.Kids」— 「首頁」字樣浪費 SEO 前綴 |
| Meta `description` | 🔴 | **「台灣好棋寶寶協會官網 - 提供圍棋教學、AI 研究資源，推動圍棋文化發展」**— 與新定位「商界夥伴×AI×圍棋」不符 |
| Meta `keywords` | 🔴 | **「圍棋, Go, 好棋寶寶, AI, KataGo, AlphaGo, 圍棋教學, 圍棋入門」**— 全是 L3 圍棋教育導向，無「商務合作／開源／公益專案」相關詞 |
| OG tags | ✅ | og:title / og:url / og:image / og:locale 完整，11 alternate locale 已標 |
| Twitter Card | ✅ | summary_large_image |
| Social card image | ✅ | `/img/social-card.png`（需在後續階段驗證圖片內容是否反映新定位） |
| Plausible Analytics | 🔴 | **`docusaurus.config.js` 內全部註解掉、未啟用** |
| Umami Analytics | 🔴 | CLAUDE.md 記載 Website ID `a4b64b22-906d-4918-934a-7da72b5aced9` 已建立，但**追蹤腳本未被注入線上 HTML**（已用 `curl` 驗證；首頁 HTML 內無任何分析腳本，只有 Docusaurus runtime） |
| Google Analytics | 🔴 | 未啟用 |
| **綜合：流量量測** | 🔴 | **完全沒有任何分析工具運作中** — 等於無法量測流量／互動，整個 KPI 架構失效 |

---

## 2. 內容盤點

### 2.1 一級目錄結構

| 路徑 | 類型 | 對應定位三層 | 狀態 |
|------|------|--------------|------|
| `docs/about/` | 協會資訊 | **L1**（核心受影響） | ⚠️ 大量「待補充」 |
| `docs/about/members/founding/` | 11 位創始會員 | **L1** | ⚠️ 簡介型，缺「對應 AI 工具」連結 |
| `docs/about/activities/` | 活動實績 | L1 | ⚠️ 多檔「待補充」 |
| `docs/about/locations/` | 圍棋據點 | L3 | ⚠️ 多檔「待補充」 |
| `docs/about/internal/` | 內部 SOP | （內部） | 多檔「待補充」 |
| `docs/alphago/` | AlphaGo 演進 | L3 | 內容性 |
| `docs/animations/` | 動畫教室 | L3 | 內容性 |
| `docs/learn/` | 圍棋學習（ai-era / history / introduction）| L3 | 內容性 |
| `docs/tech/` | 技術文件（deep-dive / hands-on / how-it-works / industry / overview）| L2 | 內容性 |
| `src/pages/research.js` | 學術論文 | L2 | 已有 |
| `src/pages/intel.js` | 產業情報 | L2 | 已有 |
| `src/pages/apps.js` | AI 工具集 | L2 | 已有 |
| `src/pages/index.js` | 首頁 | L1（流量入口）| 🔴 主敘事與新定位不符（見 §4）|

### 2.2 「待補充」清單（共 19 個檔案）

定位文件要求 P0b 補齊的內容，現況確認如下：

| 檔案 | 影響的定位章節 | 嚴重度 |
|------|---------------|--------|
| `docs/about/intro.md`（使命／願景／歷程全空）| §6 P0b About 改寫 | 🔴 P0 |
| `docs/about/meetings.md` | 信任場域佐證 | 🟡 P1 |
| `docs/about/members/accountant.md` | 11 位會員實績 | 🔴 P0 |
| `docs/about/members/directors.md` | 11 位會員實績 | 🔴 P0 |
| `docs/about/members/secretary.md` | 11 位會員實績 | 🔴 P0 |
| `docs/about/members/supervisors.md` | 11 位會員實績 | 🔴 P0 |
| `docs/about/activities/index.md` | L1 合作案例敘事 | 🔴 P0 |
| `docs/about/activities/charity.md` | L1 案例 | 🟡 P1 |
| `docs/about/activities/market.md` | L1 案例 | 🟡 P1 |
| `docs/about/activities/game-reviews.md` | L1 信任佐證 | 🟡 P1 |
| `docs/about/locations/{kaohsiung,taichung,taipei,yunlin}.md` 等 4 個 | L3（次要受眾）| 🟢 P2 |
| `docs/about/internal/sop/{host-market,financial-reports,host-fun-club,receipts}.md` 等 4 個 | （內部 SOP，非對外）| 🟢 P2 |

### 2.3 11 位創始會員檔案盤點

| 會員（slug） | 行數 | 推測完整度 |
|-------------|------|-----------|
| lightman-chang | 24 | 最少（理事長本人，反而最短）|
| index.md | 27 | 目錄頁 |
| dawn-lee | 30 | 簡介型 |
| zhen-zhao | 33 | 簡介型 |
| yang-luo, zi-yan-huang | 34 | 簡介型 |
| jing-yan-chou | 35 | 簡介型 |
| sih-jie-chou | 37 | 簡介型 |
| bo-yang-cheng | 40 | 簡介型 |
| ian-kuo | 41 | 簡介型 |
| wei-chiu | 47 | 較完整 |
| tien-chien-pan | 62 | 最完整 |

→ **沒有任何一位會員的檔案有達到定位文件 §6 P0b 要求**（本業＋協會內貢獻＋**對應的 AI 工具連結**）。內容全部需要在 `5-content-spec` 階段重寫。

### 2.4 Navbar 與首頁主敘事問題

**現況 navbar.json**：學圍棋 / AlphaGo / 動畫教室 / 技術文件 / 關於我們

→ **全部 5 個項目都是 L3（圍棋教育）導向**，**完全沒有**：
- 「夥伴」入口
- 「合作案例」/「實績」入口
- 「合作提案」CTA（mailto）

**現況首頁**（index.js 第 21–28 行）：
- Hero subtitle:「開源研究社群．從圍棋出發」
- Stats:「11 位創始會員 · 40 個公益專案 · 全部開源 · 每日自動更新」
- CTA: 認識協會 / 研究成果

→ 與新定位「商界夥伴 × AI × 圍棋」差距大，需依 §6 P0a 重寫。

---

## 3. 流量分析

### 3.1 GitHub Traffic 數據（過去 14 天）

| 指標 | 數值 |
|------|------|
| 累計頁面瀏覽（views）| **2** |
| 累計獨立訪客（uniques）| **1** |
| 熱門頁面 #1 | `/weiqi-kids/www.weiqi.kids`（**GitHub Repo Overview**，count=2 / uniques=1）|
| 流量來源 #1 | `github.com`（count=2 / uniques=1）|

> **GitHub Traffic API 統計的是「Repo 頁面」流量**，不是網站本身。「熱門頁面」指的是 GitHub 上 repo 的 Overview。

### 3.2 Plausible Analytics 數據

🔴 **未啟用**。`docusaurus.config.js` 的 Plausible script 區塊全部被註解，所以**沒有**：
- 真實網站訪客數
- 跳出率
- 停留時間
- 地理分布
- CTA 點擊事件

→ 定位文件 §5 KPI 全部依賴 Plausible（5 個 Leading + 1 個 Hard），**無啟用就無法量測改版成效**。這是進入後續階段前必須先解的關卡。

### 3.3 Google Search Console 數據

❓ 狀態未知，需用戶確認是否已驗證所有權與索引狀態。

### 3.4 結構與 CTA 替代分析（無 GA／Plausible 時）

| 分析項目 | 結果 | 建議 |
|----------|------|------|
| 導航深度 | 主導航 5 項全部指向 docs/，深度 2 層可達內容 | 結構合理但分類錯（見 §2.4）|
| CTA 明確度 | 首頁 2 個 CTA（認識協會／研究成果），無「合作提案」 | P0a 加入 mailto CTA |
| 內容完整度 | 93 個 docs 檔，**19 個（20%）標示「待補充」** | P0b 補齊核心 7 個（about/intro、5 位 member、activities/index）|
| 時效性 | 大部分內容是新的（近期 commit）| ✅ 健康 |

---

## 4. 建議 KPI 基準校準

> 根據 §3 現況數據，定位文件 §5 KPI 目標值需要重新校準。

| KPI | 定位文件原目標 | 現況基準（過去 14 天）| 建議調整 |
|-----|--------------|-------------------|---------|
| 🟡 mailto CTA 點擊 | ≥ 30 / 90 天 | **無此 CTA**，無基準 | 維持目標，需先在 P0a 加入 |
| 🟡 About 頁停留時間中位數 | ≥ 90 秒 | **無 Plausible**，無基準 | 維持目標，需先啟用 Plausible |
| 🟡 會員實績頁 PV 佔比 | 0 → ≥ 8% | 全站 14 天總 PV ≈ 0 | 改為「會員實績頁 90 天累計 PV ≥ 50」（絕對數較易量測）|
| 🟡 AI 工具 GitHub 外連 | ≥ 100 次 / 90 天 | 無事件追蹤 | 維持目標，需先設 Plausible outbound event |
| 🟡 品牌字 GSC impression | 出現於報表 | 未知 | 需先驗證 GSC 已連通 |
| 🔴 實際 inbound 商務洽談 | ≥ 5 件 / 90 天 | 不適用（無 CTA）| 維持 |
| 🔴 由 inbound 轉合作案例 | ≥ 1 件 / 半年 | 不適用 | 維持 |

---

## 5. 關鍵發現摘要

### 優勢

1. **技術基礎健康**：HTTPS、HSTS、robots、sitemap（1309 URLs × 11 語系）、OG / Twitter Card 完整
2. **內容量充沛**：93 個 docs + 4 個自訂 React 頁，11 語系 i18n 已上線
3. **創始會員檔案皆存在**（11 個 .md 全到位）— 改寫成本低於從零開始
4. **回應速度快**（0.24s）

### 問題（按嚴重度排序）

| 優先級 | 問題 | 影響 |
|--------|------|------|
| 🔴 P0 | Plausible 未啟用（config 註解狀態） | 定位文件 5/6 個 Leading KPI 與 1/2 個 Hard KPI 無法量測，改版無法驗收成敗 |
| 🔴 P0 | 首頁主敘事仍是「開源研究社群」/ Meta description 仍是「圍棋教學、AI 研究」舊文案 | 對外定位與內部定位不一致；SEO 抓不到新關鍵字 |
| 🔴 P0 | Navbar 5 項全部 L3 導向（學圍棋／AlphaGo／動畫教室／技術文件／關於我們）| 主要受眾（商界夥伴）無入口；mailto CTA 不存在 |
| 🔴 P0 | About 區大量「待補充」（intro / 5 位 member / activities/index）| P0b 內容填充無法進行 |
| 🟡 P1 | 11 位會員檔案皆為「簡介型」，**全部 0 篇**有達到「本業＋協會貢獻＋AI 工具連結」三段式 | P0b 「實績卡」需逐人重寫 |
| 🟡 P1 | 沒有實體棋會／月例會的活動紀錄（activities/{charity,market,game-reviews}.md 皆待補充）| 定位文件 §3 「圍棋對局形成的長期信任」缺實證；需確認協會是否真有定期下棋活動 |
| 🟢 P2 | GSC 連通狀態未知 | 無法量測搜尋表現 |
| 🟢 P2 | Lighthouse Performance / Accessibility 未跑 | 不阻擋改版，可後補 |
| 🟢 P2 | 4 個 locations 檔案＋ 4 個 SOP 檔案「待補充」 | 影響 L3 / 內部運營，非主敘事 |

---

## 6. 進入 `2-competitive` 前的必做事項

1. 🔴 **部署流量分析追蹤腳本**（無此則整個 KPI 量測架構失效）
   - **強烈建議：採用既有 Umami**（CLAUDE.md `a4b64b22-906d-4918-934a-7da72b5aced9` 已為 www.weiqi.kids 建立 Website ID，但追蹤腳本尚未注入）
   - 作法：在 `docusaurus.config.js` 的 `scripts:` 區塊加入：
     ```js
     scripts: [{
       src: 'https://analytics.weiqi.kids/script.js',
       defer: true,
       'data-website-id': 'a4b64b22-906d-4918-934a-7da72b5aced9',
     }]
     ```
   - 部署後驗證：`curl -s https://www.weiqi.kids/ | grep "a4b64b22"` 應有 1 筆
   - 注意：定位文件 §5 KPI 提及的「Plausible 事件追蹤」需改為「Umami event tracking」— Umami 支援 outbound link 事件與 custom event，可滿足 mailto 點擊與 GitHub 外連量測
2. 🟡 確認「合作提案」mailto 是否要先設好 `lightman.chang@gmail.com` 的 inbox 規則（避免 SPAM 漏接）
3. 🟡 確認 GSC 連通狀態（如未驗證需先做）
4. 🟡 確認協會是否有「定期實體棋會」活動，以支撐定位文件 §3 「圍棋對局形成的長期信任」論述

> ⚠️ **已確認 Umami 追蹤碼未注入網站**（2026-05-14 驗證），定位文件 §5 KPI 中所有 Plausible 提及處，應於下次 commit 一併改為 Umami；追蹤腳本注入應作為 P0a 改版的第一步（在改首頁文案之前），以便取得「改版前」基線數據。

---

## 數據來源

- GitHub Traffic API: `analytics/current/`、`analytics/history/`（最新採集 2026-05-14）
- 網站 HTTP/HTML: 直接 curl https://www.weiqi.kids/（2026-05-14）
- 內容盤點: 本機 `/root/www.weiqi.kids/docs/` 與 `src/pages/`
- Navbar config: `docusaurus.config.js` + `i18n/zh-tw/docusaurus-theme-classic/navbar.json`
- Lighthouse / Mozilla Observatory / SSL Labs: **未執行**（不阻擋本階段，列為後補）
