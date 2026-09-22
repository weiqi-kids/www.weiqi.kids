# Weiqi.Kids 專案指南

## Agent skills

使用 Matt Pocock 工程技能前，讀取 `docs/agents/skills.md`，取得 Codex 呼叫方式與技能來源。

### Issue tracker

本專案使用 GitHub Issues。建立規格、拆票、讀取或更新 issue 前，讀取 `docs/agents/issue-tracker.md`。

### Triage labels

使用五種預設 triage 標籤。分類 issue 或套用標籤前，讀取 `docs/agents/triage-labels.md`。

### Domain docs

使用 single-context 結構。工程技能探索程式碼前，讀取 `docs/agents/domain.md`。

## 專案概述

台灣好棋寶寶協會官網（https://www.weiqi.kids/）。網站正在改版：

- **新站**：`site/`，Astro 7，部署於 Cloudflare（帳戶 `weiqi.kids`）。三條服務線：協會、好棋寶寶棋聚、AI 共學營，另有公開知識。只做繁體中文。
- **舊站**：repo 根目錄的 Docusaurus（`docs/`、`i18n/`、`src/`、`static/`），正式網域切換前仍由 GitHub Pages 服務，作為回退。舊站只做緊急修正，不再新增內容。

改版的名詞與產品規則以 `CONTEXT.md` 為準，架構決策見 `docs/adr/`，規格入口是 `revamp/README.md`。分期見 ADR 0007。

### 新站目錄

```
site/
├── src/
│   ├── content/        # 內容集合：knowledge、association、gatherings、topics、members、partners
│   ├── pages/          # 路由（/association、/gatherings、/camp、/knowledge…）
│   ├── components/     # Astro 元件；D3 圖表為 React island
│   ├── layouts/
│   └── styles/         # 只允許 variables.css、global.css
├── public/             # 靜態檔、_redirects（建置時產生）
├── scripts/            # 設計／內容守門、轉址產生器
├── worker/             # Worker on-demand 路由（報名 API 等）
└── wrangler.jsonc
```

## 常用指令

```bash
# 新站
cd site
pnpm dev                      # 開發伺服器
pnpm build                    # 守門檢查 + 產生轉址 + 建置
pnpm run deploy               # 部署到 Cloudflare（staging）

# 舊站（僅緊急修正）
pnpm start                    # 根目錄
GIT_USER=weiqi-kids USE_SSH=true pnpm run deploy
```

## 新站設計規範（build 自動檢查，違規即失敗）

1. 字級只用 `var(--text-*)`，最小 18px，不寫 px 字級。
2. 顏色只寫在 `src/styles/variables.css`（OKLCH，hex fallback）。
3. 不用 `!important`。
4. 不用外部 CDN（字型、JS 都自託管）。
5. `src/` 下的 CSS 檔只允許 `styles/variables.css`、`styles/global.css`；元件樣式寫 scoped `<style>`。
6. 內容去 AI 腔（`scripts/check-content.mjs`）。

---

## 流量監控

網站使用三種數據來源監控流量：

### 數據來源

| 來源 | 用途 | 數據位置 |
|------|------|----------|
| **GitHub Traffic API** | 頁面瀏覽、流量來源 | `analytics/` |
| **Plausible Analytics** | 即時訪客、行為分析 | Plausible Dashboard |
| **Google Search Console** | 搜尋關鍵字、排名 | GSC Dashboard |

### GitHub Traffic 數據收集

- 由 GitHub Actions 每日自動執行（`.github/workflows/collect-traffic.yml`）
- 原始數據：`analytics/raw/` （保留 30 天）
- 歷史數據：`analytics/history/daily-views.json`、`daily-clones.json`
- 當前快照：`analytics/current/popular-paths.json`、`referrers.json`

### 流量分析指令

```bash
# 產生流量分析報告
./revamp/tools/analyze-traffic.sh

# 查看熱門頁面
cat analytics/current/popular-paths.json | jq '.'

# 查看流量來源
cat analytics/current/referrers.json | jq '.'

# 查看歷史瀏覽趨勢
cat analytics/history/daily-views.json | jq '.views[-7:]'
```

---

## 改版規格

- 入口：`revamp/README.md`；進度：`revamp/2026-09-20-progress-status.md`。
- `revamp/0-positioning` 到 `5-content-spec` 的無日期檔案是舊方向，已封存，不作為規格來源。

---

## 任務完成品質關卡

回報完成前確認以下項目；未通過就先修正。

### 1. 連結

- 新增／修改的內部與外部連結都正常，無 404。
- 舊網址轉址（`site/public/_redirects`）涵蓋被改動的頁面。

### 2. SEO／AEO（新站，依 `CONTEXT.md` 的 SEO 共識）

- `<title>` ≤ 60 字、`meta description` ≤ 155 字、canonical、OG、`twitter:card`。
- 結構化資料依頁面類型：協會頁 `Organization`，成員／講師 `Person`，棋聚 `Event`，課程 `Course`，知識文章 `Article`，每頁 `BreadcrumbList`。不把棋聚標成 `Course`。
- 有常見問題的頁面加 `FAQPage`；有步驟教學加 `HowTo`。
- 重要事實必須是頁面上的可讀文字，不能只放在 JSON-LD 或圖片。
- 課程頁寫出費用、資格、1 堂教學＋3 堂實作、期間；棋聚頁寫出日期、地點、形式、報名方式。
- 登入後頁面、論壇、管理區 `noindex`，不進 sitemap。

### 3. 內容

- 列出預計修改的檔案，逐一確認已更新。
- 不出現舊站的「免費」「無會員費」說法。
- AI 協助整理的內容要經人工核對並註明來源、日期與作者。

### 4. Git

- 變更已 commit，message 清楚；已 push（除非另有指示）。

### 5. SOP

- 原始任務每一步都已執行，沒有「之後再處理」的項目。

---

## 緊急修正 SOP（舊站，切換正式網域前適用）

> **重要**：涉及人名、職稱、公司名稱等敏感資訊的修正，可能造成名譽損害或法律問題，必須最高優先級處理。

### 執行流程

1. **修改檔案後立即部署**
   - 不能只 push 到 main 分支
   - 必須同時部署到 GitHub Pages

2. **標準部署指令**
   ```bash
   GIT_USER=weiqi-kids pnpm run deploy
   ```

3. **部署失敗時的快速替代方案**
   - 如果標準部署失敗或網路不穩定，直接推送已編譯的 build 目錄：
   ```bash
   cd build && git init && git remote add origin https://github.com/weiqi-kids/www.weiqi.kids.git && git add -A && git commit -m "Deploy" && git push -f origin HEAD:gh-pages
   ```

4. **部署完成後立即驗證**
   - 開啟線上網站確認內容已更新
   - 如有快取問題，按 Ctrl+Shift+R 強制刷新
   - 確認無誤才能回報完成

### 禁止事項

- ❌ 只 push 到 main 而不部署
- ❌ 部署失敗後反覆重新編譯（浪費時間）
- ❌ 未驗證線上內容就回報完成

---

## 從 Slack 收到指令時的處理原則

當透過 Slack Bot 收到指令時：

1. **理解意圖**：不要逐字執行，優先理解使用者想要達成什麼
2. **確認影響範圍**：修改前說明將影響哪些檔案
3. **執行修改**：完成檔案修改
4. **部署網站**：依照「緊急修正 SOP」執行部署
5. **驗證結果**：確認線上網站已更新

### Commit Message 格式

使用繁體中文描述，格式：`<type>: <簡短描述>`

- `fix:` 修正錯誤
- `feat:` 新增功能
- `docs:` 文件更新
- `chore:` 維護性更新

### 若指令模糊

列出你的理解並說明你將做什麼，讓使用者確認後再執行。
