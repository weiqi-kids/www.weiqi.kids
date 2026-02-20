# Weiqi.Kids 專案指南

## 專案概述

台灣好棋寶寶協會官網（https://www.weiqi.kids/），使用 Docusaurus 3.8 建置的多語系靜態網站。

### 技術棧

- **框架**：Docusaurus 3.8.1
- **部署**：GitHub Pages
- **多語系**：11 種語言（zh-tw, zh-cn, zh-hk, en, ja, ko, es, pt, hi, id, ar）
- **搜尋**：@easyops-cn/docusaurus-search-local
- **圖表**：Mermaid

### 目錄結構

```
www.weiqi.kids/
├── docs/                    # 繁體中文原始文件
│   ├── for-players/         # 給圍棋棋友
│   ├── for-engineers/       # 給 AI 工程師
│   ├── evolution/           # 圍棋 AI 演進整理
│   └── aboutus.md           # 協會介紹
├── i18n/                    # 多語系翻譯
│   └── {locale}/
│       ├── docusaurus-theme-classic/
│       │   └── navbar.json  # Navbar 翻譯
│       ├── docusaurus-plugin-content-docs/
│       │   └── current/     # 文件翻譯
│       └── code.json        # UI 字串翻譯
├── src/
│   ├── components/          # React 組件
│   ├── pages/               # 自訂頁面
│   └── theme/               # 主題覆寫
├── static/                  # 靜態資源
├── seo/                     # SEO/AEO 規則文件
├── analytics/               # 流量數據收集
│   ├── raw/                 # 原始 API 回應（按日期）
│   ├── history/             # 歷史數據（累積）
│   ├── current/             # 最新快照
│   └── scripts/             # 數據處理腳本
├── revamp/                  # 網站改版流程
│   ├── 0-positioning/       # 品牌定位
│   ├── 1-discovery/         # 網站健檢
│   ├── 2-competitive/       # 競品分析
│   ├── 3-analysis/          # 受眾與差距分析
│   ├── 4-strategy/          # 改版策略
│   ├── 5-content-spec/      # 內容規格
│   ├── final-review/        # 最終驗收
│   └── tools/               # 自動化工具
└── docusaurus.config.js     # 網站設定
```

---

## 常用指令

```bash
# 開發
pnpm start                    # 啟動開發伺服器（僅預設語系）
pnpm start -- --locale en     # 啟動特定語系

# 建置
pnpm build                    # 建置所有語系

# 部署
GIT_USER=weiqi-kids USE_SSH=true pnpm run deploy

# 翻譯
pnpm write-translations       # 產生翻譯骨架

# 流量分析
./revamp/tools/analyze-traffic.sh  # 產生流量分析報告
```

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

## 多語系開發注意事項

1. **原始文件**：所有內容先在 `docs/` 目錄編寫（繁體中文）
2. **翻譯檔案**：放在 `i18n/{locale}/docusaurus-plugin-content-docs/current/`
3. **Navbar 翻譯**：修改 `i18n/{locale}/docusaurus-theme-classic/navbar.json`
4. **UI 字串**：修改 `i18n/{locale}/code.json`
5. **sidebar_position**：每個語系的對應文件需要相同的 frontmatter

---

## 改版流程（Revamp Workflow）

當需要進行網站改版時，請依照 `revamp/` 目錄的結構化流程執行。

### 流程總覽

```
0-Positioning → 1-Discovery → 2-Competitive → 3-Analysis → 4-Strategy → 5-Content-Spec → 執行 → Final-Review
     ↓              ↓             ↓              ↓            ↓              ↓                       ↓
  Review ✓      Review ✓      Review ✓      Review ✓     Review ✓       Review ✓                Review ✓
```

### 階段說明

| 階段 | 目的 | 輸出 |
|------|------|------|
| **0-positioning** | 釐清品牌定位、核心價值 | 定位文件 |
| **1-discovery** | 盤點現有內容 + 技術健檢 | 健檢報告 + KPI |
| **2-competitive** | 分析競爭對手 | 競品分析報告 |
| **3-analysis** | 受眾分析 + 內容差距 | 差距分析報告 |
| **4-strategy** | 改版計劃 + 優先級排序 | 改版計劃書 |
| **5-content-spec** | 每頁內容規格 | 內容規格書 |
| **final-review** | 驗收執行結果 | 驗收報告 |

### 使用方式

```bash
# 依序執行各階段
請以 Writer 角色，參照 revamp/0-positioning/CLAUDE.md 執行品牌定位分析
請以 Reviewer 角色，參照 revamp/0-positioning/review/CLAUDE.md 檢查上述輸出

# 繼續下一階段...

# 最後驗收
請以 Reviewer 角色，參照 revamp/final-review/CLAUDE.md 驗收執行結果
```

### 自動化工具

| 工具 | 用途 | 指令 |
|------|------|------|
| `site-audit.sh` | 網站健檢（效能、安全、SEO） | `./revamp/tools/site-audit.sh https://example.com` |
| `competitive-audit.sh` | 競品分析比較 | `./revamp/tools/competitive-audit.sh <our-url> <competitor1> <competitor2>` |
| `analyze-traffic.sh` | 流量數據分析 | `./revamp/tools/analyze-traffic.sh` |

詳細流程說明請參照 `revamp/CLAUDE.md`。

---

## 任務完成品質關卡

> **重要**：在執行完所有任務、向使用者回報「完成」之前，必須先執行以下檢查。
>
> 全部通過才能回報完成，否則必須先修正問題。

---

### 1. 連結檢查

- [ ] 所有新增/修改的內部連結正常，無 404
- [ ] 所有新增/修改的外部連結正常
- [ ] 無死連結或斷裂連結

---

### 2. SEO + AEO 標籤檢查

#### 2.1 Meta 標籤

- [ ] `<title>` 存在且 ≤ 60 字，含核心關鍵字
- [ ] `<meta name="description">` 存在且 ≤ 155 字
- [ ] `og:title`, `og:description`, `og:image`, `og:url` 存在
- [ ] `og:type` = "article"
- [ ] `article:published_time`, `article:modified_time` 存在（ISO 8601 格式）
- [ ] `twitter:card` = "summary_large_image"

#### 2.2 JSON-LD Schema（7 種必填）

| Schema | 必填欄位 |
|--------|----------|
| WebPage | speakable（至少 7 個 cssSelector） |
| Article | isAccessibleForFree, isPartOf（含 SearchAction）, significantLink |
| Person | knowsAbout（≥2）, hasCredential（≥1）, sameAs（≥1） |
| Organization | contactPoint, logo（含 width/height） |
| BreadcrumbList | position 從 1 開始連續編號 |
| FAQPage | 3-5 個 Question + Answer |
| ImageObject | license, creditText |

#### 2.3 條件式 Schema（依內容判斷）

| Schema | 觸發條件 | 必填欄位 |
|--------|----------|----------|
| HowTo | 有步驟教學 | step, totalTime |
| Recipe | 有食譜 | recipeIngredient, recipeInstructions |
| VideoObject | 有嵌入影片 | duration, thumbnailUrl |
| ItemList | 有排序清單（「N 大」「TOP」） | itemListElement |
| Review | 有評測內容 | itemReviewed, reviewRating |
| AggregateRating | 有多則評論 | ratingValue, ratingCount |
| Product | 有商品頁 | offers, brand |
| Event | 有活動資訊 | startDate, location |
| Course | 有課程內容 | provider, offers |
| LocalBusiness | 有店家資訊 | address, openingHoursSpecification |

#### 2.4 SGE/AEO 標記（AI 引擎優化）

| 標記 | 要求 |
|------|------|
| `.key-answer` | 每個 H2 必須有，含 `data-question` 屬性 |
| `.key-takeaway` | 文章重點摘要（2-3 個） |
| `.expert-quote` | 專家引言（至少 1 個） |
| `.actionable-steps` | 行動步驟清單 |
| `.comparison-table` | 比較表格（若有） |

#### 2.5 E-E-A-T 信號

- [ ] Person Schema 有專業認證（hasCredential）
- [ ] 至少 2 個高權威外部連結（.gov、學術期刊、專業協會）

#### 2.6 YMYL 檢查（健康/財務/法律內容適用）

- [ ] `lastReviewed` 欄位（最後審核日期）
- [ ] `reviewedBy` 欄位（審核者資訊）
- [ ] 免責聲明（醫療/財務/法律）

---

### 3. 內容更新確認

- [ ] 列出本次預計修改的所有檔案
- [ ] 逐一確認每個檔案都已正確更新
- [ ] 修改內容與任務要求一致
- [ ] 無遺漏項目

---

### 4. Git 狀態檢查

- [ ] 所有變更已 commit
- [ ] commit message 清楚描述本次變更
- [ ] 已 push 到 Github（除非另有指示）
- [ ] 遠端分支已更新

---

### 5. SOP 完成度檢查

- [ ] 回顧原始任務需求
- [ ] 原訂 SOP 每個步驟都已執行
- [ ] 無遺漏的待辦項目
- [ ] 無「之後再處理」的項目

---

## 檢查報告格式

完成檢查後，輸出以下格式：

```
## 完成檢查報告

| 類別 | 狀態 | 問題（如有） |
|------|------|-------------|
| 連結檢查 | ✅/❌ | |
| Meta 標籤 | ✅/❌ | |
| Schema（必填） | ✅/❌ | |
| Schema（條件式） | ✅/❌/N/A | |
| SGE/AEO 標記 | ✅/❌ | |
| E-E-A-T 信號 | ✅/❌ | |
| YMYL | ✅/❌/N/A | |
| 內容更新 | ✅/❌ | |
| Git 狀態 | ✅/❌ | |
| SOP 完成度 | ✅/❌ | |

**總結**：X/Y 項通過，狀態：通過/未通過
```

---

## 檢查未通過時

1. **不回報完成**
2. 列出所有未通過項目
3. 立即修正問題
4. 重新執行檢查
5. 全部通過才能說「完成」

---

## 任務開始時

接到新任務時，先建立本次檢查清單：

```
## 本次任務檢查清單

- 任務目標：[描述]
- 預計修改檔案：
  - [ ] 檔案1
  - [ ] 檔案2
- 預計新增內容：
  - [ ] 內容1
  - [ ] 內容2
- 適用的條件式 Schema：[列出]
- 是否為 YMYL 內容：是/否
```

---

## 參考文件

完整規則請參照：

### SEO/AEO
- `seo/CLAUDE.md` - SEO + AEO 規則庫
- `seo/writer/CLAUDE.md` - Writer 執行流程
- `seo/review/CLAUDE.md` - Reviewer 檢查清單

### 改版流程
- `revamp/CLAUDE.md` - 改版流程總覽
- `revamp/0-positioning/CLAUDE.md` - 品牌定位
- `revamp/1-discovery/CLAUDE.md` - 網站健檢
- `revamp/2-competitive/CLAUDE.md` - 競品分析
- `revamp/3-analysis/CLAUDE.md` - 受眾與差距分析
- `revamp/4-strategy/CLAUDE.md` - 改版策略
- `revamp/5-content-spec/CLAUDE.md` - 內容規格
- `revamp/final-review/CLAUDE.md` - 最終驗收
