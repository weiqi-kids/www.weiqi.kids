# Analytics 流量數據收集

此目錄儲存網站流量數據，由 GitHub Actions 每日自動收集。

## 目錄結構

```
analytics/
├── raw/                          # 原始 API 回應（按日期，保留 30 天）
│   ├── views-YYYY-MM-DD.json     # 每日瀏覽數據
│   ├── clones-YYYY-MM-DD.json    # 每日 clone 數據
│   ├── paths-YYYY-MM-DD.json     # 熱門頁面
│   └── referrers-YYYY-MM-DD.json # 流量來源
├── history/                      # 歷史數據（累積）
│   ├── daily-views.json          # 每日瀏覽數累積
│   └── daily-clones.json         # 每日 clone 數累積
├── current/                      # 最新快照
│   ├── popular-paths.json        # 最新熱門頁面
│   └── referrers.json            # 最新流量來源
├── scripts/
│   └── merge-traffic-data.js     # 數據合併腳本
└── README.md                     # 本文件
```

## 數據來源

### GitHub Traffic API

GitHub Traffic API 提供過去 14 天的數據，無法回溯更早。因此我們每日收集並累積成歷史數據。

**API 端點**：
- `/repos/{owner}/{repo}/traffic/views` - 瀏覽數
- `/repos/{owner}/{repo}/traffic/clones` - Clone 數
- `/repos/{owner}/{repo}/traffic/popular/paths` - 熱門頁面 (Top 10)
- `/repos/{owner}/{repo}/traffic/popular/referrers` - 流量來源 (Top 10)

**限制**：
- 需要 `repo` 權限的 Personal Access Token
- 數據只保留 14 天滑動窗口
- 熱門頁面和流量來源只顯示 Top 10

## GitHub Actions Workflow

每日 UTC 00:00 自動執行 `.github/workflows/collect-traffic.yml`：

1. 抓取 GitHub Traffic API 數據
2. 儲存原始數據到 `raw/`
3. 執行 `merge-traffic-data.js` 合併歷史數據
4. 複製最新快照到 `current/`
5. 自動 commit 並 push

**手動觸發**：
```bash
gh workflow run collect-traffic.yml
```

## 使用方式

### 查看流量分析報告

```bash
./revamp/tools/analyze-traffic.sh
```

### 查看熱門頁面

```bash
cat analytics/current/popular-paths.json | jq '.'
```

輸出範例：
```json
[
  {
    "path": "/docs/for-players/",
    "title": "給圍棋棋友",
    "count": 150,
    "uniques": 80
  }
]
```

### 查看流量來源

```bash
cat analytics/current/referrers.json | jq '.'
```

輸出範例：
```json
[
  {
    "referrer": "github.com",
    "count": 100,
    "uniques": 50
  }
]
```

### 查看歷史瀏覽趨勢

```bash
# 最近 7 天
cat analytics/history/daily-views.json | jq '.views[-7:]'

# 所有歷史數據
cat analytics/history/daily-views.json | jq '.views'
```

輸出範例：
```json
[
  {
    "timestamp": "2026-02-13T00:00:00Z",
    "count": 45,
    "uniques": 20
  }
]
```

## 設定步驟

### 1. 建立 GitHub Personal Access Token

1. 前往 https://github.com/settings/tokens
2. 點擊 "Generate new token (classic)"
3. 勾選 `repo` 權限
4. 複製 token

### 2. 新增 Repository Secret

1. 前往 repo Settings > Secrets and variables > Actions
2. 新增 secret：`TRAFFIC_TOKEN`
3. 貼上 token

### 3. 測試 Workflow

```bash
# 手動觸發
gh workflow run collect-traffic.yml

# 查看執行狀態
gh run list --workflow=collect-traffic.yml
```

## 與 Revamp 流程整合

在執行 `revamp/1-discovery` 階段時，流量數據會自動納入分析：

```bash
# Discovery 階段會讀取流量數據
請以 Writer 角色，參照 revamp/1-discovery/CLAUDE.md 執行網站健檢
```

## 其他數據來源

### Plausible Analytics

隱私友好的即時分析，需另外設定：
- Cloud: https://plausible.io/
- Self-hosted: Docker Compose

設定後在 `docusaurus.config.js` 取消註解 Plausible script。

### Google Search Console

搜尋關鍵字和排名數據：

1. 前往 https://search.google.com/search-console
2. 新增資源：`https://www.weiqi.kids/`
3. 下載驗證 HTML 檔案到 `static/`
4. 部署並完成驗證

## 維護

- **原始數據清理**：`merge-traffic-data.js` 會自動清理 30 天前的原始檔案
- **歷史數據**：永久累積，不會自動清理
- **手動清理**：如需清理歷史數據，直接編輯 `history/*.json`
