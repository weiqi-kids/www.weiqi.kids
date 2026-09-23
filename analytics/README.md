# Analytics 多管道流量收集

協會的流量數據分散在五個地方：網站、Google 搜尋、GitHub repo、YouTube 頻道、LINE 官方帳號。
這個目錄把它們每天各自抓回來、累積成歷史，再每週彙整成一份中文週報。

## 排程（2026-09-23 更新）

每日收集與週報掛在**本機 cron**（`/etc/cron.d/seo-ops-www-weiqi-kids`），憑證直接讀主機上的檔案，
不需要 GitHub Secrets，也沒有 GitHub Actions workflow：

- 台北 09:00：`analytics/scripts/collect-all.sh`（YouTube / LINE@ / GA4 / GSC）
- 週一台北 09:30：`analytics/scripts/weekly-report.mjs`

log 在 `analytics/logs/`（不進 git）。GitHub Actions 只保留既有的 GitHub Traffic 收集。

## 憑證現況（2026-09-23 更新）

GA4 與 Search Console **不需要另外建立服務帳號**：主機上既有的 `~/.config/ga4-insights/sa-key.json` 已經具備
GA4 資源 `weiqi-kids`（G-16V1KSEH6W）與 `sc-domain:weiqi.kids` 的權限，收集器會自動找到它。
若日後要把好棋寶寶的數據與其他站台的服務帳號分開，再另建一把金鑰放到 `~/.config/weiqi-kids/google-sa.json`，
收集器會優先使用該路徑。

## 目錄結構

```
analytics/
├── config.json                   # 非機密設定（資源 ID、站台、保留天數）
├── raw/                          # 原始 API 回應（按日期，保留 30 天）
│   ├── views-YYYY-MM-DD.json     # GitHub 瀏覽
│   ├── clones-YYYY-MM-DD.json    # GitHub clone
│   ├── paths-YYYY-MM-DD.json     # GitHub 熱門頁面
│   ├── referrers-YYYY-MM-DD.json # GitHub 流量來源
│   ├── ga4-YYYY-MM-DD.json       # GA4
│   ├── gsc-YYYY-MM-DD.json       # Search Console
│   ├── youtube-YYYY-MM-DD.json   # YouTube
│   └── line-YYYY-MM-DD.json      # LINE 官方帳號
├── history/                      # 每日累積（以日期去重，可重跑）
│   ├── daily-views.json          # GitHub 瀏覽
│   ├── daily-clones.json         # GitHub clone
│   ├── ga4.json
│   ├── gsc.json
│   ├── youtube.json
│   └── line.json
├── current/                      # 最新快照
│   ├── popular-paths.json        # GitHub 熱門頁面
│   ├── referrers.json            # GitHub 流量來源
│   ├── ga4-summary.json / ga4-top-pages.json / ga4-outbound-clicks.json
│   ├── gsc-summary.json / gsc-top-queries.json / gsc-top-pages.json
│   ├── youtube-channel.json / youtube-videos.json / youtube-analytics.json
│   ├── line-oa.json / line-demographic.json
│   └── <來源>-status.json        # 該來源是否已接上（週報靠這個標示「尚未接上」）
├── reports/
│   ├── YYYY-Www.md               # 多管道週報（新版，ISO 週次）
│   └── weekly-YYYY-Www.md        # 舊版 GitHub Traffic 週報
└── scripts/
    ├── lib/collector.mjs         # 共用：目錄慣例、日期、落地、缺憑證處理
    ├── lib/google-data.mjs       # 共用：Google 服務帳號 JWT + GA4/GSC API
    ├── collect-ga4.mjs
    ├── collect-gsc.mjs
    ├── collect-youtube.mjs
    ├── collect-line.mjs
    ├── weekly-report.mjs
    └── merge-traffic-data.js     # GitHub Traffic 的歷史合併（既有）
```

## 資料來源一覽

| 來源 | 收集器 | 指標 | 憑證 | 狀態 |
|------|--------|------|------|------|
| **GA4**（`G-16V1KSEH6W`，新舊站共用） | `collect-ga4.mjs` | 工作階段、活躍使用者、瀏覽、熱門頁、站外點擊事件 | Google 服務帳號 | 待站主授權 |
| **Search Console** | `collect-gsc.mjs` | 曝光、點擊、CTR、平均排名、熱門查詢／著陸頁 | Google 服務帳號（同一把） | 待站主授權 |
| **YouTube**（`@goodmoveassociation`） | `collect-youtube.mjs` | 訂閱數、影片數、總觀看；每支影片的觀看／按讚／留言 | 既有 OAuth 權杖 | 已接上 |
| **YouTube Analytics**（選配） | 同上 | 每日觀看時間、流量來源 | 另一把帶 `yt-analytics.readonly` 的權杖 | 待站主授權 |
| **LINE 官方帳號**（`@685hqmrm`） | `collect-line.mjs` | 好友數、有效觸及、封鎖數、訊息送達、受眾輪廓 | Messaging API 長效權杖 | 已接上 |
| **GitHub Traffic** | `collect-traffic.yml` + `merge-traffic-data.js` | repo 瀏覽、clone、熱門路徑、referrer | `TRAFFIC_TOKEN` | 已接上 |

### 共通行為

每個收集器都能單獨執行，而且**憑證缺少時會印出中文說明、以 exit 0 結束**，不會讓每日排程整包失敗。
同時它會把「尚未接上」寫進 `current/<來源>-status.json`，週報讀到就會明確標示，不會拿估算值填空。

```bash
node analytics/scripts/collect-youtube.mjs          # 抓今天
node analytics/scripts/collect-line.mjs             # 抓前一天（LINE insight 有延遲）
node analytics/scripts/collect-line.mjs 2026-09-20  # 指定日期
node analytics/scripts/collect-ga4.mjs
node analytics/scripts/collect-gsc.mjs
node analytics/scripts/weekly-report.mjs            # 上一個完整週
node analytics/scripts/weekly-report.mjs 2026-09-23 # 指定日期所屬的那一週
```

---

## 憑證

**所有機密只放 `~/.config/weiqi-kids/`（chmod 600），或 CI 的 GitHub Secrets。一律不進 repo。**

| 檔案 | 用途 | 對應 GitHub Secret |
|------|------|--------------------|
| `google-sa.json` | GA4 + Search Console 服務帳號金鑰 | `GOOGLE_SA_JSON` |
| `youtube-client.json` | YouTube OAuth 用戶端 | `YOUTUBE_CLIENT_JSON` |
| `youtube-token.json` | YouTube OAuth 權杖（含 refresh token） | `YOUTUBE_TOKEN_JSON` |
| `youtube-analytics-token.json` | （選配）帶 `yt-analytics.readonly` 的權杖 | `YOUTUBE_ANALYTICS_TOKEN_JSON` |
| `line-token.txt` | LINE Messaging API 長效存取權杖 | `LINE_CHANNEL_ACCESS_TOKEN` |
| —（既有） | GitHub Traffic API 的 PAT | `TRAFFIC_TOKEN` |

Secret 的值就是檔案的完整內容（JSON 檔請整份貼上，`line-token.txt` 貼那一行字串）。

---

## 站主要做的授權步驟

### A. GA4 授權步驟

GA4 的 Data API 不吃網站上的評量 ID，要另外開一個服務帳號。

1. 開 <https://console.cloud.google.com/>，選一個專案（沒有就新建，名稱例如 `weiqi-kids-analytics`）。
2. 左側「API 和服務」→「程式庫」，搜尋並啟用這兩個 API：
   - **Google Analytics Data API**
   - **Google Search Console API**（下一節也要用，一起開）
3. 左側「IAM 與管理」→「服務帳戶」→「建立服務帳戶」。
   - 名稱填 `analytics-collector`，角色可以留空，直接「完成」。
4. 點進剛建好的服務帳戶 →「金鑰」→「新增金鑰」→「建立新的金鑰」→ 選 **JSON** → 下載。
5. 把下載的檔案存成 `~/.config/weiqi-kids/google-sa.json`，然後 `chmod 600` 它。
6. 複製服務帳戶的 email（長得像 `analytics-collector@專案.iam.gserviceaccount.com`）。
7. 開 <https://analytics.google.com/> →左下角「管理」→ 選到 `G-16V1KSEH6W` 那個資源 →「資源存取管理」→ 右上角「+」→「新增使用者」。
   - 貼上服務帳戶 email，角色勾 **檢視者**，取消勾「通知新使用者」，按「新增」。
8. 確認 `analytics/config.json` 的 `ga4.propertyId` 是正確的**資源 ID**（在「資源設定」頁最上方，是一串數字，目前填 `458470883`）。
9. 驗證：`node analytics/scripts/collect-ga4.mjs`，看到一行 `[ga4] ...｜工作階段 N` 就成功了。

### B. Search Console 授權步驟

1. 前四步跟上面一樣（同一個服務帳戶、同一把金鑰，不用再做一次）。
2. 開 <https://search.google.com/search-console> → 選 `weiqi.kids` 資源 → 左下「設定」→「使用者和權限」→「新增使用者」。
   - 貼上服務帳戶 email，權限選 **完整**（受限也可以，但完整比較不會卡）。
3. 確認 `analytics/config.json` 的 `gsc.siteUrl` 與 Search Console 上的資源型態一致：
   - 網域資源寫 `sc-domain:weiqi.kids`
   - 網址前置字元資源寫 `https://www.weiqi.kids/`
4. 驗證：`node analytics/scripts/collect-gsc.mjs`。

> GSC 資料有 2～3 天延遲，這是 Google 端的限制，收集器預設抓「今天往前 3 天」結束的 7 天區間。

### C. YouTube Analytics 授權步驟（選配）

目前的權杖只有 `https://www.googleapis.com/auth/youtube` scope，夠抓頻道與影片的公開統計，
但抓不到「觀看時間」「流量來源」這類只有頻道擁有者看得到的數據。要補的話：

1. 在同一個 Google Cloud 專案啟用 **YouTube Analytics API**。
2. 用既有的 `youtube-client.json`，跑一次裝置授權流程，scope 改成
   `https://www.googleapis.com/auth/yt-analytics.readonly`。
   （可比照 `site/scripts/youtube-upload.py` 的 `auth` 流程，把 `SCOPE` 換掉。）
3. 授權後的 JSON 存成 `~/.config/weiqi-kids/youtube-analytics-token.json`，`chmod 600`。
4. 驗證：`node analytics/scripts/collect-youtube.mjs`，輸出不再出現「未設定 YouTube Analytics 權杖」。

沒做這步不影響其他數據，週報會明確標示這一段尚未接上。

### D. LINE 官方帳號

權杖已經設定好（`~/.config/weiqi-kids/line-token.txt`）。要重新產生時：

1. 開 <https://developers.line.biz/console/> → 選 `好棋寶寶` 的 Provider → Messaging API channel。
2. 「Messaging API」分頁最下方「Channel access token (long-lived)」→ Issue／Reissue。
3. 覆寫 `~/.config/weiqi-kids/line-token.txt`，`chmod 600`。

> LINE insight 端點有一天延遲，而且好友數未達 LINE 的統計門檻時，`demographic` 會回 `available: false`、
> `message/delivery` 會只回 `{"status":"ready"}`。收集器把這些都當成正常情況處理，記成空值，不報錯。

### E. GitHub Secrets（給 Actions 用）

repo → Settings → Secrets and variables → Actions → New repository secret，依上面「憑證」表新增：

- `GOOGLE_SA_JSON`
- `YOUTUBE_CLIENT_JSON`
- `YOUTUBE_TOKEN_JSON`
- `YOUTUBE_ANALYTICS_TOKEN_JSON`（選配）
- `LINE_CHANNEL_ACCESS_TOKEN`
- `TRAFFIC_TOKEN`（既有，GitHub Traffic 用）

---

## GitHub Actions

| Workflow | 排程 | 做什麼 |
|----------|------|--------|
| `.github/workflows/collect-traffic.yml` | 每日 UTC 00:00 | GitHub Traffic API（既有） |
| `.github/workflows/collect-channels.yml` | 每日 UTC 00:30 | GA4／GSC／YouTube／LINE，週一多產一份週報 |
| `.github/workflows/traffic-report.yml` | 每週一 UTC 01:00 | 舊版 GitHub Traffic 週報（既有） |

commit 訊息都帶 `[skip ci]`，避免數據 commit 觸發網站重建。

```bash
gh workflow run collect-channels.yml                      # 手動收集
gh workflow run collect-channels.yml -f report=true       # 收集並產週報
gh run list --workflow=collect-channels.yml
```

---

## UTM 命名規則

跨管道推廣連到網站時，連結要帶 UTM 參數，GA4 才分得出人是從哪個管道來的。
**規則：一律小寫、用半形連字號，不要用底線或空白。**

```
https://www.weiqi.kids/課程頁/?utm_source=line-oa&utm_medium=message&utm_campaign=2026-autumn-camp
```

### utm_source（人從哪個平台來）

| 值 | 用在哪 |
|----|--------|
| `youtube` | YouTube 影片說明欄、頻道首頁、置頂留言 |
| `line-oa` | LINE 官方帳號 `@685hqmrm`（群發訊息、圖文選單、主頁） |
| `line-community` | LINE 社群 |
| `instagram` | Instagram |
| `facebook` | Facebook 粉專／社團 |
| `threads` | Threads |

### utm_medium（在那個平台的哪個位置）

| 值 | 用在哪 |
|----|--------|
| `description` | 影片說明欄、貼文的說明區塊 |
| `message` | 主動推播的訊息（LINE 群發、圖文選單） |
| `profile` | 個人／頻道／帳號的簡介欄、主頁連結 |
| `post` | 一般貼文 |
| `story` | 限時動態 |

### utm_campaign（這次在推什麼）

用活動或課程代號，格式 `YYYY-代號`，同一檔活動不管發在哪個平台都用同一個值，這樣才彙總得起來。

| 範例 | 意思 |
|------|------|
| `2026-autumn-camp` | 2026 秋季學習營 |
| `2026-carnival` | 圍棋嘉年華 |
| `2026-membership` | 會員招募 |
| `evergreen-intro` | 沒有檔期的常態介紹連結 |

### 為什麼站內連結不要加 UTM

站內連結（weiqi.kids 的某頁連到 weiqi.kids 的另一頁）**絕對不要加 UTM**，原因有三個：

1. **會把一次造訪切成兩次。** GA4 看到 UTM 參數就認定這是一個新的流量來源，於是結束原本的工作階段、開一個新的。
   一個人從首頁點到課程頁，本來是一次造訪兩個頁面，加了 UTM 之後變成兩次造訪，工作階段數虛胖、跳出率失真、
   轉換路徑也被切斷，完全看不出他是從哪裡進站的。
2. **會蓋掉真正的來源。** 原本記錄的「從 YouTube 來」會被站內的 UTM 覆寫成「從自己來」，
   等於把最重要的獲客歸因資訊親手刪掉。
3. **本來就不需要。** 站內動線 GA4 用 `page_referrer` 和事件（例如 CTA 點擊）就看得到，
   要區分哪個按鈕帶來的點擊，用事件參數，不要用 UTM。

UTM 只用在**外部平台連進網站**的那一步。

---

## 週報

```bash
node analytics/scripts/weekly-report.mjs
```

輸出 `analytics/reports/YYYY-Www.md`（ISO 週次，週一起算），含五節：
網站（GA4）、搜尋（GSC）、YouTube、LINE 官方帳號、GitHub repo 流量。

沒接上的來源會寫成：

```
> **尚未接上** — 尚未授權，缺服務帳號金鑰（…）。
>
> 取得方式：…
```

**不會**拿估算值或去年同期去填空。數字缺就是缺。

---

## 維護

- **原始數據**：各收集器跑完會自動清掉 30 天前的同來源 raw 檔（天數在 `config.json` 的 `retention.rawDays`）。
- **歷史數據**：永久累積，以日期去重，同一天重跑會覆蓋不會長出重複列。
- **重跑補資料**：`node analytics/scripts/collect-line.mjs 2026-09-20` 這樣指定日期即可。
  GA4／GSC 一次回整個 7 天區間，會自動回填中間漏掉的日子。

## 其他數據來源

### Plausible Analytics

隱私友好的即時分析，目前未啟用。要用的話在 Plausible 建立站台後，於網站模板掛上 script。

### 正式網域切換前的注意事項

新站在 `site/`（Astro + Cloudflare Worker），staging 為
<https://weiqi-kids-staging.weiqi-kids-site.workers.dev>，staging 不送 GA4 數據也不開放索引。
`www.weiqi.kids` 切換到新站之前，GA4 與 GSC 抓到的仍是舊 Docusaurus 站的數據，看趨勢時要記得這件事。
