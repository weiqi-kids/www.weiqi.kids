# Cloudflare 遷移與回退計畫（規劃版）

版本：0.1（2026-09-20）  
狀態：平台遷移規劃；尚未建立 Cloudflare 資源或部署程式。

## 1. 目前與目標

| 階段 | 前端 | 動態資料 | 部署 |
| --- | --- | --- | --- |
| 現況 | Docusaurus 靜態網站 | 無帳號、課程或討論資料層 | GitHub Pages |
| 過渡 | Astro 公開頁面可先獨立驗證 | Workers／D1／R2 以測試環境驗證 | 保留 GitHub Pages 回退 |
| 目標 | Astro 公開頁面與登入後介面 | Workers API、D1、R2 | Cloudflare Pages／Workers，依正式部署測試決定 |

## 2. Cloudflare 元件分工

| 元件 | 放置內容 | 主要限制 |
| --- | --- | --- |
| Pages 或 Astro adapter | 公開頁面與登入後 UI | 不直接決定課程資格 |
| Workers | OAuth callback、Email magic link、API、權限與內容操作；可使用 Email Service `send_email` binding 發送登入信 | 所有敏感操作都要在伺服器檢查權限 |
| D1 | 帳號、登入綁定、會員、報名、課程、實作營、文章與申訴 | 不放圖片二進位內容 |
| R2 | 作品圖片與附件 | 下載 URL 必須經權限檢查或短期簽名 |
| Access | 協會管理員內部工具的入口 | 不代替學員課程與實作營細部授權 |
| Turnstile | 報名、登入連結申請、發文等濫用防護候選 | 不代替身分驗證與權限檢查 |

## 3. 遷移順序

### Step 0：只讀盤點

- 確認 Cloudflare 帳戶、DNS、方案、網域與管理員身分。
- 若採 Cloudflare Email Service，確認 Workers Paid plan、寄件網域、SPF／DKIM／DMARC 與 Beta 服務限制。
- 建立環境命名：`dev`、`staging`、`production`；不共用資料庫與 R2 bucket。
- 確認 GitHub Pages 現站可繼續部署與回退。

### Step 1：公開頁面驗證

- 先建立 Astro 公開頁面與繁中內容。
- 不接登入、不接付款、不接討論資料。
- 以 preview domain 驗證 SEO、路由、無障礙與內容對照。

### Step 2：一門課的後端垂直切片

- 建立帳號、LINE／Email 登入綁定、會員一年期、報名編號與人工核款狀態。
- 建立一門課、一個實作營、文章、圖片、編輯、撤回、隱藏、通知與申訴。
- 只在 staging 使用測試資料，不匯入舊站人物或作品作為學員資料。

### Step 3：小規模試用

- 由協會指定測試帳號與一門真實課程。
- 逐項驗證報名、核款、開通、自行加入實作營、版主管理與會員到期。
- 產生操作紀錄、錯誤處理與資料備份檢查表。

### Step 4：正式切換

- 確認公開 URL、DNS、憑證、sitemap、canonical 與 redirect。
- 先切換公開頁面，再啟用動態功能；保留現站回退窗口。
- 動態資料只從新系統開始建立；舊站內容依 inventory 結果逐筆轉寫或保存。

## 4. 回退條件

以下任一情況發生時，停止切換並回到 GitHub Pages 公開站：

- 登入方式無法穩定辨識同一帳號。
- 課程資格與實作營內容的權限檢查出現越權。
- R2 圖片可被未授權直接取得。
- 管理員隱藏、通知或申訴沒有可追蹤紀錄。
- SEO 核心 URL、canonical 或 sitemap 出現大量錯誤。

回退只切換公開流量，不刪除 staging 或 production 資料；資料恢復與重試必須另有備份與操作紀錄。

## 5. 不在本計畫內的操作

- 不在規格完成前建立正式 Cloudflare 資源。
- 不把 Access cookie 當作課程資格。
- 不在沒有匯入對照表與權利確認時批次匯入舊站人物、作品或會員資料。
- 不在一次部署中同時完成 Astro、設計系統、會員後端與全部內容轉移。
