# Cloudflare 平台導入盤點

盤點日期：2026-09-19。狀態：規劃候選，尚未建立 Cloudflare 資源、安裝套件或修改網站。

## 結論

Cloudflare 適合成為新網站的執行平台，但 Cloudflare Access 不應單獨承擔會員系統。Access 可作為管理員後台與受保護應用程式的身分入口；學員帳號綁定、會員資格、課程權限、實作營成員、文章與內容管理仍需由網站應用程式處理。

LINE Login 使用 OAuth 2.0／OpenID Connect，可透過 Cloudflare Access 的 Generic OIDC 接入。Cloudflare Access 的 Email One-Time PIN 是一次性 PIN，官方流程要求使用者輸入 PIN；它不完全等同本專案已確認的一次性 Email 登入連結。Cloudflare 目前另有 Email Service 的 Email Sending，可由 Worker 發送交易型信件與 magic link；官方目前標示為 Beta，需 Cloudflare DNS 與 Workers Paid plan，因此列入平台方案與風險確認，不再預設一定要接第三方寄信服務。

參考：[Cloudflare Access self-hosted application](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/)、[Cloudflare Generic OIDC](https://developers.cloudflare.com/cloudflare-one/integrations/identity-providers/generic-oidc/)、[Cloudflare One-Time PIN](https://developers.cloudflare.com/cloudflare-one/integrations/identity-providers/one-time-pin/)、[LINE Login 官方文件](https://developers.line.biz/en/docs/line-login/integrate-line-login/)。

## 現況基準

| 項目 | 目前狀態 | 對 Cloudflare 規劃的影響 |
| --- | --- | --- |
| 前端 | Docusaurus 3.9.2、React 19、靜態 GitHub Pages | 目前沒有 Worker、API 或動態資料層；Astro 7 遷移不能直接解決登入與討論功能 |
| 部署 | GitHub Actions 建置後推送 `gh-pages`；`static/CNAME` 為 `www.weiqi.kids` | 需另外規劃 Cloudflare Pages／Workers 部署與 DNS 切換；不可直接把現有 gh-pages 當成動態後端 |
| 登入 | 現有程式未發現登入、報名或會員功能 | 需建立學員帳號、LINE／Email 綁定與一次性登入連結流程 |
| 討論 | 現有程式未發現站內討論、圖片上傳或內容管理 | 需建立文章、附件、成員可見性、撤回、隱藏、通知與申訴模型 |
| 網域 | `www.weiqi.kids` | Cloudflare Zone、DNS、憑證、Cookie scope 與未來子網域策略需先確認帳戶狀態 |

## 建議的 Cloudflare 元件分工

| 元件 | 建議責任 | 備註 |
| --- | --- | --- |
| Astro | 公開繁中頁面、課程介紹、技能單張與登入後介面 | 只負責呈現與路由，不直接保存權限真相 |
| Cloudflare Workers | API、登入 callback、magic link、權限檢查、發文與內容管理 | 必須在每個寫入與讀取 API 重新檢查權限 |
| D1 | 學員、登入方式綁定、報名編號、會員、課程、實作營成員、文章、申訴紀錄 | 需先定義資料模型與 migration 策略 |
| R2 | 作品圖片與附件 | 需規劃上傳授權、檔案大小／格式、公開撤回與孤兒檔案處理 |
| Access | 協會管理員後台、內部工具或受保護管理 API | 不直接取代學員課程權限；Access JWT 仍需由 Worker 驗證 |
| Durable Objects | 只有在需要即時互動、房間狀態或高一致性協作時評估 | 第一階段討論區可先採一般 Worker + D1，不預設加入 |
| Turnstile | 公開報名、登入連結請求與發文等防濫用關卡 | 是否需要及放置位置待威脅模型確認 |

## 必須保留的領域邊界

- 登入身分、協會會員、課程資格與實作營成員不是同一個狀態。
- LINE／Email 綁定必須取得學員同意，不依相同 Email 自動合併。
- 報名編號只用來對應報名與匯款，不是登入憑證。
- 協會人工核款後開通單課；首課開通同步確認新會員資格。
- 實作營是討論區，學員可自由發文與上傳；不需要事前審核。
- 作者可編輯與撤回；協會管理員可隱藏不適當內容並通知作者，作者可申訴。
- 作品撤回公開不等於刪除營內紀錄；課程結束或停止會員後，歷史內容與互動仍保留。

## 分階段建議

### Phase 0：平台與資料風險確認

- 確認 Cloudflare 帳戶、Zone、DNS 管理權、Workers／D1／R2 可用方案與成本。
- 確認 GitHub Pages 與 Cloudflare 部署的並行、預覽與回退方式。
- 定義資料分類：公開內容、會員資料、付款證明、作品附件、管理紀錄。
- 先做資料模型與威脅模型，不建立正式登入或收款流程。

### Phase 1：繁中公開網站與內容遷移

- Astro 7 公開頁面與繁中內容架構。
- 舊站資料盤點與歷史調閱策略。
- 保留 SEO URL／轉址策略與現有搜尋資產。
- 不把會員討論功能假裝成靜態頁面完成。

### Phase 2：帳號、課程與討論區垂直切片

- LINE Login、Email magic link、同一學員帳號綁定。
- 報名編號、人工核款、首課／新課開通。
- 實作營討論區、圖片上傳、作者編輯／撤回、管理員隱藏／通知／申訴。
- 以一個實際課程完成端到端驗收，再擴充其他課程。

### Phase 3：推薦展示與營運完善

- 將推薦頁與討論區自由發表分開。
- 完成管理員工作台、通知紀錄、申訴處理與內容審計。
- 再評估 TTQS 所需的訓練紀錄與對外招生流程。

## 尚未選定的 Cloudflare 方案

- 使用 Access 作為管理員入口，或由 Worker 驗證管理員身分。
- Cloudflare Email Service Email Sending Beta 的啟用方案、寄件網域與 token 保存方式。
- D1 schema、R2 bucket 公開策略、附件病毒／格式檢查。
- Cloudflare Pages、Workers Static Assets 或其他 Astro 部署方式。
- 現有 `www.weiqi.kids` DNS 切換、快取、預覽與回退策略。

本文件是盤點與候選架構，不是技術選型 ADR；在帳戶、成本、資料模型與部署驗證完成前，不開始實作。
