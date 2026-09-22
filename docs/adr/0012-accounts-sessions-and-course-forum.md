# ADR 0012：帳號、登入與單門課程論壇的實作方式

- 狀態：已接受
- 日期：2026-09-22

## 決策

- 帳號、登入方式、會員、課程資格、實作營、論壇、隱藏、申訴與稽核都存在 D1（`site/migrations/0002_accounts_and_forum.sql`）。
- Session 由網站自己管理：cookie 只放隨機值（HttpOnly、Secure、SameSite=Lax），D1 存 hash，可撤銷，效期 30 天。所有 POST 檢查 Origin。
- Email 一次性登入連結：15 分鐘、單次使用、重發使舊連結失效、只存 hash。連結頁先顯示確認按鈕，按下才使用，避免信箱的連結掃描器把連結用掉。寄不出信就撤銷連結。
- LINE 登入用 OAuth 2.1 + OpenID（`sub` 辨識），Email 與 LINE 只在使用者登入後自行確認才綁定；不依 Email 或名稱自動合併。
- 管理員核款開通時：報名已連到帳號就用該帳號，否則用報名 Email 找帳號，找不到就預先建立 Email 帳號。學員之後用同一個 Email 登入即可看到課程。
- 管理區以 `accounts.is_admin` 控制；第一位管理員以 `wrangler d1 execute` 設定。Cloudflare Access 暫不使用。
- 圖片存 R2，只經權限檢查的路由輸出；R2 未啟用時論壇照常運作，只關閉圖片上傳。
- 通知先放站內（帳號頁）；Email 寄送服務啟用後再加寄信。
- 第一門課的講師無法滿足「先上過一門課」：管理員可填寫例外理由直接指定講師，理由寫入稽核紀錄。

## 後果

- 登入後頁面全部 `noindex`、`cache-control: private, no-store`，不進 sitemap。
- 推薦頁（講師認可＋作者同意公開）不在這一期範圍。
