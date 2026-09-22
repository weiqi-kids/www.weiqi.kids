# 新站（Astro 7 + Cloudflare Workers）

架構與分期見 `docs/adr/0007`～`0012`。

## 指令

```bash
pnpm dev                 # 本機開發
pnpm build               # 設計守門 → 內容守門 → 產生轉址 → 建置 → 站內連結檢查
pnpm test                # 單元測試
pnpm run deploy          # 部署 staging（含草稿課程）
node scripts/migrate-legacy.mjs   # 重新從舊站 docs/ 轉寫文章（會覆寫 src/content/pages/）
```

## 環境

| 項目 | staging |
|---|---|
| 網址 | https://weiqi-kids-staging.weiqi-kids-site.workers.dev |
| D1 | `weiqi-kids-staging` |
| 變數 | `SHOW_DRAFT_COURSES=1`、`MAGIC_LINK_DEBUG=1`（寄信服務未啟用時，登入連結直接顯示在頁面） |

正式環境尚未建立，切換步驟見下方。

## 常用維運

```bash
# 套用資料表變更
npx wrangler d1 migrations apply weiqi-kids-staging --remote
# 指定管理員（先用該 Email 登入一次）
npx wrangler d1 execute weiqi-kids-staging --remote --command \
  "UPDATE accounts SET is_admin=1 WHERE id=(SELECT account_id FROM login_identities WHERE provider='email' AND subject='someone@example.com')"
# LINE 登入（LINE Developers 建立 LINE Login channel，callback 設為 <網址>/auth/callback/line/）
npx wrangler secret put LINE_CHANNEL_ID
npx wrangler secret put LINE_CHANNEL_SECRET
# 報名表 Turnstile（選用）
npx wrangler secret put TURNSTILE_SECRET_KEY   # 並在建置時設定 PUBLIC_TURNSTILE_SITE_KEY
```

## 新增動態路由時

在 `wrangler.jsonc` 的 `assets.run_worker_first` 加上路徑。沒加的話，瀏覽器直接開啟該網址會被靜態資產層回 404（curl 測不出來，要帶 `Sec-Fetch-Mode: navigate` 標頭測試）。

## 開課

在 `src/content/courses/<slug>.md` 新增課程（`status: open`，4 個場次），部署後即開放報名。TTQS 整理方式見 `docs/ttqs-first-course.md`。

## 切換正式網域前要完成

1. 在 Cloudflare 後台啟用 R2（圖片上傳）。
2. 依 `revamp/4-strategy/2026-09-20-dns-migration-continuity.md` 把 `weiqi.kids` 的 DNS 搬到 Cloudflare。
3. 啟用 Email Sending：`npx wrangler email sending enable weiqi.kids`，在 wrangler 加 `send_email` binding 與 `MAIL_FROM`。
4. 在 `wrangler.jsonc` 加 `env.production`：新的 D1、R2、`routes` 指向 `www.weiqi.kids`，**不要**設定 `SHOW_DRAFT_COURSES`、`MAGIC_LINK_DEBUG`。
5. 部署正式環境、驗證後，停用舊站 GitHub Pages 部署（保留回退）。
