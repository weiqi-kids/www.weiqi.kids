# DNS 遷移前置檢核：今日實測狀態

日期：2026-09-27　·　基準文件：[`4-strategy/2026-09-20-dns-zone-inventory.md`](4-strategy/2026-09-20-dns-zone-inventory.md)、[`4-strategy/2026-09-20-dns-migration-continuity.md`](4-strategy/2026-09-20-dns-migration-continuity.md)

那兩份文件的計畫（Phase A/B/C、驗收、回退）已經完整，不重寫。這份補的是**今天實際解析與連線的結果**，用來關掉原文件列為「目前不能自行補上的資料」的部分。

起因：要在 `@weiqi.kids` 開信箱，但 Cloudflare Email Routing 需要該網域的 DNS 由 Cloudflare 託管。實測 `weiqi.kids` 的 NS 是 `ns1~ns5.linode.com`，zone 不在 Cloudflare 帳戶內（`wrangler email routing settings weiqi.kids` 回 `Could not find zone`）。所以信箱卡在 DNS 遷移之後。

## 1. 今日解析結果（全部實測）

### GitHub Pages（CNAME → `weiqi-kids.github.io.`）

`www`、`ecommerce`、`epialert`、`learn`、`risk`、`security`、`skills`、`supplement`、`trade`、`vuko`，以及三組 wildcard `*.intel`、`*.proof`、`*.z`。

裸網域 `@` 是 A × 4 → `185.199.108/109/110/111.153`。
`brighterarc` → `lightchang.github.io.`（不同 owner）。
`media.peertube` → `weiqikids-peertube.jp-osa-1.linodeobjects.com.`

### Linode 主機

| 主機名 | IP | HTTP | HTTPS | 判讀 |
| --- | --- | --- | --- | --- |
| `bible` | 172.235.205.148 | 301 | **200** | 服務正常 |
| `guessing` | 172.235.205.148 | 301 | **200** | 服務正常 |
| `listening` | 172.235.205.148 | 301 | **200** | 服務正常（另有 AAAA `2400:8905::2000:6fff:fe51:2e08`） |
| `writer` | 172.235.205.148 | 301 | **200** | 服務正常 |
| `akora` | 172.235.205.148 | 301 | 404 | 主機在，沒有站台 |
| `admin.bless`、`bless` | 172.235.205.148 | 301 | 連不上 | HTTP 轉向 HTTPS 但 TLS 失敗 |
| `analytics`、`status`、`mastodon`、`media.mastodon`、`peertube` | 172.235.205.148 | 301 | 連不上 | 同上；原盤點已標「不再使用」 |
| `chrant`、`jacob`、`miner`、`mcp`、`nginx`、`phpmyadmin`、`strapi`、`summon` | 172.237.11.63 | 連不上 | 連不上 | **整批無回應** |

Wildcard `*.ai`、`*.listening` → `172.235.205.148`。

### 兩台主機的連通性

| IP | 80 | 443 | 22 |
| --- | --- | --- | --- |
| 172.235.205.148 | 開 | 開 | 開 |
| **172.237.11.63** | 開 | **不通** | **不通** |

### 郵件與驗證

| 名稱 | 類型 | 現況 |
| --- | --- | --- |
| `@` | MX | **沒有**（與 09-20 盤點一致） |
| `@` | TXT | Brevo 驗證碼、Zoho 驗證、Google Search Console 驗證，三筆都還在 |
| `_dmarc` | TXT | `v=DMARC1; p=none; rua=mailto:rua@dmarc.brevo.com` |
| `brevo1._domainkey`、`brevo2._domainkey` | CNAME | `b1/b2.weiqi-kids.dkim.brevo.com.`，都還在 |
| `@` | CAA | **沒有** |
| `@` | SPF | **沒有**（與 09-20 盤點一致） |

## 2. 與 09-20 盤點的差異

1. **`172.237.11.63` 整台的 web 服務已經不通。** 原盤點把 `mcp`、`strapi`、`phpmyadmin` 等列為「目前使用中…遷移期間不得改 origin 或關閉」，但今天 443 與 22 都不通、8 個主機名全無回應。這 8 筆要重新確認是否還需要保留。
2. **`security` 的歸類在原盤點裡自相矛盾**：同時出現在「GitHub Pages 產品頁」與「已確認不再使用」兩列。今日實測是 CNAME → GitHub Pages，屬前者。
3. **`analytics` 等五筆確實已死**：HTTP 有 301 但 TLS 連不上，與原盤點「不再使用」的判定相符。
4. 其餘 records 與 09-20 盤點一致，沒有新增或消失。

## 3. 這次遷移要重建的 records

| 類型 | 筆數 | 說明 |
| --- | --- | --- |
| A（裸網域） | 4 | GitHub Pages |
| CNAME → GitHub Pages | 10 | `www` 與 9 個產品子網域 |
| CNAME（其他） | 2 | `brighterarc`、`media.peertube` |
| CNAME（DKIM） | 2 | Brevo |
| Wildcard CNAME | 3 | `*.intel`、`*.proof`、`*.z` |
| Wildcard A | 2 | `*.ai`、`*.listening` |
| A（Linode 單一主機） | 20 | 兩台主機，含已死的 8 筆 |
| AAAA | 2 | `listening` 與 `*.listening` |
| TXT | 4 | 三筆網域驗證 + DMARC |

合計約 **49 筆**。NS 由 Cloudflare 產生、SOA 不複製，兩者都不手動搬。

## 4. 執行順序（照原文件的 Phase A/B）

1. 在 Cloudflare 建立 `weiqi.kids` zone，匯入上述 records，**全部先設 DNS-only**。
2. 逐筆比對名稱、類型、值、TTL；本文件第 1 節可當比對基準。
3. 比對通過後才到 registrar 改 nameserver。
4. **Linode zone 不刪**，保留至少一個完整回退週期。
5. nameserver 穩定後才逐服務評估要不要開 proxy。管理介面、API、未知 origin 一律維持 DNS-only。
6. `www` 切換到新站是**另一件事**，不跟 nameserver 遷移綁在一起。

## 5. 回退

1. 在 registrar 把 nameserver 改回 `ns1~ns5.linode.com`。
2. Linode zone 全程保留未動，改回去即恢復。
3. Cloudflare zone 不刪，留作紀錄。
4. 回退後逐筆重測本文件第 1 節的所有 hostname。

## 6. 只有帳戶管理者能做的事

- 匯出 Linode 完整 zone（含 TTL 與本文件未涵蓋的 record）。
- 在 registrar 變更 nameserver。
- 決定 `172.237.11.63` 上那 8 筆已無回應的主機名要保留還是退役。
- 確認 `admin.bless`、`bless` 的 TLS 失敗是暫時的還是服務已停。
- 確認 Brevo、Zoho 驗證 records 是否仍需保留（牽涉現有寄件與帳戶）。

## 7. 這一步完成後才能做的事

`@weiqi.kids` 的信箱。zone 進 Cloudflare 之後，Email Routing 在免費方案就能開，建 `info@weiqi.kids` 轉寄到現有信箱。要從 `@weiqi.kids` 寄出則需要 Workers 付費方案，另案評估。
