# 現有服務健康檢查表

版本：0.1（2026-09-20）  
狀態：待執行；來源為使用者提供的 DNS zone。此表不會自行改 DNS 或重啟服務。

## 1. 檢查規則

每個 hostname 都要記錄：DNS 解析、HTTP／HTTPS 狀態、TLS 憑證、redirect、實際服務、owner、最後檢查時間與是否可回退。管理介面與 API 先使用 DNS-only，不開啟 Cloudflare proxy。

## 2. GitHub Pages 服務

| hostname／模式 | 目前目的地 | 檢查項目 | 狀態 |
| --- | --- | --- | --- |
| `weiqi.kids`、`www.weiqi.kids` | GitHub Pages | HTTPS、canonical、redirect、CNAME、首頁與 sitemap | 待執行 |
| `*.intel.weiqi.kids` | GitHub Pages | 實際使用 hostname、custom domain、憑證與內容 | 待執行 |
| `*.proof.weiqi.kids` | GitHub Pages | 實際使用 hostname、custom domain、憑證與內容 | 待執行 |
| `*.z.weiqi.kids` | GitHub Pages | 實際使用 hostname、custom domain、憑證與內容 | 待執行 |
| `ecommerce`、`epialert`、`learn`、`risk`、`security`、`skills`、`supplement`、`trade`、`vuko` | GitHub Pages | HTTPS、內容、外部連結與是否仍在使用 | 待執行 |
| `brighterarc` | `lightchang.github.io` | owner、HTTPS、內容與 custom domain | 待執行 |

## 3. Linode 服務

以下 hostname 已確認目前使用中。遷移前需要由服務管理者填入實際服務名稱、port、TLS、備份與回退方式：

| hostname | 目前 DNS 目的地 | 服務／owner | 檢查項目 | 狀態 |
| --- | --- | --- | --- | --- |
| `admin.bless`、`bless`、`bible`、`akora`、`guessing` | `172.235.205.148` | 待填 | 應用頁面、管理介面、TLS、origin routing | 待執行 |
| `listening`、`*.listening` | A `172.235.205.148`／AAAA `2400:8905::2000:6fff:fe51:2e08` | 待填 | IPv4、IPv6、TLS、fallback | 待執行 |
| `chrant`、`jacob`、`mcp`、`miner`、`nginx`、`phpmyadmin`、`strapi`、`summon`、`writer` | `172.237.11.63` | 待填 | 管理介面、API、TLS、存取限制與備份 | 待執行 |
| `media.peertube` | Linode Object Storage CNAME | PeerTube 媒體 | 影片、附件、TLS、跨來源設定 | 待執行 |
| `*.ai` | `172.235.205.148` | 待填 | 實際 wildcard 使用量與各 hostname routing | 待執行 |

## 4.1 已確認退役服務

以下服務已確認不再使用，不進入新 Cloudflare 運行配置：

| hostname | 目前處理 |
| --- | --- |
| `analytics.weiqi.kids` | 不再保留為新站分析服務；確認外部依賴後移除 DNS 與 script 引用 |
| `security.weiqi.kids` | 不再保留；確認外部連結後移除 DNS |
| `status.weiqi.kids` | 不再保留；確認監控替代方案後移除 DNS |
| `mastodon.weiqi.kids` | 不再保留；確認資料保存需求後移除 DNS |
| `peertube.weiqi.kids` | 不再保留；確認媒體保存需求後移除 DNS |

## 4. 完成條件

- 每個使用中 hostname 有 owner、服務名稱與回退方式。
- DNS-only 解析結果與現行服務一致。
- IPv4／IPv6、HTTPS、TLS、redirect 與附件下載均通過。
- GitHub Pages、Linode 與 Object Storage 的 origin 沒有因 Cloudflare zone 建立而改變。
- 服務管理者確認後，才能把 record 匯入 Cloudflare production zone。

## 5. 2026-09-20 只讀 HTTPS 探測結果

本輪使用公開 HTTPS 讀取工具探測；「工具不可存取」不等於服務已關閉，仍需從 Linode／服務主機端確認。

| URL | 探測結果 | 判讀 |
| --- | --- | --- |
| `https://www.weiqi.kids/` | 可讀取，回傳 Docusaurus 首頁 | 現站可達；仍有舊協會／開源主張，內容遷移時需改寫 |
| `https://ecommerce.weiqi.kids/` | 可讀取 | GitHub Pages／產品服務可達 |
| `https://learn.weiqi.kids/` | 可讀取 | GitHub Pages／產品服務可達 |
| `https://trade.weiqi.kids/` | 可讀取 | GitHub Pages／產品服務可達 |
| `https://weiqi.kids/` | 探測工具不可存取 | 需從 DNS、registrar 與 GitHub Pages 驗證裸網域 redirect |
| `https://analytics.weiqi.kids/` | 探測工具不可存取 | 已確認退役，不作為新站分析服務 |
| `https://security.weiqi.kids/` | 探測工具不可存取 | 已確認退役，不納入新 DNS 配置 |
| `https://status.weiqi.kids/` | 探測工具不可存取 | 已確認退役，不納入新 DNS 配置 |
| `https://mastodon.weiqi.kids/` | 探測工具不可存取 | 已確認退役，不納入新 DNS 配置 |
| `https://peertube.weiqi.kids/` | 探測工具不可存取 | 已確認退役，不納入新 DNS 配置 |

探測來源：公開網站讀取檢查，2026-09-20。完整結果與限制已回報，沒有執行任何 DNS 或主機變更。
