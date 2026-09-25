# `weiqi.kids` DNS zone 盤點與 Cloudflare 匯入對照

版本：0.1（2026-09-20）  
來源：使用者提供的 Linode DNS zone 匯出  
狀態：盤點完成；尚未匯入 Cloudflare 或修改 nameserver。

## 1. 敏感資料處理

原始 zone 含有 Zoho、Brevo、Google 驗證值、Brevo DKIM 目標與 SOA 管理聯絡資訊。本文件不重複保存這些完整值；Cloudflare 匯入時必須從原始 zone 原樣複製，並在匯入後驗證，不要從本文件重新建立。

## 2. 必須原樣保留的 records

| 名稱／模式 | 類型 | 目前目的地 | 服務類型 | Cloudflare 初始設定 | 風險與驗收 |
| --- | --- | --- | --- | --- | --- |
| `@` | A × 4 | GitHub Pages `185.199.108–111.153` | 裸網域舊站 | DNS-only | 驗證裸網域 HTTPS、redirect、GitHub Pages custom domain |
| `www` | CNAME | `weiqi-kids.github.io.` | GitHub Pages 舊站 | DNS-only | 驗證現站可用後才能切換到新站 |
| `*.intel` | CNAME | `weiqi-kids.github.io.` | GitHub Pages 子網域 | DNS-only | 驗證每個實際 hostname 的 GitHub Pages 專案與憑證 |
| `*.proof` | CNAME | `weiqi-kids.github.io.` | GitHub Pages 研究子網域 | DNS-only | 驗證每個實際 hostname 與憑證 |
| `*.z` | CNAME | `weiqi-kids.github.io.` | GitHub Pages 子網域 | DNS-only | 盤點實際使用的 `z` 子網域後驗證 |
| `ecommerce`、`epialert`、`learn`、`risk`、`security`、`skills`、`supplement`、`trade`、`vuko` | CNAME | `weiqi-kids.github.io.` | GitHub Pages 外部／產品頁 | DNS-only | 逐一 HTTP、HTTPS、憑證與內容驗收 |
| `brighterarc` | CNAME | `lightchang.github.io.` | 另一個 GitHub Pages 服務 | DNS-only | 確認 owner 與 custom domain 設定 |
| `media.peertube` | CNAME | `weiquikids-peertube.jp-osa-1.linodeobjects.com.` | Linode Object Storage／PeerTube 媒體 | DNS-only | 驗證影片、附件與 TLS；不要直接代理 |
| `*.ai` | A | `172.235.205.148` | Linode 服務群 | DNS-only | 逐一盤點 origin 與 Host routing |
| `*.listening` | A | `172.235.205.148` | Linode 服務群 | DNS-only | 與 `listening` AAAA 一起驗證 |
| `*.listening` | AAAA | `2400:8905::2000:6fff:fe51:2e08` | Linode IPv6 | DNS-only | 驗證 IPv6 服務與 fallback |
| `listening` | AAAA | `2400:8905::2000:6fff:fe51:2e08` | Linode IPv6 | DNS-only | 與 A record 一起驗證 |
| `admin.bless`、`akora`、`bible`、`bless`、`chrant`、`guessing`、`jacob`、`listening`、`miner`、`mcp`、`nginx`、`phpmyadmin`、`strapi`、`summon`、`writer` | A | `172.235.205.148` 或 `172.237.11.63` | **目前使用中的 Linode 主機服務** | DNS-only | 逐一確認 service、port、TLS 與 owner；遷移期間不得改 origin 或關閉；管理介面不能直接開代理 |
| `analytics`、`security`、`status`、`mastodon`、`media.mastodon`、`peertube` | `172.235.205.148` | 已確認不再使用 | 不納入新 Cloudflare 運行配置；舊 zone 暫留作回退證據 | 待確認無外部依賴後移除 DNS、文件入口與相關部署 |

## 3. 驗證與郵件 records

| 名稱 | 類型 | 用途 | 處理 |
| --- | --- | --- | --- |
| `@` | TXT | Zoho domain verification | 原樣保留；不公開完整值 |
| `@` | TXT | Brevo domain verification | 原樣保留；不公開完整值 |
| `@` | TXT | Google Search Console verification | 原樣保留；不公開完整值 |
| `_dmarc` | TXT | `p=none`、Brevo aggregate report | 原樣保留，遷移後驗證 DMARC 查詢結果 |
| `brevo1._domainkey`、`brevo2._domainkey` | CNAME | Brevo DKIM | 原樣保留，遷移後驗證 DKIM |
| MX | 未出現在提供的 zone；已確認目前沒有 `@weiqi.kids` 入站信箱 | 入站郵件 | 不新增 MX；若未來啟用信箱再另行設計 |
| SPF | 未看到 `v=spf1` TXT；Brevo 不承接 `@weiqi.kids` 入站信件 | 寄件授權 | 不自行補；若未來使用 Brevo 或 Cloudflare Email Service 寄信，再依實際寄件來源新增 |

## 4. 重要發現

1. 現有 zone 沒有 MX record，且已確認目前沒有 `@weiqi.kids` 入站信箱；遷移時不新增 MX。
2. 現有 zone 未看到 SPF TXT；不依據推測新增，未來啟用 Brevo 或 Cloudflare Email Service 寄信時再處理。
3. `www`、裸網域、`*.intel`、`*.proof`、`*.z` 與多個產品 CNAME 都依賴 GitHub Pages，Cloudflare 只能先保存 DNS，不能先改 origin。
4. `mcp`、`strapi`、`phpmyadmin`、`admin.bless` 等 A record 指向目前使用中的 Linode 主機，不能套用與 GitHub Pages 相同的切換方式；`analytics`、`security`、`status`、`mastodon`、`peertube` 已確認不再使用，另走退役流程。
5. `*.listening` 同時有 A 與 AAAA；IPv4／IPv6 都要驗證，不能只測其中一個。

## 5. 匯入與切換驗收順序

1. 以原始 zone 建立離線 checksum 與受限 snapshot。
2. 在 Cloudflare 匯入所有非 NS／SOA records；NS 由 Cloudflare 產生，SOA 不手動複製。
3. 逐筆比對名稱、類型、值、TTL 與代理狀態。
4. 先以 DNS-only 驗證 GitHub Pages、Linode、產品、研究、分析與郵件相關 records。
5. 確認 registrar nameserver 變更後，舊 Linode zone 仍保留至少一個完整回退週期。
6. nameserver 穩定後才考慮單一服務是否開啟 Cloudflare proxy；管理介面、API、郵件與未知 origin 維持 DNS-only。
7. `www` 切換到新站必須獨立於 nameserver 遷移，並保留 GitHub Pages 回退 record。

## 6. 目前不能自行補上的資料

- 每個 Linode IP 上實際運行的 service、port 與 owner。
- 每個 wildcard 下實際啟用的 hostname。
- `172.237.11.63` 上 `mcp`、`strapi`、`phpmyadmin` 等服務是否仍需公開。
