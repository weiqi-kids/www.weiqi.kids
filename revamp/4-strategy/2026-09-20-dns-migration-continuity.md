# DNS 遷移與現有服務不中斷計畫

版本：0.1（2026-09-20）  
狀態：遷移計畫；尚未修改 nameserver、DNS record 或代理設定。

## 1. 已知現況

repository 能確認的服務與網域引用包括：

| 網域／類型 | 目前可確認的資訊 | 遷移要求 |
| --- | --- | --- |
| `www.weiqi.kids` | `static/CNAME` 指向 GitHub Pages 部署；現站為 Docusaurus | DNS 遷移後仍保留 GitHub Pages record，直到新站通過切換驗收 |
| `analytics.weiqi.kids` | Docusaurus 設定引用 Umami script | 必須先確認實際 A／CNAME、TLS 與服務擁有者，再搬 DNS |
| `*.weiqi.kids` 產品子網域 | `static/llms*.txt` 列出電商、健康、學習、資安、情報等子網域 | 逐筆從現行 DNS zone 與外部服務確認，不能只依 repository 推定 |
| `*.intel.weiqi.kids` | 列出多個產業情報子網域 | 逐筆保留 record、origin、憑證與部署責任 |
| `*.proof.weiqi.kids` | 列出研究與證明子網域 | 逐筆保留 record、origin、憑證與部署責任 |
| 郵件相關 records | 已確認目前沒有 `@weiqi.kids` 入站信箱；現有驗證／DKIM records 仍需保留 | 不新增 MX；SPF 只在確認未來寄件來源後設定 |

使用者已提供目前 Linode zone 匯出；逐筆對照見 [`2026-09-20-dns-zone-inventory.md`](2026-09-20-dns-zone-inventory.md)。這份匯出是目前的遷移基準；兩台 Linode 主機上的服務目前都在使用，遷移期間必須維持可用。

## 2. 遷移前凍結清單

在 nameserver 切換前，建立一份不可省略的 DNS snapshot：

- A、AAAA、CNAME、MX、TXT、CAA、SRV、HTTPS 等所有非 NS／SOA records。
- 每筆 record 的名稱、類型、值、TTL、是否代理、用途與服務 owner。
- `www`、裸網域、`analytics`、產品子網域、研究子網域、郵件與驗證 records。
- GitHub Pages、Umami、外部產品、研究服務與郵件系統的 origin／帳戶擁有者。
- 現有 SSL／TLS 憑證、HTTP redirect、robots、sitemap 與 webhook callback 使用的 hostname。

快照要保存為受限的遷移附件；API token、密碼、私密金鑰不能寫入 repository。

## 3. 無中斷遷移順序

### Phase A：先建立 Cloudflare zone

1. 在 Cloudflare 建立 `weiqi.kids` zone。
2. 匯入現行 DNS records，逐筆與 snapshot 比對。
3. 不先刪除現有 DNS provider 的 records。
4. 初期所有外部服務 record 先設為 DNS-only，避免代理層改變 origin 行為。
5. 驗證 GitHub Pages、Umami、產品、研究與郵件 records 的解析結果。

### Phase B：切換 nameserver

1. 在 registrar 將 nameserver 改為 Cloudflare 指定值。
2. 於傳播期間持續檢查裸網域、`www`、`analytics`、所有已確認子網域與郵件收發。
3. 保留舊 DNS provider zone，不做刪除或重建。
4. nameserver 穩定後，再依服務逐筆決定是否開啟 Cloudflare proxy。

### Phase C：部署新站但維持現站回退

1. Astro 公開站先使用 staging／preview hostname 驗證。
2. `www.weiqi.kids` 先繼續指向 GitHub Pages，直到公開頁面與 SEO 驗收完成。
3. 動態 Worker 使用獨立 API hostname 或 preview hostname，避免先改主站。
4. 新站通過驗收後才切換 `www`；保留 GitHub Pages 可立即回退的 record 與部署流程。

## 4. 切換驗收

- `https://weiqi.kids` 與 `https://www.weiqi.kids` 的 canonical、redirect 與憑證正確。
- 既有 `analytics.weiqi.kids` 與所有已確認產品／研究子網域仍可用。
- 已確認沒有 MX 的現況維持不變；既有 DKIM／DMARC 驗證結果正常；未啟用的寄件服務不新增 SPF。
- GitHub Pages workflow 仍能部署現站回退版本。
- 現有 sitemap、robots、llms 檔案與重要舊 URL 沒有因 DNS 切換失效。
- API、圖片、管理後台不會因 DNS-only／proxied 狀態改變而越權或失效。

## 5. 回退

回退順序：

1. 將 `www` 回指 GitHub Pages。
2. 保留 Cloudflare zone 與 records，不刪除證據與設定。
3. 若 nameserver 本身造成問題，依 registrar 的變更紀錄恢復原 nameserver。
4. 回退後檢查現站、郵件、分析與所有外部子網域。

任何回退都不能刪除原 DNS zone、GitHub Pages 或外部服務；資料與服務問題分開處理。

## 6. 必須由帳戶管理者完成的事項

- 從現任 DNS provider 匯出完整 zone 或提供逐筆 record 清單。
- 確認所有子網域的服務 owner 與是否仍需保留。
- 確認 registrar 的 nameserver 變更權限。
- 在 Cloudflare Email Service 啟用前確認寄件網域與 SPF／DKIM／DMARC 設定。
- 提供 staging 測試所需的非正式帳號與服務聯絡人；不把 token 或密碼放入 repository。
