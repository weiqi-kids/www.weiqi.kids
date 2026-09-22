# ADR 0009：舊站網址全部轉址，不留 404（待決 D-07）

- 狀態：已接受
- 日期：2026-09-22

## 背景

舊站有中文網址與 10 種翻譯語系網址（`/en/…`、`/ja/…` 等），也有外部連結與搜尋流量。新站路徑全部改變，而且不建翻譯頁。

## 決策

- 每個舊站中文網址以 301 轉到新站對應頁；對應關係以 `revamp/5-content-spec/2026-09-20-article-migration-result.md` 為準。
- 翻譯語系網址 `/{locale}/…` 以 301 轉到該頁中文版的新網址；無對應頁時轉到新站首頁。
- `/about/internal/**` 轉到 `/association`，新站不公開內部文件。
- `/apps` → `/camp/topics/ai-applications`；`/intel` → `/camp/topics/supply-chain-intelligence`；`/research` → `/camp/topics/research-publication`。
- 轉址以 Cloudflare 靜態資產的 `_redirects` 實作，由 `site/scripts/build-redirects.mjs` 從遷移清單產生。
- 新站 canonical 一律使用新網址；舊網址不進 sitemap。

## 後果

- 舊網址的搜尋權重轉到新網址，外部連結不會斷。
- `*.intel.weiqi.kids`、`*.proof.weiqi.kids` 等子網域不在本轉址範圍，遷移期間原樣保留。
