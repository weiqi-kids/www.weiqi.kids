# 改版文件入口

本目錄保存網站改版的研究、決策與規格。執行程式或內容轉移前，先依下列順序閱讀；同一階段若有舊版與日期版文件，以「有效文件」欄為準。

## 文件來源與有效版本

| 層級 | 目的 | 有效文件 |
| --- | --- | --- |
| 專案共識 | 名詞、產品規則、長期決策 | [`/CONTEXT.md`](../CONTEXT.md) |
| 架構決策 | 記錄決策背景、理由與取捨 | [`/docs/adr/`](../docs/adr/) |
| 現況盤點 | 舊站技術、頁面與資料現況 | [`1-discovery/2026-09-15-refactor-baseline.md`](1-discovery/2026-09-15-refactor-baseline.md) |
| 受眾分析 | 共學營參與者與內容差距 | [`3-analysis/2026-09-15-learning-camp-audience.md`](3-analysis/2026-09-15-learning-camp-audience.md) |
| 重構策略 | 技術邊界、產品規則、分期 | [`4-strategy/2026-09-19-refactor-specification.md`](4-strategy/2026-09-19-refactor-specification.md) |
| 平台評估 | Cloudflare 元件與部署邊界 | [`4-strategy/2026-09-19-cloudflare-platform-assessment.md`](4-strategy/2026-09-19-cloudflare-platform-assessment.md) |
| 待決事項 | 尚未核准的會員、權限、登入、保存與 SEO 規則 | [`4-strategy/2026-09-20-open-decisions.md`](4-strategy/2026-09-20-open-decisions.md) |
| 資料與權限 | 帳號、會員、課程、實作營與內容的資料模型及權限矩陣 | [`4-strategy/2026-09-20-data-model-and-permissions.md`](4-strategy/2026-09-20-data-model-and-permissions.md) |
| Cloudflare 遷移 | 環境、遷移順序、回退條件與資料邊界 | [`4-strategy/2026-09-20-cloudflare-migration-plan.md`](4-strategy/2026-09-20-cloudflare-migration-plan.md) |
| DNS 不中斷 | 現有服務盤點、nameserver 切換、保留 GitHub Pages 與回退 | [`4-strategy/2026-09-20-dns-migration-continuity.md`](4-strategy/2026-09-20-dns-migration-continuity.md) |
| DNS zone 盤點 | 目前 Linode zone 的 records、服務分類與遷移風險 | [`4-strategy/2026-09-20-dns-zone-inventory.md`](4-strategy/2026-09-20-dns-zone-inventory.md) |
| 第一門課驗收 | 報名、登入、教材、實作營、版主與回退驗收條件 | [`4-strategy/2026-09-20-first-course-acceptance.md`](4-strategy/2026-09-20-first-course-acceptance.md) |
| Magic Link | Email 登入流程、token、session、綁定與驗收 | [`4-strategy/2026-09-20-email-magic-link-spec.md`](4-strategy/2026-09-20-email-magic-link-spec.md) |
| 現有服務驗證 | GitHub Pages、Linode、Object Storage 與子網域健康檢查 | [`4-strategy/2026-09-20-service-health-check.md`](4-strategy/2026-09-20-service-health-check.md) |
| URL 處置 | 舊 URL 群組的保留、改寫、歷史與待補欄位 | [`5-content-spec/2026-09-20-url-disposition-register.md`](5-content-spec/2026-09-20-url-disposition-register.md) |
| 內容轉寫 | 共學營首頁、課程、成員、知識與教材的轉寫規則 | [`5-content-spec/2026-09-20-content-rewrite-plan.md`](5-content-spec/2026-09-20-content-rewrite-plan.md) |
| 營運規則 | 報名、續會、討論區、公開展示與私密資料流程 | [`4-strategy/2026-09-20-operations-rules.md`](4-strategy/2026-09-20-operations-rules.md) |
| TTQS 官方研究 | 官方來源、PDDRO 指標與政府格式限制 | [`4-strategy/2026-09-20-ttqs-official-research.md`](4-strategy/2026-09-20-ttqs-official-research.md) |
| TTQS 標準課程 | 1 堂教學、3 堂實作、開課欄位與資料模型 | [`4-strategy/2026-09-20-ttqs-standard-course-template.md`](4-strategy/2026-09-20-ttqs-standard-course-template.md) |
| TTQS 匯出 | 官方表單對齊的 ODT／PDF、CSV／JSON 匯出包 | [`4-strategy/2026-09-20-ttqs-export-spec.md`](4-strategy/2026-09-20-ttqs-export-spec.md) |
| TTQS 欄位對照 | 官方表單欄位、課程大綱欄位與逐項匯出驗收 | [`4-strategy/2026-09-20-ttqs-field-mapping.md`](4-strategy/2026-09-20-ttqs-field-mapping.md) |
| AI 共學營課程家族 | 將 AI 工具、供應鏈情報與研究成品重新定義為課程家族與實作能力 | [`4-strategy/2026-09-21-course-family-proposal.md`](4-strategy/2026-09-21-course-family-proposal.md) |
| 內容遷移流程 | 來源分層、文章轉寫、共學營成品關聯、會員／活動匯入、URL 與驗收流程 | [`5-content-spec/2026-09-21-content-migration-workflow.md`](5-content-spec/2026-09-21-content-migration-workflow.md) |
| 實作前剩餘規格補完 | 帳號恢復、會員續會、論壇留存、附件、URL、資料保存、權利與衝突文案 | [`4-strategy/2026-09-21-remaining-spec-completion.md`](4-strategy/2026-09-21-remaining-spec-completion.md) |
| 第一門課 TTQS 表單 | 講師申請、開課、場次、論壇佐證、檢討、成果與匯出前檢查 | [`4-strategy/2026-09-20-first-course-ttqs-form-draft.md`](4-strategy/2026-09-20-first-course-ttqs-form-draft.md) |
| 第一門課試點 | 第一位講師使用正式流程驗證整條鏈路 | [`4-strategy/2026-09-20-first-course-acceptance.md`](4-strategy/2026-09-20-first-course-acceptance.md) |
| 舊站轉移 | 資料分類、去處與驗收規則 | [`5-content-spec/2026-09-19-legacy-content-migration-spec.md`](5-content-spec/2026-09-19-legacy-content-migration-spec.md) |
| 舊站轉移清單 | 來源凍結、逐筆欄位、分類、權利、URL、轉寫與驗收清單 | [`5-content-spec/2026-09-20-legacy-migration-checklist.md`](5-content-spec/2026-09-20-legacy-migration-checklist.md) |
| 資訊架構與網站地圖 | 協會、成員、共學營、論壇與管理區的內容邊界 | [`5-content-spec/2026-09-20-information-architecture-and-sitemap.md`](5-content-spec/2026-09-20-information-architecture-and-sitemap.md) |
| 棋聚服務線 | 好棋寶寶棋聚與 AI 共學營的分開建模與資料邊界 | [`../docs/adr/0011-association-gatherings-separate-from-learning-camp.md`](../docs/adr/0011-association-gatherings-separate-from-learning-camp.md) |
| SEO／AEO／GEO 架構檢查 | 三條服務線的索引、結構化資料、可回答性與 AI grounding 要求 | [`5-content-spec/2026-09-20-seo-aeo-geo-architecture-audit.md`](5-content-spec/2026-09-20-seo-aeo-geo-architecture-audit.md) |
| 來源 inventory | 舊站來源群組與盤點規則 | [`5-content-spec/2026-09-19-legacy-content-inventory.md`](5-content-spec/2026-09-19-legacy-content-inventory.md) |
| 逐頁 inventory | 每個舊站來源檔案的初步處置候選 | [`5-content-spec/2026-09-20-legacy-page-inventory.md`](5-content-spec/2026-09-20-legacy-page-inventory.md) |
| 完整轉移清單（網站地圖版） | 153 筆舊站來源及 AI 工具、產業供應鏈、學術研究子項目的初版分類（已被 v2 取代） | [`5-content-spec/2026-09-20-legacy-migration-map.md`](5-content-spec/2026-09-20-legacy-migration-map.md) |
| 有效網站樹狀地圖 v2 | 依協會、共學營課程、公開知識、課程案例與管理區重新規劃所有舊站內容 | [`5-content-spec/2026-09-20-site-tree-v2.md`](5-content-spec/2026-09-20-site-tree-v2.md) |
| 文章遷移結果 | 139 篇舊站文章逐篇對應新網站地圖分類、新路徑與遷移動作 | [`5-content-spec/2026-09-20-article-migration-result.md`](5-content-spec/2026-09-20-article-migration-result.md) |
| 內容對照 | 舊站來源到新站角色的對照 | [`5-content-spec/2026-09-20-content-disposition-map.md`](5-content-spec/2026-09-20-content-disposition-map.md) |
| 知識資料轉移草案 | `learn`、`alphago`、`tech` 64 份內容的逐份分類建議 | [`5-content-spec/2026-09-20-knowledge-migration-draft.md`](5-content-spec/2026-09-20-knowledge-migration-draft.md) |
| 主張衝突 | 舊站免費、會員、合作與課程文案的衝突盤點 | [`5-content-spec/2026-09-20-legacy-claims-conflict-audit.md`](5-content-spec/2026-09-20-legacy-claims-conflict-audit.md) |
| URL 計畫 | 舊 URL、新 URL、301、canonical、noindex 與歷史保存原則 | [`5-content-spec/2026-09-20-url-migration-plan.md`](5-content-spec/2026-09-20-url-migration-plan.md) |
| 權利登錄 | 人物、合作、作品、影像與公開狀態確認欄位 | [`5-content-spec/2026-09-20-content-rights-register.md`](5-content-spec/2026-09-20-content-rights-register.md) |

## 已封存的舊版階段文件

以下文件保留作為歷史記錄，內容不再作為本次共學營重構的規格來源：

- [`0-positioning/positioning.md`](0-positioning/positioning.md)
- [`1-discovery/discovery.md`](1-discovery/discovery.md)
- [`2-competitive/competitive.md`](2-competitive/competitive.md)
- [`3-analysis/analysis.md`](3-analysis/analysis.md)
- [`4-strategy/strategy.md`](4-strategy/strategy.md)
- [`5-content-spec/content-spec.md`](5-content-spec/content-spec.md)

這些文件描述的是先前的「商界夥伴／開源研究」方向。若與有效文件衝突，以 `CONTEXT.md`、ADR 與本入口列出的日期版文件為準。

## 工作閘門

1. 文件入口與版本狀態確認完成。
2. 舊站逐頁 inventory 完成，所有項目都有處置與理由。
3. 新舊 URL、歷史資料保存位置、公開權利與 SEO 處理完成確認。
4. 一門課的資料模型、權限矩陣與 Cloudflare 邊界完成確認。
5. 才能進入 Astro、設計系統、Cloudflare 後端或資料轉移實作。

本入口本身不授權刪除舊資料、公開人物資料或建立 Cloudflare 資源。

目前進度見 [`2026-09-20-progress-status.md`](2026-09-20-progress-status.md)。
