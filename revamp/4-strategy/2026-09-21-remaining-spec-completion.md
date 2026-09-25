# 實作前剩餘規格補完

> 狀態：規格完成。本文補齊帳號、會員、內容管理、附件、URL、保存、權利與寄信邊界；實際建立資源、匯入資料與測試仍屬開發執行階段。

## 1. 帳號綁定與恢復

- 每個人只有一個內部 `account_id`；Email 與 LINE 是登入方式，不是兩個會員。
- Email Magic Link 登入後，使用者必須在帳號設定中明確確認，才能綁定 LINE。
- 綁定與解除綁定都要求目前登入 session 再驗證一次。
- 系統不依顯示名稱、機構名稱或相同 Email 自動合併帳號。
- 解除最後一個登入方式前，必須先新增另一個可用登入方式。
- 無法登入時由管理員執行人工恢復；管理員不得直接改用另一個帳號覆蓋原帳號。
- 所有綁定、解除、恢復與管理員操作保留稽核紀錄。

## 2. 會員續會與付款確認

會員資料保存：`member_id`、`account_id`、會員狀態、有效起訖日、付款方式、匯款識別、人工確認者、確認時間與備註。

- 首次 6,000 元：建立一年會員效期並開通首課。
- 後續每門課 6,000 元：既有會員也須付款；不重複建立首課入會紀錄。
- 管理員收到 LINE@ 匯款紀錄後人工標記付款確認。
- 會員到期：不能報名新課，但保留既有課程、論壇與歷史紀錄的原權限。
- 續會：延長會員有效期，不建立新帳號、不刪除舊課程紀錄。

## 3. 論壇內容管理與留存

內容狀態：`published`、`withdrawn`、`hidden`、`restored`、`deleted`。

隱藏原因固定使用：垃圾內容、個資、侵權、錯誤資訊、違反課程規則、作者要求與其他管理原因。

- 作者可編輯與撤回自己的內容。
- 講師可隱藏自己負責課程的學員內容。
- 管理員可隱藏、通知作者、處理申訴與恢復內容。
- 隱藏不等於刪除；原文與附件保留在受限狀態，供申訴、TTQS 與稽核使用。
- 作者撤回後不再出現在一般論壇，但課程管理與佐證索引仍保留必要紀錄。
- 真正刪除需管理員操作並留下原因；涉及 TTQS 佐證、付款或申訴的資料，在保存期限結束前不可刪除。

## 4. 附件與課程佐證

- 圖片、教材與作品放在受權限控制的 R2 bucket，不使用可猜測的公開 URL。
- 每個附件保存 `owner_id`、`course_id`、`post_id`、檔案類型、大小、雜湊、建立時間、公開狀態與刪除狀態。
- 學員可上傳課堂側拍；講師或管理員可將附件關聯到場次與 TTQS 指標。
- 公開成果只使用作者同意公開且已由管理員設定公開的附件。
- 課程結束後論壇仍可互動；附件不因課程結束自動刪除。
- 刪除附件前先從公開頁撤下，保留管理區稽核與必要的 TTQS 證據索引。

## 5. URL、canonical 與保存

每個舊 URL 最終只能有一種處置：

| 處置 | 使用時機 |
|---|---|
| 301 | 有明確的新頁面與等價內容 |
| canonical／改寫 | 舊內容合併到新主題，但仍需保留可讀入口 |
| noindex 保留 | 需要保存但不應進公開搜尋的歷史或管理內容 |
| 410／移除 | 重複、空白、失效且沒有保存價值；保留處置紀錄 |

固定對應：

- `/about/activities/game-reviews` → `/knowledge/go/reviews`
- `/about/activities/partners/**` → `/association/partners/**`
- `/learn/**`、`/alphago/**`、`/tech/**` → `/knowledge/**`
- `/apps` → `/camp/topics/ai-applications`
- `/intel` → `/camp/topics/supply-chain-intelligence`
- `/research` → 有課程關聯時 `/camp/courses/[course-slug]/history`，無關聯時受限保存
- `/about/internal/**` → 管理區，不進公開 sitemap

## 6. 舊資料保存與匯入格式

每筆來源建立一筆 manifest：

```text
source_id
source_path
old_url
source_type
title
author_or_owner
source_hash
captured_at
new_content_type
new_route
migration_action
visibility
rights_status
redirect_action
import_status
```

- 原始 Markdown、MDX、JSON、JS、圖片與附件保存為唯讀快照。
- 新站內容以結構化資料匯入；原始來源 ID 永遠保留，方便回溯。
- 匯入採 staging → 驗證 → production，不直接覆蓋正式資料。
- 每批匯入產生成功、跳過、錯誤與待人工處理報告。

## 7. 權利與衝突文案

人物、合作夥伴、棋聚照片、作品、研究、程式碼與外部連結都要有 rights register 欄位：權利人、授權範圍、公開狀態、來源、確認日期與撤回方式。

舊站以下說法不得原樣進新站：

- 免費參加或無會員費。
- 尚未啟動實體棋會。
- 任何未經目前課程規格確認的招生承諾。
- 將舊研究或工具描述成目前課程成果。

## 8. 實作前規格結論

以上規則已足以開始技術設計、資料 schema、表單與匯入器實作。尚未執行的事項是建立 Cloudflare 資源、寫程式、實際匯入、DNS 切換與測試，不再視為規格缺口。
