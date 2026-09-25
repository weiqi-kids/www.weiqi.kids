# 共學營資料模型與權限矩陣（規劃版）

版本：0.1（2026-09-20）  
狀態：Phase 0 規格草案；尚未建立 D1、Workers 或登入流程。

## 1. 模型邊界

網站帳號、協會會員、課程資格與實作營成員是不同概念，資料模型必須分開保存：

- **帳號**：辨識登入者，可用 LINE 或 Email 登入。
- **會員**：協會的一年期身分；會員到期影響新課報名，不刪除既有學習紀錄。
- **報名**：某位帳號對某門課提出的申請，包含報名編號與付款狀態。
- **課程資格**：協會人工確認收款後開通的某門課權限。
- **講師申請**：已上過一門 6,000 元標準課程的學員，才能在網站提出成為講師或開課的申請；申請需附上資格與課程資料，經管理員審核後才取得講師身分或建立正式課程。
- **實作營成員**：取得課程資格後，自行加入特定實作營的人。
- **內容**：實作營內的文章、圖片與作品；作者可編輯或撤回。
- **公開展示**：作品經講師認可且作者同意後，才可放到推薦頁。

## 2. 核心資料表草案

| 資料 | 重要欄位 | 說明 |
| --- | --- | --- |
| `accounts` | `id`、`status`、`created_at` | 網站帳號，不等同會員 |
| `login_identities` | `account_id`、`provider`、`provider_subject`、`email` | LINE／Email 登入綁定；同一帳號可有兩種方式 |
| `memberships` | `account_id`、`started_at`、`expires_at`、`status` | 一年一續會；首課開通時建立或延長 |
| `courses` | `id`、`title`、`topic`、`price`、`status` | 每月主題課程；不直接放單一講師欄位 |
| `course_instructors` | `course_id`、`account_id`、`role`、`responsibilities`、`display_order`、`status` | 課程與講師的多對多關聯；可記錄主講、共同講師、實作回覆或協作者 |
| `instructor_applications` | `id`、`account_id`、`profile_data`、`qualification_evidence`、`course_proposal`、`status`、`reviewer_id`、`reviewed_at` | 成為講師／提出課程的網站申請、附件與管理員審核軌跡 |
| `registrations` | `id`、`account_id`、`course_id`、`registration_no`、`payment_status` | 報名與人工核款對應 |
| `payment_records` | `id`、`registration_id`、`amount`、`submitted_at`、`confirmed_at`、`status`、`reviewer_id` | 匯款紀錄、人工確認、補件與狀態事件；原始私密附件另存 |
| `receipts` | `id`、`registration_id`、`receipt_no`、`issued_at`、`status`、`file_key` | 課程付款收據的產生、查詢與保存 |
| `operations_sops` | `id`、`title`、`version`、`content`、`status`、`approved_by` | 管理員使用的共學營營運 SOP 與版本核准紀錄 |
| `course_access` | `account_id`、`course_id`、`opened_at`、`source_registration_id` | 人工確認收款後的課程資格 |
| `practice_camps` | `id`、`course_id`、`starts_at`、`ends_at`、`status` | 每門課對應的實作營討論空間 |
| `camp_memberships` | `camp_id`、`account_id`、`joined_at` | 學員自行加入；需要課程資格 |
| `posts` | `id`、`camp_id`、`author_id`、`status`、`body` | 實作營文章與作品描述 |
| `attachments` | `id`、`post_id`、`storage_key`、`mime_type` | R2 圖片與附件索引 |
| `moderation_actions` | `post_id`、`actor_id`、`action`、`reason`、`notified_at` | 隱藏、恢復、通知與申訴軌跡 |
| `appeals` | `post_id`、`author_id`、`reason`、`status` | 作者申訴管理員隱藏的紀錄 |
| `recommendations` | `post_id`、`instructor_id`、`author_consent_at`、`status` | 與實作營自由發文分開的推薦展示 |

資料表名稱是規劃用語，進入 schema 設計時仍需依實際資料庫命名規則確認。

## 3. 權限矩陣

| 行動 | 未登入 | 已登入但無資格 | 已開通課程 | 實作營成員 | 協會管理員 | 講師 |
| --- | --- | --- | --- | --- | --- | --- |
| 看公開共學營介紹 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 看完整技能單張 | — | — | ✓ | ✓ | ✓ | ✓ |
| 申請課程 | — | ✓ | ✓ | ✓ | 代辦 | ✓ |
| 申請成為講師／提出開課 | — | —（須先上過 6,000 元課程） | ✓（須有既有課程紀錄） | ✓ | 可審核 | 可補充申請資料 |
| 審核講師申請 | — | — | — | — | ✓ | — |
| 看自己報名與付款狀態 | — | ✓ | ✓ | ✓ | 可管理 | 可查看必要資訊 |
| 加入實作營 | — | — | ✓ | 已加入 | 可管理 | 可管理 |
| 看實作營文章與圖片 | — | — | — | ✓ | ✓ | ✓ |
| 發文與上傳圖片 | — | — | — | ✓ | 代為管理 | ✓ |
| 編輯自己的內容 | — | — | — | ✓ | 可處理 | ✓ |
| 撤回自己的內容 | — | — | — | ✓ | 可處理 | ✓ |
| 隱藏不適當內容 | — | — | — | — | ✓ | ✓（自己負責的課程論壇） |
| 提出申訴 | — | — | — | ✓ | — | — |
| 認可作品推薦 | — | — | — | — | — | ✓ |
| 同意作品公開推薦 | — | — | — | ✓ | 代記錄 | — |
| 看公開推薦頁 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 報名新課 | — | 需有效會員 | 需有效會員 | 需有效會員 | 代辦 | 需有效會員 |
| 查看既有學習紀錄 | — | 依原課程／實作營成員資格 | 依資格 | ✓ | ✓ | ✓ |

「—」代表該角色沒有此權限；管理員的代辦與處理仍須留下操作紀錄。會員到期不會自動清除既有 `course_access`、`camp_memberships` 或內容資料。

## 4. 狀態轉移

### 會員

`pending_payment` → `active`（首課收款確認） → `renewal_due` → `active`（續會確認）或 `expired`。

`expired` 只阻止新課報名；既有課程、實作營與歷史內容依原資格保留。

### 報名與課程

`registration_submitted` → `payment_pending` → `payment_confirmed` → `course_opened`。

付款未確認不等於課程開通；課程開通後學員才能自行加入對應實作營。

### 講師申請

`draft` → `submitted` → `under_review` → `approved`、`changes_requested` 或 `rejected`。

只有具備至少一門已上課的 6,000 元標準課程紀錄，且講師申請狀態為 `approved`，才能被加入正式課程的講師關聯並進入 TTQS 開課表單；同一門課可以加入多位已核准講師。管理員的審核意見、附件、操作者與時間都要保留。

### 文章與作品

`published` ↔ `edited`；作者可轉為 `withdrawn`，管理員可轉為 `hidden`。`hidden` 必須產生通知與可申訴紀錄，不等於永久刪除。

推薦展示另有 `candidate` → `instructor_approved` → `author_consented` → `published` 狀態，不改變實作營內原文章的自由發文狀態。

## 5. 仍需在 schema 前確認的技術細節

- Cloudflare Email Service Email Sending 的啟用方案、寄件網域、有效期限與重發規則。
- LINE／Email 綁定衝突時的人工處理流程。
- 會員續會是否沿用原 `account_id`，以及付款對帳的人工欄位。
- 實作營結束後，`camp_memberships` 是否永久保留或有撤銷狀態。
- R2 附件大小、格式、病毒掃描與刪除／保留規則。
