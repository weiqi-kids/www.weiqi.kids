# AI 共學營開發計畫（接手用）

更新：2026-10-11。給接手開發的 Claude Code（Opus 5.5）執行。目標：讓 ADR 0014、0015 的整條線在 staging 真的跑得通，第一批示範成果上線，之後才切正式站。

## 怎麼用這份

開一個新的 Claude Code session（Opus 5.5），貼這句：

> 讀 `revamp/2026-10-11-camp-dev-plan.md`，從「進度紀錄」最後一個完成的任務接著做下一個。每做完一個任務就在進度紀錄寫一行（日期、任務編號、結果或卡點），commit 並 push。卡住就停下來問我。

每個任務都有「驗收」，驗收沒過不算完成。發現 bug 就修（skill 文字在 `site/plugins/`、網頁在 `site/src/pages/`、MCP 在 `site/src/server/mcp.ts`），改完 `cd site && pnpm run deploy` 重新部署 staging，再重跑那個任務。

對照文件：

- 整條線每一步對到 skill 段落、MCP 工具、網頁：`revamp/2026-10-11-camp-operation-flow.md`（簡稱「流程」，T1、L1 等編號都指它）。
- 規則：`CONTEXT.md`、`docs/adr/0014`、`docs/adr/0015`。
- 待辦總表：`revamp/STATUS.md` G 節。

## 現況

程式與兩套 skill 都寫好、部署在 staging（https://weiqi-kids-staging.weiqi-kids-site.workers.dev ，D1 `weiqi-kids-staging-camp`）。測過的：MCP 登入授權與讀取工具、ADR 0014 的開團到結案（網頁端）。**沒測過的：skill 帶著人跑的每一段**（講師上架、學員試做、每週進度、分享、講師回覆、版本送審）。沒有任何公開成果，協會 4 組成果是草稿。

## 已決定，不再問

- 講師與學員各一套 skill，加好棋寶寶 MCP；AI 只存草稿，本人在網站按確定（ADR 0015）。
- 講師上架：skill 帶講師整理成公開 repo，8 項必填，每個版本管理員審過才給學員。
- 講師要是會員；學員不用入會。
- 所有會員一年一續，沒有「不會到期」的特例。
- 匯款帳號等真的要開團再設，現在不處理。
- 第一批示範：協會 4 組成果，每組一個範本 repo（第 5 階段的表）。
- 既有會員補登：先用「等他們登入後，管理員在 `/admin/members/` 補登」，不另做功能。

## 站主要先準備（一次給齊）

1. 一台有瀏覽器的電腦跑 Claude Code（第 0 階段的 MCP OAuth 登入要開瀏覽器）。在伺服器上跑也可以，但 `/mcp` 登入時要把印出的網址拿到有瀏覽器的地方開，回呼用 SSH port forward。
2. 第 1 階段講師 skill 會問你 4 題（用 AI 前後差別、適合誰與基礎、本業與動機、自己跑起來要準備什麼），到時候直接答。
3. 網站上「送審」「通過」「確定送出」這幾個按鈕依設計要本人按，每個階段會列出來；你按，或讓 agent 用 Claude in Chrome 以你的登入狀態按。
4. 學員測試帳號用 `lightman.chang+learner@gmail.com`（Email 登入，不用入會）；登入連結由 agent 從 Gmail 讀，或你貼給它。

## 第 0 階段：接線

### 0.1 Claude Code 連上好棋寶寶 MCP

- 在 `intel-template` 的 clone 裡：`/plugin marketplace add weiqi-kids/www.weiqi.kids`、`/plugin install haoqi-teacher@haoqi`。
- `/mcp` 對 `haoqi` 做 OAuth 登入，用站主帳號。
- 驗收：呼叫 `whoami` 回傳站主名稱、有效會員、管理員。

### 0.2 確認查資料的方法

- `cd site && npx wrangler d1 execute weiqi-kids-staging-camp --remote --json --command "..."`（帳戶 `9d17db3ab1021898e4873621c147d0ad`）。
- 驗收：查得到 `showcases` 4 筆草稿。

## 第 1 階段：講師上架（流程 T1～T6、A1～A2）

### 1.1 用講師 skill 上架「23 條產業供應鏈情報網」

- 在 `intel-template` clone 裡對 Claude Code 說「把這個放上好棋寶寶」，照 skill 走：掃機密、補結構（`skills/<名稱>/SKILL.md`、`sheets/`、README 8 項、LICENSE）、push、`save_showcase_draft`（更新既有草稿 `supply-chain`，不要再建一筆）、開團設定（開團人數設 1，方便後面測）、TTQS、`register_skill_version`。
- 注意 `intel-template` 的 `sync-downstream.yml` 只同步 `.claude/skills`、三個 workflow 與 `site/index.html`，新加的檔案不會推到 23 個下游 repo；不要去動那幾個路徑。
- 驗收：repo 有 8 項資料、至少 3 張技能單張、給學員 AI 的 skill、LICENSE；D1 的 `supply-chain` 草稿 8 項都有值、`repo_url` 指到 repo；`skill_versions` 有一筆 draft。
- 站主要按：`/account/teach/showcases/<id>/` 的「送審」與「送審這個版本」。

### 1.2 管理員審核

- 站主在 `/admin/showcases/<id>/` 通過成果、`/admin/versions/<id>/` 通過版本。agent 先檢查版本審核頁有列出改了哪些檔案、定時更新、連到哪些網站、疑似機密。
- 驗收：`/camp/r/supply-chain/` 公開（不登入也看得到），有安裝說明與那句可複製的話；`get_showcase` 回傳的 commit 是審過那一版。

### 1.3 修 bug、記紀錄

- 1.1、1.2 每個問錯、少做、跑不動的地方，記在流程第八節，修好重部署，重跑。
- 驗收：從「把這個放上好棋寶寶」到公開，不用人工繞路。

## 第 2 階段：學員試做（流程 L1～L4）

### 2.1 第二個帳號裝學員 skill

- 另開一個 Claude Code（或同一台的另一個目錄），`/plugin install haoqi-learner@haoqi`，`/mcp` 用 `lightman.chang+learner@gmail.com` 登入。
- 驗收：`whoami` 回傳學員帳號、非會員。

### 2.2 照成果頁那句話試做

- 貼 `/camp/r/supply-chain/` 的那句話。skill 要先列出會讀寫什麼、連哪裡、要花多少錢，同意後在學員的 GitHub（用站主的 GitHub 另建 `test-intel` 之類的 repo）建一份、填設定、換成另一個產業、開定時更新與 Pages。
- 驗收：學員 repo 的 Daily Data Ingest 跑過一次、Pages 看得到資料。
- 中途故意問一個問題，走 `draft_question` → `/account/drafts/<id>/` 按確定 → 出現在 `/camp/r/supply-chain/forum/`。

### 2.3 修 bug、記紀錄

同 1.3。

## 第 3 階段：開團到結案（流程 L5～L10、T7～T10、S1～S5）

### 3.1 登記到簽到

- 學員 `register_interest`（或網頁）→ 滿 1 人開團 → 學員在 `/camp/g/<id>/` 填匯款 → 站主在 `/admin/payments/` 核款 → `/camp/g/<id>/room/` 簽到。
- 驗收：每一步都有站內通知；真手機（iPhone 先加到主畫面）收到推播。

### 3.2 實作與分享

- 學員 AI：「交這週的進度」→ `draft_practice` → 網站確定送出；再做一次作品分享 `draft_work_share`，勾「講師認可後同意公開到成果頁」。
- 驗收：兩篇都出現在 `/camp/g/<id>/forum/`，講師收到推播。

### 3.3 講師回覆

- 站主的 Claude Code 一打開，skill 應主動 `cohort_progress` → `read_post` → `draft_reply` 給草稿網址。站主按確定、按「認可這份作品」。
- 驗收：回覆出現在論壇；學員作品出現在 `/camp/r/supply-chain/`。另外測一次不用 AI 的常用回覆按鈕。

### 3.4 結案

- 學員填問卷；站主匯出 TTQS、在 `/admin/payouts/` 付講師；確認成果重新開放登記。
- 驗收：匯出檔包含開課表單、簽到、問卷、論壇佐證；成果頁又能登記想學。

## 第 4 階段：版本更新（流程 T11、A2）

- 在 `intel-template` 改一個檔案，對講師 skill 說「更新好棋寶寶上的版本」→ 站主送審 → 站主通過。
- 驗收：版本審核頁顯示「跟上一版的差異」只有那個檔案；`get_showcase` 回傳新 commit。

## 第 5 階段：另外 3 組成果上架

每組照第 1 階段 1.1～1.2 走一次（開團人數改回 5）。範本 repo：

| 成果草稿 slug | 範本 repo | 說明 |
|---|---|---|
| `monitoring` | `weiqi-kids/agent.cyber-security` | 6 個監測站挑這個當範本，其他 5 個列在成果展示 |
| `ai-tools` | `weiqi-kids/agent.supplement-product` | 4 個工具挑這個當範本 |
| `research` | `weiqi-kids/proof` | 9 篇研究是成品，`proof` 是做法 |

驗收：4 組成果都公開，`/camp/` 成果牆看得到 4 項。

## 第 6 階段：插畫

- 共學營頁面（`/camp/`、`/camp/ai/`、`/camp/how-it-works/`、`/camp/teach/`、成果頁）依新流程重畫插畫，產圖方法照 `revamp/2026-09-24-illustration-redraw-spec.md`，逐張人工開圖審。
- 驗收：圖上沒有舊規則（免費、入會才能上課、固定課表之類）。

## 第 7 階段：正式站切換前

- 兩套 skill 的 `.mcp.json` 與 `/camp/ai/` 的 MCP 網址改成正式站，重建 skill zip（`site/scripts/build-skill-zips.mjs`）。
- Worker 設 `GITHUB_TOKEN`（唯讀）。
- 其餘切換條件見 `STATUS.md`「一次切換」。

## 進度紀錄

| 日期 | 任務 | 結果或卡點 |
|---|---|---|
| 2026-10-11 | 計畫建立 | 還沒開始第 0 階段 |
