# AI 共學營完整操作流程（ADR 0014、0015）

更新：2026-10-11。這份是從講師、學員、管理員三個角色走完整條線的操作流程，每一步對到實際做好的東西（skill 段落、MCP 工具、網頁）與測試狀態。待辦順序以這份為準，`STATUS.md` G 節只列摘要。

## 三樣工具

| 工具 | 誰裝 | 在哪 | 做什麼 |
|---|---|---|---|
| 好棋寶寶 MCP | 講師、學員 | 網站 `/mcp`，用網站帳號 OAuth 登入（`/oauth/authorize/`） | 兩套 skill 對網站做事的管道。只存草稿，不送出 |
| 好棋寶寶講師 skill | 講師 | `site/plugins/haoqi-teacher`（Claude Code 外掛／zip） | 整理 repo、建成果草稿、登記版本、看學員進度、擬回覆 |
| 好棋寶寶學員 skill | 學員 | `site/plugins/haoqi-learner`（Claude Code 外掛／zip） | 找成果、照講師做法在自己帳號建一份、每週進度、分享草稿 |

安裝說明頁：`/camp/ai/`（Claude 網頁版、Claude Code、ChatGPT 三種路徑；不支援 skill 的工具改用 MCP 的 `get_guide` 拿同樣引導）。

MCP 工具 19 個：`whoami` `get_guide` `find_showcases` `get_showcase` `record_learning_need` `register_interest` `cancel_interest` `my_learning` `draft_question` `draft_practice` `draft_work_share` `list_posts` `read_post` `draft_reply` `my_showcases` `save_showcase_draft` `register_skill_version` `cohort_progress` `list_learning_needs`。

## 一、講師：上架成果

前提：講師是有效會員（`whoami` 會查）。

| # | 講師做什麼 | AI 做什麼（講師 skill 段落） | MCP／網頁 | 狀態 |
|---|---|---|---|---|
| T1 | 裝好棋寶寶 MCP 與講師 skill，登入網站按同意 | | `/camp/ai/`、`/oauth/authorize/` | 完成，staging 測過登入授權 |
| T2 | 在自己的專案資料夾對 AI 說「把這個放上好棋寶寶」 | `whoami` 確認會員；`get_guide(repo_structure)` 拿結構與協會範例 | `whoami`、`get_guide` | 完成，測過 |
| T3 | 回答 AI 的問題（一次一題） | 「1. 整理成 repo」：集中檔案→裝 git→註冊 GitHub→初始化→掃機密→建公開 repo、第一次 push | 講師電腦／GitHub | 寫好，**沒測** |
| T4 | 確認 AI 擬的草稿 | 「2. 補齊結構」：給學員 AI 的 skill、程式、定時更新、網站範本、設定範例、技能單張、README 8 項、LICENSE。四項要講師自己答：用 AI 前後差別、適合誰與基礎、本業與動機、要準備什麼 | 講師 repo | 寫好，**沒測** |
| T5 | 回答開團設定（一次一題）、確認 TTQS 表單 | 「3. 存成草稿」：`save_showcase_draft` 填 8 項、repo、授權、開團設定、TTQS；`register_skill_version` 登記這次 push | `save_showcase_draft`、`register_skill_version` | 完成，**沒測** |
| T6 | 開 AI 給的網址，看預覽，按「送審」成果、按「送審這個版本」 | 把網址給講師 | `/account/teach/showcases/<id>/`（5 步：成果與 8 項資料、開團設定、TTQS、送審、GitHub 版本；缺項會列出） | 完成 |

## 二、管理員：審核

| # | 做什麼 | 網頁 | 狀態 |
|---|---|---|---|
| A1 | 審成果：8 項資料、開團設定、TTQS 表單；通過就公開，開始接受登記想學 | `/admin/showcases/<id>/` | 完成 |
| A2 | 審版本：頁面自動列出跟上一個審過版本相比改了哪些檔案、定時更新設定、會連到哪些網站、疑似機密；通過後學員的 AI 讀這一版 | `/admin/versions/<id>/` | 完成，**沒測**（要先有 T5） |

## 三、學員：試做、登記、上課、分享

學員不用入會，任何 Email 或 LINE 帳號都可以。

| # | 學員做什麼 | AI 做什麼（學員 skill 段落） | MCP／網頁 | 狀態 |
|---|---|---|---|---|
| L1 | 在成果頁複製那一句話（「用好棋寶寶學員 skill，陪我照○○的『○○』（成果代碼 xxx）做一份我自己的版本」），或直接對 AI 說自己的工作問題 | 「找成果」：`get_showcase` 直接開始；或 `find_showcases` 列幾項讓學員選；找不到經同意 `record_learning_need` | `/camp/r/<slug>/`、`find_showcases`、`get_showcase`、`record_learning_need` | 完成，讀取測過 |
| L2 | 同意 AI 列出的「會讀寫哪些檔案、連到哪些網站、定時更新、要花多少錢」 | 「試做」第 1 步：照 `get_showcase` 的 `connects_to`、`scheduled_updates`、「要準備什麼」說明 | `get_showcase` | 完成，**沒測**（要先有公開成果） |
| L3 | 在自己的 GitHub 建一份、填設定、改成自己的題目 | 「試做」第 2～4 步：照審過那一版的 skill 做；確認定時更新與網站跑起來 | 學員 GitHub | 寫好，**沒測** |
| L4 | 卡住：同意後到網站按確定 | 「卡住」：`draft_question` 存草稿給連結 | `/account/drafts/<id>/` → 成果討論區 `/camp/r/<slug>/forum/` | 完成，**沒測** |
| L5 | 做不來：登記想學（免費、可取消） | 「做不來」：說明團費、上課方式、人數、排時間；`register_interest` | `register_interest`、`cancel_interest`，或網頁 `/camp/r/<slug>/` 的「登記想學」 | 完成，網頁端測過 |
| L6 | 滿人數→依講師設定選時段或投票→收通知→網站上填匯款日期與末五碼 | 不經 AI | `/camp/g/<id>/`、推播與站內通知 | 完成，staging 端對端測過（真手機推播沒測） |
| L7 | 主題教學簽到 | 不經 AI | `/camp/g/<id>/room/` | 完成，測過 |
| L8 | 3 週實作：每週交進度 | 「開團以後」：`my_learning` 看截止；`draft_practice` 存草稿給連結 | `/account/drafts/<id>/` → 課程論壇 `/camp/g/<id>/forum/` | 完成，**沒測** |
| L9 | 做出來：看草稿、按「確定送出」、可勾「講師認可後同意公開到成果頁」 | 「做出來了」：主動整理（做了什麼、連結或截圖、怎麼做、心得）；`draft_work_share` | `/account/drafts/<id>/` | 完成，**沒測** |
| L10 | 填結束問卷 | 不經 AI | `/camp/g/<id>/room/survey/` | 完成，測過 |

## 四、講師：開團以後

| # | 講師做什麼 | AI 做什麼（講師 skill 段落） | MCP／網頁 | 狀態 |
|---|---|---|---|---|
| T7 | 收到「某某分享了作品」推播，或打開 AI | 「開團以後」：一打開先 `cohort_progress`，有沒回的就 `read_post`→`draft_reply` 擬好草稿 | `cohort_progress`、`read_post`、`draft_reply` | 完成，**沒測** |
| T8 | 開草稿網址按「確定送出」；作品分享在同一頁按「認可這份作品」 | | `/account/drafts/<id>/`、論壇文章頁 | 完成 |
| T9 | 沒用 AI 的講師：論壇文章頁直接點常用回覆 | | `/camp/g/<id>/forum/<post>/` quick-reply | 完成 |
| T10 | 學員作品經講師認可＋作者同意→自動加到成果頁 | | `/camp/r/<slug>/` | 完成，**沒測** |
| T11 | 改了 repo 說「更新好棋寶寶上的版本」 | 「更新版本」：掃機密、push、`register_skill_version`，請講師在網站送審 | 同 T5、T6、A2 | 完成，**沒測** |
| T12 | 想開新課看學員記下來的問題 | `list_learning_needs` | `/account/teach/needs/` | 完成，**沒測** |

## 五、系統與管理員：結案

| # | 做什麼 | 網頁 | 狀態 |
|---|---|---|---|
| S1 | 核款、退款 | `/admin/payments/` | 完成，測過 |
| S2 | 第 1、2、3 週實作提醒推播 | 自動 | 完成，測過 |
| S3 | TTQS 佐證匯出（開課表單、簽到、問卷、論壇佐證） | 管理區 | 完成，測過 |
| S4 | 整團結束付講師 50% | `/admin/payouts/` | 完成，測過 |
| S5 | 一團結束，成果自動重新開放登記想學 | 自動 | 完成，測過 |

## 六、第一次完整實測：順序

用協會的「23 條產業供應鏈情報網」（repo `weiqi-kids/intel-template`）走一遍。站主同時是講師（會員）與管理員；學員用第二個 Email 帳號，不用入會。

1. **講師上架（T1～T6）**：站主在 `intel-template` 的 clone 裡開 Claude Code，裝 `haoqi-teacher@haoqi`，說「把這個放上好棋寶寶」，照 skill 一題一題答到送審。開團設定的開團人數先設 1，方便後面測開團。
2. **管理員審核（A1、A2）**：審成果、審版本，通過。成果頁 `/camp/r/supply-chain/` 公開。
3. **學員試做（L1～L4）**：第二個帳號裝 `haoqi-learner@haoqi`，貼成果頁那句話，照做到定時更新跑起來；中途故意發一則提問測 L4。
4. **登記到開團（L5～L7、S1）**：登記想學→滿 1 人開團→填匯款→管理員核款→簽到。
5. **實作與分享（L8～L9、S2）**：交一次進度、交一次作品分享，按確定送出。
6. **講師回覆（T7～T10）**：站主回講師的 Claude Code，確認 AI 主動擬了回覆草稿；送出、認可作品；看成果頁有沒有出現學員作品。
7. **結案（L10、S3～S5）**：填問卷、匯出 TTQS、付款、確認成果重新開放登記。
8. **更新版本（T11、A2）**：改 repo 一個檔案，走一次版本送審。

每一步卡住或問得不對，記在下面「實測紀錄」，改 skill 或網頁後重跑那一步。

## 七、協會另外 3 組成果

每組成果一個 repo，當作講師的範本 repo；其他網站列在「成果展示」。建議：

| 成果 | 範本 repo | 備註 |
|---|---|---|
| 6 個即時情報監測站 | 從 `agent.cyber-security` 或 `agent.risk-and-responsibility` 選一個整理成範本 | 6 個站各自一個 repo，要挑一個當範本 |
| 4 個給一般人用的 AI 工具 | `agent.supplement-product` | 4 個工具結構不同，挑一個當範本 |
| 9 篇 AI 輔助的數學與 AI 研究 | `proof`（三技能工作區） | 研究 repo 是成品，`proof` 才是做法 |

每組都照第六節第 1、2 步，由站主當講師用講師 skill 上架。

## 八、實測紀錄

（還沒開始）

## 九、正式網域切換時

- 兩套 skill 的 `.mcp.json` 與 `/camp/ai/` 的 MCP 網址從 staging 改成正式站。
- Worker 設 `GITHUB_TOKEN`（唯讀），版本審核頁才不會被 GitHub API 限流。
