# 舊站知識資料轉移分類草案（已由逐篇遷移結果取代）

> 本文件保留作為早期分類研究；有效逐篇結果請以 [`2026-09-20-article-migration-result.md`](2026-09-20-article-migration-result.md) 與網站樹狀地圖 v2 為準。

> 狀態：轉移分類草案。這份文件是逐份閱讀 `docs/learn`、`docs/alphago`、`docs/tech` 後整理的清單；來源盤點與既有規格確認已完成，尚未搬檔、改寫或發布。

## 1. 盤點範圍

- 已檢視 64 份 Markdown/MDX 內容：`docs/learn` 16 份、`docs/alphago` 21 份、`docs/tech` 27 份。
- `_category_.json` 是舊文件導覽設定，不列入內容轉移數量；新站將由內容模型與路由取代。
- 本次不重新開啟已完成的來源、權利、URL、SEO 或技術邊界確認；清單只描述轉移時的內容角色與處理方式。

## 2. 新網站地圖分類

本清單直接使用新網站地圖的內容分類，不再使用 K1～K5 代碼。

| 新網站地圖分類 | 新站路徑 | 內容範圍 |
|---|---|---|
| 公開知識／圍棋入門 | `/knowledge/go/introduction/` | 規則、禮儀、術語、開局 |
| 公開知識／圍棋歷史 | `/knowledge/go/history/` | 人類圍棋史與歷史脈絡 |
| 公開知識／AI 圍棋歷史 | `/knowledge/ai-go/history/` | AI 圍棋發展、AlphaGo、KataGo 歷史 |
| 公開知識／AI 時代 | `/knowledge/ai/` | AI 時代、跨產業學習與應用觀念 |
| 公開知識／AlphaGo 歷史與概念 | `/knowledge/ai-go/alphago/` | AlphaGo 發展、對局與核心概念 |
| 公開知識／AlphaGo 技術原理 | `/knowledge/ai-go/alphago/technical/` | 模型、訓練、MCTS、神經網路 |
| 公開知識／AI 圍棋技術總覽 | `/knowledge/ai-go/overview/` | AI 能力、生態與時間軸 |
| 公開知識／AI 運作原理 | `/knowledge/ai-go/how-it-works/` | AI 如何運作、KataGo 創新 |
| 公開知識／KataGo 操作 | `/knowledge/ai-go/kata-go/hands-on/` | 安裝、使用、整合 |
| 公開知識／AI 圍棋技術深入 | `/knowledge/ai-go/technical/` | 原始碼、訓練、部署、評估與論文 |
| 公開知識／AI 產業案例 | `/knowledge/ai/industry/` | 產業應用、轉型與台灣案例 |
| AI 共學營課程教材 | `/camp/courses/[course-slug]/materials/` | 只有講師選入正式課程後，才從上述知識轉成課程教材 |
| 舊站資料／歷史回憶 | `/association/history/` 或管理區保存 | 不適合公開知識用途的舊資料 |

以上是網站地圖中的實際內容分類；「課程教材」不是這次 64 份舊文章的直接去處，必須等講師建立課程並選用後才建立關聯。

## 3. 逐份分類建議

> 閱讀方式：先看下面這張「轉移清單總表」。`舊站來源` 是目前檔案或舊網站資料的位置；`新站位置` 是建議搬到新網站的內容區。`待確認` 表示目前只提出去向，尚未代表已核准或已完成轉移。

### 3.0 轉移清單總表

| 清單項目 | 舊站來源 | 新站位置 | 處理方式 | 目前狀態 |
|---|---|---|---|---|
| 學圍棋入口 | `docs/learn/index.md` | `/knowledge/go/` | 作為圍棋知識入口，重新整理導覽 | 待確認 |
| 圍棋入門 | `docs/learn/introduction/**`（6 份） | `/knowledge/go/introduction/` | 整理成公開知識文章，保留規則、禮儀、術語與開局內容 | 可轉移，需編輯 |
| 人類圍棋史 | `docs/learn/history/human-history/**`（3 份） | `/knowledge/go/history/` | 作為歷史資料；補年代、來源與作者 | 可轉移，需補來源 |
| AI 圍棋史 | `docs/learn/history/ai-history/**`（3 份） | `/knowledge/ai-go/history/` | 作為公開歷史知識；轉移時沿用明確截止日期 | 可轉移 |
| AI 時代與學習 | `docs/learn/ai-era/**`（3 份） | `/knowledge/ai/` | 保留公開觀念；操作內容可成為課程素材候選 | 可轉移，需編輯 |
| AlphaGo 歷史與概念 | `docs/alphago/01-05`、`20`、`index`（7 份） | `/knowledge/ai-go/alphago/` | 公開為歷史／概念系列；轉移時保留歷史來源與日期 | 可轉移 |
| AlphaGo 技術原理 | `docs/alphago/06-19`（14 份） | `/knowledge/ai-go/alphago/technical/` | 保留為技術知識；可供未來課程挑選，不直接掛課 | 可轉移 |
| 技術總覽 | `docs/tech/index.md`、`overview/**`（4 份） | `/knowledge/ai-go/overview/` | 作為公開技術入口與搜尋入口；轉移時套用新 URL | 可轉移 |
| AI 運作原理 | `docs/tech/how-it-works/**`（2 份） | `/knowledge/ai-go/how-it-works/` | 改寫成一般讀者可理解的公開文章 | 可轉移，需編輯 |
| KataGo 操作 | `docs/tech/hands-on/**`（4 份） | `/knowledge/ai-go/kata-go/hands-on/` | 保留既有版本標示；可列為實作課程素材候選 | 可轉移 |
| 技術深入研究 | `docs/tech/deep-dive/**`（13 份，含索引） | `/knowledge/ai-go/technical/` | 保留技術文章；轉移時保留程式碼、版本與指令限制 | 可轉移 |
| 產業與台灣案例 | `docs/tech/industry/**`（4 份） | `/knowledge/ai/industry/` | 轉移已確認的案例；保留來源、授權與時效欄位 | 可轉移 |

這張表的意思是：**舊站資料不會原樣變成新站頁面**。例如 `docs/tech/hands-on/setup.md` 會先列在「KataGo 操作」清單，建議新位置是 `/knowledge/ai-go/kata-go/hands-on/`；轉移時沿用既有版本與連結資料。

接下來的逐份表格，是把上面每一列拆成 64 個來源項目，方便逐筆決定「保留、改寫、歸檔或不轉移」。

### 3.1 圍棋與 AI 學習內容（16 份）

| 舊站來源 | 標題 | 新網站地圖分類 | 轉移處理 |
|---|---|---|---|
| `docs/learn/index.md` | 學圍棋 | 公開知識／圍棋與 AI 知識入口 | 作為圍棋學習入口；重寫導覽，不直接承諾共學營課程。 |
| `docs/learn/introduction/index.md` | 入門篇 | 公開知識／圍棋入門 | 合併為入門知識索引。 |
| `docs/learn/introduction/rules.md` | 圍棋規則 | 公開知識／圍棋入門 | 保留基礎內容，檢查規則表述。 |
| `docs/learn/introduction/etiquette.md` | 對弈禮儀 | 公開知識／圍棋入門 | 保留為入門文章。 |
| `docs/learn/introduction/terminology.md` | 圍棋術語 | 公開知識／圍棋入門 | 保留，建立可搜尋術語頁。 |
| `docs/learn/introduction/first-10-moves.md` | 開局概念 | 公開知識／圍棋入門 | 保留；避免寫成 AI 課程招生內容。 |
| `docs/learn/history/index.md` | 圍棋 AI 演進整理 | 公開知識／圍棋與 AI 知識入口 | 作為時間軸入口；每段標示年代。 |
| `docs/learn/history/human-history/index.md` | 人類圍棋發展史 | 公開知識／圍棋歷史 | 保留歷史脈絡，轉移時帶上年代與來源。 |
| `docs/learn/history/human-history/ancient.md` | 古代圍棋 | 公開知識／圍棋歷史 | 歷史文章，轉移時帶上來源與日期。 |
| `docs/learn/history/human-history/japan-china-korea.md` | 近現代圍棋 | 公開知識／圍棋歷史 | 歷史文章，轉移時保留人物與年代。 |
| `docs/learn/history/ai-history/index.md` | AI 圍棋發展史 | 公開知識／AI 圍棋歷史 | 作為 AI 圍棋歷史索引，標示內容截止日期。 |
| `docs/learn/history/ai-history/alphago-era.md` | AlphaGo 時代（2015-2017） | 公開知識／AI 圍棋歷史 | 保留為歷史階段，查核數字與引用。 |
| `docs/learn/history/ai-history/katago-era.md` | KataGo 時代（2019-現在） | 公開知識／AI 圍棋歷史 | 轉移時改用既定的明確截止日期。 |
| `docs/learn/ai-era/index.md` | AI 時代 | 公開知識／AI 時代 | 作為 AI 時代主題入口，重寫品牌脈絡。 |
| `docs/learn/ai-era/changes.md` | AI 帶來的變化 | 公開知識／AI 時代 | 可成為跨產業 AI 觀察文章，需更新案例。 |
| `docs/learn/ai-era/learning-with-ai.md` | 用 AI 學棋 | 公開知識／AI 時代 | 公開版保留觀念；操作流程可拆為未來課程素材。 |

### 3.2 AlphaGo 內容（21 份）

| 舊站來源 | 標題 | 新網站地圖分類 | 轉移處理 |
|---|---|---|---|
| `docs/alphago/index.md` | AlphaGo | 公開知識／AlphaGo 歷史與概念 | 作為系列入口，交代系列適用讀者與截止日期。 |
| `docs/alphago/01-birth-of-alphago.mdx` | AlphaGo 的誕生 | 公開知識／AlphaGo 歷史與概念 | 歷史文章；轉移時保留論文、年份與引文。 |
| `docs/alphago/02-key-matches.mdx` | 關鍵對局回顧 | 公開知識／AlphaGo 歷史與概念 | 歷史文章；轉移時沿用既有棋譜與著作權欄位。 |
| `docs/alphago/03-move-37.mdx` | 「神之一手」深度分析 | 公開知識／AlphaGo 歷史與概念 | 可公開，但要區分歷史描述與解讀。 |
| `docs/alphago/04-why-go-is-hard.mdx` | 圍棋為什麼難？ | 公開知識／AlphaGo 歷史與概念 | 適合作為 AI 入門文章。 |
| `docs/alphago/05-traditional-limits.mdx` | 傳統方法的極限 | 公開知識／AlphaGo 歷史與概念 | 保留概念，避免把歷史限制寫成當代絕對結論。 |
| `docs/alphago/06-board-representation.mdx` | 棋盤狀態表示 | 公開知識／AlphaGo 技術原理 | 技術文章；可拆成 AI 課程講義單張。 |
| `docs/alphago/07-policy-network.mdx` | Policy Network 詳解 | 公開知識／AlphaGo 技術原理 | 技術文章；轉移時保留公式與架構版本。 |
| `docs/alphago/08-value-network.mdx` | Value Network 詳解 | 公開知識／AlphaGo 技術原理 | 技術文章；轉移時保留公式與架構版本。 |
| `docs/alphago/09-input-features.mdx` | 輸入特徵設計 | 公開知識／AlphaGo 技術原理 | 技術文章；適合作為模型設計單張候選。 |
| `docs/alphago/10-cnn-and-go.mdx` | CNN 與圍棋的結合 | 公開知識／AlphaGo 技術原理 | 技術文章；檢查模型描述是否限於 AlphaGo 版本。 |
| `docs/alphago/11-supervised-learning.mdx` | 監督學習階段 | 公開知識／AlphaGo 技術原理 | 技術文章；補原始論文與術語。 |
| `docs/alphago/12-reinforcement-intro.mdx` | 強化學習入門 | 公開知識／AlphaGo 技術原理 | 公開入門版可保留，公式與例子另作技術版。 |
| `docs/alphago/13-self-play.mdx` | 自我對弈 | 公開知識／AlphaGo 技術原理 | 適合作為實作課程概念單張。 |
| `docs/alphago/14-mcts-neural-combo.mdx` | MCTS 與神經網路的結合 | 公開知識／AlphaGo 技術原理 | 技術文章；轉移時保留演算法說法與來源。 |
| `docs/alphago/15-puct-formula.mdx` | PUCT 公式詳解 | 公開知識／AlphaGo 技術原理 | 技術文章；公式需逐項驗證。 |
| `docs/alphago/16-alphago-zero.mdx` | AlphaGo Zero 概述 | 公開知識／AlphaGo 技術原理 | 公開概念文章；歷史數字需查核。 |
| `docs/alphago/17-dual-head-resnet.mdx` | 雙頭網路與殘差網路 | 公開知識／AlphaGo 技術原理 | 技術文章；標明適用模型與來源。 |
| `docs/alphago/18-training-from-scratch.mdx` | 從零訓練的過程 | 公開知識／AlphaGo 技術原理 | 技術文章；不要直接當作可重現教學，除非補齊環境。 |
| `docs/alphago/19-distributed-systems.mdx` | 分散式系統與 TPU | 公開知識／AlphaGo 技術原理 | 可公開為歷史技術文章；雲端硬體數字需查核。 |
| `docs/alphago/20-legacy-and-impact.mdx` | AlphaGo 的遺產 | 公開知識／AlphaGo 歷史與概念 | 公開歷史總結；補目前影響的時間界線。 |

### 3.3 技術內容（27 份）

| 舊站來源 | 標題 | 新網站地圖分類 | 轉移處理 |
|---|---|---|---|
| `docs/tech/index.md` | 技術文件 | 公開知識／AI 圍棋技術總覽 | 作為技術知識入口，不等同課程頁。 |
| `docs/tech/overview/index.md` | 圍棋 AI 能做什麼？ | 公開知識／AI 圍棋技術總覽 | 適合公開給非工程背景讀者。 |
| `docs/tech/overview/landscape.md` | 圍棋 AI 生態全景圖 | 公開知識／AI 圍棋技術總覽 | 保留架構，查核專案現況與連結。 |
| `docs/tech/overview/timeline.md` | 圍棋 AI 發展時間軸 | 公開知識／AI 圍棋技術總覽 | 保留時間軸；補截止日期與來源。 |
| `docs/tech/how-it-works/index.md` | 一篇文章搞懂圍棋 AI | 公開知識／AI 運作原理 | 公開總覽文章，適合 SEO/AEO 入口。 |
| `docs/tech/how-it-works/katago-innovations.md` | KataGo 的關鍵創新 | 公開知識／AI 運作原理 | 技術文章；查核 KataGo 版本與論文。 |
| `docs/tech/hands-on/index.md` | 30 分鐘跑起第一個圍棋 AI | 公開知識／KataGo 操作 | 操作入口；必須重做環境與版本驗證。 |
| `docs/tech/hands-on/setup.md` | KataGo 完整安裝指南 | 公開知識／KataGo 操作 | 先驗證作業系統、模型、下載連結與指令。 |
| `docs/tech/hands-on/basic-usage.md` | KataGo 基本使用 | 公開知識／KataGo 操作 | 可作實作教材候選；確認目前 CLI/API。 |
| `docs/tech/hands-on/integration.md` | 整合到你的專案 | 公開知識／KataGo 操作 | 可作實作教材候選；補支援範例與安全限制。 |
| `docs/tech/deep-dive/index.md` | 給想深入研究的人 | 公開知識／AI 圍棋技術深入 | 技術深度索引，標明先備知識。 |
| `docs/tech/deep-dive/source-code.md` | KataGo 原始碼導讀 | 公開知識／AI 圍棋技術深入 | 版本綁定強；指定 commit 或版本後再公開。 |
| `docs/tech/deep-dive/build-from-scratch.md` | 從零打造圍棋 AI | 公開知識／AI 圍棋技術深入 | 適合作為課程素材候選；目前不宣稱可直接重現。 |
| `docs/tech/deep-dive/neural-network.md` | 神經網路架構詳解 | 公開知識／AI 圍棋技術深入 | 技術單張候選；查核公式與示例。 |
| `docs/tech/deep-dive/mcts-implementation.md` | MCTS 實作細節 | 公開知識／AI 圍棋技術深入 | 技術單張候選；確認程式碼與偽代碼。 |
| `docs/tech/deep-dive/training.md` | KataGo 訓練機制解析 | 公開知識／AI 圍棋技術深入 | 版本與硬體相依；需來源與截止日期。 |
| `docs/tech/deep-dive/distributed-training.md` | 分散式訓練架構 | 公開知識／AI 圍棋技術深入 | 可成為進階課程素材候選；先查核可行環境。 |
| `docs/tech/deep-dive/evaluation.md` | 評估與基準測試 | 公開知識／AI 圍棋技術深入 | 可成為實作單張；需定義資料集與指標版本。 |
| `docs/tech/deep-dive/gpu-optimization.md` | GPU 後端與優化 | 公開知識／AI 圍棋技術深入 | 硬體／驅動版本敏感，先查核。 |
| `docs/tech/deep-dive/quantization-deploy.md` | 模型量化與部署 | 公開知識／AI 圍棋技術深入 | 適合作為部署實作候選；補可重現環境。 |
| `docs/tech/deep-dive/custom-rules.md` | 自訂規則與變體 | 公開知識／AI 圍棋技術深入 | 可作應用實作候選；確認支援範圍。 |
| `docs/tech/deep-dive/papers.md` | 關鍵論文導讀 | 公開知識／AI 圍棋技術深入 | 保留導讀架構；逐筆確認論文連結與引用格式。 |
| `docs/tech/deep-dive/contributing.md` | 參與開源社群 | 公開知識／AI 圍棋技術深入 | 連結與貢獻流程可能改變，查核後再公開。 |
| `docs/tech/industry/index.md` | 圍棋 AI 產業現況 | 公開知識／AI 產業案例 | 可做產業知識入口；需明確資料截止日。 |
| `docs/tech/industry/applications.md` | 實際應用案例 | 公開知識／AI 產業案例 | 案例真實性、授權與時效需逐案確認。 |
| `docs/tech/industry/ai-transformation.md` | AI 轉型啟示 | 公開知識／AI 產業案例 | 可作跨產業觀察；不能直接代表共學營承諾。 |
| `docs/tech/industry/taiwan.md` | 台灣案例 | 公開知識／AI 產業案例 | 需確認個案授權、來源與目前狀態。 |

## 4. 第一版整體去向

### 建議先進入公開知識區

`docs/learn/introduction/**`、`docs/learn/ai-era/**`、`docs/tech/overview/**`、`docs/tech/how-it-works/**`，以及 AlphaGo 的歷史與概念篇。這些內容最適合讓訪客理解圍棋、AI 與跨產業學習背景；轉移時依既有內容規格編輯。

### 建議保留為技術知識區

`docs/alphago/06-19` 與 `docs/tech/deep-dive/**`、`docs/tech/hands-on/**`。它們可服務有工程背景的讀者，也可支援日後講師設計課程；在未綁定課程前，不應出現在某一場共學營的教材清單中。

### 建議作為未來課程素材候選

技術文章中的「問題、原因、可能解法、建議流程」可以拆成講師的單張教材，但必須經過講師確認、版本固定、實作驗證，並在正式課程建立後才掛到該課程。舊文章本身不是 TTQS 課程紀錄，也不能直接宣稱是本期課程教材。

### 建議先放入歷史／回憶脈絡

人類圍棋史、AlphaGo 時代、早期 AI 時間軸與 AlphaGo 影響等內容，可以保留為公開歷史知識或舊站資料索引；必須標示年代與內容截止日期，避免訪客誤以為是目前服務或最新技術。

### 轉移時要保留的內容限制

以下不是未完成的工作，而是匯入新站時不可遺失的既有內容欄位：

1. 數字、年份、公式與技術版本的既有說明。
2. 外部連結、論文、軟體下載與操作指令的既有來源。
3. 案例、棋譜、圖片、程式碼與引用的既有權利標註。
4. 內容與協會課程、講師承諾或招生文案之間的既有邊界。

## 5. 我建議的搬遷順序

1. **保留來源**：不刪除舊檔，建立來源 ID、原始路徑、原標題與既有來源狀態。
2. **轉移公開知識**：依照上方網站地圖分類建立新站內容，補作者、日期、來源與結構化資料。
3. **保留技術限制**：標示既有技術版本與先備知識，不把技術文章誤當成課程頁。
4. **建立課程素材關聯**：只有講師在正式課程表單選用後，才把文章改成該課程教材。
5. **保存歷史資料**：歷史內容保留日期、來源與舊站脈絡；不把它當成目前服務承諾。

## 6. 目前不應自行做的事

- 不把 64 份舊文章全部直接搬成新課程內容。
- 不把技術文章自動當成 TTQS 的課程教材、佐證或學員成果。
- 不把舊站的「現在」「最新」「完整安裝」等字眼原樣保留。
- 不因為文章放在公開知識區，就推定協會對其中所有案例或技術結果背書。

## 7. 待你挑選的範圍

這一版先請你挑選「分類方向」而不是逐句修文。你可以直接指出：

- 哪些公開知識內容不想保留。
- 哪些歷史內容要移到舊站資料索引，而不是公開知識。
- 哪些技術內容應優先成為第一門課的素材候選。
- 哪些內容即使已有既定分類，也不應搬到新站。

確認分類後，下一份文件再做每一篇的欄位級轉移清單：新 slug、文章類型、作者、日期、來源、SEO/AEO/GEO 欄位、是否需要重寫，以及是否與課程或論壇資料關聯。
