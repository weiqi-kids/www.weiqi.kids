# 首圖圖說人工核對表

建立日期：2026-09-24　·　待核對：60 則　·　整理方式：AI 判讀插畫內容後撰寫，尚未經人工確認

## 這份表要做什麼

網站上 60 張保留的首圖各配了一段圖說，寫的是「這張圖在說明的那件事」，而不是畫面描述（畫面描述在 `alt` 屬性裡）。
圖說由 AI 看圖與頁面內容後撰寫，依專案品質關卡需經人工核對後才算定稿。

核對時只問兩件事：

1. **圖說講的事，圖上真的畫得出來嗎？** 畫面上沒有的東西不能寫。
2. **圖說講的事，頁面內容支持嗎？** 尤其是成員與合作夥伴頁，牽涉真實人物的職業與服務內容。

改法：直接改頁面 frontmatter 的 `heroCaption` 欄位（內容頁），或改 `.astro` 檔裡的 `<figcaption>`（靜態頁）。

> 成員與合作夥伴那 29 則已先做過一輪事實核對，移除了所有人名與第三人稱代名詞，並修掉 6 處頁面找不到依據的敘述。仍建議由熟悉當事人的人再看一次。

## 圍棋與圍棋 AI 知識（29 則）

| 核 | 頁面 | 圖說 | 圖檔 |
|:--:|---|---|---|
| ☐ | [AlphaGo 的誕生](/knowledge/ai-go/alphago/birth-of-alphago/) | 把棋盤局面交給神經網路判斷，是 AlphaGo 最初的構想（示意） | `ag-birth-of-alphago` |
| ☐ | [棋盤狀態表示](/knowledge/ai-go/alphago/board-representation/) | 棋盤被拆成一層層特徵平面，每一層只記錄一種棋子的位置 | `ag-board-representation` |
| ☐ | [分散式系統與 TPU](/knowledge/ai-go/alphago/distributed-systems/) | 訓練與推理的運算量，由成排機器和專用加速卡一起分攤 | `ag-distributed-systems` |
| ☐ | [雙頭網路與殘差網路](/knowledge/ai-go/alphago/dual-head-resnet/) | 共享主幹算完後分成兩個頭，一邊輸出下一手，一邊輸出勝率（示意） | `ag-dual-head-resnet` |
| ☐ | [AlphaGo 的遺產](/knowledge/ai-go/alphago/legacy-and-impact/) | 同一套技術從圍棋擴散到蛋白質結構、藥物與天文研究（示意） | `ag-legacy-and-impact` |
| ☐ | [MCTS 與神經網路的結合](/knowledge/ai-go/alphago/mcts-neural-combo/) | 神經網路裝進搜尋樹的主幹，讓每個分支的局面都能被評估（示意） | `ag-mcts-neural-combo` |
| ☐ | [自我對弈](/knowledge/ai-go/alphago/self-play/) | 同一個程式分飾黑白雙方對局，從每盤結果裡累積經驗（示意） | `ag-self-play` |
| ☐ | [從零訓練的過程](/knowledge/ai-go/alphago/training-from-scratch/) | 從隨機亂下到棋形完整，棋力在三天內一路長大（示意） | `ag-training-from-scratch` |
| ☐ | [Value Network 詳解](/knowledge/ai-go/alphago/value-network/) | 秤出目前局面黑白哪一方勝率較高，就是價值網路的工作（示意） | `ag-value-network` |
| ☐ | [實際應用案例](/knowledge/ai/industry/applications/) | 線上平台把勝率與領地分析畫在棋盤上，用來帶著棋友覆盤 | `ai-applications` |
| ☐ | [圍棋 AI 演進整理](/knowledge/ai-go/evolution/) | 運算能力一階一階往上墊高，圍棋 AI 才走到今天的樣子（示意） | `ai-go-evolution` |
| ☐ | [AI 圍棋發展史](/knowledge/ai-go/history/) | 早期圍棋程式、棋盤局面與冠軍獎盃，串起這段發展歷程 | `ai-go-history` |
| ☐ | [一篇文章搞懂圍棋 AI](/knowledge/ai-go/how-it-works/) | 棋盤局面送進神經網路，再輸出這一手該下在哪裡 | `ai-go-how-it-works` |
| ☐ | [圍棋 AI 能做什麼？](/knowledge/ai-go/overview/) | AI 在每一手後算出雙方勝率，用一條黑白長條顯示誰領先 | `ai-go-overview` |
| ☐ | [用 AI 學棋](/knowledge/ai/learning-with-ai/) | 棋譜載入 AI 分析介面後，畫面會直接標出建議的下一手 | `ai-learning-with-ai` |
| ☐ | [台灣案例](/knowledge/ai/industry/taiwan/) | 台灣雖然市場規模小，仍有幾個在地的圍棋 AI 應用案例（示意） | `ai-taiwan` |
| ☐ | [AI 轉型啟示](/knowledge/ai/industry/ai-transformation/) | 圍棋走過的 AI 轉型歷程，可以套用到其他各行各業（示意） | `ai-transformation-lesson` |
| ☐ | [對弈禮儀](/knowledge/go/introduction/etiquette/) | 對局開始前的問候，是每位棋友應具備的基本素養 | `go-etiquette` |
| ☐ | [開局概念](/knowledge/go/introduction/first-10-moves/) | 開局的前幾手先分佔四個角，因為角落圍地的效率最高 | `go-first-10-moves` |
| ☐ | [圍棋規則](/knowledge/go/introduction/rules/) | 白子周圍四口氣全被黑子堵住，這顆棋就要從棋盤上拿走 | `go-rules` |
| ☐ | [AI](/knowledge/ai/) | 圍棋 AI 的做法可以延伸到醫療、製造與教育等產業（示意） | `knowledge-ai` |
| ☐ | [公開知識](/knowledge/) | 這裡的文章從圍棋入門一路延伸到神經網路與 AI 應用 | `knowledge-index` |
| ☐ | [自訂規則與變體](/knowledge/ai-go/technical/custom-rules/) | 規則集與棋盤尺寸都是可替換的設定，不同大小的棋盤都能對弈 | `tech-custom-rules` |
| ☐ | [分散式訓練架構](/knowledge/ai-go/technical/distributed-training/) | 各地志願者的電腦各自自我對弈，把棋譜送回中央伺服器彙整 | `tech-distributed-training` |
| ☐ | [評估與基準測試](/knowledge/ai-go/technical/evaluation/) | 靠大量對局累積的勝負算出 Elo，再用計時衡量搜索速度 | `tech-evaluation` |
| ☐ | [MCTS 實作細節](/knowledge/ai-go/technical/mcts-implementation/) | 盤面一層層分支成搜索樹，走得越多的路徑越亮（示意） | `tech-mcts-implementation` |
| ☐ | [神經網路架構詳解](/knowledge/ai-go/technical/neural-network/) | 盤面從左側輸入網路，經過層層節點後分出多個輸出頭（示意） | `tech-neural-network` |
| ☐ | [模型量化與部署](/knowledge/ai-go/technical/quantization-deploy/) | 量化把龐大的模型壓縮成小尺寸，才塞得進手機執行（示意） | `tech-quantization-deploy` |
| ☐ | [KataGo 訓練機制解析](/knowledge/ai-go/technical/training/) | 自我對弈產生的棋譜投入訓練，煉出更強的模型再回頭對弈（示意） | `tech-training` |

## 協會成員與合作夥伴（29 則）

| 核 | 頁面 | 圖說 | 圖檔 |
|:--:|---|---|---|
| ☐ | [陳阿美](/association/members/a-mei-chen/) | 保單健檢、儲蓄與保障規劃，是這位成員的專業領域（示意） | `member-a-mei-chen` |
| ☐ | [陳秉辰](/association/members/bing-chen-chen/) | 從草圖到成品模型，對應的是產品設計企劃這個專業領域 | `member-bing-chen-chen` |
| ☐ | [鄭博陽](/association/members/bo-yang-cheng/) | 整脊調理與一對一健身課程結合的體態調整，是這裡提供的服務 | `member-bo-yang-cheng` |
| ☐ | [蕭誠緯](/association/members/cheng-wei-hsiao/) | 串起藥局通路與醫事人員的整合媒合平台，是這位成員的專業領域（示意） | `member-cheng-wei-hsiao` |
| ☐ | [李郁萱](/association/members/dawn-lee/) | 逐項查核、留下紀錄再取得認證，就是 ISO 系統輔導的工作 | `member-dawn-lee` |
| ☐ | [吳芳圳](/association/members/fang-chuan-wu/) | 把現金、不動產與後續傳承一起盤算，就是財稅資產規劃的內容（示意） | `member-fang-chuan-wu` |
| ☐ | [姜封豪](/association/members/feng-hao-chiang/) | 把運動與營養放進家庭醫學門診，是這位醫師主打的方向 | `member-feng-hao-chiang` |
| ☐ | [吳軒儒](/association/members/hsuan-ju-wu/) | 短影音製作要備齊攝影機、三腳架與播放用的直式手機 | `member-hsuan-ju-wu` |
| ☐ | [郭尚諺](/association/members/ian-kuo/) | 古典吉他演奏，是加一吉他團長累積 29 年的專業領域 | `member-ian-kuo` |
| ☐ | [廖宜鋒](/association/members/james-liao/) | 記帳、財稅簽證與工商登記，都在帳簿、計算機與印章之間完成 | `member-james-liao` |
| ☐ | [周敬彥](/association/members/jing-yan-chou/) | 螢幕上向上的折線，對應行銷規劃要交出的業績成長 | `member-jing-yan-chou` |
| ☐ | [毛子](/association/members/mao-zi/) | 繪圖板上由黑白棋子組成的星球，對應這位成員的插畫專長 | `member-mao-zi` |
| ☐ | [李尚澔](/association/members/shang-hao-lee/) | 沙拉低 GI 飲食與手作果醬，是這家食療藝文餐廳的招牌 | `member-shang-hao-lee` |
| ☐ | [周思傑](/association/members/sih-jie-chou/) | 天秤與法典之間，是契約審查與法律風險控管的工作現場 | `member-sih-jie-chou` |
| ☐ | [潘天健](/association/members/tien-chien-pan/) | 試管與細胞分子圖，對應中西醫整合醫療與細胞分子矯正學會的職務（示意） | `member-tien-chien-pan` |
| ☐ | [吳荻](/association/members/wu-di/) | 畫架上的動物角色草稿，對應這位成員的插畫專長 | `member-wu-di` |
| ☐ | [羅揚](/association/members/yang-luo/) | 高濃度魚油這類保健食品，正是要把觀念講清楚的對象 | `member-yang-luo` |
| ☐ | [謝一肇](/association/members/yi-chao-hsieh/) | 商品上架、下單到出貨包裹，是品牌數位行銷要顧的網店流程 | `member-yi-chao-hsieh` |
| ☐ | [謝依純](/association/members/yi-chun-hsieh/) | 錄音室的麥克風與隔音牆，是配音工作完成的地方 | `member-yi-chun-hsieh` |
| ☐ | [吳宜芳](/association/members/yi-fang-wu/) | 碼頭的貨櫃、貨輪與飛機，對應海空物流與理貨報關的環節 | `member-yi-fang-wu` |
| ☐ | [顏怡璇](/association/members/yi-hsuan-yen/) | 打開的茶禮盒與茶葉罐，是 B2B 客製化茶禮的樣子 | `member-yi-hsuan-yen` |
| ☐ | [詹益萬](/association/members/yi-wan-chan/) | 配線箱、監視器與電話總機，就是弱電工程涵蓋的範圍 | `member-yi-wan-chan` |
| ☐ | [陳于玲](/association/members/yu-ling-chen/) | 立體布景的大公仔從零件到上色完成，成為可以拍照的景點 | `member-yu-ling-chen` |
| ☐ | [趙珍](/association/members/zhen-zhao/) | 在中藥櫃前抓取藥材、秤重配伍，是針藥結合治療的用藥這一端 | `member-zhen-zhao` |
| ☐ | [黃子彥](/association/members/zi-yan-huang/) | 依中醫經典配方燉煮的草本藥膳，把食療觀念放進日常飲食 | `member-zi-yan-huang` |
| ☐ | [敬達](/association/partners/jing-da/) | 官網開發從版面規劃開始，先把首頁的各個區塊模組排定 | `partner-jing-da` |
| ☐ | [毛子](/association/partners/mao-zi/) | 黑白棋子化身為漂浮宇宙的星球，成為圍棋主題的手機桌布 | `partner-mao-zi` |
| ☐ | [吳荻](/association/partners/wu-di/) | 協會吉祥物的狐狸與狸貓角色，都是從畫架上的設計線稿開始 | `partner-wu-di` |
| ☐ | [謝依純](/association/partners/yi-chun/) | 宣傳影片的旁白在錄音室對稿錄製，用聲音傳達圍棋文化 | `partner-yi-chun` |

## 協會、共學營與單元首頁（2 則）

| 核 | 頁面 | 圖說 | 圖檔 |
|:--:|---|---|---|
| ☐ | [協會成員](/association/members/) | 協會成員來自醫療、設計、行銷、教育等不同領域（示意） | `association-members` |
| ☐ | [參加流程](/camp/how-it-works/) | 從填報名表、匯款、上主題教學到動手做出作品的四個步驟 | `camp-how` |

## 核對完成後

1. 在本檔標記完成日期與核對人。
2. 依專案品質關卡，於相關頁面註明整理來源、日期與作者。
