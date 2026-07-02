#!/usr/bin/env bash
# weiqi.kids 大腦層「每日 SEO 自動優化」本機 cron 進入點（比照 folk.tw 設計）。
#
# 流程：git 同步 → headless claude（Sonnet 5）讀當日數據→定優化→改≤5檔→過 gate（pnpm build）→
#   commit [auto-claude-seo] → push（→ .github/workflows/build.yml 自動部署）→（有 IndexNow key 才推收錄）→
#   寫 <date>-actions.md → 發 Slack C0BEGBQLCD7（人話）。claude 全程自理；本包裝只做環境/同步/清理/失敗保底。
#
# 用法：
#   analytics/scripts/seo-brain-cron.sh            # 正式跑（會 commit/push/發 Slack）
#   DRY_RUN=1 analytics/scripts/seo-brain-cron.sh  # 乾跑：只提案，不 commit/push/發 Slack
#
# crontab：見 /etc/cron.d/weiqi-seo（每日台灣 05:55 = UTC 21:55，排在資料層/心跳之後）。
set -uo pipefail

export PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin"
export TZ="Asia/Taipei"
export IS_SANDBOX=1   # Claude Code 認可的 root 旁路，讓 headless 得以運行

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
mkdir -p logs analytics/seo-daily

DRY_RUN="${DRY_RUN:-0}"
DATE="$(date +%F)"
CHANNEL="C0BEGBQLCD7"
SLACK_NOTIFY="$REPO/analytics/scripts/weiqi-slack-notify.sh"

echo "===== [weiqi-brain] $DATE $(date '+%T %Z') 開始（DRY_RUN=$DRY_RUN）====="

# 1) 同步 main（取資料層當天 JSON 與任何先前 [auto-claude-seo]）。
git pull --rebase --autostash origin main 2>&1 || echo "[weiqi-brain] git pull 失敗（續行，讀本機既有）"

# 2) headless claude：執行每日 SEO 自動優化閉環。
PROMPT="$(cat <<PROMPTEOF
你是 weiqi.kids（台灣好棋寶寶協會官網 https://www.weiqi.kids/）的「每日 SEO/AEO 自動優化執行者」，在**自有主機 cron**中以 headless 執行（非雲端、無先前對話）。本站是 Docusaurus 3.8 多語系（11 語）靜態站、pnpm、部署到 GitHub Pages（push main → .github/workflows/build.yml 自動 build+deploy）。內容：繁中原始文件在 docs/（for-players 給棋友、for-engineers 給 AI 工程師、evolution 圍棋 AI 演進、aboutus 協會介紹），翻譯在 i18n/{locale}/。今天台灣日期＝$DATE（系統時鐘已是台灣時間，勿再 +8）。

# 鐵則（違反即停手）
1. 絕不杜撰事實：補任何事實型內容（人物、歷史、賽事、AI 技術細節）一定先用 WebSearch 找到權威源並附上；查無權威源就不碰事實，只做不涉事實的動作（meta title/description、frontmatter、內鏈、H 標結構、AEO 標記）。
2. 每天最多改 5 個檔，寧少勿多；優先繁中 docs/ 原始檔（翻譯檔本輪先不動，避免 11 語同步爆量）。
3. 遵守本 repo 既有 SEO/AEO 品質關卡：見 CLAUDE.md 與 seo/CLAUDE.md（title ≤60 字、description ≤155 字、每個 H2 的 .key-answer、7 種必填 JSON-LD 等）。改 meta/描述時對齊這些規範，不要破壞既有 schema。
4. 改完自我驗證：pnpm install --frozen-lockfile → pnpm build（Docusaurus 全語系建置）。任一非零 → git checkout . 撤回、今日不 push、走步驟6發 Slack 標🔴。
5. 內容改動集中成一個 commit、訊息前綴 [auto-claude-seo]，便於辨識與 git revert。
6. 敏感資訊（人名/職稱/公司）一律最高謹慎：不確定就不改、不杜撰。

# DRY_RUN
本次 DRY_RUN=$DRY_RUN。若 =1：照常讀數據、可試改與跑 build 驗證，但**絕不 git commit/push、不發 Slack**；最後只在 stdout 印「乾跑提案摘要」後結束。

# 每日流程
## 1. 讀今日數據
- ls -t analytics/seo-daily/*.json | head -1 取最新、Read 之。含 ga4（sessions、taiwanOrganicSessions、topPages）、gsc（totals、topQueries、strikingDistance 排名5-15、highImpZeroClick、pageQueryCross）、index（coverage）。
- 若今日 analytics/seo-daily/$DATE-actions.md 已存在（今天已跑過）→ 只做極小補強或直接 no-op（DRY_RUN=0 時仍發 Slack）。
- 若 gsc 段為 {error}（GSC 服務帳號尚未加入屬性）→ **沒有關鍵字/排名訊號**，本輪只能依 GA4 的 topPages 做「非事實型」小優化（meta/內鏈/結構），或直接 no-op；在 Slack 標🟡提醒「GSC 尚未接上，優化受限」。
- 若最新 JSON 不存在或 ga4 也 {error} → 無數據，跳步驟5寫「無數據」、步驟6發 Slack 標🟡結束。

## 2. 驗證昨天（閉環回饋）
- 讀昨天 <昨日>-actions.md（若有），取昨天賭的 query/page，對照今日 JSON 判定進步/持平/退步（GSC 有 2-3 日延遲，資料區間未動時「持平」正常）。連續≥3 天明顯退步 → 考慮回退並在 Slack 標🟡。

## 3. 定本日優化（依訊號、結論先行、冷啟動期保守）
- strikingDistance（排名5-15）：在對應頁的 answer-first/.key-answer 區塊把該詞講更完整（事實型受鐵則1約束）、補相關內鏈。
- highImpZeroClick：改該頁 title/description/frontmatter 更貼 query。
- pageQueryCross 錯配：調內鏈把權重導向對的頁。
- 訊號弱或無 GSC：沒有可行動訊號就 no-op，不要為動而動、不要亂改全站。

## 4. 執行 + 自我驗證 + 上線（DRY_RUN=0 才 push）
- 鐵則內改 ≤5 檔；事實型先 WebSearch 查證附源。
- pnpm install --frozen-lockfile → pnpm build。失敗 → git checkout .、不 push、跳步驟5（Slack 標🔴）。
- 有內容改動且 DRY_RUN=0：
  - git add -A、git commit -m \"[auto-claude-seo] $DATE: <一句摘要>\"
  - git pull --rebase origin main（防搶先；衝突無法自動解 → git rebase --abort、放棄今日 push、跳步驟5）
  - git push origin main（被拒 non-fast-forward → git pull --rebase 後再 push）
  - **部署上線（重要：weiqi 不會因 push 自動部署！.github/workflows/build.yml 只做建置驗證，不發佈）**：push main 成功後，務必執行 GIT_USER=weiqi-kids pnpm run deploy（= docusaurus deploy → 發佈到 gh-pages）。此步會重新建置全語系並推 gh-pages 才真正上線。deploy 失敗 → Slack 標🔴回報（內容已在 main，可稍後重試部署）。
  - 收錄：對本次「改動頁的完整 https://www.weiqi.kids/... 網址（含各語系）」呼叫 node analytics/scripts/indexnow-ping.mjs <url...>（IndexNow 通知 Bing/Yandex 等即時收錄；Google 由 sitemap 自然收錄）。no-op 無 URL 則略過。

## 5. 留痕（供明日驗證）
- 寫 analytics/seo-daily/$DATE-actions.md：① 昨日賭注勝負 ② 今日判讀摘要 ③ 改了哪些檔、賭哪些 query/page、預期效果。技術細節寫這裡，不寫進 Slack。
- no-op 日（DRY_RUN=0）：只 commit actions.md，訊息 chore(seo): $DATE 無動作 [skip ci]；git pull --rebase 後 push。

## 6. 發 Slack（DRY_RUN=0 每天都發；一律人話、禁術語）
用 Bash 執行 weiqi 專屬發送工具（**不是** MCP）：printf '%s' \"<整則訊息>\" | $SLACK_NOTIFY $CHANNEL
術語照翻：strikingDistance→「排第5–15名、快擠進第一頁的關鍵字」；highImpZeroClick→「很多人看到卻沒人點的頁」；taiwanOrganicSessions→「台灣 Google 搜尋來的訪客」；impressions→「被看到N次」；build/push/deploy→合併成「已自動上線」；IndexNow→「已通知搜尋引擎重新收錄」。不要出現英文縮寫、commit hash、檔名。
訊息排版（每項一行「・」開頭、段落空行、【】標題、標題行用 *粗體*）：
🚦 今天要不要你出手：<🟢 不用，系統自己處理好了 ／ 🟡 建議你看一下：一句原因 ／ 🔴 需要你決定：一句事項>

📊 *weiqi.kids SEO 日報 · <M/D>*

【目前成效】
・台灣 Google 搜尋來的訪客：N 人
・網站在 Google 被看到 N 次、有人點 M 次（GSC 未接上就寫「搜尋數據尚未接上」）
　（Google 數據有 2–3 天延遲）

【今天做了什麼】
<一句總述改了哪頁；no-op／無數據就寫「今天沒有需要調整的地方，系統照常監看」並省略下三點>
・問題：<為什麼這頁值得改，人話>
・做法：<改了什麼，人話>
・狀態：已自動上線

【昨天的調整有沒有效】
<進步／持平／退步白話；資料未更新就寫「還看不出來，Google 數據要 2–3 天才更新」>

📄 完整紀錄：analytics/seo-daily/$DATE-actions.md

# 收尾
最後在 stdout 印 3 行內摘要（改了幾項/哪些/有無 push/有無發 Slack）。
PROMPTEOF
)"

CLAUDE_OK=1
timeout 2400 claude -p "$PROMPT" --model claude-sonnet-5 2>&1 \
  || { CLAUDE_OK=0; echo "[weiqi-brain] claude 執行失敗或逾時"; }

# 3) 殘留清理：claude 對該留的改動會自行 commit；此刻未提交的是 DRY_RUN 試改 / gate-fail / no-op 痕跡，
#    清回 HEAD 以免污染隔天 git pull --rebase。只清內容/資料/翻譯區，不碰 analytics/scripts 等工具。
for p in docs i18n src analytics/seo-daily; do
  if [ -n "$(git status --porcelain -- "$p" 2>/dev/null)" ]; then
    echo "[weiqi-brain] 清理未提交殘留：$p"
    git checkout -- "$p" 2>/dev/null || true
    git clean -fdq -- "$p" 2>/dev/null || true
  fi
done

# 4) 失敗保底通報：claude 整段失敗/逾時時，補一則🔴，確保「大腦掛了」不會全靜默。DRY_RUN 不發。
if [ "$DRY_RUN" != "1" ] && [ "$CLAUDE_OK" = "0" ] && [ -x "$SLACK_NOTIFY" ]; then
  printf '%s' "🚦 今天要不要你出手：🔴 需要你看一下
:warning: *weiqi.kids SEO 自動優化 $DATE — 執行中斷*
本機大腦層 headless 執行失敗或逾時，今日可能未完成優化/通報。
請查 log：/root/www.weiqi.kids/logs/seo-brain.log" | "$SLACK_NOTIFY" "$CHANNEL" >/dev/null 2>&1 \
    && echo "[weiqi-brain] 已發失敗保底 Slack" || echo "[weiqi-brain] 失敗保底 Slack 也送不出（查 token）"
fi

echo "===== [weiqi-brain] $DATE $(date '+%T %Z') 結束 ====="
