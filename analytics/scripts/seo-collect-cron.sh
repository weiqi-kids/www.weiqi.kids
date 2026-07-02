#!/usr/bin/env bash
# weiqi.kids 資料層「收集數據」本機 cron 進入點。
# 純資料流、無 AI：拉 GA4 + GSC → 產 analytics/seo-daily/<台灣日期>.json → commit [skip ci] → push。
# 下游（心跳、大腦層）讀此 JSON。GA4 已通；GSC 待 SA 加入屬性後自動有訊號（失敗只記 error 不中斷）。
#
# 用法：analytics/scripts/seo-collect-cron.sh
# crontab：見 /etc/cron.d/weiqi-seo（每日台灣 04:35，排在心跳/大腦之前）。
set -uo pipefail

export PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin"
export TZ="Asia/Taipei"

# 站別參數（非機密）＋ SA 金鑰路徑（既有 GA4 SA，folk 同一把）。
export GA4_PROPERTY_ID="${GA4_PROPERTY_ID:-458470883}"
export GSC_SITE_URL="${GSC_SITE_URL:-sc-domain:weiqi.kids}"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/root/.config/ga4-insights/sa-key.json}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
mkdir -p logs analytics/seo-daily
DATE="$(date +%F)"

echo "===== [weiqi-collect] $DATE $(date '+%T %Z') 開始 ====="

# 1) 同步 main（避免與其他本機 push / GitHub Action 衝突）
git pull --rebase --autostash origin main 2>&1 || echo "[weiqi-collect] git pull 失敗（續行）"

# 2) 產今日 JSON（任一段失敗只記 error，仍寫檔、退出碼 0）
if ! node analytics/scripts/seo-daily.mjs; then
  echo "[weiqi-collect] ✗ seo-daily.mjs 執行失敗，今日不 commit"
  echo "===== [weiqi-collect] $DATE 結束（失敗）====="
  exit 1
fi

# 3) commit + push（[skip ci] 不觸發部署；資料檔不影響網站）
git add analytics/seo-daily/
if git diff --cached --quiet; then
  echo "[weiqi-collect] 無變更，略過 commit"
else
  git commit -q -m "chore(seo): 每日數據 ${DATE} [skip ci]"
  git pull --rebase --autostash origin main 2>&1 || true
  git push origin main 2>&1 || echo "[weiqi-collect] push 失敗（下次再試）"
fi

echo "===== [weiqi-collect] $DATE $(date '+%T %Z') 結束 ====="
