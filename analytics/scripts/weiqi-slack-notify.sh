#!/usr/bin/env bash
# weiqi.kids 專屬 Slack 通報小工具（本機 cron / headless 流程用）。
# 沿用「好棋寶寶 Claude 助手」bot（此 App 本就是 weiqi 品牌）；與其他站分離只在頻道不同。
#
# 用法：
#   analytics/scripts/weiqi-slack-notify.sh <CHANNEL_ID> "訊息（mrkdwn）"
#   echo "多行訊息" | analytics/scripts/weiqi-slack-notify.sh <CHANNEL_ID>
#
# Token 來源（優先序）：
#   1. 環境變數 SLACK_BOT_TOKEN
#   2. /root/.config/weiqi-kids/slack-bot-token（若日後獨立 bot）
#   3. /root/.config/folk-tw/slack-bot-token（共用「好棋寶寶」bot；目前用這個）
# Bot 需 chat:write（有 chat:write.public 則發公開頻道免邀請）。
# 設計：缺 token / 失敗都「不中斷呼叫端」，印警告並回非零，呼叫端應以 `|| true` 包住。
set -uo pipefail

CHANNEL="${1:-}"
if [ -z "$CHANNEL" ]; then
  echo "[weiqi-slack] 用法：weiqi-slack-notify.sh <CHANNEL_ID> \"訊息\"（或 stdin）" >&2
  exit 2
fi
shift
if [ $# -gt 0 ]; then TEXT="$*"; else TEXT="$(cat)"; fi
if [ -z "${TEXT//[$'\t\r\n ']/}" ]; then
  echo "[weiqi-slack] 訊息為空，略過" >&2
  exit 2
fi

TOKEN="${SLACK_BOT_TOKEN:-}"
if [ -z "$TOKEN" ]; then
  for f in /root/.config/weiqi-kids/slack-bot-token /root/.config/folk-tw/slack-bot-token; do
    if [ -r "$f" ]; then TOKEN="$(tr -d ' \t\r\n' < "$f")"; break; fi
  done
fi
if [ -z "$TOKEN" ]; then
  echo "[weiqi-slack] 找不到 Slack token（SLACK_BOT_TOKEN 或 token 檔），略過發送" >&2
  exit 1
fi

RESP="$(curl -sS -X POST https://slack.com/api/chat.postMessage \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data "$(node -e 'const c=process.argv[1],t=process.argv[2];process.stdout.write(JSON.stringify({channel:c,text:t,unfurl_links:false,unfurl_media:false}))' "$CHANNEL" "$TEXT")" 2>&1)"

if printf '%s' "$RESP" | grep -q '"ok":true'; then
  TS="$(printf '%s' "$RESP" | grep -oE '"ts":"[0-9.]+"' | head -1 | cut -d'"' -f4)"
  echo "[weiqi-slack] ✅ 已發送到 $CHANNEL（ts=$TS）"
else
  echo "[weiqi-slack] ❌ 發送失敗：$RESP" >&2
  exit 1
fi
