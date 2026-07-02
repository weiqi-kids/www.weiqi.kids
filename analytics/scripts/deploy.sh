#!/usr/bin/env bash
# weiqi.kids 發佈腳本：pnpm build → 把 build/ 推上 gh-pages（用 origin 乾淨 URL + gh 憑證助手）。
#
# 為何不用 `GIT_USER=weiqi-kids pnpm run deploy`：docusaurus deploy 會把遠端改寫成
#   https://weiqi-kids@github.com/...，繞開本機 gh 憑證助手 → headless 無密碼、push 失敗
#   （fatal: could not read Password）。直接推 build/ 到 gh-pages 用 origin URL 才能無人值守運作。
# build 兼作 gate：build 失敗即不發佈、回非零，呼叫端據此處理。
set -uo pipefail
export PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
ORIGIN="$(git config --get remote.origin.url)"

echo "[deploy] pnpm install…"
pnpm install --frozen-lockfile >/dev/null 2>&1 || { echo "[deploy] install 失敗"; exit 1; }
echo "[deploy] pnpm build…"
pnpm build || { echo "[deploy] build 失敗，不發佈"; exit 1; }

cd build || { echo "[deploy] 無 build/"; exit 1; }
rm -rf .git
git init -q
git checkout -q -b gh-pages
git add -A
git -c user.email="deploy@weiqi.kids" -c user.name="weiqi-kids-deploy" commit -qm "Deploy $(date -u '+%F %T') [skip ci]"
if git push -f "$ORIGIN" gh-pages:gh-pages 2>&1 | tail -3; then
  rm -rf .git
  echo "[deploy] ✅ 已發佈到 gh-pages"
else
  rm -rf .git
  echo "[deploy] ✗ gh-pages 推送失敗"
  exit 1
fi
