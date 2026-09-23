#!/bin/bash
# 依序跑各管道收集器。單一來源失敗不影響其他來源。
set -u
cd "$(dirname "$0")/../.." || exit 1
for s in collect-youtube collect-line collect-ga4 collect-gsc; do
  echo "== $(date -u +%FT%TZ) $s"
  /usr/bin/node "analytics/scripts/$s.mjs" || echo "  （$s 失敗，略過）"
done
