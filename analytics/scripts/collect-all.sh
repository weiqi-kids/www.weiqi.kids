#!/bin/bash
# 依序跑各管道收集器。單一來源失敗不影響其他來源。
set -u
cd "$(dirname "$0")/../.." || exit 1
# 2026-11-09 站主裁示停掉 YouTube 收集（OAuth 自 2026-09-30 起 invalid_grant，不再授權）。要恢復把 collect-youtube 加回清單並重新授權。
for s in collect-line collect-ga4 collect-gsc; do
  echo "== $(date -u +%FT%TZ) $s"
  /usr/bin/node "analytics/scripts/$s.mjs" || echo "  （$s 失敗，略過）"
done
