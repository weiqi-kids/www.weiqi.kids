#!/bin/bash
# ============================================================================
# Traffic Analysis Script for Claude CLI
# ============================================================================
# Analyzes website traffic data collected by GitHub Actions workflow.
# Outputs formatted report for use in revamp/1-discovery phase.
#
# Usage:
#   ./revamp/tools/analyze-traffic.sh
#
# Prerequisites:
#   - jq installed (brew install jq)
#   - analytics/ directory with collected data
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANALYTICS_DIR="$PROJECT_ROOT/analytics"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check dependencies
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed.${NC}"
    echo "Install with: brew install jq"
    exit 1
fi

echo "=============================================="
echo "        網站流量分析報告"
echo "        Website Traffic Analysis Report"
echo "=============================================="
echo ""
echo "生成時間 Generated: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================================
# Popular Pages Analysis
# ============================================================================
echo -e "${BLUE}## 熱門頁面 Popular Pages (過去 14 天)${NC}"
echo ""

if [ -f "$ANALYTICS_DIR/current/popular-paths.json" ]; then
    PAGE_COUNT=$(jq 'length' "$ANALYTICS_DIR/current/popular-paths.json")

    if [ "$PAGE_COUNT" -gt 0 ]; then
        echo "| 排名 | 頁面路徑 | 瀏覽數 | 獨立訪客 |"
        echo "|------|----------|--------|----------|"

        jq -r 'to_entries | .[] | "| \(.key + 1) | \(.value.path) | \(.value.count) | \(.value.uniques) |"' \
            "$ANALYTICS_DIR/current/popular-paths.json"
    else
        echo "(無數據 No data available)"
    fi
else
    echo -e "${YELLOW}(尚未收集數據 - 請先執行 GitHub Action)${NC}"
fi

echo ""

# ============================================================================
# Referrer Analysis
# ============================================================================
echo -e "${BLUE}## 流量來源 Traffic Sources${NC}"
echo ""

if [ -f "$ANALYTICS_DIR/current/referrers.json" ]; then
    REF_COUNT=$(jq 'length' "$ANALYTICS_DIR/current/referrers.json")

    if [ "$REF_COUNT" -gt 0 ]; then
        echo "| 排名 | 來源 | 訪問數 | 獨立訪客 |"
        echo "|------|------|--------|----------|"

        jq -r 'to_entries | .[] | "| \(.key + 1) | \(.value.referrer) | \(.value.count) | \(.value.uniques) |"' \
            "$ANALYTICS_DIR/current/referrers.json"
    else
        echo "(無外部流量來源 No referrer data)"
    fi
else
    echo -e "${YELLOW}(尚未收集數據 - 請先執行 GitHub Action)${NC}"
fi

echo ""

# ============================================================================
# Historical Trends
# ============================================================================
echo -e "${BLUE}## 瀏覽趨勢 View Trends${NC}"
echo ""

if [ -f "$ANALYTICS_DIR/history/daily-views.json" ]; then
    TOTAL_DAYS=$(jq '.views | length' "$ANALYTICS_DIR/history/daily-views.json")

    if [ "$TOTAL_DAYS" -gt 0 ]; then
        echo "### 最近 7 天 Last 7 Days"
        echo ""
        echo "| 日期 | 瀏覽數 | 獨立訪客 |"
        echo "|------|--------|----------|"

        jq -r '.views | .[-7:] | .[] | "| \(.timestamp | split("T")[0]) | \(.count) | \(.uniques) |"' \
            "$ANALYTICS_DIR/history/daily-views.json"

        echo ""

        # Calculate totals
        TOTAL_VIEWS=$(jq '[.views[].count] | add // 0' "$ANALYTICS_DIR/history/daily-views.json")
        TOTAL_UNIQUES=$(jq '[.views[].uniques] | add // 0' "$ANALYTICS_DIR/history/daily-views.json")
        LAST_UPDATED=$(jq -r '.lastUpdated' "$ANALYTICS_DIR/history/daily-views.json")

        echo "### 累計統計 Cumulative Stats"
        echo ""
        echo "- 總瀏覽數 Total Views: $TOTAL_VIEWS"
        echo "- 總獨立訪客 Total Unique Visitors: $TOTAL_UNIQUES"
        echo "- 數據天數 Days of Data: $TOTAL_DAYS"
        echo "- 最後更新 Last Updated: $LAST_UPDATED"
    else
        echo "(無歷史數據 No historical data)"
    fi
else
    echo -e "${YELLOW}(尚未收集歷史數據)${NC}"
fi

echo ""

# ============================================================================
# Clone Statistics
# ============================================================================
echo -e "${BLUE}## Clone 統計 (開發者關注度)${NC}"
echo ""

if [ -f "$ANALYTICS_DIR/history/daily-clones.json" ]; then
    CLONE_DAYS=$(jq '.clones | length' "$ANALYTICS_DIR/history/daily-clones.json")

    if [ "$CLONE_DAYS" -gt 0 ]; then
        TOTAL_CLONES=$(jq '[.clones[].count] | add // 0' "$ANALYTICS_DIR/history/daily-clones.json")
        UNIQUE_CLONERS=$(jq '[.clones[].uniques] | add // 0' "$ANALYTICS_DIR/history/daily-clones.json")

        echo "- 總 Clone 數: $TOTAL_CLONES"
        echo "- 獨立 Cloner 數: $UNIQUE_CLONERS"
    else
        echo "(無 clone 數據)"
    fi
else
    echo -e "${YELLOW}(尚未收集數據)${NC}"
fi

echo ""

# ============================================================================
# Data Quality Check
# ============================================================================
echo -e "${BLUE}## 數據品質檢查 Data Quality${NC}"
echo ""

RAW_FILES=$(find "$ANALYTICS_DIR/raw" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
echo "- 原始數據檔案數: $RAW_FILES"

if [ -d "$ANALYTICS_DIR/current" ]; then
    CURRENT_FILES=$(ls "$ANALYTICS_DIR/current"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "- 當前快照檔案數: $CURRENT_FILES"
fi

if [ -d "$ANALYTICS_DIR/history" ]; then
    HISTORY_FILES=$(ls "$ANALYTICS_DIR/history"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "- 歷史數據檔案數: $HISTORY_FILES"
fi

echo ""

# ============================================================================
# Insights & Recommendations
# ============================================================================
echo -e "${BLUE}## 洞察與建議 Insights${NC}"
echo ""

if [ -f "$ANALYTICS_DIR/current/popular-paths.json" ]; then
    TOP_PAGE=$(jq -r '.[0].path // "N/A"' "$ANALYTICS_DIR/current/popular-paths.json")
    if [ "$TOP_PAGE" != "N/A" ] && [ "$TOP_PAGE" != "null" ]; then
        echo "1. 最熱門頁面: $TOP_PAGE"
        echo "   - 建議：確保此頁面內容最新、SEO 優化完善"
    fi
fi

if [ -f "$ANALYTICS_DIR/current/referrers.json" ]; then
    TOP_REF=$(jq -r '.[0].referrer // "N/A"' "$ANALYTICS_DIR/current/referrers.json")
    if [ "$TOP_REF" != "N/A" ] && [ "$TOP_REF" != "null" ]; then
        echo "2. 主要流量來源: $TOP_REF"
        echo "   - 建議：加強與此渠道的連結，維持流量來源多樣性"
    fi
fi

echo ""
echo "=============================================="
echo "報告結束 End of Report"
echo "=============================================="
