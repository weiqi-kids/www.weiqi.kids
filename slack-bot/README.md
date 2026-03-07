# Weiqi.Kids Slack Bot

透過 Slack 頻道與 Claude CLI 互動，實現對話式網站維護。

## 系統需求

- Node.js 18+
- Claude CLI（已安裝並認證）
- Slack App（Socket Mode）

## 安裝步驟

### 1. 安裝相依套件

```bash
cd slack-bot
npm install
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env 填入 Slack tokens
```

### 3. 啟動 Bot

```bash
# 直接啟動
npm start

# 使用 pm2 背景運行
pm2 start server.js --name weiqi-slack-bot

# 設定開機自動啟動
pm2 startup
pm2 save
```

## Slack App 設定

1. 前往 https://api.slack.com/apps 建立新 App
2. 開啟 **Socket Mode**
3. 產生 **App-Level Token**（`connections:write` scope）
4. 前往 **OAuth & Permissions**，加入：
   - `chat:write`
   - `channels:history`
   - `groups:history`（私密頻道必要）
5. 安裝 App 到 Workspace
6. 前往 **Event Subscriptions** → 加入 `message.groups`
7. 將 Bot 邀請加入維護頻道

## 使用範例

在 Slack 頻道輸入：

```
把張饒輝的職稱改成「圍棋人科技股份有限公司 技術顧問」，然後部署
```

Bot 會：
1. 修改相關檔案
2. Git commit & push
3. 執行部署
4. 回報結果

## 監控與維護

```bash
# 查看狀態
pm2 status

# 查看日誌
pm2 logs weiqi-slack-bot

# 重啟
pm2 restart weiqi-slack-bot
```

## 安全性

- 限制可使用的 Slack 使用者（`ALLOWED_USERS`）
- 限制 Claude 可用工具（禁止 rm -rf、sudo 等）
- 使用檔案鎖避免並發衝突
- 輸入驗證防止危險指令
