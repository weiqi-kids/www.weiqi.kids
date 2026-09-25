# Email Magic Link 登入規格（規劃版）

版本：0.1（2026-09-20）  
狀態：登入流程草案；尚未建立 Worker、D1 或 Email Service。

## 1. 流程

1. 使用者在登入頁輸入 Email。
2. Worker 對請求做基本限流與濫用檢查。
3. Worker 產生高熵隨機 token，只保存 token hash、帳號候選、用途、到期時間與使用狀態。
4. Worker 使用 Cloudflare Email Service `send_email` binding 寄出一次性登入連結。
5. 使用者點擊連結，Worker 檢查 token hash、用途、到期時間與 `used_at`。
6. 驗證成功後將 token 標記為已使用，建立網站 session，並導向原本要求的頁面。
7. Email 登入不會自動把帳號與 LINE 身分合併；綁定必須在已登入狀態下由使用者明確確認。

## 2. 預設安全規則

| 項目 | 預設規格 | 原因 |
| --- | --- | --- |
| 連結有效期限 | 15 分鐘 | 降低連結外洩後的可用時間 |
| 使用次數 | 一次 | 驗證成功或明確撤銷後立即失效 |
| 同一 Email 重發 | 新連結使舊連結失效；短時間內限制重發 | 避免多個有效 token 與濫用寄信 |
| Token 保存 | 只保存 hash，不保存明文 token | D1 泄漏時降低直接登入風險 |
| Session | HttpOnly、Secure、SameSite cookie；伺服器端可撤銷 | 不把登入狀態放在可被前端讀取的 localStorage |
| 登入錯誤 | 不透露 Email 是否已有帳號 | 避免帳號枚舉 |
| 記錄 | 保存申請、寄送、成功、失敗與撤銷事件 | 支援客服與安全稽核 |
| 限流 | 依 IP、Email hash、裝置／session 綜合限制 | 防止大量寄信與 token 猜測 |

## 3. 帳號與 LINE 綁定

- Email 登入成功後，只能進入已綁定該 Email 的帳號，或依明確的註冊流程建立新帳號。
- LINE 登入使用 LINE provider subject 辨識，不用顯示名稱或 Email 自動合併。
- 將 LINE 綁定到已登入的 Email 帳號時，要求目前 session 仍有效，並再次確認操作。
- 發現 Email 或 LINE 已被其他帳號使用時，停止自動合併，交由協會管理流程處理。
- 綁定、解除綁定與帳號恢復都留下安全事件紀錄。

## 4. Cloudflare 配置前提

- 網域使用 Cloudflare DNS。
- 啟用 Cloudflare Email Service Email Sending，設定寄件網域與 SPF／DKIM／DMARC。
- Worker 使用 `send_email` binding；寄件失敗要回傳可理解的重試訊息，不顯示內部錯誤。
- Email Service Beta 的配額、退信、抑制名單與事件記錄在 staging 先驗證。
- Magic Link 的 token 與 session 資料放 D1；信件內容不包含付款資料或其他私密資訊。

## 5. 驗收

- 同一連結第二次使用必定失效。
- 過期連結失效，且不會建立 session。
- 重新申請後，舊連結失效。
- 不存在的 Email 與已存在的 Email 對外回應一致。
- Email 登入與 LINE 登入在明確綁定後指向同一帳號。
- 未完成綁定時，兩種登入不會共用課程或實作營權限。
- 寄信服務暫時不可用時，不會產生可登入但使用者收不到的有效狀態。

