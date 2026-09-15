# Matt Pocock 工程技能

## 專案設定

使用工程技能前，讀取同目錄的 `issue-tracker.md`、`triage-labels.md` 與 `domain.md`。
此專案已完成 setup；後續可直接修改這三份設定。

## Codex 使用方式

在 Codex 對話中使用 `$技能名稱` 加上任務描述，或從 `/skills` 選擇技能。
技能文件中的 `/setup-matt-pocock-skills`、`/tdd` 等寫法代表技能名稱；執行時載入對應 `SKILL.md`。

本環境使用 `/root/.agents/skills/<技能名稱>/SKILL.md`，安裝來源為 `mattpocock/skills`。
若有 `/root/.codex/skills/` 同名副本，以前述 `.agents` 版本為準。
其他環境先從使用者的 `~/.agents/skills/` 或專案 `.agents/skills/` 尋找。
選單找不到時，直接要求 Codex 讀取上述完整檔案路徑並依照內容執行；相對參考路徑以該技能目錄解析。

| 技能 | 輸入與用途 |
| --- | --- |
| `setup-matt-pocock-skills` | 切換 tracker 或重新設定專案 |
| `grill-with-docs` | 提供設計構想；載入 `grilling` 與 `domain-modeling`，提問並記錄決策 |
| `to-spec` | 將現有討論整理成規格並發布到 tracker |
| `to-tickets` | 提供規格或 issue；拆成有依賴關係的工作票 |
| `implement` | 提供規格或工作票；載入所需的 `tdd`、`code-review`，實作並提交 |
| `code-review` | 提供比較基準（例如 `main`）與規格來源；審查規範與需求符合度 |

`grill-with-docs` 提及的 Skill tool，在 Codex 中以讀取兩個相應技能文件並遵循其流程實現。
`code-review` 的三點比較針對已提交的分支變更；審查未提交工作時，需在任務中明確指定。
