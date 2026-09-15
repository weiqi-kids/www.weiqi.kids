# Triage labels

| 技能中的角色 | GitHub 標籤 | 意義 |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | 等待維護者分類 |
| `needs-info` | `needs-info` | 等待補充資訊 |
| `ready-for-agent` | `ready-for-agent` | 規格完整，可交由 agent 實作 |
| `ready-for-human` | `ready-for-human` | 需要人工實作 |
| `wontfix` | `wontfix` | 不處理 |

技能提及角色時，使用表中的 GitHub 標籤。新規格與拆分工作票預設使用 `ready-for-agent`。
此檔定義標籤對應；首次套用前用 `gh label list` 確認遠端已有標籤，缺少時建立對應標籤，保留既有標籤的顏色與說明。
