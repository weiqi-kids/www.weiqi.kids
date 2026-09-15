# Domain docs

本專案使用 single-context 結構。

## 探索程式碼前

- 若根目錄 `CONTEXT.md` 存在，讀取其中的領域詞彙與模型。
- 若 `docs/adr/` 存在，讀取與本次修改相關的架構決策紀錄。
- 文件尚未建立時繼續工作；由 `domain-modeling` 在詞彙或決策明確後建立。

## 文件位置與使用規則

- 詞彙與領域模型：根目錄 `CONTEXT.md`。
- 架構決策：`docs/adr/NNNN-簡短名稱.md`，編號接續既有紀錄。
- Issue、規格、測試與重構提案使用 `CONTEXT.md` 已定義的詞彙。
- 新方案若與既有 ADR 衝突，指出相應紀錄及重新討論的理由。

`docs/agents/` 與 `docs/adr/` 是工程文件，已從 Docusaurus 文件收集範圍排除。
