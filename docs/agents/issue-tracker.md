# Issue tracker: GitHub

本專案的規格與工作票存放在 `weiqi-kids/www.weiqi.kids` 的 GitHub Issues，使用 `gh` CLI 操作。

## 操作慣例

以下指令從本專案根目錄執行，由 Git remote 決定 repository；在其他目錄執行時，加上 `--repo weiqi-kids/www.weiqi.kids`。

- 建立：`gh issue create --title "標題" --body-file /tmp/issue-body.md`。先將完整 Markdown 寫入檔案，保留實際換行。
- 讀取：`gh issue view <number> --json number,title,body,labels,comments,state`。
- 列表：`gh issue list --state open --json number,title,body,labels`，依任務加入 `--label` 等篩選。
- 留言：`gh issue comment <number> --body-file /tmp/issue-comment.md`。
- 標籤：`gh issue edit <number> --add-label "label"` 或 `--remove-label "label"`，名稱見 `triage-labels.md`。
- 關閉：`gh issue close <number>`。

技能要求「publish to the issue tracker」時建立 GitHub issue；要求讀取 ticket 時取得完整內容、標籤與留言。
拆票時依依賴順序建立，每張工作票包含驗收條件及 `Blocked by`。
平台支援時使用原生 issue dependency；呼叫 API 時使用 blocker 的 database ID，並保留文字依賴方便閱讀。若不可用，以 `Blocked by: #編號` 表達。

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub issue 與 PR 共用編號空間；遇到無法辨別的編號，先確認類型再選擇對應的 `gh issue` 或 `gh pr` 操作。
