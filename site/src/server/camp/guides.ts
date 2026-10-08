// 好棋寶寶 MCP 的 get_guide：不支援 skill 的 AI 工具，也能從這裡拿到同樣的引導（ADR 0015）。
import teacherSkill from '../../../plugins/haoqi-teacher/skills/haoqi-teacher/SKILL.md?raw';
import learnerSkill from '../../../plugins/haoqi-learner/skills/haoqi-learner/SKILL.md?raw';
import { EXAMPLES } from '../../lib/showcase-examples';

const REPO_STRUCTURE = `# 好棋寶寶講師 repo 的結構

講師的 repo 放在講師自己的公開 GitHub。學員的 AI 透過好棋寶寶 MCP 讀管理員審過的那個 commit，照著在學員自己的 GitHub 建一份。

\`\`\`
README.md          8 項資料（見下方），也是學員打開 repo 第一個看到的
LICENSE            講師自選的授權；不允許改作也要寫清楚
skills/<名稱>/SKILL.md   給學員 AI 看的步驟：怎麼建一份自己的、要填哪些設定、怎麼改成自己的題目、怎麼確認跑起來
scripts/           下載資料、清理、呼叫 AI、產生網頁的程式
.github/workflows/ 定時更新設定（例如每天跑一次）
site/ 或 templates/ 網站範本
config.example.*   學員要自己填的設定，例如要追的公司名單；不能放真的金鑰
sheets/            技能單張，每張一個檔案，四部分：問題、發生的原因、解決方法有哪些、我建議的流程
\`\`\`

## README 的 8 項資料

1. 成果展示：連結、截圖或短片
2. 用 AI 之前和之後差在哪裡（數字或文字）
3. 學完，學員能做出什麼
4. 適合誰、需要什麼基礎
5. 講師本業，以及為什麼做這個
6. 做法說明：讓學員自己試得到一半
7. 一張試閱的技能單張
8. 自己跑起來要準備什麼：要哪些帳號、大概每月花多少錢、要不要信用卡

## 協會成果的範例（「23 條產業供應鏈情報網」）

${Object.entries({
  成果名稱: EXAMPLES.title, 一句話介紹: EXAMPLES.summary, '1. 成果展示': EXAMPLES.demo_links, '2. 用 AI 前後': EXAMPLES.before_after,
  '3. 學完能做出什麼': EXAMPLES.learner_outcome, '4. 適合誰': EXAMPLES.audience, '4. 需要的基礎': EXAMPLES.prerequisites, '5. 講師本業': EXAMPLES.instructor_bio,
  '6. 做法說明': EXAMPLES.method, '7. 試閱技能單張：問題': EXAMPLES.sample_problem, '7. 發生的原因': EXAMPLES.sample_cause, '7. 解決方法有哪些': EXAMPLES.sample_solutions,
  '7. 我建議的流程': EXAMPLES.sample_flow, '8. 自己跑起來要準備什麼': EXAMPLES.setup_requirements, 授權: EXAMPLES.license,
}).map(([k, v]) => `### ${k}\n${v}`).join('\n\n')}

## 送審前

- 掃描機密：API 金鑰、密碼、token、個人資料、客戶資料都不能進 repo。
- 每個新版本都要講師在網站上送審、管理員審過，學員才會拿到。
`;

const WRITING = `# 寫作規則（台灣好棋寶寶協會）

用台灣繁體中文，直接、具體，寫出真正做過的事和數字。下面這些寫法一律不要用：

- 「不是 X，而是 Y」「並非…而是」「不只是…更是」「不僅…更/還/也」這類下定義或排比
- 「值得注意的是」「值得一提的是」「換句話說」
- 「綜上所述」「總的來說」「總而言之」「整體而言」這類收尾
- 「真正的問題是」「關鍵在於」
- 「隨著…的發展，」「在…的今天」這類開場
- 「至關重要」「不可或缺」「舉足輕重」「賦能」「助力」「底層邏輯」
- 「首先…其次…最後」「讓我們」「不妨」「你是否曾」
- 破折號
- 沒有出處的「研究顯示」「專家認為」「業界普遍」
`;

export const GUIDES: Record<string, string> = { teacher: teacherSkill, learner: learnerSkill, repo_structure: REPO_STRUCTURE, writing: WRITING };
