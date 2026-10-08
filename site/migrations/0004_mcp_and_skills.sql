-- 好棋寶寶講師 skill、學員 skill 與 MCP（ADR 0015）。

-- 第 8 項必填與講師的 GitHub repo。
ALTER TABLE showcases ADD COLUMN setup_requirements TEXT NOT NULL DEFAULT '';  -- 自己跑起來要準備什麼
ALTER TABLE showcases ADD COLUMN repo_url TEXT NOT NULL DEFAULT '';
ALTER TABLE showcases ADD COLUMN license TEXT NOT NULL DEFAULT '';             -- 授權說明（講師自選）
ALTER TABLE showcases ADD COLUMN allow_derivative INTEGER NOT NULL DEFAULT 1; -- 學員可否公開改作版本

-- 講師 repo 的每個版本：講師送審、管理員審過才給學員。
CREATE TABLE skill_versions (
  id TEXT PRIMARY KEY,
  showcase_id TEXT NOT NULL REFERENCES showcases(id),
  repo_url TEXT NOT NULL,
  commit_sha TEXT NOT NULL,
  note TEXT,
  status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'submitted', 'approved', 'rejected')),
  review_summary TEXT,                  -- 自動整理：變更檔案、定時更新設定、連到的網站（JSON）
  review_note TEXT,
  reviewed_by TEXT,
  reviewed_at TEXT,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX skill_versions_showcase ON skill_versions (showcase_id, created_at);

-- 學員的問題對不到成果時，經本人同意記下來給講師們看。
CREATE TABLE learning_needs (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  problem TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- AI 透過 MCP 存的草稿，本人在網站上按「確定送出」才會發出。
CREATE TABLE drafts (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  kind TEXT NOT NULL CHECK (kind IN ('question', 'practice', 'work', 'reply')),
  space TEXT NOT NULL,                  -- 要發到哪個論壇
  parent_id TEXT,                       -- 回覆的對象
  week INTEGER,                         -- 實作第幾週
  title TEXT,
  body TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'discarded')),
  post_id TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  sent_at TEXT
);
CREATE INDEX drafts_account ON drafts (account_id, status, created_at);
