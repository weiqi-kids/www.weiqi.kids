-- 講師先有成果，學員做不來就登記想學，登記人數到講師訂的開團人數才開團（ADR 0013）。
CREATE TABLE interests (
  topic_slug TEXT NOT NULL,            -- 講師成果（content: topics）
  account_id TEXT NOT NULL REFERENCES accounts(id),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (topic_slug, account_id)
);
CREATE INDEX interests_account ON interests (account_id);

-- 同一成果滿額只通知管理員一次。
CREATE TABLE interest_thresholds (
  topic_slug TEXT PRIMARY KEY,
  reached_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 講師資格改為「先有成果」：申請時附上自己做出來的成果。
ALTER TABLE instructor_applications ADD COLUMN results TEXT;
