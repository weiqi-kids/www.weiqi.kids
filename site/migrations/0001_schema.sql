-- 共學營重建（ADR 0014）。staging 資料庫清空後從這份建立。

-- 帳號與登入（沿用 ADR 0012 的設計）
CREATE TABLE accounts (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  is_admin INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE login_identities (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  provider TEXT NOT NULL CHECK (provider IN ('email', 'line')),
  subject TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  UNIQUE (provider, subject)
);
CREATE INDEX login_identities_account ON login_identities (account_id);
CREATE TABLE magic_tokens (
  token_hash TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  purpose TEXT NOT NULL CHECK (purpose IN ('login', 'link')),
  account_id TEXT,
  redirect_to TEXT,
  expires_at TEXT NOT NULL,
  used_at TEXT,
  revoked_at TEXT,
  ip_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX magic_tokens_email ON magic_tokens (email, created_at);
CREATE INDEX magic_tokens_ip ON magic_tokens (ip_hash, created_at);
CREATE TABLE oauth_states (
  state_hash TEXT PRIMARY KEY,
  purpose TEXT NOT NULL CHECK (purpose IN ('login', 'link')),
  account_id TEXT,
  redirect_to TEXT,
  expires_at TEXT NOT NULL,
  used_at TEXT
);
CREATE TABLE sessions (
  id_hash TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  expires_at TEXT NOT NULL,
  revoked_at TEXT
);
CREATE INDEX sessions_account ON sessions (account_id);

-- 會員：常年會費 6,000 元，會員資格是上架成果的條件。
CREATE TABLE memberships (
  account_id TEXT PRIMARY KEY REFERENCES accounts(id),
  started_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  updated_by TEXT,
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE membership_dues (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  amount INTEGER NOT NULL,
  paid_on TEXT NOT NULL,                -- 匯款日期 YYYY-MM-DD
  account_last5 TEXT NOT NULL,          -- 匯款帳號末五碼，只有管理員看得到
  status TEXT NOT NULL DEFAULT 'submitted' CHECK (status IN ('submitted', 'confirmed', 'rejected')),
  reviewed_by TEXT,
  reviewed_at TEXT,
  review_note TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX membership_dues_status ON membership_dues (status, created_at);

-- 講師成果：7 項必填資料＋開團設定＋TTQS 開課表單。
CREATE TABLE showcases (
  id TEXT PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  owner_id TEXT NOT NULL REFERENCES accounts(id),
  instructor_name TEXT NOT NULL,        -- 顯示在成果頁的講師名稱
  title TEXT NOT NULL DEFAULT '',
  summary TEXT NOT NULL DEFAULT '',
  demo_links TEXT NOT NULL DEFAULT '[]', -- 1 成果展示：JSON [{title, url, note}]
  before_after TEXT NOT NULL DEFAULT '',  -- 2 用 AI 之前和之後差在哪裡
  learner_outcome TEXT NOT NULL DEFAULT '', -- 3 學完能做出什麼
  audience TEXT NOT NULL DEFAULT '',      -- 4 適合誰
  prerequisites TEXT NOT NULL DEFAULT '', -- 4 需要什麼基礎
  instructor_bio TEXT NOT NULL DEFAULT '', -- 5 本業與為什麼做這個
  method TEXT NOT NULL DEFAULT '',        -- 6 做法說明（Markdown）
  sample_problem TEXT NOT NULL DEFAULT '', -- 7 試閱技能單張：問題
  sample_cause TEXT NOT NULL DEFAULT '',   --   發生的原因
  sample_solutions TEXT NOT NULL DEFAULT '', -- 解決方法有哪些
  sample_flow TEXT NOT NULL DEFAULT '',    --   我建議的流程
  -- 開團設定
  price INTEGER NOT NULL DEFAULT 6000 CHECK (price >= 6000),
  teaching_format TEXT NOT NULL DEFAULT 'online' CHECK (teaching_format IN ('online', 'onsite')),
  location TEXT NOT NULL DEFAULT '',
  schedule_mode TEXT NOT NULL DEFAULT 'slots' CHECK (schedule_mode IN ('slots', 'vote', 'fixed')),
  fixed_start TEXT,                     -- 預先訂好的主題教學時間（UTC ISO）
  teaching_minutes INTEGER NOT NULL DEFAULT 120,
  group_rule TEXT NOT NULL DEFAULT 'paid' CHECK (group_rule IN ('paid', 'signup')),
  min_size INTEGER NOT NULL DEFAULT 5 CHECK (min_size >= 1),
  max_size INTEGER,
  pay_days INTEGER NOT NULL DEFAULT 7 CHECK (pay_days BETWEEN 1 AND 60),
  vote_days INTEGER NOT NULL DEFAULT 3 CHECK (vote_days BETWEEN 1 AND 30),
  forum_open INTEGER NOT NULL DEFAULT 1,
  -- TTQS 開課表單
  ttqs_needs TEXT NOT NULL DEFAULT '',
  ttqs_goals TEXT NOT NULL DEFAULT '',
  ttqs_outline TEXT NOT NULL DEFAULT '',
  ttqs_hours TEXT NOT NULL DEFAULT '',
  ttqs_methods TEXT NOT NULL DEFAULT '',
  ttqs_evaluation TEXT NOT NULL DEFAULT '',
  ttqs_expected TEXT NOT NULL DEFAULT '',
  -- 審核
  status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'submitted', 'changes_requested', 'published', 'archived')),
  review_note TEXT,
  reviewed_by TEXT,
  reviewed_at TEXT,
  published_at TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX showcases_owner ON showcases (owner_id);
CREATE INDEX showcases_status ON showcases (status, published_at);

-- 講師先填的每週時段（台北時間）。
CREATE TABLE showcase_slots (
  id TEXT PRIMARY KEY,
  showcase_id TEXT NOT NULL REFERENCES showcases(id),
  weekday INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6), -- 0 = 週日
  start_time TEXT NOT NULL,                                 -- HH:MM
  active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX showcase_slots_showcase ON showcase_slots (showcase_id);

-- 一團。建立時把成果當下的開團設定複製一份，之後講師改設定不影響進行中的團。
CREATE TABLE cohorts (
  id TEXT PRIMARY KEY,
  showcase_id TEXT NOT NULL REFERENCES showcases(id),
  slot_id TEXT REFERENCES showcase_slots(id),
  seq INTEGER NOT NULL,                 -- 這個成果的第幾團
  state TEXT NOT NULL DEFAULT 'gathering' CHECK (state IN ('gathering', 'scheduling', 'payment', 'confirmed', 'running', 'ended', 'unfilled', 'cancelled')),
  price INTEGER,
  teaching_format TEXT,
  location TEXT,
  schedule_mode TEXT,
  group_rule TEXT,
  min_size INTEGER,
  max_size INTEGER,
  pay_days INTEGER,
  vote_days INTEGER,
  teaching_minutes INTEGER,
  threshold_at TEXT,                    -- 滿開團人數的時間
  propose_by TEXT,                      -- 投票方式：講師提出候選日期的期限
  vote_closes_at TEXT,
  teaching_start TEXT,
  teaching_end TEXT,
  meeting_url TEXT,
  pay_deadline TEXT,
  confirmed_at TEXT,
  ends_at TEXT,                         -- 主題教學後滿 1 個月
  closed_at TEXT,                       -- 結營、流團或取消的時間
  close_reason TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX cohorts_showcase ON cohorts (showcase_id, state);
CREATE INDEX cohorts_state ON cohorts (state);

-- 登記想學：屬於一個登記中的團。
CREATE TABLE interests (
  cohort_id TEXT NOT NULL REFERENCES cohorts(id),
  account_id TEXT NOT NULL REFERENCES accounts(id),
  carried INTEGER NOT NULL DEFAULT 0,   -- 上一團額滿自動帶過來的
  overflow INTEGER NOT NULL DEFAULT 0,  -- 想報名時已額滿，下一團自動帶過去
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (cohort_id, account_id)
);
CREATE INDEX interests_account ON interests (account_id);

-- 投票方式：講師提出的候選日期與登記者的投票。
CREATE TABLE vote_options (
  id TEXT PRIMARY KEY,
  cohort_id TEXT NOT NULL REFERENCES cohorts(id),
  starts_at TEXT NOT NULL
);
CREATE TABLE votes (
  option_id TEXT NOT NULL REFERENCES vote_options(id),
  account_id TEXT NOT NULL REFERENCES accounts(id),
  PRIMARY KEY (option_id, account_id)
);

-- 報名與匯款。匯款日期與末五碼只有管理員看得到。
CREATE TABLE enrollments (
  id TEXT PRIMARY KEY,
  code TEXT NOT NULL UNIQUE,            -- 報名編號，匯款附言用
  cohort_id TEXT NOT NULL REFERENCES cohorts(id),
  account_id TEXT NOT NULL REFERENCES accounts(id),
  amount INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'awaiting_payment' CHECK (status IN ('awaiting_payment', 'submitted', 'confirmed', 'refund_pending', 'refunded', 'cancelled')),
  paid_on TEXT,
  account_last5 TEXT,
  submitted_at TEXT,
  confirmed_by TEXT,
  confirmed_at TEXT,
  refund_reason TEXT,
  refunded_by TEXT,
  refunded_at TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  UNIQUE (cohort_id, account_id)
);
CREATE INDEX enrollments_status ON enrollments (status, submitted_at);
CREATE INDEX enrollments_account ON enrollments (account_id);

-- 主題教學簽到（TTQS 佐證）。
CREATE TABLE attendance (
  cohort_id TEXT NOT NULL REFERENCES cohorts(id),
  account_id TEXT NOT NULL REFERENCES accounts(id),
  checked_in_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (cohort_id, account_id)
);

-- 3 週實作每週繳交：1 題目、2 第一版、3 成果。
CREATE TABLE practice_submissions (
  cohort_id TEXT NOT NULL REFERENCES cohorts(id),
  account_id TEXT NOT NULL REFERENCES accounts(id),
  week INTEGER NOT NULL CHECK (week BETWEEN 1 AND 3),
  post_id TEXT NOT NULL,
  submitted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (cohort_id, account_id, week)
);

-- 已寄出的提醒，避免重複寄。
CREATE TABLE reminders_sent (
  key TEXT PRIMARY KEY,
  sent_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 技能單張：跟著成果，每團沿用；每次修改留版本。
CREATE TABLE materials (
  id TEXT PRIMARY KEY,
  showcase_id TEXT NOT NULL REFERENCES showcases(id),
  title TEXT NOT NULL,
  problem TEXT NOT NULL,
  cause TEXT NOT NULL,
  solutions TEXT NOT NULL,
  flow TEXT NOT NULL,
  position INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1,
  archived INTEGER NOT NULL DEFAULT 0,
  updated_by TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX materials_showcase ON materials (showcase_id, position);
CREATE TABLE material_versions (
  material_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  body TEXT NOT NULL,                   -- JSON {title, problem, cause, solutions, flow}
  edited_by TEXT NOT NULL,
  edited_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (material_id, version)
);

-- 論壇：space = 'showcase:{id}'（公開討論區）或 'cohort:{id}'（課程論壇）。
CREATE TABLE posts (
  id TEXT PRIMARY KEY,
  space TEXT NOT NULL,
  author_id TEXT NOT NULL REFERENCES accounts(id),
  parent_id TEXT REFERENCES posts(id),
  title TEXT,
  body TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'published' CHECK (status IN ('published', 'withdrawn', 'hidden')),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX posts_space ON posts (space, parent_id, created_at);
CREATE TABLE post_revisions (
  post_id TEXT NOT NULL,
  revised_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  title TEXT,
  body TEXT NOT NULL,
  revised_by TEXT NOT NULL
);
CREATE TABLE attachments (
  id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL REFERENCES posts(id),
  space TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  r2_key TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  size INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  deleted_at TEXT
);
CREATE INDEX attachments_post ON attachments (post_id);
CREATE TABLE moderation_actions (
  id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL REFERENCES posts(id),
  actor_id TEXT NOT NULL,
  actor_role TEXT NOT NULL CHECK (actor_role IN ('instructor', 'admin')),
  action TEXT NOT NULL CHECK (action IN ('hide', 'restore')),
  reason TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE appeals (
  id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL REFERENCES posts(id),
  author_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'upheld', 'restored')),
  resolved_by TEXT,
  resolved_at TEXT,
  resolution_note TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 學員作品推薦：講師認可＋作者同意公開，兩者都有才出現在成果頁。
CREATE TABLE showcase_works (
  post_id TEXT PRIMARY KEY REFERENCES posts(id),
  showcase_id TEXT NOT NULL REFERENCES showcases(id),
  approved_by TEXT,
  approved_at TEXT,
  author_consent_at TEXT
);
CREATE INDEX showcase_works_showcase ON showcase_works (showcase_id);

-- 結營問卷（TTQS 反應評估）。
CREATE TABLE surveys (
  cohort_id TEXT NOT NULL REFERENCES cohorts(id),
  account_id TEXT NOT NULL REFERENCES accounts(id),
  overall INTEGER NOT NULL CHECK (overall BETWEEN 1 AND 5),
  instructor INTEGER NOT NULL CHECK (instructor BETWEEN 1 AND 5),
  materials INTEGER NOT NULL CHECK (materials BETWEEN 1 AND 5),
  practice INTEGER NOT NULL CHECK (practice BETWEEN 1 AND 5),
  learned TEXT NOT NULL,
  suggestion TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (cohort_id, account_id)
);

-- 講師分潤：結營時算出，管理員匯出後標記。
CREATE TABLE payouts (
  cohort_id TEXT PRIMARY KEY REFERENCES cohorts(id),
  instructor_id TEXT NOT NULL REFERENCES accounts(id),
  gross INTEGER NOT NULL,
  amount INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'due' CHECK (status IN ('due', 'paid')),
  paid_by TEXT,
  paid_at TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 站內通知與網頁推播訂閱。
CREATE TABLE notifications (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  message TEXT NOT NULL,
  link TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  read_at TEXT
);
CREATE INDEX notifications_account ON notifications (account_id, created_at);
CREATE TABLE push_subscriptions (
  endpoint TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  p256dh TEXT NOT NULL,
  auth TEXT NOT NULL,
  user_agent TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX push_subscriptions_account ON push_subscriptions (account_id);

CREATE TABLE audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor_id TEXT,
  action TEXT NOT NULL,
  target TEXT,
  detail TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
