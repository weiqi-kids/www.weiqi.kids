-- 第三期：帳號、登入、會員、課程資格、實作營論壇（ADR 0012）。
CREATE TABLE accounts (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  is_admin INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 同一帳號可有 Email 與 LINE 兩種登入方式；不依 Email 或名稱自動合併。
CREATE TABLE login_identities (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  provider TEXT NOT NULL CHECK (provider IN ('email', 'line')),
  subject TEXT NOT NULL,               -- email：小寫 Email；line：LINE userId
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  UNIQUE (provider, subject)
);
CREATE INDEX login_identities_account ON login_identities (account_id);

-- Email 一次性登入連結：只存 token hash；15 分鐘、單次使用、重發使舊連結失效。
CREATE TABLE magic_tokens (
  token_hash TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  purpose TEXT NOT NULL CHECK (purpose IN ('login', 'link')),
  account_id TEXT,                     -- purpose=link 時為要綁定 Email 的帳號
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

-- 一年一續會；到期只影響新課報名。
CREATE TABLE memberships (
  account_id TEXT PRIMARY KEY REFERENCES accounts(id),
  started_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  updated_by TEXT,
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

ALTER TABLE enrollments ADD COLUMN account_id TEXT;
ALTER TABLE enrollments ADD COLUMN reviewed_by TEXT;
ALTER TABLE enrollments ADD COLUMN reviewed_at TEXT;
ALTER TABLE enrollments ADD COLUMN review_note TEXT;

-- 協會核款後開通的課程資格。
CREATE TABLE course_access (
  account_id TEXT NOT NULL REFERENCES accounts(id),
  course_slug TEXT NOT NULL,
  source_code TEXT,
  opened_by TEXT,
  opened_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (account_id, course_slug)
);

-- 課程講師（多對多）。
CREATE TABLE course_staff (
  course_slug TEXT NOT NULL,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  role TEXT NOT NULL DEFAULT 'instructor' CHECK (role IN ('instructor', 'co_instructor')),
  added_by TEXT,
  added_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (course_slug, account_id)
);

-- 學員自行加入實作營，不需協會再確認。
CREATE TABLE camp_members (
  course_slug TEXT NOT NULL,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  joined_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (course_slug, account_id)
);

-- 技能單張：講師撰寫；每次修改新增版本。
CREATE TABLE materials (
  id TEXT PRIMARY KEY,
  course_slug TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  position INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'published' CHECK (status IN ('published', 'archived')),
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX materials_course ON materials (course_slug, position);
CREATE TABLE material_versions (
  material_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  edited_by TEXT NOT NULL,
  edited_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  PRIMARY KEY (material_id, version)
);

-- 課程論壇：主題文與回覆共用一張表。
CREATE TABLE posts (
  id TEXT PRIMARY KEY,
  course_slug TEXT NOT NULL,
  author_id TEXT NOT NULL REFERENCES accounts(id),
  parent_id TEXT REFERENCES posts(id),
  title TEXT,
  body TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'published' CHECK (status IN ('published', 'withdrawn', 'hidden')),
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX posts_course ON posts (course_slug, parent_id, created_at);
CREATE TABLE post_revisions (
  post_id TEXT NOT NULL,
  revised_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  title TEXT,
  body TEXT NOT NULL,
  revised_by TEXT NOT NULL
);

-- 圖片放 R2；網址不可猜測，下載一律經權限檢查。
CREATE TABLE attachments (
  id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL REFERENCES posts(id),
  course_slug TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  r2_key TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  size INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  deleted_at TEXT
);
CREATE INDEX attachments_post ON attachments (post_id);

-- 隱藏／恢復紀錄。隱藏原因使用固定分類。
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

-- 站內通知（Email 寄送服務啟用前，作者在帳號頁看到）。
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

CREATE TABLE instructor_applications (
  id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(id),
  profile TEXT NOT NULL,
  experience TEXT NOT NULL,
  topic TEXT NOT NULL,
  audience TEXT NOT NULL,
  proposal TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'submitted' CHECK (status IN ('submitted', 'changes_requested', 'approved', 'rejected')),
  reviewer_id TEXT,
  reviewed_at TEXT,
  review_note TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 所有登入綁定、核款、開通、隱藏、申訴、審核操作的稽核紀錄。
CREATE TABLE audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor_id TEXT,
  action TEXT NOT NULL,
  target TEXT,
  detail TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
