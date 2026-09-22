-- 第二期：課程報名（ADR 0010）。付款確認在 LINE@ 人工處理，網站不存匯款資料。
CREATE TABLE enrollments (
  code TEXT PRIMARY KEY,                -- 報名編號 WK-YYYYMM-XXXX
  course_slug TEXT NOT NULL,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  line_name TEXT,
  membership TEXT NOT NULL CHECK (membership IN ('new', 'member')),
  note TEXT,
  status TEXT NOT NULL DEFAULT 'submitted' CHECK (status IN ('submitted', 'paid', 'activated', 'cancelled')),
  ip_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX enrollments_course ON enrollments (course_slug, created_at);
CREATE INDEX enrollments_ip ON enrollments (ip_hash, created_at);
