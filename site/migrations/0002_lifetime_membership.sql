-- 不會到期的會員（例如創會會長）：不需繳常年會費，會員資格一直有效。
ALTER TABLE memberships ADD COLUMN lifetime INTEGER NOT NULL DEFAULT 0;
