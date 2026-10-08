-- 取消不會到期的會員：所有會員都一年一續，沒有特例。
ALTER TABLE memberships DROP COLUMN lifetime;
