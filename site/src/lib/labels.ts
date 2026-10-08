// 畫面上顯示的狀態一律用中文；資料庫內部值不直接顯示給使用者。
export const SHOWCASE_STATUS: Record<string, string> = { draft: '草稿', submitted: '審核中', changes_requested: '需要補件', published: '已公開', archived: '已下架' };
export const COHORT_STATE: Record<string, string> = {
  gathering: '登記中', scheduling: '排時間', payment: '等匯款', confirmed: '已成團', running: '上課中', ended: '已結營', unfilled: '沒有成團', cancelled: '講師取消',
};
export const ENROLLMENT_STATUS: Record<string, string> = {
  awaiting_payment: '待匯款', submitted: '已匯款，等管理員核對', confirmed: '已核款', refund_pending: '待退款', refunded: '已退款', cancelled: '已取消',
};
export const DUES_STATUS: Record<string, string> = { submitted: '等管理員核對', confirmed: '已核款', rejected: '核對不符' };
export const POST_STATUS: Record<string, string> = { published: '已發布', withdrawn: '已撤回', hidden: '已隱藏' };
export const FORMAT_LABEL: Record<string, string> = { online: '線上直播', onsite: '實體上課' };
export const SCHEDULE_LABEL: Record<string, string> = { slots: '講師先填每週時段，學員登記時選', vote: '滿人數後講師提出候選日期，登記的人投票參考', fixed: '講師預先訂好日期' };
export const RULE_LABEL: Record<string, string> = { paid: '匯款期限內付款滿人數才成團', signup: '登記滿人數就成團，付款的人上課' };
export const PRACTICE_WEEKS: Record<number, string> = { 1: '第 1 週：題目', 2: '第 2 週：第一版', 3: '第 3 週：成果' };

export const label = (map: Record<string, string>, code: string | null | undefined) => (code && map[code]) || '未知狀態';

export const AUDIT_ACTION: Record<string, string> = {
  'account.create': '建立帳號', 'account.rename': '改顯示名稱', 'identity.link': '綁定登入方式', 'identity.unlink': '解除登入方式',
  'login.email': 'Email 登入', 'login.line': 'LINE 登入', 'magic.sent': '寄出登入信', 'magic.send_failed': '登入信寄送失敗', 'push.subscribe': '開啟推播',
  'dues.submit': '填會費匯款', 'dues.confirm': '會費核款', 'dues.reject': '會費核對不符',
  'showcase.create': '建立成果', 'showcase.edit': '修改成果', 'showcase.slots': '修改時段', 'showcase.submit': '成果送審', 'showcase.publish': '成果審核通過', 'showcase.changes': '成果要求補件', 'showcase.archive': '成果下架',
  'interest.add': '登記想學', 'interest.remove': '取消登記',
  'cohort.payment': '開團，開始匯款', 'cohort.propose': '提出候選日期', 'cohort.set_date': '決定上課日期', 'cohort.meeting_url': '更新直播連結', 'cohort.cancel': '講師取消', 'cohort.ended': '結營', 'cohort.unfilled': '沒有成團', 'cohort.cancelled': '團取消', 'cohort.run_now': '手動推進各團',
  'enrollment.create': '報名', 'enrollment.submit_payment': '填匯款資料', 'enrollment.confirm': '核款', 'enrollment.reject': '核對不符', 'enrollment.refund_request': '申請退款', 'enrollment.refunded': '已退款',
  'payout.paid': '分潤已匯出', 'attendance.checkin': '簽到', 'survey.submit': '填問卷',
  'material.save': '儲存技能單張', 'material.archive': '移除技能單張', 'work.approve': '講師認可作品', 'work.consent': '作者同意公開', 'work.withdraw_consent': '作者撤回公開',
  'post.create': '發文', 'post.edit': '修改文章', 'post.withdraw': '撤回文章', 'post.hide': '隱藏內容', 'post.restore': '恢復內容', 'appeal.create': '提出申訴', 'appeal.upheld': '申訴：維持隱藏', 'appeal.restored': '申訴：恢復內容',
  'admin.grant': '設為管理員', 'admin.revoke': '取消管理員',
};
