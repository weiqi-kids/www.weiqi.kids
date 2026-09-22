// 畫面上顯示的狀態一律用中文；資料庫內部值不直接顯示給使用者。
export const ENROLLMENT_STATUS: Record<string, string> = { submitted: '待核款', paid: '已核款', activated: '已開通', cancelled: '已取消' };
export const APPLICATION_STATUS: Record<string, string> = { submitted: '審核中', changes_requested: '需要補件', approved: '已核准', rejected: '未核准' };
export const COURSE_STATUS: Record<string, string> = { draft: '草稿（只在 staging 顯示）', open: '開放報名', running: '進行中', closed: '已結營' };
export const POST_STATUS: Record<string, string> = { published: '已發布', withdrawn: '已撤回', hidden: '已隱藏' };

export const label = (map: Record<string, string>, code: string | null | undefined) => (code && map[code]) || '未知狀態';
