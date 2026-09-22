// 全站共用的組織資訊與導覽。
export const site = {
  name: '台灣好棋寶寶協會',
  legalName: '社團法人台灣好棋寶寶協會',
  alternateName: 'Taiwan Good Go Baby Association',
  url: 'https://www.weiqi.kids',
  email: 'lightman.chang@gmail.com',
  logo: { url: 'https://www.weiqi.kids/img/social-card.png', width: 1200, height: 630 },
  // 協會 LINE@ 加入連結。尚未提供時頁面改為顯示 Email 聯絡方式。
  lineOfficialUrl: null,
};

export const courseRules = {
  fee: 6000,
  teaching: 1,
  practice: 3,
  duration: '1 個月',
  membershipTerm: '1 年',
};

export const nav = [
  { label: '協會', href: '/association/' },
  { label: '好棋寶寶棋聚', href: '/gatherings/' },
  { label: 'AI 共學營', href: '/camp/' },
  { label: '公開知識', href: '/knowledge/' },
];
