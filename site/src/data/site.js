// 全站共用的組織資訊與導覽。
// GA4 沿用舊站的評量 ID，歷史資料才接得起來。
export const ga4Id = 'G-16V1KSEH6W';

export const site = {
  name: '台灣好棋寶寶協會',
  legalName: '社團法人台灣好棋寶寶協會',
  alternateName: 'Good Move Association',
  youtube: 'https://www.youtube.com/@GoodMoveAssociation',
  url: 'https://www.weiqi.kids',
  email: 'lightman.chang@gmail.com',
  logo: { url: 'https://www.weiqi.kids/media/brand/logo-stacked.png', width: 704, height: 590 },
  // 協會 LINE@（好棋寶寶，@685hqmrm）：提交匯款紀錄與私密資料。
  lineOfficialUrl: 'https://line.me/R/ti/p/@685hqmrm',
  lineOfficialId: '@685hqmrm',
  // 好棋寶寶 LINE 社群：棋聚消息與交流。
  lineCommunityUrl: 'https://line.me/ti/g2/ZXE1nUXuSbWKFfv0ZduAVvjPQvdpurLE2LjjTg',
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
