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

export const association = {
  // 立案與法人登記（內政部 台內團字第1130005685號；臺中地院 113年度法登社字第14號）
  foundedDate: '2024-01-05',
  registeredDate: '2024-02-20',
  corporateDate: '2024-02-29',
  registrationNo: '台內團字第 1130005685 號',
  courtCaseNo: '臺灣臺中地方法院 113 年度法登社字第 14 號',
  address: '臺中市西區臺灣大道二段 239 號 13 樓',
  purpose: '成為圍棋文化前進的推手',
  missions: [
    '各城市棋友交流，以圍棋為載體，提升不同文化包容性',
    '圍棋授課教材升級',
    '促進圍棋與其他產業交流活動',
    '專業、職業棋手投入公益活動',
  ],
  board: { directors: 9, supervisors: 3, term: 4 },
  chairman: { name: '張饒輝', term: '2024-01-05～2028-01-04', slug: 'lightman-chang' },
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
