// 網站使用的影片與照片。影片放在協會 YouTube 頻道；照片從 ref/ 挑選壓縮後放在 public/media/。
export const channelUrl = 'https://www.youtube.com/@GoodMoveAssociation';

// id：YouTube 影片 ID；duration：秒
export const videos = {
  promo: { id: 'xvi2ME0H9pE', title: '好棋寶寶 台灣美學 圍棋藝術 推廣影片', duration: 107 },
  promoLang: [
    { id: 'cLS82wfyQmI', label: 'English' },
    { id: 'RGS0jQw-tGw', label: '日本語' },
    { id: 'cA6zsoWwBNU', label: '한국어' },
    { id: 'GfeC0_jQle4', label: 'Español' },
    { id: 'zII37MLzTVY', label: 'Français' },
    { id: 'hQkkSLxuX-c', label: 'Bahasa Indonesia' },
  ],
  intro: { id: '6f1DTtoWG9I', title: '哈嚕！大家好！我們是好棋寶寶協會！', duration: 11 },
  carnival: { id: 'WRv1unFIo4c', title: '好棋寶寶圍棋嘉年華', duration: 158 },
  reviews: [
    { id: 'myzm4hLSQLg', title: '來點覆盤《對局時候的抗壓力》圍棋與營養學跨界探討', duration: 1397 },
    { id: 'zj8mC-675J4', title: '來點覆盤《運動》圍棋老師與脊椎矯正專家談久坐', duration: 1431 },
    { id: 'swmno_9Z73c', title: '來點覆盤《時間管理》圍棋與工業設計，新版圍棋計時器首度亮相', duration: 1732 },
  ],
  proReviews: [
    { id: 'upsXBB0VHjk', title: '蔡丞韋 職業五段 覆盤 01', duration: 84 },
    { id: 'jZBo2CFagPw', title: '蔡丞韋 職業五段 覆盤 02', duration: 95 },
    { id: '1wW9TXtHOpU', title: '蔡丞韋 職業五段 覆盤 03', duration: 179 },
    { id: 'zqZwPCf96z4', title: '陳禧 職業五段 純棋分享', duration: 289 },
    { id: 'yXUVdOlixBU', title: '林新為 職業三段 純棋分享', duration: 300 },
    { id: 'nbKZH7AY5dQ', title: '詹宜典 職業三段 純棋分享', duration: 345 },
    { id: 'enJfSS1IJKo', title: '詹宜典 職業三段 純棋與圍棋', duration: 147 },
    { id: '6fRI78jLs-I', title: '洪萱安 7 段（洪門棋院院長）純棋分享', duration: 436 },
    { id: 'rWF3g5g261E', title: '圍棋棋譜講解：3 盤實戰（角部死活、打劫、官子）｜新為老師', duration: 1418 },
  ],
  teaching: [
    { id: 'mBwJ36xpeBI', title: '如何下圍棋？基礎教學', duration: 432 },
    { id: 'dmDS8adouGA', title: '消除餘味：把棋提起來', duration: 58 },
    { id: 'CRhS0PT7_oc', title: '圍棋的起源：堯帝創棋的傳說', duration: 169 },
  ],
  funClub: [
    { id: 'KD9XpGnkkDM', title: '台中場 2023/11/11 前導影片' },
    { id: 'QWH78PsHJGg', title: '台中 好棋同樂會 活動紀錄 2023/11/11', duration: 71 },
    { id: 'q9iYyfa0APo', title: '好棋同樂會 一田視覺 微距攝影', duration: 158 },
    { id: 'Jt0BgMqvkQU', title: '好棋同樂會 心匠藝造 體驗金工之美', duration: 169 },
    { id: '_c-ndKgOn4I', title: '台中場 2023/12/17 前導影片' },
    { id: '-hKcneEpLcg', title: '台中 好棋同樂會 活動紀錄 2023/12/17', duration: 40 },
    { id: 'TBhUBoQEsE4', title: '2024 一月棋友聚 前導' },
    { id: 'SNV1fXEx56g', title: '2024 一月棋友聚 活動' },
  ],
};

// 照片集：key 對應 public/media/<key>/NN.webp
export const galleries = {
  carnival: {
    title: '第一屆好棋寶寶盃親子圍棋嘉年華（2024 年 6 月，和平高中）',
    photos: ['舞台主視覺', '全場大合照', '比賽場地全景', '對弈中的會場', '獎盃', '棋盤上的對局', '對局中的棋子', '工作人員與來賓合影', '圍棋主題花器', '開場吉他演出'],
  },
  taipei: {
    title: '113 年臺北市體育總會全國圍棋公開賽（2024 年 1 月 28 日，蓬萊國小）',
    photos: ['賽前的比賽場地', '頒獎台與獎盃', '計時器與棋罐', '主辦與協辦單位合影', '比賽進行中', '落子特寫', '裁判與工作人員'],
  },
  qingshui: {
    title: '臺中市清水區 113 年區長盃全國圍棋公開賽',
    photos: ['參賽者與來賓大合照', '工作人員合影'],
  },
  funclub1217: {
    title: '好棋同樂會 健康領域主題講座（2023 年 12 月 17 日，台中）',
    photos: ['講者分享', '講者與參加者合影', '友誼對局', '棋友對弈', '第二位講者介紹'],
  },
  brand2020: {
    title: '品牌棋聚（2020 年 8 月 24 日）',
    photos: ['棋聚會場', '多桌同時對弈', '棋友對局', '友誼對局'],
  },
  central04: {
    title: '中部棋友聚會（2020 年）',
    photos: ['棋友大合照', '夜間棋聚', '多桌對弈'],
  },
};

// 內容頁（依網址）要附加的照片集與影片
export const mediaByPath = {
  '/gatherings/archive/': { galleries: ['carnival', 'taipei', 'qingshui', 'brand2020', 'central04'], videos: ['carnival'] },
  '/gatherings/archive/fun-club/': { galleries: ['funclub1217'], videoList: 'funClub' },
};
