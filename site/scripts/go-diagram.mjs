// 圍棋棋盤 SVG 產生器。
//
// 為什麼要有這個：站上的吉祥物插畫把棋子畫在格子裡而不是線的交叉點，
// 棋盤路數與星位也不對。棋盤是確定性幾何，用程式畫就不會錯。
//
// 座標用圍棋慣例：直行 A–T 跳過 I，橫列由下往上 1 起算。

const COLS = 'ABCDEFGHJKLMNOPQRST';

const STAR_POINTS = {
  19: ['D4', 'K4', 'Q4', 'D10', 'K10', 'Q10', 'D16', 'K16', 'Q16'],
  13: ['D4', 'K4', 'G7', 'D10', 'K10'],
  9: ['C3', 'G3', 'E5', 'C7', 'G7'],
  17: [],
};

/** 'Q16' → { col: 15, row: 15 }（皆為 0 起算，row 由下往上） */
export function parsePoint(pt, size) {
  const m = /^([A-HJ-T])(\d{1,2})$/.exec(pt);
  if (!m) throw new Error(`座標格式錯誤：${pt}`);
  const col = COLS.indexOf(m[1]);
  const row = Number(m[2]) - 1;
  if (col < 0 || col >= size) throw new Error(`${pt} 的直行超出 ${size} 路棋盤`);
  if (row < 0 || row >= size) throw new Error(`${pt} 的橫列超出 ${size} 路棋盤`);
  return { col, row };
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function renderBoard({
  size = 19,
  stones = [],
  marks = [],
  width = 720,
  background = '#e8b96a',
  line = '#2b2118',
  padding = 0.62, // 以格距為單位的外緣留白，容納座標標示
} = {}) {
  if (!STAR_POINTS[size]) throw new Error(`不支援 ${size} 路棋盤`);

  // 幾何：格距 gap，棋盤左下角第一條線在 (pad, pad)
  const cells = size - 1;
  const gap = width / (cells + padding * 2);
  const pad = gap * padding;
  const height = width;
  const r = gap * 0.47; // 棋子半徑

  // 交叉點座標；SVG 的 y 往下增加，所以橫列要翻轉
  const X = (col) => pad + col * gap;
  const Y = (row) => pad + (cells - row) * gap;

  const out = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width.toFixed(2)} ${height.toFixed(2)}" width="100%" role="img">`);
  out.push(`<rect x="0" y="0" width="${width.toFixed(2)}" height="${height.toFixed(2)}" fill="${background}"/>`);

  // 格線
  const lw = Math.max(1, gap * 0.045);
  for (let i = 0; i < size; i++) {
    const edge = i === 0 || i === size - 1 ? lw * 1.6 : lw;
    out.push(`<line x1="${X(0).toFixed(2)}" y1="${Y(i).toFixed(2)}" x2="${X(cells).toFixed(2)}" y2="${Y(i).toFixed(2)}" stroke="${line}" stroke-width="${edge.toFixed(2)}"/>`);
    out.push(`<line x1="${X(i).toFixed(2)}" y1="${Y(0).toFixed(2)}" x2="${X(i).toFixed(2)}" y2="${Y(cells).toFixed(2)}" stroke="${line}" stroke-width="${edge.toFixed(2)}"/>`);
  }

  // 星位
  for (const pt of STAR_POINTS[size]) {
    const { col, row } = parsePoint(pt, size);
    out.push(`<circle cx="${X(col).toFixed(2)}" cy="${Y(row).toFixed(2)}" r="${(gap * 0.11).toFixed(2)}" fill="${line}"/>`);
  }

  // 座標標示
  const fs = gap * 0.42;
  for (let i = 0; i < size; i++) {
    const letter = COLS[i];
    const num = String(i + 1);
    const common = `font-family="'Noto Sans TC','PingFang TC','Microsoft JhengHei',system-ui,sans-serif" font-size="${fs.toFixed(2)}" fill="${line}" text-anchor="middle" dominant-baseline="middle"`;
    out.push(`<text x="${X(i).toFixed(2)}" y="${(Y(cells) - pad * 0.55).toFixed(2)}" ${common}>${letter}</text>`);
    out.push(`<text x="${X(i).toFixed(2)}" y="${(Y(0) + pad * 0.55).toFixed(2)}" ${common}>${letter}</text>`);
    out.push(`<text x="${(X(0) - pad * 0.55).toFixed(2)}" y="${Y(i).toFixed(2)}" ${common}>${num}</text>`);
    out.push(`<text x="${(X(cells) + pad * 0.55).toFixed(2)}" y="${Y(i).toFixed(2)}" ${common}>${num}</text>`);
  }

  // 棋子
  for (const s of stones) {
    const { col, row } = parsePoint(s.pt, size);
    const cx = X(col), cy = Y(row);
    const black = s.color === 'black';
    out.push(`<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${r.toFixed(2)}" fill="${black ? '#141210' : '#fbfaf7'}" stroke="${line}" stroke-width="${(lw * 1.1).toFixed(2)}"/>`);
    if (s.label !== undefined && s.label !== null && s.label !== '') {
      out.push(`<text x="${cx.toFixed(2)}" y="${cy.toFixed(2)}" font-family="'Noto Sans TC','PingFang TC','Microsoft JhengHei',system-ui,sans-serif" font-size="${(r * 1.05).toFixed(2)}" font-weight="700" fill="${black ? '#fbfaf7' : '#141210'}" text-anchor="middle" dominant-baseline="central">${esc(s.label)}</text>`);
    }
  }

  // 標記與引線標籤
  for (const m of marks) {
    const { col, row } = parsePoint(m.pt, size);
    const cx = X(col), cy = Y(row);
    const mr = r * 0.62;
    const colour = m.color ?? '#c0392b';
    const sw = (lw * 1.6).toFixed(2);
    if (m.type === 'square') {
      out.push(`<rect x="${(cx - mr).toFixed(2)}" y="${(cy - mr).toFixed(2)}" width="${(mr * 2).toFixed(2)}" height="${(mr * 2).toFixed(2)}" fill="none" stroke="${colour}" stroke-width="${sw}"/>`);
    } else if (m.type === 'triangle') {
      const p = `${cx.toFixed(2)},${(cy - mr).toFixed(2)} ${(cx - mr).toFixed(2)},${(cy + mr * 0.8).toFixed(2)} ${(cx + mr).toFixed(2)},${(cy + mr * 0.8).toFixed(2)}`;
      out.push(`<polygon points="${p}" fill="none" stroke="${colour}" stroke-width="${sw}"/>`);
    } else if (m.type === 'cross') {
      out.push(`<line x1="${(cx - mr).toFixed(2)}" y1="${(cy - mr).toFixed(2)}" x2="${(cx + mr).toFixed(2)}" y2="${(cy + mr).toFixed(2)}" stroke="${colour}" stroke-width="${sw}"/>`);
      out.push(`<line x1="${(cx + mr).toFixed(2)}" y1="${(cy - mr).toFixed(2)}" x2="${(cx - mr).toFixed(2)}" y2="${(cy + mr).toFixed(2)}" stroke="${colour}" stroke-width="${sw}"/>`);
    } else {
      out.push(`<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${mr.toFixed(2)}" fill="none" stroke="${colour}" stroke-width="${sw}"/>`);
    }
    if (m.text) {
      // 標籤方向：預設放右側，太靠右邊界改放左側；可用 side 指定
      const right = m.side ? m.side === 'right' : col < size - 5;
      const lx = cx + (right ? gap * 1.35 : -gap * 1.35);
      out.push(`<line x1="${(cx + (right ? mr : -mr)).toFixed(2)}" y1="${cy.toFixed(2)}" x2="${(lx - (right ? gap * 0.15 : -gap * 0.15)).toFixed(2)}" y2="${cy.toFixed(2)}" stroke="${colour}" stroke-width="${lw.toFixed(2)}"/>`);
      out.push(`<text x="${lx.toFixed(2)}" y="${cy.toFixed(2)}" font-family="'Noto Sans TC','PingFang TC','Microsoft JhengHei',system-ui,sans-serif" font-size="${(gap * 0.72).toFixed(2)}" font-weight="700" fill="${colour}" text-anchor="${right ? 'start' : 'end'}" dominant-baseline="central" paint-order="stroke" stroke="${background}" stroke-width="${(gap * 0.18).toFixed(2)}">${esc(m.text)}</text>`);
    }
  }

  out.push('</svg>');
  return out.join('\n');
}

export { COLS, STAR_POINTS };
