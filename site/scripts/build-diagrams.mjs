// 產生棋盤示意圖。改圖請改這裡，不要手改 SVG。
import { writeFileSync, mkdirSync } from 'node:fs';
import { renderBoard } from './go-diagram.mjs';

mkdirSync('public/media/diagrams', { recursive: true });

const diagrams = {
  // 圍棋術語：棋盤上四個有固定名稱的位置
  'go-terminology': renderBoard({
    size: 19,
    // 四個點分散在盤面不同區域，標籤才不會疊在一起
    stones: [
      { pt: 'K10', color: 'black' },
      { pt: 'Q16', color: 'black' },
      { pt: 'D17', color: 'black' },
      { pt: 'C3', color: 'black' },
    ],
    marks: [
      { pt: 'K10', text: '天元', side: 'right' },
      { pt: 'Q16', text: '星位', side: 'left' },
      { pt: 'D17', text: '小目', side: 'right' },
      { pt: 'C3', text: '三三', side: 'right' },
    ],
  }),

  // 圍棋規則：白子四口氣被填滿，提子
  'go-capture': renderBoard({
    size: 9,
    stones: [
      { pt: 'E5', color: 'white' },
      { pt: 'E6', color: 'black', label: '1' },
      { pt: 'D5', color: 'black', label: '2' },
      { pt: 'F5', color: 'black', label: '3' },
      { pt: 'E4', color: 'black', label: '4' },
    ],
    marks: [{ pt: 'E5', type: 'square' }],
  }),

  // 開局十手：先佔四個角
  'go-opening-corners': renderBoard({
    size: 19,
    stones: [
      { pt: 'Q16', color: 'black', label: '1' },
      { pt: 'D4', color: 'white', label: '2' },
      { pt: 'Q4', color: 'black', label: '3' },
      { pt: 'D16', color: 'white', label: '4' },
    ],
  }),
};

for (const [name, svg] of Object.entries(diagrams)) {
  const path = `public/media/diagrams/${name}.svg`;
  writeFileSync(path, svg + '\n');
  console.log(`  ${path}  ${(svg.length / 1024).toFixed(1)} KB`);
}
