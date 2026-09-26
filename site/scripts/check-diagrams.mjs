// 驗證棋盤 SVG 的格線數量，以及每顆棋子都落在線的交叉點。
import { readdirSync, readFileSync } from 'node:fs';

const directory = 'public/media/diagrams';
const files = readdirSync(directory)
  .filter((name) => name.endsWith('.svg'))
  .sort();

if (files.length === 0) {
  console.error(`FAIL ${directory} 沒有 SVG 檔案`);
  process.exit(1);
}

const attributes = (tag) =>
  Object.fromEntries(
    [...tag.matchAll(/([\w:-]+)="([^"]*)"/g)].map((match) => [match[1], match[2]]),
  );

let failed = false;

for (const file of files) {
  const svg = readFileSync(`${directory}/${file}`, 'utf8');
  const xCoordinates = new Set();
  const yCoordinates = new Set();
  const errors = [];

  for (const match of svg.matchAll(/<line\b[^>]*>/g)) {
    const line = attributes(match[0]);
    const required = ['x1', 'y1', 'x2', 'y2'];
    if (required.some((name) => line[name] === undefined)) {
      errors.push(`無法解析 line：${match[0]}`);
      continue;
    }

    if (line.x1 === line.x2) xCoordinates.add(Number(line.x1));
    if (line.y1 === line.y2) yCoordinates.add(Number(line.y1));
  }

  let stones = 0;
  for (const match of svg.matchAll(/<circle\b[^>]*>/g)) {
    const circle = attributes(match[0]);
    if (circle.fill !== '#141210' && circle.fill !== '#fbfaf7') continue;

    stones += 1;
    const cx = Number(circle.cx);
    const cy = Number(circle.cy);
    if (!xCoordinates.has(cx) || !yCoordinates.has(cy)) {
      errors.push(`棋子 (${circle.cx}, ${circle.cy}) 不在格線交叉點`);
    }
  }

  if (xCoordinates.size !== yCoordinates.size) {
    errors.push(`直線 ${xCoordinates.size} 條，橫線 ${yCoordinates.size} 條`);
  }

  if (errors.length > 0) {
    failed = true;
    console.error(`FAIL ${file}`);
    for (const error of errors) console.error(`  ${error}`);
  } else {
    console.log(
      `PASS ${file}: ${xCoordinates.size}×${yCoordinates.size} 格線，${stones} 顆棋子都在交叉點`,
    );
  }
}

if (failed) process.exit(1);
