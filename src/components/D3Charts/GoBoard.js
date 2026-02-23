/**
 * GoBoard - 圍棋棋盤基礎組件
 * 提供 19×19 棋盤的 SVG 渲染，支援棋子顯示與互動
 */
import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const BOARD_SIZE = 19;
const STAR_POINTS = [
  [3, 3], [3, 9], [3, 15],
  [9, 3], [9, 9], [9, 15],
  [15, 3], [15, 9], [15, 15]
];

export default function GoBoard({
  size = 400,
  stones = [],  // [{x, y, color: 'black'|'white'}]
  highlights = [],  // [{x, y, intensity: 0-1}]
  labels = [],  // [{x, y, text}]
  onCellClick = null,
  showCoordinates = true,
}) {
  const svgRef = useRef(null);
  const margin = showCoordinates ? 30 : 15;
  const boardSize = size - margin * 2;
  const cellSize = boardSize / (BOARD_SIZE - 1);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // 建立主繪圖區域
    const g = svg.append('g')
      .attr('transform', `translate(${margin}, ${margin})`);

    // 繪製棋盤背景
    g.append('rect')
      .attr('x', -cellSize / 2)
      .attr('y', -cellSize / 2)
      .attr('width', boardSize + cellSize)
      .attr('height', boardSize + cellSize)
      .attr('fill', '#dcb35c')
      .attr('rx', 4);

    // 繪製格線
    const gridGroup = g.append('g').attr('class', 'grid');

    // 水平線
    for (let i = 0; i < BOARD_SIZE; i++) {
      gridGroup.append('line')
        .attr('class', 'grid-line')
        .attr('x1', 0)
        .attr('y1', i * cellSize)
        .attr('x2', (BOARD_SIZE - 1) * cellSize)
        .attr('y2', i * cellSize);
    }

    // 垂直線
    for (let i = 0; i < BOARD_SIZE; i++) {
      gridGroup.append('line')
        .attr('class', 'grid-line')
        .attr('x1', i * cellSize)
        .attr('y1', 0)
        .attr('x2', i * cellSize)
        .attr('y2', (BOARD_SIZE - 1) * cellSize);
    }

    // 繪製星位
    const starGroup = g.append('g').attr('class', 'star-points');
    STAR_POINTS.forEach(([x, y]) => {
      starGroup.append('circle')
        .attr('class', 'star-point')
        .attr('cx', x * cellSize)
        .attr('cy', y * cellSize)
        .attr('r', cellSize / 8);
    });

    // 繪製高亮（熱力圖效果）
    if (highlights.length > 0) {
      const highlightGroup = g.append('g').attr('class', 'highlights');
      highlights.forEach(({ x, y, intensity }) => {
        highlightGroup.append('rect')
          .attr('class', 'heatmap-cell')
          .attr('x', x * cellSize - cellSize / 2)
          .attr('y', y * cellSize - cellSize / 2)
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', d3.interpolateReds(intensity))
          .attr('opacity', intensity * 0.7);
      });
    }

    // 繪製棋子
    const stoneGroup = g.append('g').attr('class', 'stones');
    stones.forEach(({ x, y, color }) => {
      const stoneClass = color === 'black' ? 'stone-black' : 'stone-white';

      // 陰影效果
      stoneGroup.append('circle')
        .attr('cx', x * cellSize + 2)
        .attr('cy', y * cellSize + 2)
        .attr('r', cellSize * 0.45)
        .attr('fill', 'rgba(0,0,0,0.2)');

      // 棋子本體
      stoneGroup.append('circle')
        .attr('class', stoneClass)
        .attr('cx', x * cellSize)
        .attr('cy', y * cellSize)
        .attr('r', cellSize * 0.45);
    });

    // 繪製標籤
    if (labels.length > 0) {
      const labelGroup = g.append('g').attr('class', 'labels');
      labels.forEach(({ x, y, text }) => {
        const stone = stones.find(s => s.x === x && s.y === y);
        const textColor = stone?.color === 'black' ? '#fff' : '#000';

        labelGroup.append('text')
          .attr('x', x * cellSize)
          .attr('y', y * cellSize)
          .attr('dy', '0.35em')
          .attr('text-anchor', 'middle')
          .attr('fill', textColor)
          .attr('font-size', cellSize * 0.5)
          .attr('font-weight', 'bold')
          .text(text);
      });
    }

    // 座標標示
    if (showCoordinates) {
      const coordGroup = svg.append('g').attr('class', 'coordinates');
      const letters = 'ABCDEFGHJKLMNOPQRST'; // 跳過 I

      // 頂部字母
      for (let i = 0; i < BOARD_SIZE; i++) {
        coordGroup.append('text')
          .attr('x', margin + i * cellSize)
          .attr('y', margin / 2)
          .attr('text-anchor', 'middle')
          .attr('fill', '#666')
          .attr('font-size', 10)
          .text(letters[i]);
      }

      // 左側數字
      for (let i = 0; i < BOARD_SIZE; i++) {
        coordGroup.append('text')
          .attr('x', margin / 2)
          .attr('y', margin + i * cellSize)
          .attr('dy', '0.35em')
          .attr('text-anchor', 'middle')
          .attr('fill', '#666')
          .attr('font-size', 10)
          .text(BOARD_SIZE - i);
      }
    }

    // 點擊事件
    if (onCellClick) {
      g.append('g')
        .attr('class', 'click-targets')
        .selectAll('rect')
        .data(d3.range(BOARD_SIZE * BOARD_SIZE))
        .enter()
        .append('rect')
        .attr('x', d => (d % BOARD_SIZE) * cellSize - cellSize / 2)
        .attr('y', d => Math.floor(d / BOARD_SIZE) * cellSize - cellSize / 2)
        .attr('width', cellSize)
        .attr('height', cellSize)
        .attr('fill', 'transparent')
        .attr('cursor', 'pointer')
        .on('click', (event, d) => {
          const x = d % BOARD_SIZE;
          const y = Math.floor(d / BOARD_SIZE);
          onCellClick({ x, y });
        });
    }

  }, [size, stones, highlights, labels, showCoordinates, onCellClick, cellSize, margin, boardSize]);

  return (
    <div className="go-board-container">
      <svg
        ref={svgRef}
        width={size}
        height={size}
        className="go-board"
      />
    </div>
  );
}
