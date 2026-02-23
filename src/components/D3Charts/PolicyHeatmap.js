/**
 * PolicyHeatmap - Policy Network 輸出視覺化
 * 在棋盤上以熱力圖方式顯示每個位置的落子機率
 */
import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import BrowserOnly from '@docusaurus/BrowserOnly';

const BOARD_SIZE = 19;

// 預設的機率分布範例
const EXAMPLE_POLICIES = {
  empty: generateUniformPolicy(),
  corner: generateCornerPolicy(),
  move37: generateMove37Policy(),
};

function generateUniformPolicy() {
  const policy = [];
  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      policy.push({ x, y, prob: 1 / 361 });
    }
  }
  return policy;
}

function generateCornerPolicy() {
  const policy = [];
  const corners = [[3, 3], [3, 15], [15, 3], [15, 15]];
  const approaches = [[2, 4], [4, 2], [2, 14], [4, 16], [14, 2], [16, 4], [14, 16], [16, 14]];

  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      let prob = 0.001;
      // 星位高機率
      if (corners.some(([cx, cy]) => cx === x && cy === y)) {
        prob = 0.15;
      }
      // 掛角中等機率
      else if (approaches.some(([ax, ay]) => ax === x && ay === y)) {
        prob = 0.05;
      }
      // 邊線較低
      else if (x === 0 || x === 18 || y === 0 || y === 18) {
        prob = 0.0005;
      }
      policy.push({ x, y, prob });
    }
  }
  return normalizePolicy(policy);
}

function generateMove37Policy() {
  // 模擬 AlphaGo 第 37 手時的機率分布
  const policy = [];
  const move37 = { x: 9, y: 4 }; // 五路肩衝位置
  const alternativeMoves = [[3, 2], [15, 2], [10, 10], [8, 6]];

  for (let y = 0; y < BOARD_SIZE; y++) {
    for (let x = 0; x < BOARD_SIZE; x++) {
      let prob = 0.001;

      // 第 37 手位置
      if (x === move37.x && y === move37.y) {
        prob = 0.08; // 不是最高，但顯著
      }
      // 其他候選位置
      else if (alternativeMoves.some(([ax, ay]) => ax === x && ay === y)) {
        prob = 0.12;
      }
      // 中央區域有一定機率
      else if (x >= 5 && x <= 13 && y >= 5 && y <= 13) {
        prob = 0.005 + Math.random() * 0.01;
      }

      policy.push({ x, y, prob });
    }
  }
  return normalizePolicy(policy);
}

function normalizePolicy(policy) {
  const total = policy.reduce((sum, p) => sum + p.prob, 0);
  return policy.map(p => ({ ...p, prob: p.prob / total }));
}

function PolicyHeatmapInner({
  initialPosition = 'corner',
  stones = [],
  highlightMoves = [],
  size = 450,
  showTopN = 5,
  interactive = true,
}) {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const [policy, setPolicy] = useState(EXAMPLE_POLICIES[initialPosition] || EXAMPLE_POLICIES.corner);
  const [selectedCell, setSelectedCell] = useState(null);

  const margin = 35;
  const boardSize = size - margin * 2;
  const cellSize = boardSize / (BOARD_SIZE - 1);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin}, ${margin})`);

    // 棋盤背景
    g.append('rect')
      .attr('x', -cellSize / 2)
      .attr('y', -cellSize / 2)
      .attr('width', boardSize + cellSize)
      .attr('height', boardSize + cellSize)
      .attr('fill', '#dcb35c')
      .attr('rx', 4);

    // 找出最大機率值用於正規化顏色
    const maxProb = Math.max(...policy.map(p => p.prob));

    // 繪製熱力圖
    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
      .domain([0, maxProb]);

    const heatmapGroup = g.append('g').attr('class', 'heatmap');

    heatmapGroup.selectAll('rect')
      .data(policy)
      .enter()
      .append('rect')
      .attr('class', 'heatmap-cell')
      .attr('x', d => d.x * cellSize - cellSize / 2)
      .attr('y', d => d.y * cellSize - cellSize / 2)
      .attr('width', cellSize)
      .attr('height', cellSize)
      .attr('fill', d => colorScale(d.prob))
      .attr('opacity', d => 0.3 + (d.prob / maxProb) * 0.6)
      .attr('cursor', interactive ? 'pointer' : 'default')
      .on('mouseover', function(event, d) {
        if (!interactive) return;
        d3.select(this).attr('stroke', '#333').attr('stroke-width', 2);

        const tooltip = d3.select(tooltipRef.current);
        tooltip
          .style('display', 'block')
          .style('left', `${event.pageX + 10}px`)
          .style('top', `${event.pageY - 10}px`)
          .html(`位置: ${String.fromCharCode(65 + d.x)}${19 - d.y}<br>機率: ${(d.prob * 100).toFixed(2)}%`);
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke', 'none');
        d3.select(tooltipRef.current).style('display', 'none');
      })
      .on('click', function(event, d) {
        if (!interactive) return;
        setSelectedCell(d);
      });

    // 繪製格線
    const gridGroup = g.append('g').attr('class', 'grid');

    for (let i = 0; i < BOARD_SIZE; i++) {
      gridGroup.append('line')
        .attr('class', 'grid-line')
        .attr('x1', 0)
        .attr('y1', i * cellSize)
        .attr('x2', (BOARD_SIZE - 1) * cellSize)
        .attr('y2', i * cellSize)
        .attr('stroke', '#333')
        .attr('stroke-width', 0.5)
        .attr('opacity', 0.5);

      gridGroup.append('line')
        .attr('class', 'grid-line')
        .attr('x1', i * cellSize)
        .attr('y1', 0)
        .attr('x2', i * cellSize)
        .attr('y2', (BOARD_SIZE - 1) * cellSize)
        .attr('stroke', '#333')
        .attr('stroke-width', 0.5)
        .attr('opacity', 0.5);
    }

    // 繪製棋子
    const stoneGroup = g.append('g').attr('class', 'stones');
    stones.forEach(({ x, y, color }) => {
      stoneGroup.append('circle')
        .attr('cx', x * cellSize)
        .attr('cy', y * cellSize)
        .attr('r', cellSize * 0.45)
        .attr('fill', color === 'black' ? '#1a1a1a' : '#f5f5f5')
        .attr('stroke', color === 'black' ? '#000' : '#333')
        .attr('stroke-width', 1);
    });

    // 標示 Top N 高機率位置
    const topN = [...policy]
      .sort((a, b) => b.prob - a.prob)
      .slice(0, showTopN);

    const labelGroup = g.append('g').attr('class', 'top-labels');
    topN.forEach((move, idx) => {
      // 如果該位置已有棋子則跳過
      if (stones.some(s => s.x === move.x && s.y === move.y)) return;

      labelGroup.append('circle')
        .attr('cx', move.x * cellSize)
        .attr('cy', move.y * cellSize)
        .attr('r', cellSize * 0.3)
        .attr('fill', 'rgba(255,255,255,0.8)')
        .attr('stroke', '#e74c3c')
        .attr('stroke-width', 2);

      labelGroup.append('text')
        .attr('x', move.x * cellSize)
        .attr('y', move.y * cellSize)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .attr('fill', '#e74c3c')
        .attr('font-size', cellSize * 0.4)
        .attr('font-weight', 'bold')
        .text(idx + 1);
    });

    // 座標標示
    const coordGroup = svg.append('g').attr('class', 'coordinates');
    const letters = 'ABCDEFGHJKLMNOPQRST';

    for (let i = 0; i < BOARD_SIZE; i++) {
      coordGroup.append('text')
        .attr('x', margin + i * cellSize)
        .attr('y', margin / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#666')
        .attr('font-size', 10)
        .text(letters[i]);

      coordGroup.append('text')
        .attr('x', margin / 2)
        .attr('y', margin + i * cellSize)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .attr('fill', '#666')
        .attr('font-size', 10)
        .text(BOARD_SIZE - i);
    }

  }, [policy, stones, showTopN, interactive, cellSize, margin, boardSize]);

  const handleScenarioChange = (scenario) => {
    setPolicy(EXAMPLE_POLICIES[scenario] || EXAMPLE_POLICIES.corner);
  };

  return (
    <div>
      {interactive && (
        <div className="d3-controls">
          <button
            className={initialPosition === 'empty' ? 'active' : ''}
            onClick={() => handleScenarioChange('empty')}
          >
            均勻分布
          </button>
          <button
            className={initialPosition === 'corner' ? 'active' : ''}
            onClick={() => handleScenarioChange('corner')}
          >
            開局星位
          </button>
          <button
            className={initialPosition === 'move37' ? 'active' : ''}
            onClick={() => handleScenarioChange('move37')}
          >
            第 37 手
          </button>
        </div>
      )}

      <div className="go-board-container">
        <svg
          ref={svgRef}
          width={size}
          height={size}
          className="go-board"
        />
      </div>

      <div
        ref={tooltipRef}
        className="d3-tooltip"
        style={{ display: 'none', position: 'fixed' }}
      />

      {selectedCell && (
        <div className="d3-legend">
          <div className="d3-legend-item">
            已選擇: {String.fromCharCode(65 + selectedCell.x)}{19 - selectedCell.y}
            — 機率: {(selectedCell.prob * 100).toFixed(2)}%
          </div>
        </div>
      )}

      <div className="d3-legend">
        <div className="d3-legend-item">
          <div className="d3-legend-color" style={{ background: '#ffffb2' }} />
          低機率
        </div>
        <div className="d3-legend-item">
          <div className="d3-legend-color" style={{ background: '#fd8d3c' }} />
          中機率
        </div>
        <div className="d3-legend-item">
          <div className="d3-legend-color" style={{ background: '#bd0026' }} />
          高機率
        </div>
      </div>
    </div>
  );
}

export default function PolicyHeatmap(props) {
  return (
    <BrowserOnly fallback={<div>載入中...</div>}>
      {() => <PolicyHeatmapInner {...props} />}
    </BrowserOnly>
  );
}
