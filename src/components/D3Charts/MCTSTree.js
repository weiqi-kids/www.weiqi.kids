/**
 * MCTSTree - MCTS 搜索樹視覺化
 * 展示蒙地卡羅樹搜索的節點結構與選擇過程
 */
import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import BrowserOnly from '@docusaurus/BrowserOnly';

// 範例 MCTS 樹數據
const EXAMPLE_TREE = {
  name: 'Root',
  visits: 1600,
  value: 0.55,
  prior: 1,
  children: [
    {
      name: 'D4',
      visits: 800,
      value: 0.62,
      prior: 0.35,
      selected: true,
      children: [
        { name: 'Q16', visits: 400, value: 0.58, prior: 0.3 },
        { name: 'R4', visits: 300, value: 0.65, prior: 0.25, selected: true },
        { name: 'C16', visits: 100, value: 0.55, prior: 0.2 },
      ]
    },
    {
      name: 'Q4',
      visits: 500,
      value: 0.52,
      prior: 0.30,
      children: [
        { name: 'D16', visits: 300, value: 0.50, prior: 0.28 },
        { name: 'Q16', visits: 200, value: 0.54, prior: 0.22 },
      ]
    },
    {
      name: 'D16',
      visits: 200,
      value: 0.48,
      prior: 0.20,
    },
    {
      name: 'Q16',
      visits: 100,
      value: 0.45,
      prior: 0.15,
    },
  ]
};

function MCTSTreeInner({
  data = EXAMPLE_TREE,
  width = 700,
  height = 450,
  showPUCT = true,
  cPuct = 1.5,
  interactive = true,
}) {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [currentCPuct, setCurrentCPuct] = useState(cPuct);

  const margin = { top: 40, right: 40, bottom: 40, left: 40 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // 計算 PUCT 分數
  const calculatePUCT = (node, parentVisits) => {
    if (!parentVisits) return 0;
    const Q = node.value;
    const P = node.prior;
    const N = node.visits;
    const U = currentCPuct * P * Math.sqrt(parentVisits) / (1 + N);
    return Q + U;
  };

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // 建立樹形佈局
    const treeLayout = d3.tree()
      .size([innerWidth, innerHeight - 50]);

    const root = d3.hierarchy(data);
    treeLayout(root);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // 繪製連線
    const linkGroup = g.append('g').attr('class', 'links');

    linkGroup.selectAll('path')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', d => `link ${d.target.data.selected ? 'selected' : ''}`)
      .attr('fill', 'none')
      .attr('stroke', d => d.target.data.selected ? '#4a90d9' : '#999')
      .attr('stroke-width', d => d.target.data.selected ? 3 : 1.5)
      .attr('d', d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y)
      );

    // 繪製節點
    const nodeGroup = g.append('g').attr('class', 'nodes');

    const nodes = nodeGroup.selectAll('g')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.x}, ${d.y})`)
      .attr('cursor', interactive ? 'pointer' : 'default')
      .on('mouseover', function(event, d) {
        if (!interactive) return;

        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', 30);

        const parentVisits = d.parent ? d.parent.data.visits : d.data.visits;
        const puct = calculatePUCT(d.data, parentVisits);

        const tooltip = d3.select(tooltipRef.current);
        tooltip
          .style('display', 'block')
          .style('left', `${event.pageX + 15}px`)
          .style('top', `${event.pageY - 10}px`)
          .html(`
            <strong>${d.data.name}</strong><br>
            訪問次數 (N): ${d.data.visits}<br>
            平均價值 (Q): ${d.data.value.toFixed(3)}<br>
            先驗機率 (P): ${(d.data.prior * 100).toFixed(1)}%<br>
            ${showPUCT ? `PUCT 分數: ${puct.toFixed(3)}` : ''}
          `);
      })
      .on('mouseout', function() {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', 25);

        d3.select(tooltipRef.current).style('display', 'none');
      })
      .on('click', function(event, d) {
        if (!interactive) return;
        setSelectedNode(d.data);
      });

    // 節點圓形
    nodes.append('circle')
      .attr('r', 25)
      .attr('fill', d => d.data.selected ? '#4a90d9' : '#fff')
      .attr('stroke', d => {
        if (d.data.selected) return '#2c5282';
        const intensity = d.data.visits / data.visits;
        return d3.interpolateBlues(0.3 + intensity * 0.5);
      })
      .attr('stroke-width', d => d.data.selected ? 3 : 2);

    // 節點標籤（名稱）
    nodes.append('text')
      .attr('dy', -5)
      .attr('text-anchor', 'middle')
      .attr('fill', d => d.data.selected ? '#fff' : '#333')
      .attr('font-size', 11)
      .attr('font-weight', 'bold')
      .text(d => d.data.name);

    // 節點標籤（訪問次數）
    nodes.append('text')
      .attr('dy', 10)
      .attr('text-anchor', 'middle')
      .attr('fill', d => d.data.selected ? '#fff' : '#666')
      .attr('font-size', 9)
      .text(d => `N=${d.data.visits}`);

    // 標題
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 14)
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text('MCTS 搜索樹');

    // 繪製選中路徑說明
    if (showPUCT) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', height - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('fill', '#666')
        .text('藍色路徑：PUCT 選擇的最佳路徑');
    }

  }, [data, width, height, showPUCT, currentCPuct, interactive, innerWidth, innerHeight]);

  return (
    <div>
      {showPUCT && interactive && (
        <div className="d3-controls">
          <div className="d3-slider">
            <label>c_puct: {currentCPuct.toFixed(1)}</label>
            <input
              type="range"
              min="0.5"
              max="3"
              step="0.1"
              value={currentCPuct}
              onChange={(e) => setCurrentCPuct(parseFloat(e.target.value))}
            />
          </div>
        </div>
      )}

      <div className="mcts-tree-container">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="mcts-tree"
        />
      </div>

      <div
        ref={tooltipRef}
        className="d3-tooltip"
        style={{ display: 'none', position: 'fixed' }}
      />

      {selectedNode && (
        <div className="d3-legend" style={{ background: '#f5f5f5', padding: '1rem', borderRadius: '4px' }}>
          <strong>已選擇節點: {selectedNode.name}</strong>
          <div>訪問次數: {selectedNode.visits}</div>
          <div>平均價值: {selectedNode.value.toFixed(3)}</div>
          <div>先驗機率: {(selectedNode.prior * 100).toFixed(1)}%</div>
        </div>
      )}

      <div className="d3-legend">
        <div className="d3-legend-item">
          <div className="d3-legend-color" style={{ background: '#4a90d9' }} />
          選中路徑
        </div>
        <div className="d3-legend-item">
          <div className="d3-legend-color" style={{ background: '#fff', border: '2px solid #999' }} />
          其他節點
        </div>
        <div className="d3-legend-item">
          <span style={{ fontSize: '12px' }}>節點大小 ∝ 訪問次數</span>
        </div>
      </div>
    </div>
  );
}

export default function MCTSTree(props) {
  return (
    <BrowserOnly fallback={<div>載入中...</div>}>
      {() => <MCTSTreeInner {...props} />}
    </BrowserOnly>
  );
}
