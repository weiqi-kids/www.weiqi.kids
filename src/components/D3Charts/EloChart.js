/**
 * EloChart - ELO 評分成長曲線
 * 展示訓練過程中棋力的變化
 */
import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import BrowserOnly from '@docusaurus/BrowserOnly';

// AlphaGo Zero 訓練曲線數據（近似值）
const ALPHAGO_ZERO_DATA = [
  { hours: 0, elo: 0, label: '隨機' },
  { hours: 3, elo: 1000, label: '發現規則' },
  { hours: 6, elo: 2000 },
  { hours: 12, elo: 3000, label: '發現定式' },
  { hours: 24, elo: 4000 },
  { hours: 36, elo: 4500, label: '超越 Fan Hui' },
  { hours: 48, elo: 5000 },
  { hours: 60, elo: 5200, label: '超越 Lee Sedol' },
  { hours: 72, elo: 5400, label: '超越原版 AlphaGo' },
];

// 人類參考線
const HUMAN_LEVELS = [
  { elo: 2700, label: '業餘強豪' },
  { elo: 3500, label: 'Fan Hui (職業二段)' },
  { elo: 4500, label: 'Lee Sedol (世界冠軍)' },
  { elo: 5000, label: '原版 AlphaGo' },
];

// 監督學習數據
const SL_DATA = [
  { epochs: 0, elo: 0 },
  { epochs: 10, elo: 1500 },
  { epochs: 20, elo: 2500 },
  { epochs: 30, elo: 3000 },
  { epochs: 40, elo: 3200 },
  { epochs: 50, elo: 3300 },
];

// 自我對弈數據
const SELF_PLAY_DATA = [
  { games: 0, elo: 3300 },
  { games: 1000, elo: 3800 },
  { games: 5000, elo: 4200 },
  { games: 10000, elo: 4500 },
  { games: 50000, elo: 4800 },
  { games: 100000, elo: 5000 },
];

function EloChartInner({
  mode = 'zero',  // 'zero', 'sl', 'selfplay'
  width = 600,
  height = 400,
  animated = true,
  showMilestones = true,
}) {
  const svgRef = useRef(null);
  const [currentMode, setCurrentMode] = useState(mode);

  const margin = { top: 40, right: 100, bottom: 60, left: 70 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // 選擇數據
    let data, xLabel, xDomain;
    if (currentMode === 'zero') {
      data = ALPHAGO_ZERO_DATA;
      xLabel = '訓練時間（小時）';
      xDomain = [0, 80];
    } else if (currentMode === 'sl') {
      data = SL_DATA;
      xLabel = '訓練輪數（Epochs）';
      xDomain = [0, 60];
    } else {
      data = SELF_PLAY_DATA;
      xLabel = '自我對弈局數';
      xDomain = [0, 120000];
    }

    // 比例尺
    const xScale = currentMode === 'selfplay'
      ? d3.scaleLog().domain([1, xDomain[1]]).range([0, innerWidth])
      : d3.scaleLinear().domain(xDomain).range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, 6000])
      .range([innerHeight, 0]);

    // 主繪圖區域
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // 繪製網格
    const gridGroup = g.append('g').attr('class', 'grid');

    gridGroup.selectAll('.grid-line-y')
      .data(yScale.ticks(6))
      .enter()
      .append('line')
      .attr('class', 'grid-line-y')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', d => yScale(d))
      .attr('y2', d => yScale(d))
      .attr('stroke', '#ddd')
      .attr('stroke-dasharray', '3,3');

    // 繪製人類參考線
    if (showMilestones && currentMode === 'zero') {
      const levelGroup = g.append('g').attr('class', 'human-levels');

      HUMAN_LEVELS.forEach(level => {
        levelGroup.append('line')
          .attr('x1', 0)
          .attr('x2', innerWidth)
          .attr('y1', yScale(level.elo))
          .attr('y2', yScale(level.elo))
          .attr('stroke', '#e74c3c')
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.6);

        levelGroup.append('text')
          .attr('x', innerWidth + 5)
          .attr('y', yScale(level.elo))
          .attr('dy', '0.35em')
          .attr('fill', '#e74c3c')
          .attr('font-size', 10)
          .text(level.label);
      });
    }

    // 繪製曲線
    const line = d3.line()
      .x(d => {
        if (currentMode === 'zero') return xScale(d.hours);
        if (currentMode === 'sl') return xScale(d.epochs);
        return xScale(Math.max(1, d.games));
      })
      .y(d => yScale(d.elo))
      .curve(d3.curveMonotoneX);

    // 面積
    const area = d3.area()
      .x(d => {
        if (currentMode === 'zero') return xScale(d.hours);
        if (currentMode === 'sl') return xScale(d.epochs);
        return xScale(Math.max(1, d.games));
      })
      .y0(innerHeight)
      .y1(d => yScale(d.elo))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(data)
      .attr('class', 'area')
      .attr('fill', '#4a90d9')
      .attr('opacity', 0.1)
      .attr('d', area);

    const path = g.append('path')
      .datum(data)
      .attr('class', 'line')
      .attr('fill', 'none')
      .attr('stroke', '#4a90d9')
      .attr('stroke-width', 3)
      .attr('d', line);

    // 動畫效果
    if (animated) {
      const totalLength = path.node().getTotalLength();
      path
        .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
        .attr('stroke-dashoffset', totalLength)
        .transition()
        .duration(2000)
        .ease(d3.easeLinear)
        .attr('stroke-dashoffset', 0);
    }

    // 繪製里程碑點
    if (showMilestones && currentMode === 'zero') {
      const milestones = data.filter(d => d.label);

      const milestoneGroup = g.append('g').attr('class', 'milestones');

      milestoneGroup.selectAll('circle')
        .data(milestones)
        .enter()
        .append('circle')
        .attr('cx', d => xScale(d.hours))
        .attr('cy', d => yScale(d.elo))
        .attr('r', 6)
        .attr('fill', '#e74c3c')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

      milestoneGroup.selectAll('text')
        .data(milestones)
        .enter()
        .append('text')
        .attr('x', d => xScale(d.hours))
        .attr('y', d => yScale(d.elo) - 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#333')
        .attr('font-size', 10)
        .text(d => d.label);
    }

    // X 軸
    const xAxis = currentMode === 'selfplay'
      ? d3.axisBottom(xScale).ticks(5, '~s')
      : d3.axisBottom(xScale);

    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${innerHeight})`)
      .call(xAxis);

    g.append('text')
      .attr('class', 'axis-label')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 45)
      .attr('text-anchor', 'middle')
      .attr('fill', '#666')
      .text(xLabel);

    // Y 軸
    g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale).ticks(6));

    g.append('text')
      .attr('class', 'axis-label')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -50)
      .attr('text-anchor', 'middle')
      .attr('fill', '#666')
      .text('ELO 評分');

    // 標題
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 14)
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text(currentMode === 'zero' ? 'AlphaGo Zero 訓練曲線'
           : currentMode === 'sl' ? '監督學習棋力成長'
           : '自我對弈棋力成長');

  }, [currentMode, width, height, animated, showMilestones, innerWidth, innerHeight]);

  return (
    <div>
      <div className="d3-controls">
        <button
          className={currentMode === 'zero' ? 'active' : ''}
          onClick={() => setCurrentMode('zero')}
        >
          AlphaGo Zero
        </button>
        <button
          className={currentMode === 'sl' ? 'active' : ''}
          onClick={() => setCurrentMode('sl')}
        >
          監督學習
        </button>
        <button
          className={currentMode === 'selfplay' ? 'active' : ''}
          onClick={() => setCurrentMode('selfplay')}
        >
          自我對弈
        </button>
      </div>

      <div className="elo-chart-container">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="elo-chart"
        />
      </div>

      <div className="d3-legend">
        <div className="d3-legend-item">
          <div className="d3-legend-color" style={{ background: '#4a90d9' }} />
          ELO 評分
        </div>
        {showMilestones && currentMode === 'zero' && (
          <>
            <div className="d3-legend-item">
              <div className="d3-legend-color" style={{ background: '#e74c3c' }} />
              里程碑
            </div>
            <div className="d3-legend-item">
              <div className="d3-legend-color" style={{ background: '#e74c3c', opacity: 0.3 }} />
              人類水平參考線
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function EloChart(props) {
  return (
    <BrowserOnly fallback={<div>載入中...</div>}>
      {() => <EloChartInner {...props} />}
    </BrowserOnly>
  );
}
