/**
 * WinrateSpread - 2017 年 Leela 實測：同一局面在三種條件下的勝率差
 *
 * 資料來源：協會 2017 年 3–4 月的測試紀錄，原始 SGF 共 313 檔，
 * 三組共同的（局面、候選點）組合 92 筆。完整資料見
 * /data/leela-2017-winrates.json
 */
import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

const SETS = [
  { key: 'mba30', label: 'MacBook Air／30 秒', color: 'var(--chart-blue)' },
  { key: 'cj30', label: 'CJSCOPE／30 秒', color: 'var(--chart-green)' },
  { key: 'mba60', label: 'MacBook Air／60 秒', color: 'var(--chart-red)' },
];

function WinrateSpreadInner({ mode = 'spread', width = 700, height = 420 }) {
  const svgRef = useRef(null);
  const [rows, setRows] = useState(null);
  const [failed, setFailed] = useState(false);
  const [currentMode, setCurrentMode] = useState(mode);

  useEffect(() => {
    let alive = true;
    fetch('/data/leela-2017-winrates.json')
      .then((r) => r.json())
      .then((d) => { if (alive) setRows(d.資料); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!svgRef.current || !rows) return;
    const margin = { top: 28, right: 24, bottom: 52, left: currentMode === 'spread' ? 52 : 92 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const axisText = (sel) => sel.selectAll('text')
      .attr('fill', 'var(--chart-ink)').attr('font-size', 12);
    const axisLine = (sel) => sel.selectAll('path,line').attr('stroke', 'var(--chart-grid)');

    if (currentMode === 'spread') {
      // 92 筆的差距分布：每一筆一個點，依差距排序
      const sorted = [...rows].sort((a, b) => a.spread - b.spread);
      const x = d3.scaleLinear().domain([0, d3.max(sorted, (d) => d.spread) * 1.05]).range([0, innerWidth]);
      const y = d3.scaleLinear().domain([0, sorted.length - 1]).range([innerHeight, 0]);
      const median = d3.median(sorted, (d) => d.spread);

      g.append('g').attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).ticks(6).tickFormat((v) => `${v}`))
        .call(axisText).call(axisLine);
      g.append('g').call(d3.axisLeft(y).ticks(5)).call(axisText).call(axisLine);

      g.append('text').attr('x', innerWidth / 2).attr('y', innerHeight + 40)
        .attr('text-anchor', 'middle').attr('fill', 'var(--chart-ink)').attr('font-size', 13)
        .text('三組之間的勝率最大差距（百分點）');
      g.append('text').attr('transform', 'rotate(-90)').attr('x', -innerHeight / 2).attr('y', -38)
        .attr('text-anchor', 'middle').attr('fill', 'var(--chart-ink)').attr('font-size', 13)
        .text('第幾筆（依差距排序）');

      g.append('line').attr('x1', x(median)).attr('x2', x(median))
        .attr('y1', 0).attr('y2', innerHeight)
        .attr('stroke', 'var(--chart-ink-muted)').attr('stroke-dasharray', '4 4');
      g.append('text').attr('x', x(median) + 6).attr('y', 14)
        .attr('fill', 'var(--chart-ink-soft)').attr('font-size', 12)
        .text(`中位數 ${median.toFixed(2)}`);

      g.selectAll('circle').data(sorted).enter().append('circle')
        .attr('cx', (d) => x(d.spread)).attr('cy', (_, i) => y(i)).attr('r', 3.5)
        .attr('fill', (d) => (d.spread >= 10 ? 'var(--chart-red)' : 'var(--chart-blue)'))
        .attr('fill-opacity', 0.85)
        .append('title')
        .text((d) => `第 ${d.move} 手 ${d.pt}：${d.mba30} / ${d.cj30} / ${d.mba60}（差 ${d.spread}）`);

      sorted.filter((d) => d.spread >= 10).forEach((d) => {
        const i = sorted.indexOf(d);
        g.append('text').attr('x', x(d.spread) - 8).attr('y', y(i) + 4)
          .attr('text-anchor', 'end').attr('fill', 'var(--chart-ink)').attr('font-size', 12)
          .text(`第 ${d.move} 手 ${d.pt}`);
      });
    } else {
      // 差距最大的十筆：三組並排比較
      const top = [...rows].sort((a, b) => b.spread - a.spread).slice(0, 10);
      const y = d3.scaleBand().domain(top.map((d) => `第 ${d.move} 手 ${d.pt}`))
        .range([0, innerHeight]).padding(0.28);
      const x = d3.scaleLinear().domain([0, 70]).range([0, innerWidth]);
      const ySub = d3.scaleBand().domain(SETS.map((s) => s.key)).range([0, y.bandwidth()]).padding(0.15);

      g.append('g').attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).ticks(7).tickFormat((v) => `${v}%`))
        .call(axisText).call(axisLine);
      g.append('g').call(d3.axisLeft(y)).call(axisText).call(axisLine);
      g.append('text').attr('x', innerWidth / 2).attr('y', innerHeight + 42)
        .attr('text-anchor', 'middle').attr('fill', 'var(--chart-ink)').attr('font-size', 13)
        .text('Leela 給這個候選點的勝率');

      g.append('line').attr('x1', x(50)).attr('x2', x(50)).attr('y1', 0).attr('y2', innerHeight)
        .attr('stroke', 'var(--chart-ink-muted)').attr('stroke-dasharray', '4 4');

      top.forEach((d) => {
        const row = g.append('g').attr('transform', `translate(0,${y(`第 ${d.move} 手 ${d.pt}`)})`);
        SETS.forEach((s) => {
          row.append('rect')
            .attr('x', 0).attr('y', ySub(s.key))
            .attr('width', x(d[s.key])).attr('height', ySub.bandwidth())
            .attr('fill', s.color).attr('fill-opacity', 0.85)
            .append('title').text(`${s.label}：${d[s.key]}%`);
          row.append('text')
            .attr('x', x(d[s.key]) + 5).attr('y', ySub(s.key) + ySub.bandwidth() / 2 + 4)
            .attr('fill', 'var(--chart-ink)').attr('font-size', 11)
            .text(`${d[s.key]}`);
        });
      });
    }
  }, [rows, currentMode, width, height]);

  if (failed) return <div className="chart"><p className="chart-caption">圖表資料載入失敗，資料本身在 <a href="/data/leela-2017-winrates.json">leela-2017-winrates.json</a>。</p></div>;

  return (
    <div className="chart">
      <div className="d3-controls">
        <button type="button" className={currentMode === 'spread' ? 'active' : ''} onClick={() => setCurrentMode('spread')} aria-pressed={currentMode === 'spread'}>92 筆的差距分布</button>
        <button type="button" className={currentMode === 'top' ? 'active' : ''} onClick={() => setCurrentMode('top')} aria-pressed={currentMode === 'top'}>差距最大的十筆</button>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} width="100%" height="auto" role="img"
        aria-label="2017 年 Leela 在三種機器與思考秒數下，對同一盤棋 92 個候選點給出的勝率差異" />
      {currentMode === 'top' && (
        <div className="d3-legend">
          {SETS.map((s) => (
            <div className="d3-legend-item" key={s.key}>
              <div className="d3-legend-color" style={{ background: s.color }} />
              {s.label}
            </div>
          ))}
        </div>
      )}
      <p className="chart-caption">
        資料：協會 2017 年 3–4 月的 Leela 測試紀錄，三組共同的候選點 92 筆。
        原始數據可下載：<a href="/data/leela-2017-winrates.json">leela-2017-winrates.json</a>
      </p>
    </div>
  );
}

export default function WinrateSpread(props) {
  return <WinrateSpreadInner {...props} />;
}
