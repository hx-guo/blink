"use client";

import { scaleLinear } from "d3-scale";
import { useState } from "react";

export type Bin = { label: string; value: number; tick?: string };

/** 单系列柱状图：细柱、柱间 2px 间隙、圆角只在数据端、悬停提示。 */
export default function Bars({
  bins,
  height = 180,
  yLabel,
  color = "var(--series-1)",
  unit = "",
}: {
  bins: Bin[];
  height?: number;
  yLabel?: string;
  color?: string;
  unit?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 720;
  const M = { top: 10, right: 8, bottom: 26, left: 40 };
  const max = Math.max(1, ...bins.map((b) => b.value));
  const y = scaleLinear().domain([0, max]).nice().range([height - M.bottom, M.top]);
  const step = (W - M.left - M.right) / Math.max(1, bins.length);
  const gap = Math.min(2, step * 0.3);
  const barW = Math.max(1, step - gap);
  const radius = Math.min(4, barW / 2);

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full h-auto" role="img" aria-label={yLabel}>
        {y.ticks(4).map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <text x={M.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--text-muted)" className="tabular">
              {t}
            </text>
          </g>
        ))}
        <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} stroke="var(--axis)" />
        {bins.map((b, i) => {
          const x = M.left + i * step + gap / 2;
          const top = y(b.value);
          const h = y(0) - top;
          const r = Math.min(radius, h);
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={M.left + i * step} y={M.top} width={step} height={height - M.top - M.bottom} fill="transparent" />
              {h > 0 && (
                <path
                  d={`M${x},${y(0)} V${top + r} Q${x},${top} ${x + r},${top} H${x + barW - r} Q${x + barW},${top} ${x + barW},${top + r} V${y(0)} Z`}
                  fill={color}
                  opacity={hover == null || hover === i ? 1 : 0.55}
                />
              )}
              {b.tick && (
                <text x={x + barW / 2} y={height - M.bottom + 15} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
                  {b.tick}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hover != null && (
        <div
          className="tooltip"
          style={{
            left: `${((M.left + (hover + 0.5) * step) / W) * 100}%`,
            top: 0,
            transform: hover > bins.length * 0.7 ? "translateX(-105%)" : "translateX(8px)",
          }}
        >
          <div style={{ fontWeight: 600 }}>{bins[hover].label}</div>
          <div className="tabular">{bins[hover].value}{unit}</div>
        </div>
      )}
    </div>
  );
}
