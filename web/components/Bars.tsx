"use client";

import { scaleLinear } from "d3-scale";
import { useState } from "react";
import { useWidth } from "@/lib/useWidth";

/** 一根柱：`err` 为 1σ 误差（可选），`tick` 为柱左沿的刻度标签（可选） */
export type Bin = { label: string; value: number; err?: number; tick?: string; detail?: string };

/**
 * 直方图，与首页直方图同一笔法：墨色、箱与箱紧挨、底线与外向刻度、淡网格。按容器的像素宽度画，
 * 字号不随缩放变小。y 轴从 0 到数据的最大值（含误差）取整。
 */
export default function Bars({
  bins,
  height = 200,
  unit = "",
  format = (v: number) => String(v),
  label,
  endTick,
}: {
  bins: Bin[];
  height?: number;
  unit?: string;
  /** y 轴与提示里的数值格式 */
  format?: (v: number) => string;
  label: string;
  /** 最后一根柱右沿的刻度标签 */
  endTick?: string;
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const M = { top: 8, right: 2, bottom: 28, left: 46 };
  const top = Math.max(...bins.map((b) => b.value + (b.err ?? 0)));
  const y = scaleLinear().domain([0, top > 0 ? top : 1]).nice(4).range([height - M.bottom, M.top]);
  const step = (W - M.left - M.right) / Math.max(1, bins.length);
  // 直方图：相邻箱紧挨着，柱宽即箱宽
  const barW = step;
  const x = (i: number) => M.left + i * step;

  return (
    <div ref={ref} className="relative">
      <svg width={W} height={height} className="block" role="img" aria-label={label}>
        {y.ticks(4).map((t) => (
          <g key={t}>
            {t > 0 && <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={0.75} />}
            <text x={M.left - 8} y={y(t)} dy="0.35em" textAnchor="end" fontSize={13} fill="var(--text-secondary)" className="tabular">
              {format(t)}
            </text>
          </g>
        ))}
        {bins.map((b, i) => {
          const h = y(0) - y(b.value);
          const on = hover == null || hover === i;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={M.left + i * step} y={M.top} width={step} height={height - M.top - M.bottom} fill="transparent" />
              {h > 0 && <rect x={x(i)} y={y(b.value)} width={barW} height={h} shapeRendering="crispEdges" fill={hover === i ? "var(--accent)" : "var(--ink)"} opacity={on ? 0.55 : 0.3} />}
              {b.err != null && b.err > 0 && (
                <line
                  x1={x(i) + barW / 2}
                  x2={x(i) + barW / 2}
                  y1={y(Math.max(0, b.value - b.err))}
                  y2={y(b.value + b.err)}
                  stroke="var(--ink)"
                  strokeWidth={0.9}
                  opacity={on ? 0.8 : 0.4}
                />
              )}
              {b.tick && (
                <g>
                  <line x1={M.left + i * step} x2={M.left + i * step} y1={y(0)} y2={y(0) + 4} stroke="var(--border-strong)" strokeWidth={0.75} />
                  <text x={M.left + i * step} y={y(0) + 19} textAnchor="middle" fontSize={13} fill="var(--text-secondary)" className="tabular">
                    {b.tick}
                  </text>
                </g>
              )}
            </g>
          );
        })}
        {endTick && (
          <g>
            <line x1={W - M.right} x2={W - M.right} y1={y(0)} y2={y(0) + 4} stroke="var(--border-strong)" strokeWidth={0.75} />
            <text x={W - M.right} y={y(0) + 19} textAnchor="end" fontSize={13} fill="var(--text-secondary)" className="tabular">
              {endTick}
            </text>
          </g>
        )}
        <line x1={M.left} x2={W - M.right} y1={y(0) + 0.5} y2={y(0) + 0.5} stroke="var(--border-strong)" strokeWidth={0.75} />
      </svg>
      {hover != null && (
        <div
          className="tooltip"
          style={{
            left: x(hover) + barW / 2,
            top: 0,
            transform: hover > bins.length * 0.6 ? "translateX(calc(-100% - 10px))" : "translateX(10px)",
          }}
        >
          <div style={{ fontWeight: 600 }}>{bins[hover].label}</div>
          <div className="tabular">
            {format(bins[hover].value)}
            {bins[hover].err != null && ` ± ${format(bins[hover].err!)}`}
            {unit}
          </div>
          {bins[hover].detail && <div className="tabular secondary">{bins[hover].detail}</div>}
        </div>
      )}
    </div>
  );
}
