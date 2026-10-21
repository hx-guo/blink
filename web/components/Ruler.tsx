"use client";

import { useState } from "react";

export type RulerBin = { label: string; count: number };

/**
 * 竖直刻度尺：事件按当前选中的量排序后等分成若干箱，一箱一根刻线（颜色由浅到深）。
 * 悬停的箱附近刻线鱼眼放大并在右侧标出数值；点击锁定该箱，再点一次取消。
 */
export default function Ruler({
  title,
  bins,
  height,
  active,
  locked,
  onHover,
  onToggle,
}: {
  title: string;
  bins: RulerBin[];
  height: number;
  active: number | null;
  locked: number | null;
  onHover: (bin: number | null) => void;
  onToggle: (bin: number) => void;
}) {
  const [pointerY, setPointerY] = useState<number | null>(null);
  const W = 170;
  const TOP = 8;
  const span = height - 2 * TOP;
  const n = bins.length;
  const yOf = (k: number) => TOP + (n <= 1 ? span / 2 : (k / (n - 1)) * span);
  const focusY = active != null ? yOf(active) : pointerY;
  const marks = n ? [0, Math.round((n - 1) * 0.25), Math.round((n - 1) * 0.5), Math.round((n - 1) * 0.75), n - 1] : [];
  const binAt = (y: number) => Math.max(0, Math.min(n - 1, Math.round(((y - TOP) / span) * (n - 1))));

  return (
    <svg
      width={W}
      height={height}
      viewBox={`0 0 ${W} ${height}`}
      role="slider"
      aria-label={`${title} distribution`}
      aria-valuemin={0}
      aria-valuemax={Math.max(0, n - 1)}
      aria-valuenow={active ?? undefined}
      style={{ cursor: "pointer", flex: "none" }}
      onMouseMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const y = ((e.clientY - r.top) / r.height) * height;
        setPointerY(y);
        onHover(binAt(y));
      }}
      onMouseLeave={() => {
        setPointerY(null);
        onHover(null);
      }}
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        onToggle(binAt(((e.clientY - r.top) / r.height) * height));
      }}
    >
      <text x={14} y={height / 2} transform={`rotate(-90 14 ${height / 2})`} textAnchor="middle" fontSize={15} fontWeight={700} fill="var(--text-primary)">
        {title}
      </text>
      {bins.map((b, k) => {
        const y = yOf(k);
        // 鱼眼：离焦点越近刻线越长、向两侧鼓出
        const bulge = focusY == null ? 0 : 12 * Math.exp(-(((y - focusY) / 10) ** 2));
        const x0 = 38 - bulge / 2;
        const len = 24 + bulge;
        const t = n <= 1 ? 1 : k / (n - 1);
        return (
          <line
            key={k}
            x1={x0}
            x2={x0 + len}
            y1={y}
            y2={y}
            stroke="var(--accent)"
            strokeOpacity={0.15 + 0.85 * t}
            strokeWidth={k === locked ? 3 : 1.6}
          />
        );
      })}
      {marks.map((k) => (
        <g key={`m${k}`} opacity={active != null && Math.abs(yOf(k) - yOf(active)) < 16 ? 0 : 1}>
          <circle cx={84} cy={yOf(k)} r={3} fill="var(--axis)" />
          <text x={92} y={yOf(k)} dy="0.32em" fontSize={10} fill="var(--text-secondary)" className="tabular">{bins[k].label}</text>
        </g>
      ))}
      {active != null && bins[active] && (
        <g>
          <circle cx={84} cy={yOf(active)} r={3.5} fill="var(--accent)" />
          <text x={92} y={yOf(active)} dy="0.32em" fontSize={10.5} fontWeight={600} fill="var(--accent)" className="tabular">
            {bins[active].label}
          </text>
        </g>
      )}
    </svg>
  );
}
