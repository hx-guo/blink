"use client";

import { useMemo, useRef, useState } from "react";
import Sup from "./Sup";
import { useViewScale } from "@/lib/useWidth";

const W = 960;
const PLOT_H = 112;
const DAY = 86400000;

/**
 * 范围选择：事件按所选量的直方图，按住拖动选一段，点一下选中那一根柱子。时间按 UTC 日取整，
 * 对数量（时长、虚警率）在 log10 上分箱与选取。`selection` 用原始单位（毫秒或原值）。
 */
export default function RangeBrush({
  kind,
  values,
  selection,
  onChange,
  tick,
  bins = 96,
}: {
  kind: "time" | "log";
  values: number[];
  selection: [number, number] | null;
  onChange: (s: [number, number] | null) => void;
  /** 刻度标签，参数为原始单位 */
  tick: (v: number) => string;
  bins?: number;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [scaleRef, k] = useViewScale<HTMLDivElement>(W);
  const H = PLOT_H + 28 * k;
  const [drag, setDrag] = useState<{ from: number; to: number; x0: number } | null>(null);
  const t = (v: number) => (kind === "log" ? Math.log10(v) : v);
  const inv = (u: number) => (kind === "log" ? 10 ** u : u);

  const { lo, hi, counts, ticks } = useMemo(() => {
    const ts = values.map(t).filter(Number.isFinite);
    let lo = Math.min(...ts), hi = Math.max(...ts);
    if (kind === "time") {
      lo = Date.UTC(new Date(lo).getUTCFullYear(), new Date(lo).getUTCMonth(), 1);
      hi = Date.UTC(new Date(hi).getUTCFullYear(), new Date(hi).getUTCMonth() + 1, 1);
    } else {
      lo = Math.floor(lo * 4) / 4;
      hi = Math.ceil(hi * 4) / 4;
    }
    const counts = new Array(bins).fill(0);
    for (const u of ts) counts[Math.min(bins - 1, Math.floor(((u - lo) / (hi - lo)) * bins))]++;
    const ticks: number[] = [];
    if (kind === "time") {
      for (let y = new Date(lo).getUTCFullYear() + 1; Date.UTC(y, 0, 1) < hi; y++) ticks.push(Date.UTC(y, 0, 1));
    } else {
      const span = Math.ceil(hi) - Math.floor(lo);
      const step = span > 40 ? 10 : span > 16 ? 5 : span > 8 ? 2 : 1;
      for (let e = Math.ceil(lo / step) * step; e <= hi; e += step) ticks.push(e);
    }
    return { lo, hi, counts, ticks };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, kind, bins]);

  const x = (u: number) => ((u - lo) / (hi - lo)) * W;
  const uAt = (clientX: number) => {
    const r = svgRef.current!.getBoundingClientRect();
    return lo + Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * (hi - lo);
  };
  const snap = (a: number, b: number): [number, number] => {
    const [u0, u1] = a < b ? [a, b] : [b, a];
    if (kind === "time") return [Math.floor(u0 / DAY) * DAY, Math.ceil(u1 / DAY) * DAY];
    return [u0, u1];
  };
  const maxCount = Math.max(1, ...counts);
  const binW = W / bins;

  const shown: [number, number] | null = drag ? snap(drag.from, drag.to) : selection ? [t(selection[0]), t(selection[1])] : null;
  const inSel = (k: number) => !shown || (lo + (k + 1) * (hi - lo) / bins > shown[0] && lo + (k * (hi - lo)) / bins < shown[1]);

  return (
    <div ref={scaleRef}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto select-none"
        style={{ touchAction: "none", cursor: "crosshair" }}
        role="slider"
        aria-label="Drag to select a range"
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture?.(e.pointerId);
          const u = uAt(e.clientX);
          setDrag({ from: u, to: u, x0: e.clientX });
        }}
        onPointerMove={(e) => drag && setDrag({ ...drag, to: uAt(e.clientX) })}
        onPointerUp={(e) => {
          if (!drag) return;
          if (Math.abs(e.clientX - drag.x0) < 4) {
            // 点一下：选中那根柱子
            const k = Math.min(bins - 1, Math.floor(((drag.from - lo) / (hi - lo)) * bins));
            const [a, b] = snap(lo + (k * (hi - lo)) / bins, lo + ((k + 1) * (hi - lo)) / bins);
            onChange([inv(a), inv(b)]);
          } else {
            const [a, b] = snap(drag.from, drag.to);
            onChange([inv(a), inv(b)]);
          }
          setDrag(null);
        }}
      >
        {counts.map((c, k) => {
          const h = c ? Math.max(1.5, (c / maxCount) * (PLOT_H - 6)) : 0;
          return (
            <rect
              key={k}
              x={k * binW}
              y={PLOT_H - h}
              width={binW}
              shapeRendering="crispEdges"
              height={h}
              fill={shown && inSel(k) ? "var(--accent)" : "var(--ink)"}
              opacity={shown ? (inSel(k) ? 0.85 : 0.18) : 0.42}
            >
              <title>{`${c} TGFs, ${tick(inv(lo + (k * (hi - lo)) / bins))} – ${tick(inv(lo + ((k + 1) * (hi - lo)) / bins))}`}</title>
            </rect>
          );
        })}
        <line x1={0} x2={W} y1={PLOT_H + 0.5} y2={PLOT_H + 0.5} stroke="var(--border-strong)" strokeWidth={0.75} />
        {ticks.map((u) => (
          <g key={u}>
            <line x1={x(u)} x2={x(u)} y1={PLOT_H} y2={PLOT_H + 4 * k} stroke="var(--border-strong)" strokeWidth={0.75 * k} />
            <text x={x(u)} y={PLOT_H + 19 * k} textAnchor="middle" fontSize={13 * k} fill="var(--text-secondary)" className="tabular">
              <Sup text={tick(inv(u))} svg />
            </text>
          </g>
        ))}
        {shown && (
          <g pointerEvents="none">
            <rect x={x(shown[0])} y={0} width={Math.max(1, x(shown[1]) - x(shown[0]))} height={PLOT_H} fill="var(--accent-soft)" />
            <line x1={x(shown[0])} x2={x(shown[0])} y1={0} y2={PLOT_H} stroke="var(--accent)" strokeWidth={0.75} />
            <line x1={x(shown[1])} x2={x(shown[1])} y1={0} y2={PLOT_H} stroke="var(--accent)" strokeWidth={0.75} />
          </g>
        )}
      </svg>
    </div>
  );
}
