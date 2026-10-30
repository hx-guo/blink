"use client";

import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import type { LightCurve as LC } from "@/lib/types";
import { useWidth } from "@/lib/useWidth";

export const BIN_CHOICES = [10, 20, 50, 100];
const M = { top: 8, right: 2, bottom: 30, left: 46 };

/** 窗口标记：候选窗铺浅底，最佳区间两条细线 */
function Marks({ x, y0, y1, window, best }: { x: (t: number) => number; y0: number; y1: number; window: [number, number]; best: [number, number] }) {
  return (
    <g pointerEvents="none">
      <rect x={x(window[0])} y={y0} width={Math.max(1, x(window[1]) - x(window[0]))} height={y1 - y0} fill="var(--wash)" />
      {best.map((t) => (
        <line key={t} x1={x(t)} x2={x(t)} y1={y0} y2={y1} stroke="var(--text-primary)" strokeWidth={0.75} strokeDasharray="3 3" />
      ))}
    </g>
  );
}

function XAxis({ x, y, ticks, fmt }: { x: (t: number) => number; y: number; ticks: number[]; fmt: (t: number) => string }) {
  return (
    <g>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={y} y2={y + 4} stroke="var(--border-strong)" strokeWidth={0.75} />
          <text x={x(t)} y={y + 19} textAnchor="middle" fontSize={13} fill="var(--text-secondary)" className="tabular">{fmt(t)}</text>
        </g>
      ))}
    </g>
  );
}

/**
 * 候选附近的逐事例数据：上面是计数直方图（全部事例墨色，其中 ACD 着火的叠成深墨），下面是每个
 * 事例的时间与能道，两图共用时间轴。候选窗铺浅底，最佳区间用虚线标出。
 */
export function EventLightCurve({
  lc,
  binUs,
  window,
  best,
}: {
  lc: LC;
  binUs: number;
  /** 候选窗、最佳区间相对峰值的起止（µs） */
  window: [number, number];
  best: [number, number];
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = 220;
  const HS = 150;

  const { bins, lo, hi } = useMemo(() => {
    // 候选窗两侧各 1 ms；更远的事例（导出到 ±5 ms）只是本底
    const lo = Math.floor((window[0] - 1000) / binUs) * binUs;
    const hi = Math.ceil((window[1] + 1000) / binUs) * binUs;
    const n = Math.max(1, Math.round((hi - lo) / binUs));
    const bins = Array.from({ length: n }, (_, i) => ({ t0: lo + i * binUs, all: 0, acd: 0 }));
    for (const [t, , , acd] of lc.events) {
      const k = Math.floor((t - lo) / binUs);
      if (k >= 0 && k < n) {
        bins[k].all += 1;
        if (acd) bins[k].acd += 1;
      }
    }
    return { bins, lo, hi };
  }, [lc, binUs, window]);

  const x = scaleLinear().domain([lo, hi]).range([M.left, W - M.right]);
  const y = scaleLinear().domain([0, Math.max(2, ...bins.map((b) => b.all))]).nice(4).range([H - M.bottom, M.top]);
  const inRange = lc.events.filter(([t]) => t >= lo && t < hi);
  const chMax = Math.max(...inRange.map((e) => e[2]), 1);
  const ys = scaleLinear().domain([0, chMax]).nice(3).range([HS - M.bottom, M.top]);
  const bw = x(lo + binUs) - x(lo);
  const ticks = x.ticks(8);
  const ms = (t: number) => `${+(t / 1000).toFixed(2)}`;
  const yTicks = y.ticks(4).filter(Number.isInteger);

  return (
    <div ref={ref} className="relative">
      <svg width={W} height={H} className="block" role="img" aria-label="Counts per bin around the candidate">
        <Marks x={x} y0={M.top} y1={H - M.bottom} window={window} best={best} />
        {yTicks.map((t) => (
          <g key={t}>
            {t > 0 && <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={0.75} />}
            <text x={M.left - 8} y={y(t)} dy="0.35em" textAnchor="end" fontSize={13} fill="var(--text-secondary)" className="tabular">{t}</text>
          </g>
        ))}
        {bins.map((b, i) => (
          <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <rect x={x(b.t0)} y={M.top} width={bw} height={H - M.top - M.bottom} fill="transparent" />
            {b.all > 0 && (
              <rect x={x(b.t0)} y={y(b.all)} width={bw} height={y(0) - y(b.all)} fill={hover === i ? "var(--accent)" : "var(--ink)"} opacity={0.4} shapeRendering="crispEdges" />
            )}
            {b.acd > 0 && <rect x={x(b.t0)} y={y(b.acd)} width={bw} height={y(0) - y(b.acd)} fill="var(--ink)" opacity={0.9} shapeRendering="crispEdges" />}
          </g>
        ))}
        <line x1={M.left} x2={W - M.right} y1={y(0) + 0.5} y2={y(0) + 0.5} stroke="var(--border-strong)" strokeWidth={0.75} />
        <XAxis x={x} y={y(0)} ticks={ticks} fmt={ms} />
      </svg>
      {hover != null && (
        <div
          className="tooltip"
          style={{ left: x(bins[hover].t0) + bw, top: 0, transform: hover > bins.length * 0.6 ? `translateX(calc(-100% - ${bw + 10}px))` : "translateX(10px)" }}
        >
          <div className="tabular" style={{ fontWeight: 600 }}>
            {ms(bins[hover].t0)} to {ms(bins[hover].t0 + binUs)} ms
          </div>
          <div className="tabular">
            {bins[hover].all} counts, {bins[hover].acd} with an anti-coincidence shield
          </div>
        </div>
      )}
      <svg width={W} height={HS} className="block mt-4" role="img" aria-label="Time and energy channel of each count">
        <Marks x={x} y0={M.top} y1={HS - M.bottom} window={window} best={best} />
        {ys.ticks(3).map((t) => (
          <g key={t}>
            {t > 0 && <line x1={M.left} x2={W - M.right} y1={ys(t)} y2={ys(t)} stroke="var(--grid)" strokeWidth={0.75} />}
            <text x={M.left - 8} y={ys(t)} dy="0.35em" textAnchor="end" fontSize={13} fill="var(--text-secondary)" className="tabular">{t}</text>
          </g>
        ))}
        {inRange.map(([t, , ch, acd], i) =>
          acd ? (
            <circle key={i} cx={x(t)} cy={ys(ch)} r={3} fill="var(--ink)" opacity={0.9} />
          ) : (
            <circle key={i} cx={x(t)} cy={ys(ch)} r={3} fill="none" stroke="var(--ink)" strokeWidth={0.9} opacity={0.6} />
          ),
        )}
        <line x1={M.left} x2={W - M.right} y1={ys(0) + 0.5} y2={ys(0) + 0.5} stroke="var(--border-strong)" strokeWidth={0.75} />
        <XAxis x={x} y={ys(0)} ticks={ticks} fmt={ms} />
      </svg>
    </div>
  );
}

/** [−2, +2) s 的 10 ms 计数：看候选之外有没有延展辐射（GRB 判据的输入） */
export function ContextLightCurve({ lc }: { lc: LC }) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const H = 170;
  const n = lc.context.length;
  const t = (i: number) => lc.context_start_s + i * lc.context_bin_s;
  const x = scaleLinear().domain([t(0), t(n)]).range([M.left, W - M.right]);
  const y = scaleLinear().domain([0, Math.max(1, ...lc.context)]).nice(3).range([H - M.bottom, M.top]);
  const path = lc.context.map((c, i) => `${i === 0 ? "M" : "L"}${x(t(i)).toFixed(1)},${y(c).toFixed(1)} H${x(t(i + 1)).toFixed(1)}`).join(" ");
  return (
    <div ref={ref} className="relative">
      <svg
        width={W}
        height={H}
        className="block"
        role="img"
        aria-label="Counts per 10 ms from 2 s before to 2 s after the candidate"
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const i = Math.floor((x.invert(e.clientX - r.left) - lc.context_start_s) / lc.context_bin_s);
          setHover(i >= 0 && i < n ? i : null);
        }}
        onMouseLeave={() => setHover(null)}
      >
        <rect x={x(-0.1)} y={M.top} width={x(0.3) - x(-0.1)} height={H - M.top - M.bottom} fill="var(--wash)" />
        {y.ticks(3).map((v) => (
          <g key={v}>
            {v > 0 && <line x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} stroke="var(--grid)" strokeWidth={0.75} />}
            <text x={M.left - 8} y={y(v)} dy="0.35em" textAnchor="end" fontSize={13} fill="var(--text-secondary)" className="tabular">{v}</text>
          </g>
        ))}
        <path d={path} fill="none" stroke="var(--ink)" strokeWidth={1} strokeLinejoin="round" opacity={0.85} />
        {hover != null && <line x1={x(t(hover) + lc.context_bin_s / 2)} x2={x(t(hover) + lc.context_bin_s / 2)} y1={M.top} y2={H - M.bottom} stroke="var(--accent)" strokeWidth={0.75} />}
        <line x1={M.left} x2={W - M.right} y1={y(0) + 0.5} y2={y(0) + 0.5} stroke="var(--border-strong)" strokeWidth={0.75} />
        <XAxis x={x} y={y(0)} ticks={x.ticks(8)} fmt={(v) => `${v}`} />
      </svg>
      {hover != null && (
        <div className="tooltip" style={{ left: x(t(hover)), top: 0, transform: hover > n * 0.6 ? "translateX(calc(-100% - 10px))" : "translateX(10px)" }}>
          <div className="tabular" style={{ fontWeight: 600 }}>{t(hover).toFixed(2)} s</div>
          <div className="tabular">{lc.context[hover]} counts in 10 ms</div>
        </div>
      )}
    </div>
  );
}
