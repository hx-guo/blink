"use client";

import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import type { LightCurve as LC } from "@/lib/types";
import { useMounted } from "@/lib/useMounted";

const BIN_CHOICES = [10, 20, 50, 100];

/**
 * 候选窗附近的逐事例光变曲线：全部事例与其中 ACD 符合的事例两个系列，
 * 共用一个 y 轴（同一量纲的计数），叠放而非双轴。
 */
export function EventLightCurve({
  lc,
  startUs,
  stopUs,
}: {
  lc: LC;
  /** 候选窗相对峰值的起止（µs） */
  startUs: number;
  stopUs: number;
}) {
  const mounted = useMounted();
  const [binUs, setBinUs] = useState(20);
  const [hover, setHover] = useState<number | null>(null);
  const W = 720;
  const H = 240;
  const M = { top: 12, right: 10, bottom: 34, left: 40 };

  const { bins, lo, hi } = useMemo(() => {
    // 候选窗两侧各 1 ms；更远的事例（导出到 ±5 ms）只是本底
    const lo = Math.floor((startUs - 1000) / binUs) * binUs;
    const hi = Math.ceil((stopUs + 1000) / binUs) * binUs;
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
  }, [lc, binUs, startUs, stopUs]);

  const x = scaleLinear().domain([lo, hi]).range([M.left, W - M.right]);
  const y = scaleLinear()
    .domain([0, Math.max(2, ...bins.map((b) => b.all))])
    .nice()
    .range([H - M.bottom, M.top]);
  const bw = x(lo + binUs) - x(lo);
  const gap = bw > 4 ? 1 : 0;
  if (!mounted) return <div style={{ aspectRatio: `${W} / ${H + 30}` }} />;

  return (
    <div>
      <div className="flex items-center gap-4 text-sm mb-2 flex-wrap">
        <label className="secondary">
          Bin{" "}
          <select className="control" value={binUs} onChange={(e) => setBinUs(Number(e.target.value))}>
            {BIN_CHOICES.map((b) => (
              <option key={b} value={b}>{b} µs</option>
            ))}
          </select>
        </label>
        <span className="flex items-center gap-1 secondary">
          <span style={{ width: 10, height: 10, background: "var(--series-1)", borderRadius: 2, display: "inline-block" }} />
          All selected events
        </span>
        <span className="flex items-center gap-1 secondary">
          <span style={{ width: 10, height: 10, background: "var(--series-2)", borderRadius: 2, display: "inline-block" }} />
          In coincidence with an ACD
        </span>
      </div>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Light curve">
          <rect x={x(startUs)} y={M.top} width={Math.max(1, x(stopUs) - x(startUs))} height={H - M.top - M.bottom} fill="var(--wash)" />
          {y.ticks(4).map((t) => (
            <g key={t}>
              <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
              <text x={M.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--text-muted)" className="tabular">{t}</text>
            </g>
          ))}
          {bins.map((b, i) => (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={x(b.t0)} y={M.top} width={bw} height={H - M.top - M.bottom} fill="transparent" />
              {b.all > 0 && (
                <rect x={x(b.t0) + gap / 2} y={y(b.all)} width={Math.max(0.5, bw - gap)} height={y(0) - y(b.all)} fill="var(--series-1)" opacity={hover == null || hover === i ? 1 : 0.6} />
              )}
              {b.acd > 0 && (
                <rect x={x(b.t0) + gap / 2} y={y(b.acd)} width={Math.max(0.5, bw - gap)} height={y(0) - y(b.acd)} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={gap ? 1 : 0} />
              )}
            </g>
          ))}
          <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} stroke="var(--axis)" />
          {x.ticks(8).map((t) => (
            <text key={t} x={x(t)} y={H - M.bottom + 15} textAnchor="middle" fontSize={11} fill="var(--text-muted)" className="tabular">{t / 1000}</text>
          ))}
          <text x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
            Time from peak (ms) · shaded: candidate window
          </text>
        </svg>
        {hover != null && (
          <div className="tooltip" style={{ left: `${(x(bins[hover].t0 + binUs / 2) / W) * 100}%`, top: 0, transform: hover > bins.length * 0.7 ? "translateX(-105%)" : "translateX(8px)" }}>
            <div style={{ fontWeight: 600 }} className="tabular">
              {(bins[hover].t0 / 1000).toFixed(3)} to {((bins[hover].t0 + binUs) / 1000).toFixed(3)} ms
            </div>
            <div className="tabular">{bins[hover].all} events, {bins[hover].acd} with ACD</div>
          </div>
        )}
      </div>
    </div>
  );
}

/** [−2, +2) s 的 10 ms 计数：看候选之外有没有延展辐射（GRB 判据的输入） */
export function ContextLightCurve({ lc }: { lc: LC }) {
  const mounted = useMounted();
  const [hover, setHover] = useState<number | null>(null);
  const W = 720;
  const H = 150;
  const M = { top: 10, right: 10, bottom: 34, left: 40 };
  const n = lc.context.length;
  const t = (i: number) => lc.context_start_s + i * lc.context_bin_s;
  const x = scaleLinear().domain([t(0), t(n)]).range([M.left, W - M.right]);
  const y = scaleLinear().domain([0, Math.max(1, ...lc.context)]).nice().range([H - M.bottom, M.top]);
  const path = lc.context
    .map((c, i) => `${i === 0 ? "M" : "L"}${x(t(i))},${y(c)} H${x(t(i + 1))}`)
    .join(" ");
  if (!mounted) return <div style={{ aspectRatio: `${W} / ${H}` }} />;
  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto"
        role="img"
        aria-label="Counts per 10 ms"
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const sx = ((e.clientX - r.left) / r.width) * W;
          const i = Math.floor((x.invert(sx) - lc.context_start_s) / lc.context_bin_s);
          setHover(i >= 0 && i < n ? i : null);
        }}
        onMouseLeave={() => setHover(null)}
      >
        <rect x={x(-0.1)} y={M.top} width={x(0.3) - x(-0.1)} height={H - M.top - M.bottom} fill="var(--wash)" />
        {y.ticks(3).map((v) => (
          <g key={v}>
            <line x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
            <text x={M.left - 6} y={y(v)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--text-muted)" className="tabular">{v}</text>
          </g>
        ))}
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" />
        <line x1={M.left} x2={W - M.right} y1={y(0)} y2={y(0)} stroke="var(--axis)" />
        {hover != null && <line x1={x(t(hover) + lc.context_bin_s / 2)} x2={x(t(hover) + lc.context_bin_s / 2)} y1={M.top} y2={H - M.bottom} stroke="var(--text-muted)" strokeDasharray="2 2" />}
        {x.ticks(8).map((v) => (
          <text key={v} x={x(v)} y={H - M.bottom + 15} textAnchor="middle" fontSize={11} fill="var(--text-muted)" className="tabular">{v}</text>
        ))}
        <text x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
          Time from candidate start (s) · shaded: GRB test interval
        </text>
      </svg>
      {hover != null && (
        <div className="tooltip" style={{ left: `${(x(t(hover)) / W) * 100}%`, top: 0, transform: hover > n * 0.7 ? "translateX(-105%)" : "translateX(8px)" }}>
          <div className="tabular" style={{ fontWeight: 600 }}>{t(hover).toFixed(2)} s</div>
          <div className="tabular">{lc.context[hover]} events / 10 ms</div>
        </div>
      )}
    </div>
  );
}
