"use client";

import { geoEqualEarth, geoGraticule10, geoPath } from "d3-geo";
import { scaleQuantize } from "d3-scale";
import { useMemo, useState } from "react";
import { feature } from "topojson-client";
import type { Topology } from "topojson-specification";
import land110 from "world-atlas/land-110m.json";
import { useMounted } from "@/lib/useMounted";

/** 顺序色阶（单一蓝色相，浅 → 深），来自 dataviz 参考调色板 */
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

export type RateCell = { lat0: number; lon0: number; size: number; events: number; hours: number };

const WIDTH = 960;
const HEIGHT = 470;

/**
 * 每格事件数 / 曝光小时数。曝光太少的格不上色（率没有意义），图例写明阈值。
 */
export default function RateMap({ cells, minHours }: { cells: RateCell[]; minHours: number }) {
  const mounted = useMounted();
  const [hover, setHover] = useState<{ c: RateCell; x: number; y: number } | null>(null);
  const shown = cells.filter((c) => c.hours >= minHours);
  const rates = shown.map((c) => c.events / c.hours);
  const max = Math.max(1e-9, ...rates);
  const color = scaleQuantize<string>().domain([0, max]).range(RAMP);
  const { path, land, grid, outline } = useMemo(() => {
    const projection = geoEqualEarth().fitExtent([[4, 4], [WIDTH - 4, HEIGHT - 4]], { type: "Sphere" });
    const path = geoPath(projection).digits(1);
    const topo = land110 as unknown as Topology;
    return {
      path,
      land: path(feature(topo, topo.objects.land)) ?? "",
      grid: path(geoGraticule10()) ?? "",
      outline: path({ type: "Sphere" }) ?? "",
    };
  }, []);
  if (!mounted) return <div style={{ aspectRatio: `${WIDTH} / ${HEIGHT + 40}` }} />;

  // 经纬格画成多边形（沿纬线加点，投影后边是曲的）
  const cellPath = (c: RateCell) => {
    const lon0 = c.lon0 > 180 ? c.lon0 - 360 : c.lon0;
    const steps = 4;
    const ring: [number, number][] = [];
    for (let k = 0; k <= steps; k++) ring.push([lon0 + (c.size * k) / steps, c.lat0]);
    for (let k = steps; k >= 0; k--) ring.push([lon0 + (c.size * k) / steps, c.lat0 + c.size]);
    ring.push(ring[0]);
    return path({ type: "Polygon", coordinates: [ring.reverse()] }) ?? "";
  };

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full h-auto" role="img" aria-label="TGFs per hour of exposure">
        <path d={outline} fill="var(--surface)" stroke="var(--axis)" />
        {shown.map((c, i) => (
          <path
            key={i}
            d={cellPath(c)}
            fill={color(c.events / c.hours)}
            stroke="var(--surface)"
            strokeWidth={hover?.c === c ? 0 : 0.5}
            onMouseEnter={(e) => {
              const r = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
              setHover({ c, x: ((e.clientX - r.left) / r.width) * WIDTH, y: ((e.clientY - r.top) / r.height) * HEIGHT });
            }}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        <path d={grid} fill="none" stroke="var(--grid)" strokeWidth={0.4} pointerEvents="none" />
        <path d={land} fill="none" stroke="var(--text-secondary)" strokeWidth={0.7} pointerEvents="none" />
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: `${(hover.x / WIDTH) * 100}%`, top: `${(hover.y / HEIGHT) * 100}%`, transform: `translate(${hover.x > WIDTH * 0.7 ? "-105%" : "10px"}, -50%)` }}>
          <div style={{ fontWeight: 600 }} className="tabular">
            {hover.c.lat0}° to {hover.c.lat0 + hover.c.size}°, {hover.c.lon0 > 180 ? hover.c.lon0 - 360 : hover.c.lon0}° to {(hover.c.lon0 > 180 ? hover.c.lon0 - 360 : hover.c.lon0) + hover.c.size}°
          </div>
          <div className="tabular">{(hover.c.events / hover.c.hours).toFixed(3)} TGFs per hour</div>
          <div className="tabular secondary">{hover.c.events} TGFs in {hover.c.hours.toFixed(0)} h of exposure</div>
        </div>
      )}
      <div className="flex items-center gap-2 text-xs secondary mt-2 flex-wrap">
        <span>TGFs per hour of exposure</span>
        <span className="flex">
          {RAMP.map((c) => <span key={c} style={{ width: 28, height: 10, background: c, display: "inline-block" }} />)}
        </span>
        <span className="tabular">0 – {max.toFixed(2)}</span>
        <span className="muted">· cells with less than {minHours} h of exposure are blank</span>
      </div>
    </div>
  );
}
