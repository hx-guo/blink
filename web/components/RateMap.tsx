"use client";

import { geoEquirectangular, geoPath } from "d3-geo";
import { useMemo, useState } from "react";
import { feature } from "topojson-client";
import type { Topology } from "topojson-specification";
import land50 from "world-atlas/land-50m.json";
import { useMounted } from "@/lib/useMounted";
import { useViewScale } from "@/lib/useWidth";

export type RateCell = { lat0: number; lon0: number; size: number; events: number; hours: number };

const W = 960;
const ML = 44, MB = 26, MT = 1, MR = 1;
/** 轨道倾角 43°，率图只画到 ±50° */
const LAT_MIN = -50;
const LAT_MAX = 50;
const PW = W - ML - MR;
const PH = Math.round((PW / 360) * (LAT_MAX - LAT_MIN));

/** 率的分档（每小时 TGF 数），近似对数等距；色阶由纸色到墨色，深浅两套主题各自成立 */
const STEPS = [0.02, 0.05, 0.1, 0.2, 0.5, 1];
const INK = [6, 16, 28, 42, 58, 74, 90];
const shade = (k: number) => `color-mix(in oklab, var(--ink) ${INK[k]}%, var(--page))`;
const classOf = (r: number) => STEPS.filter((s) => r >= s).length;

const lonLabel = (l: number) => (l === 0 || Math.abs(l) === 180 ? `${Math.abs(l)}°` : `${Math.abs(l)}°${l < 0 ? "W" : "E"}`);
const latLabel = (l: number) => (l === 0 ? "0°" : `${Math.abs(l)}°${l < 0 ? "S" : "N"}`);

/**
 * 每格 TGF 数 / 该格的搜索活时间（小时），与首页地图同一图框与刻度。活时间不足 `minHours`
 * 的格不画（SAA 与轨道边缘），那里的率没有意义。
 */
export default function RateMap({ cells, minHours }: { cells: RateCell[]; minHours: number }) {
  const mounted = useMounted();
  const [scaleRef, k] = useViewScale<HTMLDivElement>(PW + MR);
  const [hover, setHover] = useState<RateCell | null>(null);
  const { land, projection } = useMemo(() => {
    const topo = land50 as unknown as Topology;
    const projection = geoEquirectangular()
      .scale(PW / (2 * Math.PI))
      .translate([ML + PW / 2, MT + (LAT_MAX / (LAT_MAX - LAT_MIN)) * PH]);
    return { projection, land: geoPath(projection).digits(1)(feature(topo, topo.objects.land as never)) ?? "" };
  }, []);
  const H = MT + PH + MB * k;
  if (!mounted) return <div ref={scaleRef} style={{ aspectRatio: `${PW + MR} / ${H + 40}` }} />;

  const xy = (lon: number, lat: number) => projection([lon, lat]) ?? [0, 0];
  const west = (c: RateCell) => (c.lon0 >= 180 ? c.lon0 - 360 : c.lon0);
  const shown = cells.filter((c) => c.hours >= minHours && c.lat0 + c.size > LAT_MIN && c.lat0 < LAT_MAX);
  const hoverXY = hover ? xy(west(hover) + hover.size / 2, hover.lat0 + hover.size / 2) : null;
  const left = hoverXY ? hoverXY[0] < W * 0.62 : true;

  return (
    <div style={{ paddingLeft: k > 1.2 ? 40 : 0 }}>
      <div ref={scaleRef} className="relative">
        <svg viewBox={`${ML} 0 ${PW + MR} ${H}`} className="w-full h-auto" style={{ overflow: "visible" }} role="img" aria-label="TGFs per hour of exposure in 5° cells">
          <defs>
            <clipPath id="rate-frame">
              <rect x={ML} y={MT} width={PW} height={PH} />
            </clipPath>
          </defs>
          <g clipPath="url(#rate-frame)">
            {shown.map((c, i) => {
              const [x0, y0] = xy(west(c), c.lat0 + c.size);
              const [x1, y1] = xy(west(c) + c.size, c.lat0);
              return (
                <rect
                  key={i}
                  x={x0}
                  y={y0}
                  width={x1 - x0 + 0.3}
                  height={y1 - y0 + 0.3}
                  fill={shade(classOf(c.events / c.hours))}
                  onMouseEnter={() => setHover(c)}
                  onMouseLeave={() => setHover(null)}
                />
              );
            })}
            {/* 海岸线：纸色衬底上一道墨线，深浅格上都看得见 */}
            <path d={land} fill="none" stroke="var(--page)" strokeWidth={1.8} strokeOpacity={0.8} pointerEvents="none" />
            <path d={land} fill="none" stroke="var(--ink)" strokeWidth={0.6} strokeOpacity={0.8} pointerEvents="none" />
            {hover && hoverXY && (() => {
              const [x0, y0] = xy(west(hover), hover.lat0 + hover.size);
              const [x1, y1] = xy(west(hover) + hover.size, hover.lat0);
              return <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke="var(--accent)" strokeWidth={1.2} pointerEvents="none" />;
            })()}
          </g>
          <rect x={ML} y={MT} width={PW} height={PH} fill="none" stroke="var(--border-strong)" strokeWidth={0.75 * k} />
          {[-120, -60, 0, 60, 120].map((l) => {
            const [x] = xy(l, 0);
            return (
              <g key={l}>
                <line x1={x} x2={x} y1={MT + PH} y2={MT + PH + 4 * k} stroke="var(--border-strong)" strokeWidth={0.75 * k} />
                <text x={x} y={MT + PH + 19 * k} textAnchor="middle" fontSize={13 * k} fill="var(--text-secondary)">{lonLabel(l)}</text>
              </g>
            );
          })}
          {[-40, -20, 0, 20, 40].map((l) => {
            const [, y] = xy(0, l);
            return (
              <g key={l}>
                <line x1={ML - 4 * k} x2={ML} y1={y} y2={y} stroke="var(--border-strong)" strokeWidth={0.75 * k} />
                <text x={ML - 8 * k} y={y} dy="0.35em" textAnchor="end" fontSize={13 * k} fill="var(--text-secondary)">{latLabel(l)}</text>
              </g>
            );
          })}
        </svg>
        {hover && hoverXY && (
          <div
            className="map-tip"
            style={{
              left: `${((hoverXY[0] - ML + (left ? 14 : -14)) / (PW + MR)) * 100}%`,
              top: `${(hoverXY[1] / H) * 100}%`,
              transform: `translate(${left ? "0" : "-100%"}, -50%)`,
            }}
          >
            <div className="tabular" style={{ fontWeight: 600 }}>
              {latLabel(hover.lat0)} to {latLabel(hover.lat0 + hover.size)}, {lonLabel(west(hover))} to {lonLabel(west(hover) + hover.size)}
            </div>
            <div className="tabular">{(hover.events / hover.hours).toFixed(3)} TGFs per hour</div>
            <div className="tabular secondary">{hover.events} TGFs in {hover.hours.toFixed(0)} h</div>
          </div>
        )}
      </div>
      <div className="flex items-start gap-4 caption mt-4 flex-wrap">
        <span>TGFs per hour of exposure</span>
        <svg width={INK.length * 40 + 20} height={34} aria-hidden>
          {INK.map((_, k) => (
            <rect key={k} x={10 + k * 40} y={2} width={40} height={10} fill={shade(k)} />
          ))}
          {STEPS.map((s, k) => (
            <g key={s}>
              <line x1={10 + (k + 1) * 40} x2={10 + (k + 1) * 40} y1={12} y2={16} stroke="var(--border-strong)" strokeWidth={0.75 * k} />
              <text x={10 + (k + 1) * 40} y={29} textAnchor="middle" fontSize={13} fill="var(--text-secondary)" className="tabular">{s}</text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
