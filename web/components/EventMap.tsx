"use client";

import { geoEquirectangular, geoPath } from "d3-geo";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { feature, mesh } from "topojson-client";
import type { Topology } from "topojson-specification";
import countries110 from "world-atlas/countries-110m.json";
import { useMounted } from "@/lib/useMounted";

export type MapEvent = {
  key: string;
  lon: number;
  lat: number;
  href: string;
  provisional: boolean;
  rows: [string, string][];
};

const W = 960;
/** 右侧留给 ±倾角标注的宽度 */
const LABEL_W = 90;
/** 画到的纬度范围：南极洲与北冰洋高纬对 TGF 没有意义 */
const LAT_MIN = -58;
const LAT_MAX = 82;
/** 挂载前占位的高度（等距圆柱投影：宽 870 对 350° 经度，140° 纬度） */
export const MAP_ASPECT_H = Math.round(((W - LABEL_W) / 350) * (LAT_MAX - LAT_MIN)) + 4;

/**
 * 首页地图：灰色陆地、±轨道倾角虚线、由曝光推出的 SAA 区（曝光近零的格，模糊成一块），
 * 事件点悬停出带虚线引线的提示框，点击进事件页。`highlight` 非空时其余点淡出。
 */
export default function EventMap({
  events,
  highlight,
  inclination,
  saaCells,
}: {
  events: MapEvent[];
  highlight: Set<string> | null;
  inclination: number | null;
  saaCells: [number, number, number][];
}) {
  const mounted = useMounted();
  const router = useRouter();
  const [hover, setHover] = useState<MapEvent | null>(null);
  const { land, borders, projection, path } = useMemo(() => {
    const topo = countries110 as unknown as Topology;
    // 按宽度定比例尺，高度跟着投影走，没有上下留白
    const projection = geoEquirectangular().fitWidth(W - LABEL_W, {
      type: "MultiPoint",
      coordinates: [[-170, LAT_MIN], [180, LAT_MAX]],
    });
    const path = geoPath(projection).digits(1);
    return {
      projection,
      path,
      land: path(feature(topo, topo.objects.countries)) ?? "",
      borders: path(mesh(topo, topo.objects.countries as never, (a, b) => a !== b)) ?? "",
    };
  }, []);
  const H = Math.ceil((projection([0, LAT_MIN]) ?? [0, 0])[1]) + 4;
  if (!mounted) return <div style={{ aspectRatio: `${W} / ${MAP_ASPECT_H}`, flex: 1 }} />;

  const xy = (lon: number, lat: number) => projection([lon, lat]) ?? [0, 0];
  const saa = saaCells.map(([lat0, lon0, d]) => {
    const l = lon0 > 180 ? lon0 - 360 : lon0;
    return path({ type: "Polygon", coordinates: [[[l, lat0], [l, lat0 + d], [l + d, lat0 + d], [l + d, lat0], [l, lat0]]] }) ?? "";
  });
  const saaCenter = saaCells.length
    ? xy(
        saaCells.reduce((a, c) => a + (c[1] > 180 ? c[1] - 360 : c[1]) + c[2] / 2, 0) / saaCells.length,
        saaCells.reduce((a, c) => a + c[0] + c[2] / 2, 0) / saaCells.length,
      )
    : null;
  const hoverXY = hover ? xy(hover.lon, hover.lat) : null;
  const boxLeft = hoverXY ? hoverXY[0] < W * 0.6 : true;

  return (
    <div className="relative" style={{ flex: 1, minWidth: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Sub-satellite points of the TGFs">
        <defs>
          <pattern id="saa-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--saa)" strokeWidth="1.2" />
          </pattern>
        </defs>
        <path d={land} fill="var(--land)" />
        <path d={borders} fill="none" stroke="var(--land-edge)" strokeWidth={0.8} />
        {saa.length > 0 && (
          <g>
            {saa.map((d, i) => <path key={i} d={d} fill="url(#saa-hatch)" stroke="url(#saa-hatch)" strokeWidth={0.6} />)}
          </g>
        )}
        {saaCenter && (
          <text x={saaCenter[0]} y={saaCenter[1]} textAnchor="middle" dy="0.35em" fontSize={12} fill="var(--text-secondary)" pointerEvents="none" paintOrder="stroke" stroke="var(--surface)" strokeWidth={3}>
            South Atlantic Anomaly · no data
          </text>
        )}
        {inclination != null &&
          [inclination, -inclination].map((lat) => {
            const [x0, y] = xy(-140, lat);
            const [x1] = xy(180, lat);
            return (
              <g key={lat}>
                <line x1={x0} x2={x1} y1={y} y2={y} stroke="var(--axis)" strokeWidth={1} strokeDasharray="4 4" />
                <text x={x1 + 8} y={y} dy="0.35em" fontSize={12} fill="var(--text-muted)">
                  {Math.abs(lat)}°{lat > 0 ? "N" : "S"}
                </text>
              </g>
            );
          })}
        {events.map((e) => {
          const [x, y] = xy(e.lon, e.lat);
          const on = !highlight || highlight.has(e.key);
          return (
            <circle
              key={e.key}
              cx={x}
              cy={y}
              r={hover === e ? 5 : on && highlight ? 3.2 : 2}
              fill={e.provisional ? "var(--surface)" : "var(--accent)"}
              stroke={e.provisional ? "var(--accent)" : "none"}
              strokeWidth={1.4}
              opacity={on ? (highlight ? 1 : 0.5) : 0.06}
              style={{ cursor: "pointer" }}
              onMouseEnter={() => setHover(e)}
              onMouseLeave={() => setHover(null)}
              onClick={() => router.push(e.href)}
            />
          );
        })}
        {hover && hoverXY && (
          <line
            x1={hoverXY[0]}
            y1={hoverXY[1]}
            x2={hoverXY[0] + (boxLeft ? 130 : -130)}
            y2={hoverXY[1]}
            stroke="var(--axis)"
            strokeWidth={1}
            strokeDasharray="3 3"
            pointerEvents="none"
          />
        )}
      </svg>
      {hover && hoverXY && (
        <div
          className="absolute text-xs"
          style={{
            left: `${((hoverXY[0] + (boxLeft ? 130 : -130)) / W) * 100}%`,
            top: `${(hoverXY[1] / H) * 100}%`,
            transform: `translate(${boxLeft ? "0" : "-100%"}, -50%)`,
            background: "var(--surface)",
            border: "1px solid var(--border-strong)",
            boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
            borderRadius: 8,
            padding: "8px 10px",
            lineHeight: 1.5,
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 10,
          }}
        >
          {hover.rows.map(([k, v]) => (
            <div key={k}><strong>{k}:</strong> <span className="tabular">{v}</span></div>
          ))}
        </div>
      )}
    </div>
  );
}
