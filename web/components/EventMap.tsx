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
/** 画到的纬度范围：南极洲与北冰洋高纬对 TGF 没有意义 */
const LAT_MIN = -58;
const LAT_MAX = 75;
/** 挂载前占位的高度（等距圆柱投影：宽 W 对 360° 经度） */
export const MAP_ASPECT_H = Math.round((W / 360) * (LAT_MAX - LAT_MIN));

/** 一台仪器的曝光：cell_deg 度的格，[格下沿纬度, 格西沿经度 0..360, 活时间秒] */
export type ExposureGrid = { cell_deg: number; map: [number, number, number][] };

/**
 * 曝光底图：各仪器活时间相加后画成一张小图，由浏览器平滑放大。轨道纬度范围与 SAA 都由
 * 数据自己显出来，每颗卫星不同也不用另外标。透明度取活时间的平方根，零曝光处全透明。
 */
function exposureImage(grids: ExposureGrid[]): string | null {
  if (!grids.length || typeof document === "undefined") return null;
  const d = grids[0].cell_deg;
  const nx = 360 / d, ny = 180 / d;
  const sum = new Float64Array(nx * ny);
  for (const g of grids) {
    for (const [lat, lon, sec] of g.map) {
      const i = Math.floor((lat + 90) / d), j = Math.floor((((lon + 180) % 360) + 360) % 360 / d);
      if (i >= 0 && i < ny && j >= 0 && j < nx) sum[(ny - 1 - i) * nx + j] += sec;
    }
  }
  const max = Math.max(...sum);
  if (!(max > 0)) return null;
  const canvas = document.createElement("canvas");
  canvas.width = nx;
  canvas.height = ny;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(nx, ny);
  for (let k = 0; k < sum.length; k++) {
    const f = sum[k] / max;
    img.data[k * 4] = img.data[k * 4 + 1] = img.data[k * 4 + 2] = 255;
    img.data[k * 4 + 3] = f < 1e-3 ? 0 : Math.round(255 * (0.25 + 0.75 * Math.sqrt(f)));
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

/**
 * 首页地图：曝光底图（墨色深浅 = 搜索的活时间）、灰色陆地轮廓与事件点。点悬停出提示框，
 * 点击进事件页；`highlight` 非空时其余点淡出。
 */

export default function EventMap({
  events,
  highlight,
  exposures,
}: {
  events: MapEvent[];
  highlight: Set<string> | null;
  exposures: ExposureGrid[];
}) {
  const mounted = useMounted();
  const router = useRouter();
  const [hover, setHover] = useState<MapEvent | null>(null);
  const { land, borders, projection } = useMemo(() => {
    const topo = countries110 as unknown as Topology;
    const projection = geoEquirectangular().fitWidth(W, {
      type: "MultiPoint",
      coordinates: [[-180, LAT_MIN], [180, LAT_MAX]],
    });
    const path = geoPath(projection).digits(1);
    return {
      projection,
      land: path(feature(topo, topo.objects.countries)) ?? "",
      borders: path(mesh(topo, topo.objects.countries as never, (a, b) => a !== b)) ?? "",
    };
  }, []);
  const image = useMemo(() => (mounted ? exposureImage(exposures) : null), [mounted, exposures]);
  const H = Math.ceil((projection([0, LAT_MIN]) ?? [0, 0])[1]);
  if (!mounted) return <div style={{ aspectRatio: `${W} / ${MAP_ASPECT_H}`, flex: 1 }} />;

  const xy = (lon: number, lat: number) => projection([lon, lat]) ?? [0, 0];
  const [gx0, gy0] = xy(-180, 90);
  const [gx1, gy1] = xy(180, -90);
  const hoverXY = hover ? xy(hover.lon, hover.lat) : null;
  const boxLeft = hoverXY ? hoverXY[0] < W * 0.6 : true;

  return (
    <div className="relative" style={{ flex: 1, minWidth: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Sub-satellite points of the TGFs over the exposure of the search">
        <path d={land} fill="var(--land)" />
        {image && (
          <g style={{ color: "var(--exposure)" }}>
            <mask id="exposure-mask">
              <image href={image} x={gx0} y={gy0} width={gx1 - gx0} height={gy1 - gy0} preserveAspectRatio="none" />
            </mask>
            <rect x={gx0} y={gy0} width={gx1 - gx0} height={gy1 - gy0} fill="var(--exposure)" mask="url(#exposure-mask)" opacity={0.3} />
          </g>
        )}
        <path d={land} fill="none" stroke="var(--text-muted)" strokeWidth={0.5} strokeOpacity={0.7} />
        <path d={borders} fill="none" stroke="var(--land-edge)" strokeWidth={0.5} strokeOpacity={0.6} />
        {events.map((e) => {
          const [x, y] = xy(e.lon, e.lat);
          const on = !highlight || highlight.has(e.key);
          return (
            <circle
              key={e.key}
              cx={x}
              cy={y}
              r={hover === e ? 5 : on && highlight ? 2.8 : 2}
              fill={e.provisional ? "var(--surface)" : "var(--accent)"}
              stroke={e.provisional ? "var(--accent)" : "none"}
              strokeWidth={1.2}
              opacity={on ? (highlight ? 0.95 : 0.6) : 0.07}
              style={{ cursor: "pointer" }}
              onMouseEnter={() => setHover(e)}
              onMouseLeave={() => setHover(null)}
              onClick={() => router.push(e.href)}
            />
          );
        })}
      </svg>
      {hover && hoverXY && (
        <div
          className="absolute text-xs"
          style={{
            left: `${((hoverXY[0] + (boxLeft ? 14 : -14)) / W) * 100}%`,
            top: `${(hoverXY[1] / H) * 100}%`,
            transform: `translate(${boxLeft ? "0" : "-100%"}, -50%)`,
            background: "var(--surface)",
            border: "1px solid var(--border-strong)",
            boxShadow: "0 4px 12px rgba(40,30,10,0.14)",
            padding: "8px 10px",
            lineHeight: 1.5,
            pointerEvents: "none",
            whiteSpace: "nowrap",
            zIndex: 10,
          }}
        >
          {hover.rows.map(([k, v]) => (
            <div key={k}><span className="secondary">{k}</span> <span className="tabular">{v}</span></div>
          ))}
        </div>
      )}
    </div>
  );
}
