"use client";

import { geoEquirectangular, geoPath } from "d3-geo";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState } from "react";
import { feature } from "topojson-client";
import type { Topology } from "topojson-specification";
import land50 from "world-atlas/land-50m.json";
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
/** 图框外留给刻度标签的边 */
const ML = 44, MB = 26, MT = 1, MR = 1;
/** 画到的纬度范围：TGF 只在中低纬，高纬只留一点陆地作参照 */
const LAT_MIN = -60;
const LAT_MAX = 70;
const PW = W - ML - MR;
const PH = Math.round((PW / 360) * (LAT_MAX - LAT_MIN));
/** 挂载前占位的高度 */
export const MAP_ASPECT_H = PH + MT + MB;

/** 一台仪器的曝光：cell_deg 度的格，[格下沿纬度, 格西沿经度 0..360, 活时间秒] */
/** 经纬度框：[西经界, 东经界, 南纬界, 北纬界]，度 */
export type Region = [number, number, number, number];

export type ExposureGrid = { cell_deg: number; map: [number, number, number][] };

/**
 * 搜索覆盖的区域：有活时间的格（同纬度中位数 5 % 以上，去掉 SAA 里的零星秒数与坏星历的散点）
 * 放大、模糊后取阈值，5° 格的台阶变成光滑清晰的边。返回经度 −180..180、纬度自北向南的图。
 */
function exposureImage(grids: ExposureGrid[], fill: string): string | null {
  if (!grids.length || typeof document === "undefined") return null;
  const d = grids[0].cell_deg;
  const nx = 360 / d, ny = 180 / d;
  const sum = new Float64Array(nx * ny);
  for (const g of grids) {
    for (const [lat, lon, sec] of g.map) {
      const i = Math.floor((lat + 90) / d), j = Math.floor(((((lon + 180) % 360) + 360) % 360) / d);
      if (i >= 0 && i < ny && j >= 0 && j < nx) sum[(ny - 1 - i) * nx + j] += sec;
    }
  }
  const small = document.createElement("canvas");
  small.width = nx;
  small.height = ny;
  const sctx = small.getContext("2d")!;
  const cells = sctx.createImageData(nx, ny);
  for (let r = 0; r < ny; r++) {
    const row = Array.from(sum.subarray(r * nx, (r + 1) * nx));
    const med = [...row].sort((a, b) => a - b)[nx >> 1];
    row.forEach((v, j) => {
      cells.data[(r * nx + j) * 4 + 3] = med > 0 && v > 0.05 * med ? 255 : 0;
    });
  }
  sctx.putImageData(cells, 0, 0);
  const S = 10; // 每度 S/d 像素
  const big = document.createElement("canvas");
  big.width = nx * S;
  big.height = ny * S;
  const ctx = big.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.filter = `blur(${S * 0.6}px)`;
  ctx.drawImage(small, 0, 0, big.width, big.height);
  ctx.filter = "none";
  const img = ctx.getImageData(0, 0, big.width, big.height);
  const rgb = fill.match(/\w\w/g)!.map((h) => parseInt(h, 16));
  for (let k = 0; k < img.data.length; k += 4) {
    const on = img.data[k + 3] >= 128;
    img.data[k] = rgb[0];
    img.data[k + 1] = rgb[1];
    img.data[k + 2] = rgb[2];
    img.data[k + 3] = on ? 255 : 0;
  }
  ctx.putImageData(img, 0, 0);
  return big.toDataURL();
}

const lonLabel = (l: number) => (l === 0 || Math.abs(l) === 180 ? `${Math.abs(l)}°` : `${Math.abs(l)}°${l < 0 ? "W" : "E"}`);
const latLabel = (l: number) => (l === 0 ? "0°" : `${Math.abs(l)}°${l < 0 ? "S" : "N"}`);

/**
 * 首页地图，按论文插图的样子：图框、细经纬网、外向刻度，陆地只画轮廓。搜索覆盖的区域（有活时间
 * 的格，平滑后取阈值）铺一层纸色，轨道纬度范围与 SAA 由数据自己显出来。点悬停出提示，点击进
 * 事件页；`highlight` 非空时选中的点用朱红，其余淡出。
 */
export default function EventMap({
  events,
  highlight,
  exposures,
  region,
  onRegion,
}: {
  events: MapEvent[];
  highlight: Set<string> | null;
  exposures: ExposureGrid[];
  region: Region | null;
  /** 在空白处拖出一个框来选区域 */
  onRegion: (r: Region) => void;
}) {
  const mounted = useMounted();
  const router = useRouter();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<MapEvent | null>(null);
  const [drag, setDrag] = useState<{ a: [number, number]; b: [number, number] } | null>(null);
  const press = useRef<{ x: number; y: number; id: number } | null>(null);
  const suppressClick = useRef(false);
  const { land, projection, graticule } = useMemo(() => {
    const topo = land50 as unknown as Topology;
    const projection = geoEquirectangular()
      .scale(PW / (2 * Math.PI))
      .translate([ML + PW / 2, MT + (LAT_MAX / (LAT_MAX - LAT_MIN)) * PH]);
    const path = geoPath(projection).digits(1);
    const lines: string[] = [];
    for (let l = -150; l <= 150; l += 30) lines.push(path({ type: "LineString", coordinates: [[l, LAT_MIN], [l, 0], [l, LAT_MAX]] }) ?? "");
    for (let l = -60; l <= 60; l += 30) lines.push(path({ type: "LineString", coordinates: [[-180, l], [0, l], [180, l]] }) ?? "");
    return {
      projection,
      land: path(feature(topo, topo.objects.land as never)) ?? "",
      graticule: lines.join(""),
    };
  }, []);
  const image = useMemo(
    () => (mounted ? exposureImage(exposures, getComputedStyle(document.documentElement).getPropertyValue("--exposure").trim().slice(1) || "e6e0d3") : null),
    [mounted, exposures],
  );
  const H = MT + PH + MB;
  if (!mounted) return <div style={{ aspectRatio: `${PW + MR} / ${H}` }} />;

  const xy = (lon: number, lat: number) => projection([lon, lat]) ?? [0, 0];
  const [gx0, gy0] = xy(-180, 90);
  const [gx1, gy1] = xy(180, -90);
  const hoverXY = hover ? xy(hover.lon, hover.lat) : null;
  const boxLeft = hoverXY ? hoverXY[0] < W * 0.62 : true;
  // 屏幕坐标 → 图内坐标（夹在图框内）
  const toPlot = (cx: number, cy: number): [number, number] => {
    const pt = svgRef.current!.createSVGPoint();
    pt.x = cx;
    pt.y = cy;
    const p = pt.matrixTransform(svgRef.current!.getScreenCTM()!.inverse());
    return [Math.min(ML + PW, Math.max(ML, p.x)), Math.min(MT + PH, Math.max(MT, p.y))];
  };
  const box = drag ?? (region ? { a: xy(region[0], region[3]) as [number, number], b: xy(region[1], region[2]) as [number, number] } : null);
  const ordered = highlight ? [...events.filter((e) => !highlight.has(e.key)), ...events.filter((e) => highlight.has(e.key))] : events;

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`${ML} 0 ${PW + MR} ${H}`}
        className="w-full h-auto select-none"
        style={{ overflow: "visible", cursor: "crosshair", touchAction: "none" }}
        role="img"
        aria-label="Sub-satellite points of the TGFs over the area searched; drag to select a region"
        onPointerDown={(e) => {
          // 记下起点；移动超过几个像素才算框选，否则留给点的单击
          press.current = { x: e.clientX, y: e.clientY, id: e.pointerId };
        }}
        onPointerMove={(e) => {
          const p0 = press.current;
          if (!p0) return;
          if (!drag) {
            if (Math.hypot(e.clientX - p0.x, e.clientY - p0.y) < 5) return;
            (e.currentTarget as Element).setPointerCapture(p0.id);
            setHover(null);
            setDrag({ a: toPlot(p0.x, p0.y), b: toPlot(e.clientX, e.clientY) });
          } else setDrag({ ...drag, b: toPlot(e.clientX, e.clientY) });
        }}
        onPointerUp={() => {
          press.current = null;
          if (!drag) return;
          const [x0, x1] = [Math.min(drag.a[0], drag.b[0]), Math.max(drag.a[0], drag.b[0])];
          const [y0, y1] = [Math.min(drag.a[1], drag.b[1]), Math.max(drag.a[1], drag.b[1])];
          setDrag(null);
          // 拖完松手时浏览器还会补发一次单击，别让它打开事件页
          suppressClick.current = true;
          setTimeout(() => (suppressClick.current = false), 0);
          if (x1 - x0 < 4 || y1 - y0 < 4) return;
          const [w, n] = projection.invert!([x0, y0])!;
          const [e2, s2] = projection.invert!([x1, y1])!;
          onRegion([Math.round(w), Math.round(e2), Math.round(s2), Math.round(n)]);
        }}
      >
        <defs>
          <clipPath id="map-frame">
            <rect x={ML} y={MT} width={PW} height={PH} />
          </clipPath>
        </defs>
        <g clipPath="url(#map-frame)">
          {image && <image href={image} x={gx0} y={gy0} width={gx1 - gx0} height={gy1 - gy0} preserveAspectRatio="none" />}
          <path d={graticule} fill="none" stroke="var(--axis)" strokeWidth={0.35} strokeOpacity={0.6} />
          <path d={land} fill="none" stroke="var(--land)" strokeWidth={0.55} strokeOpacity={0.55} />
          {ordered.map((e) => {
            const [x, y] = xy(e.lon, e.lat);
            const on = !highlight || highlight.has(e.key);
            const picked = highlight && on;
            return (
              <circle
                key={e.key}
                cx={x}
                cy={y}
                r={hover === e ? 4 : picked ? 2 : 1.45}
                fill={e.provisional ? "var(--surface)" : picked ? "var(--accent)" : "var(--ink)"}
                stroke={e.provisional ? "var(--ink)" : "none"}
                strokeWidth={0.8}
                opacity={on ? (picked ? 0.9 : 0.42) : 0.08}
                style={{ cursor: "pointer" }}
                onMouseEnter={() => setHover(e)}
                onMouseLeave={() => setHover(null)}
                onClick={() => {
                  if (suppressClick.current) suppressClick.current = false;
                  else router.push(e.href);
                }}
              />
            );
          })}
        </g>
        {box && (
          <rect
            x={Math.min(box.a[0], box.b[0])}
            y={Math.min(box.a[1], box.b[1])}
            width={Math.abs(box.b[0] - box.a[0])}
            height={Math.abs(box.b[1] - box.a[1])}
            fill="var(--accent-soft)"
            stroke="var(--accent)"
            strokeWidth={0.75}
            pointerEvents="none"
          />
        )}
        <rect x={ML} y={MT} width={PW} height={PH} fill="none" stroke="var(--border-strong)" strokeWidth={0.75} />
        {[-120, -60, 0, 60, 120].map((l) => {
          const [x] = xy(l, 0);
          return (
            <g key={l}>
              <line x1={x} x2={x} y1={MT + PH} y2={MT + PH + 4} stroke="var(--border-strong)" strokeWidth={0.75} />
              <text x={x} y={MT + PH + 19} textAnchor="middle" fontSize={13} fill="var(--text-secondary)">{lonLabel(l)}</text>
            </g>
          );
        })}
        {[-30, 0, 30, 60].map((l) => {
          const [, y] = xy(0, l);
          return (
            <g key={l}>
              <line x1={ML - 4} x2={ML} y1={y} y2={y} stroke="var(--border-strong)" strokeWidth={0.75} />
              <text x={ML - 8} y={y} dy="0.35em" textAnchor="end" fontSize={13} fill="var(--text-secondary)">{latLabel(l)}</text>
            </g>
          );
        })}
      </svg>
      {hover && hoverXY && (
        <div
          className="map-tip"
          style={{
            left: `${((hoverXY[0] - ML + (boxLeft ? 12 : -12)) / (PW + MR)) * 100}%`,
            top: `${(hoverXY[1] / H) * 100}%`,
            transform: `translate(${boxLeft ? "0" : "-100%"}, -50%)`,
          }}
        >
          {hover.rows.map(([k, v], i) => (
            <div key={i}>
              {k && <span className="muted">{k} </span>}
              <span className={i === 0 ? "" : "tabular"} style={i === 0 ? { fontWeight: 600 } : undefined}>{v}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
