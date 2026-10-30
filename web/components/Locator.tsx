"use client";

import { geoEquirectangular, geoPath } from "d3-geo";
import { useMemo } from "react";
import { feature } from "topojson-client";
import type { Topology } from "topojson-specification";
import land50 from "world-atlas/land-50m.json";
import type { Coverage } from "@/lib/coverage";
import { useMounted } from "@/lib/useMounted";
import { useViewScale } from "@/lib/useWidth";

const W = 960;
const ML = 44, MB = 26, MT = 1, MR = 1;
const LAT_MIN = -60;
const LAT_MAX = 70;
const PW = W - ML - MR;
const PH = Math.round((PW / 360) * (LAT_MAX - LAT_MIN));

const lonLabel = (l: number) => (l === 0 || Math.abs(l) === 180 ? `${Math.abs(l)}°` : `${Math.abs(l)}°${l < 0 ? "W" : "E"}`);
const latLabel = (l: number) => (l === 0 ? "0°" : `${Math.abs(l)}°${l < 0 ? "S" : "N"}`);

/** 事件页的定位图：与首页地图同一图框与覆盖区，一个点，细十字线引到图框 */
export default function Locator({ lon, lat, coverage }: { lon: number; lat: number; coverage: Coverage | null }) {
  const mounted = useMounted();
  const [scaleRef, k] = useViewScale<HTMLDivElement>(PW + MR);
  const { land, projection } = useMemo(() => {
    const topo = land50 as unknown as Topology;
    const projection = geoEquirectangular()
      .scale(PW / (2 * Math.PI))
      .translate([ML + PW / 2, MT + (LAT_MAX / (LAT_MAX - LAT_MIN)) * PH]);
    return { projection, land: geoPath(projection).digits(1)(feature(topo, topo.objects.land as never)) ?? "" };
  }, []);
  const H = MT + PH + MB * k;
  if (!mounted) return <div ref={scaleRef} style={{ aspectRatio: `${PW + MR} / ${H}` }} />;
  const xy = (lo: number, la: number) => projection([lo, la]) ?? [0, 0];
  const [px, py] = xy(lon, lat);
  const polygon = (pts: [number, number][]) => `M${pts.map(([lo, la]) => xy(lo, la).map((v) => v.toFixed(1)).join(",")).join("L")}Z`;

  return (
    <div style={{ paddingLeft: k > 1.2 ? 40 : 0 }}>
      <div ref={scaleRef}>
        <svg viewBox={`${ML} 0 ${PW + MR} ${H}`} className="w-full h-auto" style={{ overflow: "visible" }} role="img" aria-label={`Sub-satellite point ${latLabel(lat)}, ${lonLabel(lon)}`}>
          <defs>
            <clipPath id="loc-frame">
              <rect x={ML} y={MT} width={PW} height={PH} />
            </clipPath>
            {coverage && (
              <mask id="loc-coverage" maskUnits="userSpaceOnUse" x={ML} y={MT} width={PW} height={PH}>
                <rect x={ML} y={xy(0, coverage.inclination)[1]} width={PW} height={xy(0, -coverage.inclination)[1] - xy(0, coverage.inclination)[1]} fill="white" />
                <path d={polygon(coverage.saa)} fill="black" />
              </mask>
            )}
          </defs>
          <g clipPath="url(#loc-frame)">
            {coverage && <rect x={ML} y={MT} width={PW} height={PH} fill="var(--exposure)" mask="url(#loc-coverage)" />}
            <path d={land} fill="none" stroke="var(--land)" strokeWidth={0.55} strokeOpacity={0.55} />
            <line x1={ML} x2={ML + PW} y1={py} y2={py} stroke="var(--accent)" strokeWidth={0.6} strokeOpacity={0.6} />
            <line x1={px} x2={px} y1={MT} y2={MT + PH} stroke="var(--accent)" strokeWidth={0.6} strokeOpacity={0.6} />
            <circle cx={px} cy={py} r={5 * Math.sqrt(k)} fill="var(--accent)" stroke="var(--page)" strokeWidth={1.5} />
          </g>
          <rect x={ML} y={MT} width={PW} height={PH} fill="none" stroke="var(--border-strong)" strokeWidth={0.75 * k} />
          {[-120, -60, 0, 60, 120].map((l) => (
            <g key={l}>
              <line x1={xy(l, 0)[0]} x2={xy(l, 0)[0]} y1={MT + PH} y2={MT + PH + 4 * k} stroke="var(--border-strong)" strokeWidth={0.75 * k} />
              <text x={xy(l, 0)[0]} y={MT + PH + 19 * k} textAnchor="middle" fontSize={13 * k} fill="var(--text-secondary)">{lonLabel(l)}</text>
            </g>
          ))}
          {[-30, 0, 30, 60].map((l) => (
            <g key={l}>
              <line x1={ML - 4 * k} x2={ML} y1={xy(0, l)[1]} y2={xy(0, l)[1]} stroke="var(--border-strong)" strokeWidth={0.75 * k} />
              <text x={ML - 8 * k} y={xy(0, l)[1]} dy="0.35em" textAnchor="end" fontSize={13 * k} fill="var(--text-secondary)">{latLabel(l)}</text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
