"use client";

import { geoEqualEarth, geoGraticule10, geoPath } from "d3-geo";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { feature } from "topojson-client";
import type { Topology } from "topojson-specification";
import land110 from "world-atlas/land-110m.json";
import { useMounted } from "@/lib/useMounted";

export type MapPoint = {
  lon: number;
  lat: number;
  color: string;
  href: string;
  title: string;
  lines: string[];
  /** 空心标记：provisional 等需要与 final 区分的状态 */
  hollow?: boolean;
};

const WIDTH = 960;
const HEIGHT = 470;

export default function WorldMap({ points, height = HEIGHT }: { points: MapPoint[]; height?: number }) {
  const router = useRouter();
  const mounted = useMounted();
  const [hover, setHover] = useState<{ p: MapPoint; x: number; y: number } | null>(null);
  const { landPath, gridPath, outline, project } = useMemo(() => {
    const land = feature(land110 as unknown as Topology, (land110 as unknown as Topology).objects.land);
    const projection = geoEqualEarth().fitExtent(
      [[4, 4], [WIDTH - 4, height - 4]],
      { type: "Sphere" },
    );
    const path = geoPath(projection).digits(1);
    return {
      landPath: path(land) ?? "",
      gridPath: path(geoGraticule10()) ?? "",
      outline: path({ type: "Sphere" }) ?? "",
      project: (lon: number, lat: number) => projection([lon, lat]),
    };
  }, [height]);

  if (!mounted) {
    return <div style={{ aspectRatio: `${WIDTH} / ${height}` }} />;
  }
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${WIDTH} ${height}`} className="w-full h-auto" role="img" aria-label="Map of TGF sub-satellite points">
        <path d={outline} fill="var(--surface)" stroke="var(--axis)" />
        <path d={gridPath} fill="none" stroke="var(--grid)" strokeWidth={0.5} />
        <path d={landPath} fill="var(--wash)" stroke="var(--axis)" strokeWidth={0.6} />
        {points.map((p, i) => {
          const xy = project(p.lon, p.lat);
          if (!xy) return null;
          return (
            <circle
              key={i}
              cx={xy[0]}
              cy={xy[1]}
              r={points.length === 1 ? 7 : hover?.p === p ? 5 : 2.6}
              fill={p.hollow ? "var(--surface)" : p.color}
              fillOpacity={p.hollow ? 1 : 0.75}
              stroke={p.hollow ? p.color : "var(--surface)"}
              strokeWidth={p.hollow ? 1.5 : 0.6}
              style={{ cursor: p.href === "#" ? "default" : "pointer" }}
              onMouseEnter={() => setHover({ p, x: xy[0], y: xy[1] })}
              onMouseLeave={() => setHover(null)}
              onClick={() => p.href !== "#" && router.push(p.href)}
            />
          );
        })}
      </svg>
      {hover && (
        <div
          className="tooltip"
          style={{
            left: `${(hover.x / WIDTH) * 100}%`,
            top: `${(hover.y / height) * 100}%`,
            transform: `translate(${hover.x > WIDTH * 0.7 ? "-105%" : "8px"}, -50%)`,
          }}
        >
          <div style={{ fontWeight: 600 }}>{hover.p.title}</div>
          {hover.p.lines.map((l) => (
            <div key={l} className="secondary">{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}
