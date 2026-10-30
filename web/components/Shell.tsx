"use client";

import Link from "next/link";
import { site } from "@/site.config";
import { BASE, instrumentHref, utc } from "@/lib/format";
import type { Index } from "@/lib/types";
import { useJson } from "@/lib/useData";

/** 多久以前，粗到小时或天 */
export function ago(t: string): string {
  const h = (Date.now() - Date.parse(t)) / 3.6e6;
  if (h < 1) return "within the hour";
  if (h < 48) return `${Math.round(h)} hours ago`;
  return `${Math.round(h / 24)} days ago`;
}

/** 页头：刊名、导航与最近一次更新（index.json，运行时取） */
export function Header() {
  const { data: index } = useJson<Index>("index.json");
  const demo = index?.instruments.some((i) => i.demo);
  const unvalidated = index?.instruments.filter((i) => !i.public) ?? [];
  const last = index?.instruments.map((i) => i.last_day).sort().slice(-1)[0];
  return (
    <header className="page pt-10">
      <div className="flex items-baseline gap-10 flex-wrap pb-4" style={{ borderBottom: "0.5px solid var(--border-strong)" }}>
        <Link href="/" className="nav" style={{ color: "var(--text-primary)" }}>
          <span style={{ fontSize: 15, letterSpacing: "0.16em", textTransform: "uppercase" }}>TGF Catalog</span>
        </Link>
        <nav className="flex items-baseline gap-6 label">
          <Link href="/" className="nav">Catalog</Link>
          {index?.instruments.map((i) => (
            <Link key={i.slug} href={instrumentHref(i.slug)} className="nav">{i.instrument}</Link>
          ))}
          <Link href="/about/" className="nav">About</Link>
        </nav>
        {index && (
          <span className="ml-auto label muted" title={`Pipeline run ${utc(index.generated_at, 0)} UTC; data through ${last}`}>
            Updated {ago(index.generated_at)}
          </span>
        )}
      </div>
      {(demo || unvalidated.length > 0) && (
        <div className="caption py-2" style={{ borderBottom: "0.5px solid var(--border)" }}>
          {demo && (
            <div>
              <strong>Demo data.</strong> Events are from the Insight-HXMT/HE catalog v6; their light curves, GRB tests
              and the exposure are simulated for development. Do not use these pages for science.
            </div>
          )}
          {unvalidated.length > 0 && (
            <div>
              <strong>Internal site.</strong> Not yet validated for release: {unvalidated.map((i) => i.instrument).join(", ")}.
            </div>
          )}
        </div>
      )}
    </header>
  );
}

export function Footer() {
  const { data: index } = useJson<Index>("index.json");
  return (
    <footer className="page pb-14">
      <div className="pt-3 flex flex-wrap justify-between gap-4 caption" style={{ borderTop: "0.5px solid var(--border-strong)" }}>
        <span>
          <Link href="/about/">About the catalog</Link>
          <span className="muted"> · </span>
          <a href={`${BASE}/THIRD-PARTY-NOTICES.txt`}>Third-party notices</a>
          {index && <span className="muted"> · pipeline {index.instruments.map((i) => i.blink_git).join(", ")}</span>}
        </span>
        <span className="flex gap-6">
          {site.email && <a href={`mailto:${site.email}`}>{site.email}</a>}
          {site.tel && <span>{site.tel}</span>}
        </span>
      </div>
    </footer>
  );
}

export function Loading({ error, what }: { error: string | null; what: string }) {
  return <p className={error ? "" : "muted"}>{error ? `Could not load ${what}: ${error}` : `Loading ${what}…`}</p>;
}
