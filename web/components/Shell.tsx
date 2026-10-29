"use client";

import Link from "next/link";
import { site } from "@/site.config";
import { instrumentHref, utc } from "@/lib/format";
import type { Index } from "@/lib/types";
import { useJson } from "@/lib/useData";
import Logo from "./Logo";

/** 多久以前，粗到小时或天 */
export function ago(t: string): string {
  const h = (Date.now() - Date.parse(t)) / 3.6e6;
  if (h < 1) return "less than an hour ago";
  if (h < 48) return `${Math.round(h)} h ago`;
  return `${Math.round(h / 24)} days ago`;
}

/** 页头：站标、导航与最近一次更新（index.json，运行时取） */
export function Header() {
  const { data: index } = useJson<Index>("index.json");
  const demo = index?.instruments.some((i) => i.demo);
  const unvalidated = index?.instruments.filter((i) => !i.public) ?? [];
  const last = index?.instruments.map((i) => i.last_day).sort().slice(-1)[0];
  return (
    <>
      <header style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="mx-auto max-w-6xl px-4 h-14 flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2 shrink-0" style={{ color: "var(--text-primary)", textDecoration: "none" }}>
            <Logo size={30} />
            <span className="text-base font-semibold tracking-tight">TGF Catalog</span>
          </Link>
          <nav className="hidden sm:flex items-center gap-6 text-sm">
            <Link href="/" className="nav">Catalog</Link>
            {index?.instruments.map((i) => (
              <Link key={i.slug} href={instrumentHref(i.slug)} className="nav">{i.instrument}</Link>
            ))}
            <Link href="/about/" className="nav">About</Link>
          </nav>
          {index && (
            <div className="ml-auto flex items-center gap-2 text-xs secondary" title={`Pipeline run ${utc(index.generated_at, 0)} UTC; data through ${last}`}>
              <span className="live-dot" aria-hidden />
              <span>Updated {ago(index.generated_at)}</span>
            </div>
          )}
        </div>
      </header>
      {(demo || unvalidated.length > 0) && (
        <div className="mx-auto max-w-6xl px-4">
          <div className="text-sm rounded-md px-3 py-2 mt-2" style={{ background: "var(--banner)" }}>
            {demo && (
              <div>
                <strong>Demo data.</strong> Events are from the Insight-HXMT/HE catalog v6; their light curves, GRB
                tests and the exposure are simulated for development. Do not use these pages for science.
              </div>
            )}
            {unvalidated.length > 0 && (
              <div>
                <strong>Internal site.</strong> Not yet validated for release:{" "}
                {unvalidated.map((i) => i.instrument).join(", ")}.
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export function Footer() {
  const { data: index } = useJson<Index>("index.json");
  return (
    <footer className="mx-auto max-w-6xl px-4 py-8 flex flex-wrap justify-between gap-4 text-sm">
      <div className="muted text-xs">
        <Link href="/about/">About the catalog</Link>
        {index && <> · pipeline {index.instruments.map((i) => i.blink_git).join(", ")}</>}
      </div>
      <div className="flex gap-6">
        {site.email && (
          <span><strong>Email:</strong> <a href={`mailto:${site.email}`} style={{ color: "inherit" }}>{site.email}</a></span>
        )}
        {site.tel && <span><strong>TEL:</strong> {site.tel}</span>}
      </div>
    </footer>
  );
}

export function Loading({ error, what }: { error: string | null; what: string }) {
  return <p className={error ? "" : "muted"}>{error ? `Could not load ${what}: ${error}` : `Loading ${what}…`}</p>;
}
