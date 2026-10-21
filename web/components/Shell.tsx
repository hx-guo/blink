"use client";

import Link from "next/link";
import { site } from "@/site.config";
import { instrumentHref, utc } from "@/lib/format";
import type { Index } from "@/lib/types";
import { useJson } from "@/lib/useData";
import Logo from "./Logo";

/** 页头：站标与各仪器的最新数据时间（index.json，运行时取） */
export function Header() {
  const { data: index } = useJson<Index>("index.json");
  const demo = index?.instruments.some((i) => i.demo);
  const unvalidated = index?.instruments.filter((i) => !i.public) ?? [];
  return (
    <>
      <header className="mx-auto max-w-6xl px-4 pt-6 pb-2 flex items-center justify-between gap-6 flex-wrap">
        <Link href="/" className="flex items-center gap-3" style={{ color: "var(--text-primary)", textDecoration: "none" }}>
          <Logo size={56} />
          <span>
            <span className="block text-2xl font-bold leading-tight">
              TGF<span style={{ color: "var(--accent)" }}>s</span> Catalog
            </span>
            <span className="block text-sm font-semibold" style={{ color: "var(--accent-strong)" }}>Find It Fast</span>
          </span>
        </Link>
        {index && (
          <div className="flex gap-10 text-sm">
            <div>
              <div className="font-semibold">Pipeline</div>
              <div className="secondary tabular text-xs mt-1">{utc(index.generated_at, 0)}</div>
            </div>
            {index.instruments.map((i) => (
              <Link key={i.slug} href={instrumentHref(i.slug)} style={{ color: "inherit", textDecoration: "none" }}>
                <div className="font-semibold">{i.instrument.replace("Insight-", "")}</div>
                <div className="secondary tabular text-xs mt-1" title="Data through">{i.last_day} 23:59:59</div>
              </Link>
            ))}
          </div>
        )}
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
