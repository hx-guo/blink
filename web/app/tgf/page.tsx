"use client";

import Link from "next/link";
import EventView from "@/components/EventView";
import { Loading } from "@/components/Shell";
import { displayName, eventHref, utc } from "@/lib/format";
import type { Detection, EventRecord, NameEntry } from "@/lib/types";
import { useJson, useParam } from "@/lib/useData";

function DetectionView({ d, many }: { d: Detection; many: boolean }) {
  const { data: event, error } = useJson<EventRecord>(d.cand_id ? `${d.slug}/events/${d.cand_id}.json` : null);
  if (!d.cand_id) {
    return (
      <div className="card p-3 text-sm secondary">
        {d.instrument}: detected at {utc(d.peak, 6)} UTC; the event is no longer in the current catalog.
      </div>
    );
  }
  return (
    <section className="flex flex-col gap-2">
      {many && <h2 className="font-semibold">{d.instrument}</h2>}
      {event ? <EventView event={event} /> : <Loading error={error} what="the event" />}
    </section>
  );
}

export default function NamePage() {
  const name = useParam("name", "tgf");
  const { data: entry, error } = useJson<NameEntry>(name ? `names/${name}.json` : null);
  if (name === "") return <p className="secondary">No name given. Find events in the <Link href="/">catalog</Link>.</p>;
  if (!entry) return <Loading error={error && `no event named ${name}`} what={name ? displayName(name) : "the event"} />;
  return (
    <div className="flex flex-col gap-4">
      <title>{`${displayName(entry.name)} · TGF Catalog`}</title>
      <div>
        <h1 className="text-2xl font-semibold">{displayName(entry.name)}</h1>
        <div className="secondary text-sm">Peak {utc(entry.peak_utc, 6)} UTC · named {utc(entry.assigned_at, 0)} UTC</div>
      </div>
      {entry.status !== "active" && (
        <div className="card p-3 text-sm" style={{ background: "var(--banner)" }}>
          {entry.status === "merged" ? (
            <>This name was merged into <Link href={eventHref({ name: entry.merged_into, cand_id: null })}>{displayName(entry.merged_into)}</Link>. It stays reserved and is not reused.</>
          ) : (
            <>This name was retracted ({entry.status_reason || "no reason recorded"}): a later reprocessing no longer selects the event. The name stays reserved and is not reused.</>
          )}
        </div>
      )}
      {entry.detections.map((d) => (
        <DetectionView key={d.instrument} d={d} many={entry.detections.length > 1} />
      ))}
    </div>
  );
}
