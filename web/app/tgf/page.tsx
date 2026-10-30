"use client";

import Link from "next/link";
import EventView from "@/components/EventView";
import { Loading } from "@/components/Shell";
import { displayName, eventHref, lat, lon, utc } from "@/lib/format";
import type { Detection, EventRecord, NameEntry } from "@/lib/types";
import { useJson, useParam } from "@/lib/useData";

const H1 = { fontSize: "clamp(32px, 4.2vw, 46px)", lineHeight: 1.12, fontWeight: 400, letterSpacing: "-0.01em" } as const;

/** 名字页的页头：名字、一句话（时间、地点、探测器）、命名时间，撤回或合并时说明 */
function Head({ entry, d }: { entry: NameEntry; d: Detection }) {
  return (
    <>
      <h1 style={H1}>{displayName(entry.name)}</h1>
      <p className="mt-6 secondary prose" style={{ maxWidth: "34em", fontSize: 18, lineHeight: 1.6 }}>
        <span className="tabular">{utc(entry.peak_utc, 6)}</span> UTC, seen by {d.instrument}
        {d.latitude != null && d.longitude != null && (
          <>
            {" "}over <span className="tabular">{lat(d.latitude)}, {lon(d.longitude)}</span>
          </>
        )}
        . Named {entry.assigned_at.slice(0, 10)}.
      </p>
      {entry.status !== "active" && (
        <p className="caption mt-4" style={{ maxWidth: "36em" }}>
          <span className="badge mr-2">{entry.status}</span>
          {entry.status === "merged" ? (
            <>
              This name was merged into <Link href={eventHref({ name: entry.merged_into, cand_id: null })}>{displayName(entry.merged_into)}</Link>. It stays
              reserved and is not reused.
            </>
          ) : (
            <>A later reprocessing no longer selects the event ({entry.status_reason || "no reason recorded"}). The name stays reserved and is not reused.</>
          )}
        </p>
      )}
    </>
  );
}

function DetectionView({ entry, d }: { entry: NameEntry; d: Detection }) {
  const { data: event, error } = useJson<EventRecord>(d.cand_id ? `${d.slug}/events/${d.cand_id}.json` : null);
  if (!d.cand_id) {
    return (
      <section className="margin-grid pt-14">
        <div />
        <div>
          <Head entry={entry} d={d} />
          <p className="caption mt-6">The event is no longer in the current catalog of {d.instrument}.</p>
        </div>
      </section>
    );
  }
  return event ? <EventView event={event} head={<Head entry={entry} d={d} />} /> : <Loading error={error} what="the event" />;
}

export default function NamePage() {
  const name = useParam("name", "tgf");
  const { data: entry, error } = useJson<NameEntry>(name ? `names/${name}.json` : null);
  if (name === "") return <p className="secondary pt-14">No name given. Find events in the <Link href="/">catalog</Link>.</p>;
  if (!entry) return <Loading error={error && `no event named ${name}`} what={name ? displayName(name) : "the event"} />;
  return (
    <div className="flex flex-col" style={{ gap: 88 }}>
      <title>{`${displayName(entry.name)} · TGF Catalog`}</title>
      {entry.detections.map((d) => (
        <DetectionView key={d.instrument} entry={entry} d={d} />
      ))}
    </div>
  );
}
