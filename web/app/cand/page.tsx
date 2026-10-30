"use client";

import Link from "next/link";
import EventView from "@/components/EventView";
import { Loading } from "@/components/Shell";
import { displayName, eventHref, lat, lon, utc } from "@/lib/format";
import type { EventRecord, Index } from "@/lib/types";
import { useJson, useParam } from "@/lib/useData";

export default function CandidatePage() {
  const id = useParam("id", "cand");
  const { data: index } = useJson<Index>("index.json");
  // cand_id 的前缀即仪器 slug
  const slug = id && index?.instruments.find((i) => id.startsWith(`${i.slug}-`))?.slug;
  const { data: event, error } = useJson<EventRecord>(slug ? `${slug}/events/${id}.json` : null);
  if (id === "") return <p className="secondary pt-14">No candidate given. Find events in the <Link href="/">catalog</Link>.</p>;
  if (!event) return <Loading error={error} what="the event" />;
  const head = (
    <>
      <h1 style={{ fontSize: "clamp(32px, 4.2vw, 46px)", lineHeight: 1.12, fontWeight: 400, letterSpacing: "-0.01em" }}>
        {event.name ? displayName(event.name) : "Provisional event"}
      </h1>
      <p className="mt-6 secondary prose" style={{ maxWidth: "34em", fontSize: 18, lineHeight: 1.6 }}>
        <span className="tabular">{utc(event.peak, 6)}</span> UTC, seen by {event.instrument} over{" "}
        <span className="tabular">{lat(event.latitude)}, {lon(event.longitude)}</span>.
        {event.name && (
          <>
            {" "}Its page is <Link href={eventHref({ name: event.name, cand_id: null })}>{displayName(event.name)}</Link>.
          </>
        )}
      </p>
    </>
  );
  return (
    <>
      <title>{`${event.name ? displayName(event.name) : "Provisional event"} · TGF Catalog`}</title>
      <EventView event={event} head={head} />
    </>
  );
}
