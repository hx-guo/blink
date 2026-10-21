"use client";

import Link from "next/link";
import EventView from "@/components/EventView";
import { Loading } from "@/components/Shell";
import type { EventRecord, Index } from "@/lib/types";
import { useJson, useParam } from "@/lib/useData";

export default function CandidatePage() {
  const id = useParam("id", "cand");
  const { data: index } = useJson<Index>("index.json");
  // cand_id 的前缀即仪器 slug
  const slug = id && index?.instruments.find((i) => id.startsWith(`${i.slug}-`))?.slug;
  const { data: event, error } = useJson<EventRecord>(slug ? `${slug}/events/${id}.json` : null);
  if (id === "") return <p className="secondary">No candidate given. Find events in the <Link href="/">catalog</Link>.</p>;
  return (
    <div className="flex flex-col gap-4">
      <title>Provisional event · TGF Catalog</title>
      <h1 className="text-2xl font-semibold">{event?.name ? "Event" : "Provisional event"}</h1>
      {event ? <EventView event={event} /> : <Loading error={error} what="the event" />}
    </div>
  );
}
