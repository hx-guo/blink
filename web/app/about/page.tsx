export const metadata = { title: "About · TGF Catalog" };

export default function About() {
  return (
    <article className="flex flex-col gap-4 max-w-3xl text-sm leading-6">
      <h1 className="text-2xl font-semibold">About the catalog</h1>
      <h2 className="text-lg font-semibold">Search</h2>
      <p>
        The search tests every time window that starts and ends on a detected event, up to 1 ms long and with at
        least 8 events, against the background rate measured within ±0.5 s (event-based transient search with the
        snapshot-stepping algorithm). Its significance is expressed as a nominal false-alarm rate fa per year.
      </p>
      <h2 className="text-lg font-semibold">Selection</h2>
      <p>
        An event enters the catalog if fa ≤ 10⁻⁵ per year, if it is not part of a train of candidates (more than 34
        candidates within ±10 min, typical of precipitating radiation-belt electrons), and if it shows no extended
        emission outside its window that is shared by the detector boxes (a gamma-ray burst). Lightning data are not
        used, so the catalog can be updated as soon as the satellite data arrive. Electron beams are flagged, not
        removed.
      </p>
      <h2 className="text-lg font-semibold">Names</h2>
      <p>
        A confirmed event is named <code>TGF yymmddAA</code>: the UTC date of its peak and two letters in the order
        the names were assigned. A published name is permanent. If a later reprocessing no longer selects the event,
        the name is marked retracted; if two events turn out to be one, one name is merged into the other. Retracted
        and merged names are never reused. Events that still wait for data are listed as provisional and have no name.
      </p>
      <h2 className="text-lg font-semibold">Instruments</h2>
      <p>
        An instrument is published after its search has passed a validation: calibration of fa with Poisson
        simulations, recovery of reference catalogs, and checks of instrumental and particle backgrounds.
      </p>
    </article>
  );
}
