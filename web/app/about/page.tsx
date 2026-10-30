import { BASE } from "@/lib/format";

export const metadata = { title: "About · TGF Catalog" };

/** catalog.csv 的列，顺序同文件 */
const COLUMNS: [string, string, string][] = [
  ["name", "", "TGF name, yymmddAA; empty while the event is provisional"],
  ["cand_id", "", "Identifier of the candidate: instrument and start time"],
  ["instrument", "", "Detector that saw the TGF"],
  ["status", "", "final, or provisional while data are missing"],
  ["pending", "", "What a provisional event waits for: not_vetted, grb_background_gap (data gap in the GRB test), neighbors_unsearched (an hour within ±10 min not yet searched)"],
  ["start, stop", "UTC", "First and last count of the candidate window"],
  ["peak", "UTC", "Middle of the best interval"],
  ["duration_us", "µs", "Best interval: first to last count of the most significant window"],
  ["count", "", "Counts in the best interval"],
  ["false_positive_per_year", "yr⁻¹", "Nominal false-alarm rate of the search"],
  ["longitude, latitude", "°", "Sub-satellite point at the peak"],
  ["altitude_km", "km", "Altitude of the satellite"],
  ["solar_time", "h:m:s", "Apparent solar time at the sub-satellite point"],
  ["neighbors_10min", "", "Other candidates within ±10 min (train test)"],
  ["grb_z", "σ", "Significance of emission from −0.1 to +0.3 s outside the candidate"],
  ["grb_max_box_share", "", "Largest excess of one detector box over the total excess (GRB test)"],
  ["electron_beam", "0 / 1", "Flagged as an electron beam"],
  ["beam_p", "", "Binomial probability of the counts that fire only the top anti-coincidence shields"],
  ["det_share_max", "", "Largest share of the window's counts from a single detector"],
  ["n, n_acd, n_acd_multi", "", "Counts in the window; of them, with any anti-coincidence shield firing; with two or more"],
  ["n_bg, n_acd_bg", "", "The same in the background, 1 s on each side, 10 ms next to the window left out"],
];

function Section({ id, title, children }: { id?: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="margin-grid section">
      <header className="margin-note">
        <h2 className="section-title">{title}</h2>
      </header>
      <div className="prose flex flex-col gap-4" style={{ maxWidth: "36em", fontSize: 16.5, lineHeight: 1.65 }}>{children}</div>
    </section>
  );
}

export default function About() {
  return (
    <article className="flex flex-col pb-10" style={{ gap: 72 }}>
      <section className="margin-grid pt-14">
        <div />
        <div>
          <h1 style={{ fontSize: "clamp(32px, 4.2vw, 46px)", lineHeight: 1.12, fontWeight: 400, letterSpacing: "-0.01em" }}>About the catalog</h1>
          <p className="mt-6 secondary prose" style={{ maxWidth: "34em", fontSize: 18, lineHeight: 1.6 }}>
            Terrestrial gamma-ray flashes found in the satellite data alone, within hours of their arrival, with the same
            selection from the first day of the mission to the last.
          </p>
        </div>
      </section>

      <Section title="Search">
        <p>
          Every time window that starts and ends on a detected count, up to 1 ms long and with at least 8 counts, is
          tested against the background measured within ±0.5 s. The significance is a nominal false-alarm rate per
          year of data.
        </p>
        <p>The search covers every hour of data with the detectors on. Its live time, cell by cell and hour by hour, is published with the catalog.</p>
      </Section>

      <Section title="Selection">
        <p>
          An event enters the catalog with a false-alarm rate of at most 10⁻⁵ per year, outside trains of candidates
          (more than 34 within ±10 min, from precipitating radiation-belt electrons), and with no emission around it that
          all detector boxes share (a gamma-ray burst).
        </p>
        <p>
          Lightning data play no part, so each TGF can be listed as soon as its data arrive. Electron beams are flagged
          and kept.
        </p>
        <p>
          An instrument is published once its search is validated: the false-alarm rate calibrated with Poisson
          simulations, reference catalogs recovered, instrumental and particle backgrounds checked.
        </p>
      </Section>

      <Section title="Names">
        <p>
          A confirmed event is named <span className="tabular">TGF yymmddAA</span>: the UTC date of its peak and two
          letters in the order the names were given. A name is permanent.
        </p>
        <p>
          If a reprocessing no longer selects an event, its name is marked retracted; if two events turn out to be one,
          one name is merged into the other. Names are never reused.
        </p>
      </Section>

      <section id="columns" className="margin-grid section">
        <header className="margin-note caption">
          <h2 className="section-title mb-3">Columns</h2>
          <p>The columns of catalog.csv and catalog.json, in order.</p>
        </header>
        <div className="table-scroll">
          <table style={{ minWidth: 520 }} className="data wrap">
            <thead>
              <tr>
                <th>Column</th>
                <th>Unit</th>
                <th>Meaning</th>
              </tr>
            </thead>
            <tbody>
              {COLUMNS.map(([k, u, m]) => (
                <tr key={k}>
                  <td className="tabular" style={{ whiteSpace: "nowrap", overflowWrap: "normal" }}>{k}</td>
                  <td className="secondary tabular" style={{ whiteSpace: "nowrap", overflowWrap: "normal" }}>{u}</td>
                  <td className="secondary" style={{ fontSize: 14.5 }}>{m}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>


      <Section title="Credits">
        <p>
          Type is Source Serif 4 by Frank Grießhammer for Adobe, under the SIL Open Font License. Coastlines are from
          Natural Earth, in the public domain. The pages are built with React, Next.js, D3 and TopoJSON, under the MIT
          and ISC licenses.
        </p>
        <p>
          The copyright notices and license texts of everything the site sends to the browser are in{" "}
          <a href={`${BASE}/THIRD-PARTY-NOTICES.txt`}>THIRD-PARTY-NOTICES.txt</a>.
        </p>
      </Section>
    </article>
  );
}
