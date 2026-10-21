#!/usr/bin/env python3
"""GRB 221009A: attach raw 1B event bytes to `blink sat extract` output.

`blink sat extract` gives (box, met, channel, det_id, pkt_idx, evt_idx) but not
the pulse width (byte1) or the event-type flag (byte7[5:4]: 00 physical,
10 = 241Am-tagged calibration). We index the CCSDS rows by (pkt_idx, evt_idx)
(payload starts at byte 6, 8 bytes/event) and verify channel and det_id match.

Input : data/p221009/eband/extract_1b.csv
        (HXMT_1B_DIR=data/1B blink sat extract 2022-10-09T13:17:00
         --before 60 --after 760 > ...)
Output: data/p221009/eband/events_obs.npz
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

OUT = Path("data/p221009/eband")
EVT_APID = {"A": "0642", "B": "0922", "C": "1686"}
BOX_OFFSET = {"A": 0, "B": 6, "C": 12}


def main():
    df = pd.read_csv(OUT / "extract_1b.csv")
    df = df[df.type == "EVT"]
    parts = []
    for box, apid in EVT_APID.items():
        d = df[df.box == box]
        f = sorted(Path(f"data/1B/2022/20221009/{apid}").glob("*.fits"))[0]
        ccsds = fits.open(f, memmap=True)[1].data["CCSDS"]
        pk = d.pkt_idx.to_numpy()
        col = 6 + 8 * d.evt_idx.to_numpy()[:, None] + np.arange(8)[None, :]
        raw = ccsds[pk[:, None], col]
        ok_ch = (raw[:, 0] == d.channel.to_numpy()).mean()
        ok_det = (((raw[:, 4] >> 1) & 0x1F) == d.det_id.to_numpy()).mean()
        print(f"{box}: {len(d):,} evts, channel match {ok_ch:.6f}, det match {ok_det:.6f}",
              file=sys.stderr)
        parts.append(dict(
            box=np.full(len(d), "ABC".index(box), np.int8),
            met=d.met.to_numpy(),
            raw_ch=raw[:, 0].astype(np.int16),
            pw=raw[:, 1].astype(np.int16),
            det=(BOX_OFFSET[box] + d.det_id.to_numpy()).astype(np.int8),
            typ=((raw[:, 7] >> 4) & 0x3).astype(np.int8),
        ))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(OUT / "events_obs.npz", **out)
    print(f"wrote {OUT/'events_obs.npz'}: {len(out['met']):,} events; "
          f"typ counts {np.unique(out['typ'], return_counts=True)}", file=sys.stderr)


if __name__ == "__main__":
    main()
