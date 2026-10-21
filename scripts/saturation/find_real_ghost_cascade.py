#!/usr/bin/env python3
"""Find a real (ds=1) SEC interval in GRB 221009A where a CRC-collision
ghost causes a greedy cascade-rejection of many real events. The same
events are accepted intact by LIS.

This script bypasses blink_cli (which already filters via LIS) and parses
the raw CCSDS packets from the 1B FITS files directly. That way the
"rejected by LIS" events are available, which is exactly what we need
to demonstrate the cascade mechanism for Fig 2.

References (reading from rust source):
  - rec_sci_data::parse_events (event-word layout)
  - crc_check (CRC-4/ITU lookup table)
"""
import os
import sys
import glob
from bisect import bisect_left
import numpy as np
from astropy.io import fits

PMOD = 524288             # 2^19
TICKS_PER_SEC = 500000    # 2 us per tick
DEAD_TICKS = TICKS_PER_SEC  # ds=1: elapsed_fwd must be <= 500000

CRC_TABLE = [0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2]


def crc_check(row):
    crc = 0
    m = 0
    cdata = 0
    for j in range(1, 16):
        if (j - 1) % 2 == 0:
            cdata = row[m]
            m += 1
            nibble = (cdata & 0xF0) >> 4
        else:
            nibble = cdata & 0xF
        crc = CRC_TABLE[crc ^ nibble]
    return crc


def parse_packet(ccsds_882):
    """Parse one CCSDS packet (882 bytes) -> list of dicts."""
    payload = ccsds_882[6:878]   # 109 × 8 = 872 bytes
    events = []
    for i in range(109):
        row = payload[i*8:(i+1)*8]
        crc_pass = crc_check(row) == (row[7] & 0x0F)
        evt = {"slot": i, "raw": bytes(row), "crc_pass": crc_pass}
        if not crc_pass:
            evt["kind"] = "ERR"
            events.append(evt)
            continue
        # Decode type field (row[7] & 0x30): 0x00/0x20 = EVT, 0x10 = SEC
        kind_bits = row[7] & 0x30
        ptime = ((row[4] & 1) << 18) | (row[5] << 10) | (row[6] << 2) | ((row[7] & 0xC0) >> 6)
        if kind_bits == 0x10:
            stime = (row[0] << 24) | (row[1] << 16) | (row[2] << 8) | row[3]
            evt["kind"] = "SEC"; evt["stime"] = stime; evt["ptime"] = ptime
        else:
            evt["kind"] = "EVT"; evt["ptime"] = ptime
            evt["channel"] = row[0]
        events.append(evt)
    return events


def lis_membership(values):
    n = len(values)
    if n == 0:
        return np.array([], dtype=bool)
    tails = []; tail_pos = []
    parent = [-1] * n
    for i, v in enumerate(values):
        pos = bisect_left(tails, v)
        if pos == len(tails):
            tails.append(v); tail_pos.append(i)
        else:
            tails[pos] = v; tail_pos[pos] = i
        if pos > 0:
            parent[i] = tail_pos[pos - 1]
    idx = tail_pos[-1]
    member = np.zeros(n, dtype=bool)
    while idx != -1:
        member[idx] = True; idx = parent[idx]
    return member


def load_1b(fits_path):
    """Return list of (pkt_idx, [event dicts])."""
    print(f"  Loading {fits_path}", file=sys.stderr)
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        ccsds_col = data["CCSDS"]
    pkts = []
    for i, row in enumerate(ccsds_col):
        # row may be raw bytes or array; ensure bytes-like
        pkt_bytes = bytes(np.asarray(row, dtype=np.uint8).tobytes()
                          if not isinstance(row, (bytes, bytearray))
                          else row)
        if len(pkt_bytes) != 882:
            continue
        pkts.append((i, parse_packet(pkt_bytes)))
    return pkts


def scan_for_cascade(pkts, min_cascade=1, ds_target=1):
    """Walk SEC -> SEC pairs in file order. For each ds=1 pair, compute
    elapsed_fwd, run greedy + LIS, score cascade. Return top candidates."""

    # Build file-order stream of (pkt_idx, slot_idx, kind, ptime, stime?)
    stream = []
    for pkt_idx, evts in pkts:
        for e in evts:
            if e["kind"] == "SEC":
                stream.append((pkt_idx, e["slot"], "SEC",
                               e["ptime"], e["stime"]))
            elif e["kind"] == "EVT":
                stream.append((pkt_idx, e["slot"], "EVT",
                               e["ptime"], None))

    sec_positions = [i for i, x in enumerate(stream) if x[2] == "SEC"]
    print(f"  Stream: {len(stream)} entries, {len(sec_positions)} SECs",
          file=sys.stderr)

    results = []
    n_ds1 = 0
    for k in range(len(sec_positions) - 1):
        ia = sec_positions[k]; ib = sec_positions[k+1]
        st_a = stream[ia][4]; st_b = stream[ib][4]
        pt_a = stream[ia][3]
        ds = st_b - st_a
        if ds != ds_target:
            continue
        n_ds1 += 1
        between = stream[ia+1:ib]
        evts = [x for x in between if x[2] == "EVT"]
        if len(evts) < 10:
            continue
        ef = np.array([(pt - pt_a) % PMOD for (_p, _s, _k, pt, _) in evts],
                      dtype=np.int64)
        valid_mask = ef <= DEAD_TICKS
        if valid_mask.sum() < 10:
            continue
        ef_v = ef[valid_mask]
        # Greedy
        accept_g = np.zeros(len(ef_v), dtype=bool)
        running_max = -1
        for i, v in enumerate(ef_v):
            if v > running_max:
                accept_g[i] = True
                running_max = int(v)
        # LIS
        accept_l = lis_membership(ef_v.tolist())
        cascade = int(((~accept_g) & accept_l).sum())
        if cascade >= min_cascade:
            results.append({
                "k": k, "ia": ia, "ib": ib, "st_a": st_a, "pt_a": pt_a,
                "cascade": cascade, "n_valid": int(valid_mask.sum()),
                "n_dead": int((~valid_mask).sum()),
                "ef": ef_v, "accept_g": accept_g, "accept_l": accept_l,
                "evts": [evts[i] for i in np.where(valid_mask)[0]],
            })
    print(f"  ds=1 pairs (with >=10 events): {n_ds1}", file=sys.stderr)
    print(f"  candidates with cascade>={min_cascade}: {len(results)}",
          file=sys.stderr)
    results.sort(key=lambda r: -r["cascade"])
    return results


def main():
    # Box A science file for 221009 (rust file_name table: sci A = code 0642)
    fits_files = sorted(glob.glob("data/1B/2022/20221009/0642/HXMT_1B_*.fits"))
    print(f"Found {len(fits_files)} Box-A sci file(s) for 20221009",
          file=sys.stderr)
    if not fits_files:
        sys.exit("no 1B files found")

    all_pkts = []
    for fp in fits_files:
        pkts = load_1b(fp)
        all_pkts.extend(pkts)

    cands = scan_for_cascade(all_pkts, min_cascade=1)
    print(f"\nTop {min(len(cands), 20)} by cascade size:", file=sys.stderr)
    for r in cands[:20]:
        n = len(r["ef"])
        ghost_idx = int(np.argmax(r["ef"] * (~r["accept_l"])))
        print(f"  k={r['k']:>4d} stime={r['st_a']:>10d}: "
              f"n_valid={r['n_valid']:>4d} dead={r['n_dead']:>3d}  "
              f"cascade={r['cascade']:>3d}  "
              f"ghost@{ghost_idx} ef={r['ef'][ghost_idx]}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
