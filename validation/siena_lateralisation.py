"""
Siena validation: per-patient lateralisation of interictal spike events (eeg-spikes output) against
the seizure lateralisation PhysioNet lists for each patient (Siena Scalp EEG Database v1.0.0,
subject_info.csv: localization, lateralization).

Usage: python validation/siena_lateralisation.py <eeg-spikes output folder> <subject_info.csv> [min_channels]

Pools every run of a patient: events led by a left vs a right channel (laterality index
(L-R)/(L+R)), and the mean per-channel spike rate over left vs right channels. Prints a table and
the agreement with the annotated side (bilateral patients left out), with a two-sided sign test.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import comb
from pathlib import Path


def sign_test(k: int, n: int) -> float:
    """Two-sided binomial test of k successes in n against p = 0.5."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main(out: Path, info: Path, min_channels: int = 1) -> None:
    side = {}
    with open(info, newline="") as fh:
        for r in csv.DictReader(fh, skipinitialspace=True):
            pid = r["patient_id"].strip().replace("PN", "")
            side[pid] = (r["localization"].strip(), r["lateralization"].strip())
    ev = defaultdict(lambda: {"L": 0, "R": 0, "Z": 0})
    minutes = defaultdict(float)
    with open(out / "spike_events.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["n_channels"]) < min_channels:
                continue
            sub = r["recording_id"].split("_")[0].replace("sub-", "")
            ev[sub][r["hemisphere"]] += 1
    with open(out / "lateralisation.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            minutes[r["recording_id"].split("_")[0].replace("sub-", "")] += float(r["minutes"])
    rates = defaultdict(lambda: {"L": [0, 0.0], "R": [0, 0.0]})
    hemi = {}
    with open(out / "spike_events.csv", newline="") as fh:
        pass
    with open(out / "spike_rates.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            sub = r["recording_id"].split("_")[0].replace("sub-", "")
            ch = r["channel"]
            h = "L" if all(int(c[-1]) % 2 == 1 for c in ch.split("-") if c[-1].isdigit()) and not ch.endswith("z") \
                and "z" not in ch else ("R" if "z" not in ch and all(int(c[-1]) % 2 == 0 for c in ch.split("-")) else "Z")
            if h in ("L", "R"):
                rates[sub][h][0] += int(r["n_spikes"])
                rates[sub][h][1] += float(r["minutes"])
    print(f"{'patient':8} {'loc':4} {'side':10} {'min':>7} {'L ev':>7} {'R ev':>7} {'LI':>7} {'rate L':>7} {'rate R':>7} agree")
    agree = n = 0
    agree_r = 0
    for sub in sorted(side):
        loc, lat = side[sub]
        if sub not in ev and sub not in rates:
            continue
        L, R = ev[sub]["L"], ev[sub]["R"]
        li = (L - R) / (L + R) if L + R else 0.0
        rl = rates[sub]["L"][0] / rates[sub]["L"][1] if rates[sub]["L"][1] else 0.0
        rr = rates[sub]["R"][0] / rates[sub]["R"][1] if rates[sub]["R"][1] else 0.0
        mark = ""
        if lat in ("L", "R"):
            n += 1
            ok = (li > 0) == (lat == "L") and li != 0
            agree += ok
            agree_r += ((rl > rr) == (lat == "L")) and rl != rr
            mark = "yes" if ok else "no"
        print(f"{sub:8} {loc:4} {lat:10} {minutes[sub]:7.0f} {L:7d} {R:7d} {li:7.3f} {rl:7.2f} {rr:7.2f} {mark}")
    print(f"\nevents: lateralisation agrees with the annotated seizure side in {agree} of {n} patients "
          f"(sign test p = {sign_test(agree, n):.3f})")
    print(f"per-channel rates: left > right matches the annotated side in {agree_r} of {n} patients "
          f"(sign test p = {sign_test(agree_r, n):.3f})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
