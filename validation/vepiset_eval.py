"""
vEpiSet validation of eeg-spikes: per-epoch detection on 4-s epochs with and without expert IED marks.

Epochs are cut as the dataset's own code cuts them (github.com/vepiset/vepiset_dataset mat2npy.py:
consecutive 4 s from 0; an epoch is IED-positive when an `!` or `!start` mark lies inside it, at
least 0.1 s from its edges). Epochs inside a `!start`-`!end` run without a mark of their own, or
holding only an `!end`, are left out. Epoch score: the number of channel detections of the spike
events that fall in it. Prints AUROC (Mann-Whitney), operating points (an epoch is called positive
when it holds an event spanning >= m channels), event-level recall of the single `!` marks (an event
within ±0.25 s), and events per minute in the patients without IEDs.

Usage: python validation/vepiset_eval.py <eeg-spikes output folder> <vepiset_labels.json> [label]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EPOCH = 4.0
EDGE = 0.1


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(out: Path, labels_path: Path, label: str = "") -> dict:
    labels = json.loads(labels_path.read_text())
    events = defaultdict(list)
    with open(out / "spike_events.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            events[r["recording_id"]].append((float(r["time_s"]), int(r["n_channels"])))
    minutes = {}
    with open(out / "lateralisation.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            minutes[r["recording_id"]] = float(r["minutes"])
    rows = []  # (label 0/1, score, max_channels)
    marks_hit = marks_n = 0
    fp_rate = []
    for rid, lab in labels.items():
        if rid not in minutes:
            continue
        dur = lab["duration_s"]
        tags = [(e["time_s"], e["text"].replace(" ", "").lower()) for e in lab["events"] if e["text"].startswith("!")]
        runs, start = [], None
        for t, txt in sorted(tags):
            if txt == "!start":
                start = t
            elif txt == "!end" and start is not None:
                runs.append((start, t))
                start = None
        ev = sorted(events.get(rid, []))
        et = np.array([t for t, _ in ev]) if ev else np.array([])
        for t, txt in tags:
            if txt == "!":
                marks_n += 1
                marks_hit += bool(len(et)) and np.min(np.abs(et - t)) <= 0.25
        if not tags:
            fp_rate.append(len(ev) / minutes[rid])
        n_ep = int(dur // EPOCH)
        for k in range(n_ep):
            a, b = k * EPOCH, (k + 1) * EPOCH
            inside = [txt for t, txt in tags if a + EDGE < t < b - EDGE]
            pos = any(txt in ("!", "!start") for txt in inside)
            if not pos and (inside or any(r0 < b and r1 > a for r0, r1 in runs)):
                continue
            in_ep = [n for t, n in ev if a <= t < b]
            rows.append((int(pos), sum(in_ep), max(in_ep) if in_ep else 0))
    y = np.array([r[0] for r in rows])
    score = np.array([r[1] for r in rows])
    mx = np.array([r[2] for r in rows])
    res = {"label": label, "epochs_pos": int(y.sum()), "epochs_neg": int((1 - y).sum()),
           "auroc": round(float(auroc(score[y == 1], score[y == 0])), 3),
           "mark_recall_250ms": round(marks_hit / max(marks_n, 1), 3), "marks": marks_n,
           "events_per_min_no_ied_patients": round(float(np.median(fp_rate)), 2) if fp_rate else None,
           "operating_points": []}
    for m in (1, 2, 3, 4, 6):
        called = mx >= m
        tp = int((called & (y == 1)).sum())
        fp = int((called & (y == 0)).sum())
        res["operating_points"].append({"min_channels": m, "sensitivity": round(tp / max(1, y.sum()), 3),
                                        "specificity": round(1 - fp / max(1, (1 - y).sum()), 3),
                                        "precision": round(tp / max(1, tp + fp), 3)})
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
    print(json.dumps(r, indent=1))
