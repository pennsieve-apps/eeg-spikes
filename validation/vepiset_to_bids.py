"""
Convert vEpiSet MAT files (Lin et al. 2025, figshare doi:10.6084/m9.figshare.28069568.v2, CC BY 4.0)
to a plain BIDS-EEG folder eeg-spikes reads, plus the expert IED annotations as JSON.

Each MAT holds `eeg_data` (29 x samples, volts, 500 Hz; channel order as in the dataset's own code,
github.com/vepiset/vepiset_dataset mat2npy.py: Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6
Fz Cz Pz PG1 PG2 A1 A2 EKG1 EKG2 EMG1-4) and `events` (time s, -, text: `!` an IED, `!start` /
`!end` a run of discharges, `Waking` / `Sleeping` the state from then on). The 19 scalp
electrodes are written in average reference (as BIDS Siena is), the rest dropped.

Usage: python validation/vepiset_to_bids.py <mat folder> <out folder>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io

ORDER = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz",
         "Cz", "Pz"]
FS = 500.0


def main(src: Path, out: Path) -> None:
    from edfio import Edf, EdfSignal

    labels = {}
    for mat in sorted(src.glob("DA*.mat")):
        m = scipy.io.loadmat(mat)
        x = m["eeg_data"][:19] * 1e6
        x = x - x.mean(axis=0, keepdims=True)
        sub = mat.stem
        d = out / f"sub-{sub}" / "eeg"
        d.mkdir(parents=True, exist_ok=True)
        sigs = []
        for row, name in zip(x, ORDER):
            r = max(1.0, float(np.max(np.abs(row))) * 1.1)
            sigs.append(EdfSignal(row, sampling_frequency=FS, label=f"{name}-Avg", physical_dimension="uV",
                                  physical_range=(-r, r)))
        Edf(sigs).write(d / f"sub-{sub}_task-ied_eeg.edf")
        (d / f"sub-{sub}_task-ied_eeg.json").write_text(json.dumps({"SamplingFrequency": FS, "PowerLineFrequency": 50,
                                                                     "EEGReference": "average"}))
        ev = []
        for row in m["events"]:
            t, text = float(str(row[0]).strip()), str(row[-1]).strip()
            ev.append({"time_s": t, "text": text})
        # the annotated state (Waking / Sleeping from each mark on) as BIDS sleep / wake events
        dur = x.shape[1] / FS
        states = [(e["time_s"], e["text"].lower()) for e in ev if e["text"].lower() in ("waking", "sleeping")]
        rows = []
        for i, (t, s) in enumerate(states):
            end = states[i + 1][0] if i + 1 < len(states) else dur
            if end > t:
                rows.append(f"{t:.3f}\t{end - t:.3f}\t{'wake' if s == 'waking' else 'sleep'}\tannotation")
        (d / f"sub-{sub}_task-ied_events.tsv").write_text("onset\tduration\ttrial_type\tsource\n" + "".join(r + "\n" for r in rows))
        labels[f"sub-{sub}_task-ied"] = {"events": ev, "duration_s": x.shape[1] / FS}
        print(sub, x.shape, sum(e["text"].startswith("!") for e in ev), "IED marks")
    (out / "vepiset_labels.json").write_text(json.dumps(labels))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
