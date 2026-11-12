"""
A small synthetic scalp EEG input that follows workflow-scalp-eeg/docs/conventions.md, for the
tests, `make run` and the app.yml review block. Needs only numpy and edfio (no MNE), so it runs
in the platform's generator image.

  <out>/linked/eeg/                    the `eeg` port: an eeg-prep-style BIDS-EEG derivative
      dataset_description.json, participants.tsv, recordings.csv
      sub-01/ses-01/eeg/sub-01_ses-01_task-szMonitoring_run-01_desc-preproc_{eeg.edf,eeg.json,channels.tsv,events.tsv}
          480 s, 19 channels 10-20 in average reference, 256 Hz: posterior alpha in wake
          (0-330 s), a seizure-like 3 Hz spike-and-wave discharge at 240-270 s, a bad segment
          at 100-106 s (artefact with sharp transients), sleep-like slow waves from 330 s;
          planted spikes: left temporal (field centred on T7) and fewer right temporal (T8),
          twice as frequent in sleep
      sub-02/ses-01/eeg/sub-02_ses-01_task-routine_run-01_desc-preproc_*
          240 s, bipolar-only (CHB-MIT style: 18 longitudinal pairs `Fp1-F7`, ...), left temporal spikes
  <out>/linked/sleep/eeg-sleep/hypnogram.csv   the `sleep` port: 30 s epochs for sub-01 (W, then N2, N3)
  <out>/plain/sub-03/ses-01/eeg/sub-03_ses-01_task-szMonitoring_run-00_{eeg.edf,eeg.json,events.tsv}
          the sub-01 signal as plain BIDS Siena writes it: `Fp1-Avg`, old names T3/T4/T5/T6, no
          channels.tsv, the seizure as a HED-SCORE `eventType` row
  <out>/truth.json                     planted spike times (peak, s), side and centre electrode

Usage: python -m tests.make_fixture <out-dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SEED = 42
FS = 256.0
ELECTRODES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
              "O1", "O2"]
# rough 2-D scalp positions (nose up), for spatially smooth background and spike fields
XY = {"Fp1": (-0.3, 0.95), "Fp2": (0.3, 0.95), "F7": (-0.8, 0.6), "F3": (-0.4, 0.5), "Fz": (0, 0.45),
      "F4": (0.4, 0.5), "F8": (0.8, 0.6), "T7": (-1.0, 0.0), "C3": (-0.5, 0.0), "Cz": (0, 0), "C4": (0.5, 0.0),
      "T8": (1.0, 0.0), "P7": (-0.8, -0.6), "P3": (-0.4, -0.5), "Pz": (0, -0.45), "P4": (0.4, -0.5),
      "P8": (0.8, -0.6), "O1": (-0.3, -0.95), "O2": (0.3, -0.95)}
OLD = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
PAIRS = [("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"), ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"),
         ("P3", "O1"), ("Fz", "Cz"), ("Cz", "Pz"), ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
         ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2")]
SPIKE_UV = 150.0          # peak negativity at the field centre
PEAK_OFFSET = 0.05        # the template's sharp peak is 50 ms after its start


def pink(n: int, rng: np.random.Generator, std: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std() * std


def field(centre: str, width: float = 0.55) -> np.ndarray:
    cx, cy = XY[centre]
    return np.array([np.exp(-((XY[e][0] - cx) ** 2 + (XY[e][1] - cy) ** 2) / (2 * width ** 2)) for e in ELECTRODES])


def spike_template(fs: float, amp: float) -> np.ndarray:
    """A surface-negative spike (sharp, ~70 ms base to base) with an after-going slow wave (~200 ms)."""
    t = np.arange(int(0.4 * fs)) / fs
    sharp = -amp * np.exp(-((t - PEAK_OFFSET) / 0.011) ** 2) + 0.25 * amp * np.exp(-((t - 0.085) / 0.015) ** 2)
    slow = 0.35 * amp * np.exp(-((t - 0.2) / 0.06) ** 2)
    return sharp + slow


def add(x: np.ndarray, fs: float, t0: float, wave: np.ndarray) -> None:
    i = int(round(t0 * fs))
    j = min(len(x), i + len(wave))
    if 0 <= i < len(x):
        x[i:j] += wave[: j - i]


def background(n: int, rng: np.random.Generator) -> np.ndarray:
    """19 x n µV: a few spatially smooth 1/f sources plus independent noise (~20 µV in all)."""
    data = np.vstack([pink(n, rng, 5.0) for _ in ELECTRODES])
    for _ in range(6):
        c = ELECTRODES[rng.integers(len(ELECTRODES))]
        data += np.outer(field(c, 0.7), pink(n, rng, 14.0))
    return data


def spike_schedule(rng, segments: list[tuple[float, float, float]], right_share: float) -> list[dict]:
    """segments: (start, end, rate_per_min of all spikes); at least 1.2 s apart; a share of them right-sided."""
    out = []
    for a, b, rate in segments:
        step = 60.0 / rate
        t = a + 1.5
        while t < b - 1.5:
            out.append({"onset": float(t + rng.uniform(-0.25, 0.25) * step)})
            t += step
    for s in out:
        s["side"] = "R" if rng.random() < right_share else "L"
        s["centre"] = "T8" if s["side"] == "R" else "T7"
        s["time_s"] = round(s["onset"] + PEAK_OFFSET, 4)
    return sorted(out, key=lambda s: s["onset"])


def plant(data: np.ndarray, fs: float, spikes: list[dict], rng) -> None:
    for s in spikes:
        amp = SPIKE_UV * rng.uniform(0.8, 1.2)
        w = spike_template(fs, amp)
        f = field(s["centre"], 0.5)
        for i in range(len(ELECTRODES)):
            if f[i] > 0.02:
                add(data[i], fs, s["onset"], f[i] * w)


def write_edf(path: Path, data: np.ndarray, names: list[str], fs: float) -> None:
    """EDF+ with per-channel physical range from the data (as eeg-prep writes it)."""
    from edfio import Edf, EdfSignal

    sigs = []
    for x, name in zip(data, names):
        r = max(1.0, float(np.max(np.abs(x))) * 1.1)
        sigs.append(EdfSignal(x, sampling_frequency=fs, label=name, physical_dimension="uV",
                              physical_range=(-r, r)))
    Edf(sigs).write(path)


def write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def hemi(e: str) -> str:
    return "Z" if e.endswith("z") else ("L" if int(e[-1]) % 2 else "R")


def region(e: str) -> str:
    if e.endswith("z"):
        return "midline"
    if e[0] == "T" or e in ("F7", "F8", "P7", "P8"):
        return "temporal"
    return {"F": "frontal", "C": "central", "P": "parietal", "O": "occipital"}[e[0]]


CHANNEL_COLS = ["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "status",
                "status_description", "contacts", "hemisphere", "region", "x", "y", "z", "original_name"]


def make(out: Path) -> Path:
    rng = np.random.default_rng(SEED)
    root = out / "linked" / "eeg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "eeg-spikes synthetic fixture", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "eeg-prep", "Description": "tests/make_fixture.py of eeg-spikes (synthetic)"}]},
        indent=2))
    write_tsv(root / "participants.tsv", ["participant_id", "age", "sex"],
              [{"participant_id": "sub-01", "age": "40", "sex": "F"}, {"participant_id": "sub-02", "age": "9", "sex": "M"}])
    truth: dict = {}
    recs = []

    # --- sub-01: referential (average), 480 s ------------------------------------------
    dur = 480.0
    n = int(FS * dur)
    t = np.arange(n) / FS
    data = background(n, rng)
    alpha_w = np.array([{"O1": 1, "O2": 1, "P3": 0.6, "P4": 0.6, "Pz": 0.7, "P7": 0.4, "P8": 0.4}.get(e, 0.05)
                        for e in ELECTRODES])
    wake = t < 330
    alpha = 20 * np.sin(2 * np.pi * 10 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / FS) * wake
    data += np.outer(alpha_w, alpha)
    sleep = t >= 330
    slow_w = np.array([{"Fz": 1, "F3": 0.9, "F4": 0.9, "Cz": 0.9, "C3": 0.8, "C4": 0.8, "Fp1": 0.7, "Fp2": 0.7}
                       .get(e, 0.5) for e in ELECTRODES])
    slow = pink(n, rng, 1.0)
    from numpy.fft import irfft, rfft, rfftfreq
    spec = rfft(slow)
    spec[rfftfreq(n, 1 / FS) > 2.5] = 0
    slow = irfft(spec, n)
    slow = slow / slow.std() * 45 * sleep
    data += np.outer(slow_w, slow)
    # seizure: 3 Hz spike-and-wave over the left hemisphere, spreading
    sz0, sz1 = 240.0, 270.0
    sw = spike_template(FS, 220.0)
    for k in np.arange(sz0, sz1, 1 / 3):
        f = field("T7", 0.9) if k < 250 else np.ones(len(ELECTRODES)) * 0.8
        for i in range(len(ELECTRODES)):
            add(data[i], FS, k, f[i] * sw)
    # bad segment: electrode pops and muscle on the right temporal electrodes
    bad0, bad1 = 100.0, 106.0
    i0, i1 = int(bad0 * FS), int(bad1 * FS)
    for e in ("T8", "F8", "P8"):
        i = ELECTRODES.index(e)
        data[i, i0:i1] += 80 * rng.standard_normal(i1 - i0)
        for k in np.arange(bad0 + 0.5, bad1 - 0.3, 0.7):
            add(data[i], FS, k, spike_template(FS, -400.0))
    # spikes: wake 0-180 s (minus the bad segment) 12/min, sleep 330-480 s 24/min; 20 % right-sided
    spikes = spike_schedule(rng, [(2.0, 98.0, 12.0), (108.0, 178.0, 12.0), (332.0, 478.0, 24.0)], 0.2)
    plant(data, FS, spikes, rng)
    data -= data.mean(axis=0, keepdims=True)  # average reference
    stem = "sub-01_ses-01_task-szMonitoring_run-01"
    d = root / "sub-01" / "ses-01" / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    write_edf(d / f"{stem}_desc-preproc_eeg.edf", data, ELECTRODES, FS)
    write_tsv(d / f"{stem}_desc-preproc_channels.tsv", CHANNEL_COLS, [
        {"name": e, "type": "EEG", "units": "µV", "low_cutoff": "0.5", "high_cutoff": "n/a", "sampling_frequency": "256",
         "status": "good", "status_description": "n/a", "contacts": e, "hemisphere": hemi(e), "region": region(e),
         "x": "", "y": "", "z": "", "original_name": f"{OLD.get(e, e)}-Avg"} for e in ELECTRODES])
    write_tsv(d / f"{stem}_desc-preproc_events.tsv", ["onset", "duration", "trial_type", "source", "channels"], [
        {"onset": f"{bad0}", "duration": f"{bad1 - bad0}", "trial_type": "bad_segment", "source": "eeg-prep",
         "channels": ""},
        {"onset": f"{sz0}", "duration": f"{sz1 - sz0}", "trial_type": "seizure", "source": "annotation",
         "channels": "T7,F7"}])
    (d / f"{stem}_desc-preproc_eeg.json").write_text(json.dumps(
        {"SamplingFrequency": FS, "PowerLineFrequency": 50, "EEGReference": "average", "EEGChannelCount": 19},
        indent=2))
    recs.append({"recording_id": stem, "subject": "01", "session": "01", "task": "szMonitoring", "run": "01",
                 "path": f"sub-01/ses-01/eeg/{stem}_desc-preproc_eeg.edf", "sfreq": "256", "n_channels": 19,
                 "n_good": 19, "duration_s": f"{dur:g}", "montage": "average", "reference": "average",
                 "kind": "routine", "start_time": "", "n_seizures": 1, "source_format": "bids-edf", "age": "40",
                 "sex": "F"})
    truth[stem] = {"spikes": [{k: s[k] for k in ("time_s", "side", "centre")} for s in spikes],
                   "seizure": [sz0, sz1], "bad_segment": [bad0, bad1], "sleep_from": 330.0, "duration": dur}
    ref_data = data

    # --- sub-02: bipolar only, 240 s --------------------------------------------------
    dur2 = 240.0
    n2 = int(FS * dur2)
    t2 = np.arange(n2) / FS
    d2 = background(n2, rng)
    d2 += np.outer(alpha_w, 20 * np.sin(2 * np.pi * 9.5 * t2))
    spikes2 = spike_schedule(rng, [(2.0, 238.0, 10.0)], 0.0)
    plant(d2, FS, spikes2, rng)
    idx = {e: i for i, e in enumerate(ELECTRODES)}
    bip = np.vstack([d2[idx[a]] - d2[idx[b]] for a, b in PAIRS])
    names = [f"{a}-{b}" for a, b in PAIRS]
    stem2 = "sub-02_ses-01_task-routine_run-01"
    d = root / "sub-02" / "ses-01" / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    write_edf(d / f"{stem2}_desc-preproc_eeg.edf", bip, names, FS)
    write_tsv(d / f"{stem2}_desc-preproc_channels.tsv", CHANNEL_COLS, [
        {"name": nm, "type": "EEG", "units": "µV", "low_cutoff": "0.5", "high_cutoff": "n/a",
         "sampling_frequency": "256", "status": "good", "status_description": "n/a", "contacts": f"{a}|{b}",
         "hemisphere": hemi(a) if hemi(a) == hemi(b) else "Z",
         "region": "temporal" if "temporal" in (region(a), region(b)) else region(a),
         "x": "", "y": "", "z": "", "original_name": nm.upper()} for nm, (a, b) in zip(names, PAIRS)])
    write_tsv(d / f"{stem2}_desc-preproc_events.tsv", ["onset", "duration", "trial_type", "source", "channels"], [])
    (d / f"{stem2}_desc-preproc_eeg.json").write_text(json.dumps(
        {"SamplingFrequency": FS, "PowerLineFrequency": 60, "EEGReference": "bipolar"}, indent=2))
    recs.append({"recording_id": stem2, "subject": "02", "session": "01", "task": "routine", "run": "01",
                 "path": f"sub-02/ses-01/eeg/{stem2}_desc-preproc_eeg.edf", "sfreq": "256", "n_channels": 18,
                 "n_good": 18, "duration_s": f"{dur2:g}", "montage": "bipolar-longitudinal", "reference": "bipolar",
                 "kind": "routine", "start_time": "", "n_seizures": 0, "source_format": "edf", "age": "9", "sex": "M"})
    truth[stem2] = {"spikes": [{k: s[k] for k in ("time_s", "side", "centre")} for s in spikes2], "duration": dur2}

    header = ["recording_id", "subject", "session", "task", "run", "path", "sfreq", "n_channels", "n_good",
              "duration_s", "montage", "reference", "kind", "start_time", "n_seizures", "source_format", "age", "sex"]
    with open(root / "recordings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerows(recs)

    # --- the sleep port: eeg-sleep's hypnogram for sub-01 ------------------------------
    sl = out / "linked" / "sleep" / "eeg-sleep"
    sl.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, start in enumerate(np.arange(0.0, dur, 30.0)):
        stage = "W" if start < 330 else ("N2" if start < 420 else "N3")
        rows.append({"recording_id": stem, "epoch": k, "start_s": f"{start:g}", "duration_s": "30", "stage": stage,
                     "confidence": "0.9", "channel": "C4"})
    with open(sl / "hypnogram.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["recording_id", "epoch", "start_s", "duration_s", "stage", "confidence",
                                           "channel"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # --- plain BIDS, as BIDS Siena writes it ---------------------------------------------
    stem3 = "sub-03_ses-01_task-szMonitoring_run-00"
    d = out / "plain" / "sub-03" / "ses-01" / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    write_edf(d / f"{stem3}_eeg.edf", ref_data, [f"{OLD.get(e, e)}-Avg" for e in ELECTRODES], FS)
    write_tsv(d / f"{stem3}_events.tsv", ["onset", "duration", "eventType", "confidence", "channels", "dateTime",
                                         "recordingDuration"],
              [{"onset": f"{sz0:.2f}", "duration": f"{sz1 - sz0:.2f}", "eventType": "sz_foc_ia", "confidence": "n/a",
                "channels": "T3,F7", "dateTime": "2016-01-01 02:18:17", "recordingDuration": f"{dur:.2f}"}])
    (d / f"{stem3}_eeg.json").write_text(json.dumps({"SamplingFrequency": 256, "PowerLineFrequency": 50,
                                                     "EEGReference": "Common Average"}, indent=2))
    truth[stem3] = truth[stem]

    (out / "truth.json").write_text(json.dumps(truth))
    return out


if __name__ == "__main__":
    print(make(Path(sys.argv[1] if len(sys.argv) > 1 else "data/fixture")))
