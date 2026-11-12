"""
Finding and reading the recordings: the BIDS-EEG derivative of eeg-prep (the `eeg` port), a
plain BIDS-EEG dataset, or loose EDF/BDF/BrainVision/FIF/EEGLAB files.

The contract is docs/conventions.md of workflow-scalp-eeg:

- `recordings.csv` (anywhere under INPUT_DIR) lists the recordings, with `path` relative to
  the folder the CSV is in, plus sfreq, montage and n_seizures;
- `<stem>[_desc-preproc]_channels.tsv` next to the signal file says which channels are good,
  their type, contacts, hemisphere and region;
- `<stem>[_desc-preproc]_events.tsv` holds `seizure`, `bad_segment`, `artifact`, `sleep` and
  `wake` rows (`trial_type`), or HED-SCORE rows (`eventType` `sz`, `sz_foc_ia`, ...) in a
  SzCORE-style BIDS dataset such as BIDS Siena;
- a CHB-MIT `chbNN-summary.txt` next to loose EDFs lists their seizures.

Signal files that no recordings.csv lists are picked up as well, with whatever sidecars they
have. Nothing here loads signal data; the callers read one time window at a time.
Adapted from processor-ieeg-spikes' `ieeg.py` (same discovery and windowing).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from processor import montage as MT

log = logging.getLogger(__name__)

SIGNAL_EXTENSIONS = (".edf", ".bdf", ".vhdr", ".fif", ".set")
# Names that are clearly not brain signals (used when there is no _channels.tsv).
NON_NEURAL_NAME = re.compile(r"^(EEG\s+)?(ECG|EKG|EMG|EOG|RESP|PHOTIC|EVENT|STATUS|TRIG|DC\d*|SPO2|PULSE|PR|OSAT|ANNOT|"
                             r"LOC|ROC|CHIN|LEG|SAO2|VNS|IBI|BURSTS|SUPPR|MK)", re.IGNORECASE)
SEIZURE_TYPES = {"seizure", "ictal", "sz"}
EXCLUDED_TYPES = {"bad_segment", "artifact"}


def walk_files(root: Path):
    """Every file under root, following symlinked folders (INPUT_DIR entries are symlinks), once."""
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                yield Path(dirpath) / name


def _float(value) -> float | None:
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(fh, delimiter="\t")]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]


@dataclass
class Recording:
    recording_id: str
    path: Path                         # the signal file (.edf, .vhdr, ...)
    rel: str                           # path relative to INPUT_DIR, for logs and provenance
    source: str                        # "recordings.csv" or "scan"
    montage: str = ""                  # recordings.csv montage (average, bipolar-longitudinal, ...)
    subject: str = ""
    channels_tsv: Path | None = None
    events_tsv: Path | None = None
    sidecar_json: Path | None = None
    extra_seizures: list[tuple[float, float]] = field(default_factory=list)  # CHB-MIT summary
    notes: list[str] = field(default_factory=list)


def signal_prefix(path: Path) -> str:
    """`sub-01_task-x_desc-preproc_eeg.edf` -> `sub-01_task-x_desc-preproc`; `foo.edf` -> `foo`."""
    stem = path.name[: -len(path.suffix)]
    return stem[: -len("_eeg")] if stem.endswith("_eeg") else stem


def recording_id_from_prefix(prefix: str) -> str:
    """The BIDS entity chain of the source recording: drop a `_desc-<label>` entity."""
    return re.sub(r"_desc-[A-Za-z0-9]+", "", prefix)


def subject_of(recording_id: str) -> str:
    m = re.match(r"sub-([A-Za-z0-9]+)", recording_id)
    return m.group(1) if m else ""


def _sidecars(rec: Recording) -> None:
    prefix = signal_prefix(rec.path)
    folder = rec.path.parent
    for attr, suffix in (("channels_tsv", "_channels.tsv"), ("events_tsv", "_events.tsv"),
                         ("sidecar_json", "_eeg.json")):
        if getattr(rec, attr) is None:
            p = folder / f"{prefix}{suffix}"
            if p.is_file():
                setattr(rec, attr, p)


def chbmit_summary(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Seizures per file name from a CHB-MIT `chbNN-summary.txt`."""
    out: dict[str, list[tuple[float, float]]] = {}
    current = None
    start = None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        m = re.match(r"\s*File Name:\s*(\S+)", line)
        if m:
            current = m.group(1)
            out.setdefault(current, [])
            continue
        m = re.match(r"\s*Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)", line)
        if m:
            start = float(m.group(1))
            continue
        m = re.match(r"\s*Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)", line)
        if m and current and start is not None:
            out[current].append((start, float(m.group(1))))
            start = None
    return out


def find_recordings(input_dir: Path) -> tuple[list[Recording], list[str], list[dict]]:
    """
    (recordings, every input file seen (relative), problems). A problem is a listed recording
    whose file is missing, or a duplicate recording id; each is a dict with recording_id,
    input and reason.
    """
    files = list(walk_files(input_dir))
    rel = lambda p: os.path.relpath(p, input_dir)  # noqa: E731
    seen = [rel(p) for p in files]
    problems: list[dict] = []
    recs: list[Recording] = []
    listed: set[str] = set()

    for csv_path in (p for p in files if p.name == "recordings.csv"):
        try:
            rows = read_csv(csv_path)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            problems.append({"recording_id": "", "input": rel(csv_path), "reason": f"unreadable recordings.csv: {exc}"})
            continue
        for row in rows:
            rid, path = row.get("recording_id", ""), row.get("path", "")
            if not rid or not path:
                continue
            sig = csv_path.parent / path
            if not sig.is_file():
                problems.append({"recording_id": rid, "input": rel(csv_path),
                                 "reason": f"recordings.csv lists {path}, which is not in the input"})
                continue
            listed.add(os.path.realpath(sig))
            recs.append(Recording(recording_id=rid, path=sig, rel=rel(sig), source="recordings.csv",
                                  montage=row.get("montage", ""), subject=row.get("subject", "") or subject_of(rid)))

    summaries: dict[str, dict] = {}
    for p in files:
        if re.match(r".+-summary\.txt$", p.name):
            summaries[str(p.parent)] = {**summaries.get(str(p.parent), {}), **chbmit_summary(p)}

    for p in files:
        if p.suffix.lower() not in SIGNAL_EXTENSIONS or os.path.realpath(p) in listed:
            continue
        prefix = signal_prefix(p)
        rid = recording_id_from_prefix(prefix)
        r = Recording(recording_id=rid, path=p, rel=rel(p), source="scan", subject=subject_of(rid))
        sz = summaries.get(str(p.parent), {}).get(p.name)
        if sz:
            r.extra_seizures = list(sz)
        recs.append(r)

    for r in recs:
        _sidecars(r)

    out: list[Recording] = []
    by_id: dict[str, Recording] = {}
    for r in sorted(recs, key=lambda r: (r.source != "recordings.csv", r.rel)):
        if r.recording_id in by_id:
            keep = by_id[r.recording_id]
            problems.append({"recording_id": r.recording_id, "input": r.rel, "status": "skipped",
                             "reason": f"duplicate recording id; {keep.rel} is used"})
            continue
        by_id[r.recording_id] = r
        out.append(r)
    out.sort(key=lambda r: r.recording_id)
    return out, seen, problems


# --- channels -------------------------------------------------------------------

@dataclass
class Channel:
    name: str                    # the name in the signal file
    label: str                   # canonical: "Fp1" or "Fp1-F7" (the _channels.tsv name of eeg-prep)
    contacts: tuple[str, ...]    # ("Fp1",) or ("Fp1", "F7")
    status: str                  # good | bad
    reason: str = ""             # why it is not analysed
    hemisphere: str = ""
    region: str = ""

    @property
    def bipolar(self) -> bool:
        return len(self.contacts) == 2


def _from_name(name: str) -> tuple[str, tuple[str, ...]]:
    parsed = MT.parse(name)
    if parsed:
        return parsed
    return name, tuple(name.split("|"))


def channels(rec: Recording, raw_names: list[str], raw_types: list[str]) -> list[Channel]:
    """
    Every channel of the recording with good/bad status, canonical label and contacts. With a
    channels.tsv, its type, status, contacts, hemisphere and region columns decide (a channel
    absent from the file is bad); without one, MNE's channel type and the name decide.
    """
    in_file = set(raw_names)
    out: list[Channel] = []
    if rec.channels_tsv is not None:
        for row in read_tsv(rec.channels_tsv):
            name = row.get("name", "")
            if not name:
                continue
            ctype = (row.get("type") or "EEG").upper()
            status = (row.get("status") or "good").lower()
            reason = ""
            if name not in in_file:
                reason = "listed in channels.tsv but not in the signal file"
            elif status == "bad":
                reason = row.get("status_description") or "status bad in channels.tsv"
                if reason.lower() == "n/a":
                    reason = "status bad in channels.tsv"
            elif ctype != "EEG":
                reason = f"type {ctype} is not analysed"
            contacts_col = row.get("contacts", "")
            parsed = MT.parse(name)
            if contacts_col and contacts_col.lower() != "n/a":
                contacts = tuple(MT.electrode(c) or c for c in contacts_col.split("|"))
                label = "-".join(contacts) if len(contacts) == 2 else (parsed[0] if parsed else name)
            elif parsed:
                label, contacts = parsed
            else:
                label, contacts = name, (name,)
            if not reason and not all(MT.electrode(c) for c in contacts):
                reason = "not a 10-20/10-10 electrode name"
            hemi = (row.get("hemisphere") or "").upper()
            reg = (row.get("region") or "").lower()
            out.append(Channel(name, label, contacts, "bad" if reason else "good", reason,
                               hemi if hemi in ("L", "R", "Z") else MT.hemisphere(contacts),
                               reg if reg and reg != "n/a" else MT.region(contacts)))
        return out
    for name, mtype in zip(raw_names, raw_types):
        ctype = mtype.upper()
        parsed = MT.parse(name)
        if ctype != "EEG":
            out.append(Channel(name, name, (name,), "bad", f"type {ctype} is not analysed"))
        elif NON_NEURAL_NAME.match(name):
            out.append(Channel(name, name, (name,), "bad", "name looks like a non-brain channel"))
        elif parsed is None:
            out.append(Channel(name, name, (name,), "bad", "not a 10-20/10-10 electrode name"))
        else:
            label, contacts = parsed
            out.append(Channel(name, label, contacts, "good", "", MT.hemisphere(contacts), MT.region(contacts)))
    # the same label twice (CHB-MIT's T8-P8-0 / T8-P8-1): keep the first
    seen: set[str] = set()
    for c in out:
        if c.status == "good":
            if c.label in seen:
                c.status, c.reason = "bad", f"duplicate of {c.label}"
            seen.add(c.label)
    return out


def is_bipolar_recording(rec: Recording, good: list[Channel]) -> bool:
    """A bipolar-only recording: recordings.csv says so, or most good channels are pairs."""
    if rec.montage:
        return rec.montage.lower().startswith("bipolar")
    return sum(c.bipolar for c in good) > len(good) / 2


# --- time intervals -------------------------------------------------------------

def events(rec: Recording) -> list[dict]:
    if rec.events_tsv is None:
        return []
    try:
        return read_tsv(rec.events_tsv)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        rec.notes.append(f"events.tsv unreadable: {exc}")
        return []


def event_type(e: dict) -> str:
    """`trial_type`, or a HED-SCORE `eventType` (`sz*` -> seizure)."""
    t = (e.get("trial_type") or "").strip().lower()
    if t and t != "n/a":
        return "seizure" if t in SEIZURE_TYPES else t
    et = (e.get("eventType") or "").strip().lower()
    if et == "sz" or et.startswith("sz_") or et.startswith("sz-"):
        return "seizure"
    return et


def seizure_intervals(rec: Recording, evs: list[dict], duration_s: float) -> list[tuple[float, float, bool]]:
    """(start, end, end_known) of every known seizure; end = recording end when its duration is unknown."""
    out = []
    for e in evs:
        if event_type(e) == "seizure":
            on, dur = _float(e.get("onset")), _float(e.get("duration"))
            if on is None:
                continue
            known = dur is not None and dur > 0
            out.append((on, on + dur if known else duration_s, known))
    for a, b in rec.extra_seizures:
        out.append((a, b, True))
    return sorted(out)


def typed_intervals(evs: list[dict], types: set[str]) -> list[tuple[float, float, str]]:
    out = []
    for e in evs:
        t = event_type(e)
        if t in types:
            on, dur = _float(e.get("onset")), _float(e.get("duration"))
            if on is not None and dur is not None and dur > 0:
                out.append((on, on + dur, t))
    return sorted(out)


def subtract(intervals: list[tuple[float, float]], cut: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """intervals minus the union of cut."""
    out = list(intervals)
    for c0, c1 in cut:
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return [(a, b) for a, b in out if b > a]


def overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total length of the intersection of two interval lists (each without self-overlap)."""
    total = 0.0
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                total += hi - lo
    return total


@dataclass
class Plan:
    segments: list[tuple[float, float]]   # analysed stretches (seizures cut out), seconds
    masks: list[tuple[float, float]]      # bad_segment / artifact intervals: detections dropped, time not counted
    description: str
    excluded: list[dict]


def analysis_plan(rec: Recording, duration_s: float, exclude_seizure: bool, seizure_margin_s: float) -> Plan:
    """
    Which parts of the recording to analyse: the whole recording minus every known seizure
    with seizure_margin_s on both sides (to the end of the recording when its end is
    unknown) when exclude_seizure. `bad_segment` and `artifact` intervals are masked: the
    detector runs through them (so its background estimate stays continuous) but what it
    finds there is dropped and their time is not counted.
    """
    evs = events(rec)
    excluded: list[dict] = []
    cuts: list[tuple[float, float]] = []
    if exclude_seizure:
        for s, e, known in seizure_intervals(rec, evs, duration_s):
            a, b = max(0.0, s - seizure_margin_s), min(duration_s, e + seizure_margin_s if known else duration_s)
            cuts.append((a, b))
            excluded.append({"start_s": round(a, 3), "end_s": round(b, 3),
                             "reason": "seizure" + ("" if known else " (end unknown: cut to the end of the recording)")})
    masks = []
    for a, b, t in typed_intervals(evs, EXCLUDED_TYPES):
        a, b = max(0.0, a), min(duration_s, b)
        if b > a:
            masks.append((a, b))
            excluded.append({"start_s": round(a, 3), "end_s": round(b, 3), "reason": t})
    segs = subtract([(0.0, duration_s)], cuts)
    n_sz = sum(1 for x in excluded if x["reason"].startswith("seizure"))
    n_bad = len(masks)
    desc = "whole recording"
    if n_sz:
        desc += f" minus {n_sz} seizure(s) ±{seizure_margin_s:g} s"
    if n_bad:
        desc += f"{',' if n_sz else ''} minus {n_bad} bad segment(s)/artifact(s) ({sum(b - a for a, b in masks):.0f} s)"
    return Plan(segs, masks, desc, excluded)


def chunks(segments: list[tuple[float, float]], window_s: float, pad_s: float, min_s: float,
           edge_s: float = 0.0):
    """
    Split segments into analysis windows. Yields (core_start, core_end, read_start, read_end)
    in seconds: detections are kept when they fall in the core; the read interval adds pad_s
    on each side where the segment allows it (never across excluded time). edge_s is trimmed
    from both ends of each segment (a detector that cannot see its first and last second).
    Segments shorter than min_s are skipped (yielded with read_start = None so the caller can
    log them).
    """
    for s0, s1 in segments:
        c0, c1 = s0 + edge_s, s1 - edge_s
        if s1 - s0 < min_s or c1 <= c0:
            yield s0, s1, None, None
            continue
        n = max(1, math.ceil((c1 - c0) / window_s - 1e-9))
        step = (c1 - c0) / n
        for i in range(n):
            a, b = c0 + i * step, c0 + (i + 1) * step
            ra, rb = max(s0, a - pad_s), min(s1, b + pad_s)
            yield a, b, ra, rb


def sidecar(rec: Recording) -> dict:
    if rec.sidecar_json is None:
        return {}
    try:
        d = json.loads(rec.sidecar_json.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def line_freq(rec: Recording) -> float | None:
    v = _float(sidecar(rec).get("PowerLineFrequency"))
    return v if v in (50.0, 60.0) else None


def sleep_annotations(evs: list[dict]) -> list[tuple[float, float, str]]:
    """Annotated `sleep` / `wake` intervals as (start, end, stage): `sleep` (stage unknown) or `W`."""
    return [(a, b, "W" if t == "wake" else "sleep") for a, b, t in typed_intervals(evs, {"sleep", "wake"})]


# --- signal ---------------------------------------------------------------------

def open_raw(path: Path):
    """The recording, lazily (nothing loaded). EDF/BDF/BrainVision/FIF/EEGLAB through MNE."""
    import mne

    return mne.io.read_raw(str(path), preload=False, verbose="error")


def read_window(raw, picks: list[int], start: int, stop: int, max_bytes: int = 256 << 20):
    """
    Samples [start, stop) of the picked channels as float32 microvolts, shape (channels,
    samples). Read in groups of channels so the float64 buffer MNE returns stays under
    max_bytes.
    """
    import numpy as np

    n = stop - start
    out = np.empty((len(picks), n), dtype=np.float32)
    group = max(1, int(max_bytes // max(1, n * 8)))
    for i in range(0, len(picks), group):
        block = raw.get_data(picks=picks[i:i + group], start=start, stop=stop)
        out[i:i + group] = block * 1e6  # MNE returns volts
        del block
    return out
