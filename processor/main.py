"""
eeg-spikes: interictal epileptiform discharge (spike) detection on clinical scalp EEG.

Reads the BIDS-EEG derivative of eeg-prep (or a plain BIDS-EEG dataset, or loose EDF/BDF/
BrainVision/FIF/EEGLAB files) and, optionally, eeg-sleep's hypnogram.csv from INPUT_DIR, and
writes to OUTPUT_DIR/eeg-spikes/ (or OUTPUT_DIR/eeg-spikes/<LABEL>/):

  spikes.csv          recording_id, channel, time_s, amplitude_uv, duration_ms, detector, sleep_stage, event_id
  spike_rates.csv     recording_id, channel, n_spikes, minutes, rate_per_min, detector
  spike_events.csv    recording_id, event_id, time_s, n_channels, channels, leading_channel, hemisphere, region,
                      detector, sleep_stage
  lateralisation.csv  recording_id, n_left, n_right, n_midline, laterality_index, dominant_region, ...
  rates_by_stage.csv  recording_id, channel, stage, n_spikes, minutes, rate_per_min (when stages are known)
  features.csv        recording_id, channel, feature (spikerate_<detector>), value
  figures/*.png, summary.json, provenance.json

One recording at a time, one time window (WINDOW_S) of the channels at a time, channels in
parallel (THREADS). Spikes are detected on the longitudinal bipolar montage derived in-app
from referential data (MONTAGE=bipolar, the default) or on the channels as stored.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _prepare_environment() -> None:
    """Lambda-safe start: writable HOME/TMPDIR, caches under TMPDIR, one BLAS thread per process."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    tmp = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    home = Path(os.environ.get("HOME") or tmp / "home")
    for d in (tmp, home):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    os.environ["TMPDIR"] = str(tmp)
    tempfile.tempdir = None
    mpl = os.environ.get("MPLCONFIGDIR")
    if not mpl or not os.access(mpl, os.W_OK):
        target = tmp / "mpl"
        target.mkdir(parents=True, exist_ok=True)
        # reuse the font cache built into the image (read-only) so matplotlib does not rebuild it
        for src in (Path(mpl) if mpl else None, Path("/opt/eeg-spikes/mpl")):
            if src and src.is_dir():
                for f in src.glob("fontlist-*.json"):
                    try:
                        (target / f.name).write_bytes(f.read_bytes())
                    except OSError:
                        pass
        os.environ["MPLCONFIGDIR"] = str(target)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(tmp / "numba"))
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_DATA", str(tmp / "mne_data"))
    os.environ.setdefault("MNE_USE_CUDA", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")


_prepare_environment()

import bisect  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import numpy as np  # noqa: E402

from processor import detect as D  # noqa: E402
from processor import eeg as E  # noqa: E402
from processor import montage as MT  # noqa: E402
from processor import resources as R  # noqa: E402
from processor import stages as S  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("eeg-spikes")
log.setLevel(logging.INFO)
for noisy in ("mne", "numba", "joblib", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "eeg-spikes"
REPO_ROOT = Path(__file__).resolve().parent.parent
PAD_S = 10.0     # context read on each side of a window (never across excluded time)
EDGE_S = 1.0     # Janca ignores the first and last second it is given
PROGRESS_S = 60.0

SPIKES_HEADER = ["recording_id", "channel", "time_s", "amplitude_uv", "duration_ms", "detector", "sleep_stage",
                 "event_id"]
RATES_HEADER = ["recording_id", "channel", "n_spikes", "minutes", "rate_per_min", "detector"]
EVENTS_HEADER = ["recording_id", "event_id", "time_s", "n_channels", "channels", "leading_channel", "hemisphere",
                 "region", "detector", "sleep_stage"]
LAT_HEADER = ["recording_id", "n_left", "n_right", "n_midline", "laterality_index", "dominant_region", "n_events",
              "minutes", "rate_left_per_min", "rate_right_per_min", "rate_laterality_index", "dominant_channel",
              "montage", "detector"]
STAGE_HEADER = ["recording_id", "channel", "stage", "n_spikes", "minutes", "rate_per_min"]
FEATURES_HEADER = ["recording_id", "channel", "feature", "value"]

CITATIONS = ["10.1007/s10548-014-0379-1", "10.1016/j.clinph.2011.09.023", "10.3389/fnins.2013.00267",
             "10.1111/j.1528-1157.1998.tb01329.x"]

# Memory per parallel worker (a loky process: Python + epycom/numba + one channel of a window) and
# for the main process (MNE, the window of every channel, figures); see README "Runtime".
WORKER_MB = 250
MAIN_MB = 600


def version_of(root: Path) -> str:
    try:
        return (root / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("epycom", "mne", "numpy", "scipy", "joblib", "numba", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1 / 1024 / 1024 if sys.platform == "darwin" else 1 / 1024  # bytes on macOS, KiB on Linux
    return round(max(r, c) * scale, 1)


def _bool(name: str, value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"{name} must be true or false (got {value!r})")


PARAM_NAMES = ("DETECTOR", "JANCA_K1", "MONTAGE", "EXCLUDE_SEIZURE", "SEIZURE_MARGIN_S", "WINDOW_S",
               "GROUP_WINDOW_MS", "MIN_EVENT_CHANNELS", "THREADS", "LABEL")


def env_params() -> dict:
    """Parameters from the environment; raises ValueError on a bad value."""
    e = lambda k, d: (os.environ.get(k) or d).strip()  # noqa: E731
    raw = {
        "DETECTOR": e("DETECTOR", "janca").lower(),
        "JANCA_K1": e("JANCA_K1", "3.65"),
        "MONTAGE": e("MONTAGE", "bipolar").lower(),
        "EXCLUDE_SEIZURE": e("EXCLUDE_SEIZURE", "true"),
        "SEIZURE_MARGIN_S": e("SEIZURE_MARGIN_S", "60"),
        "WINDOW_S": e("WINDOW_S", "300"),
        "GROUP_WINDOW_MS": e("GROUP_WINDOW_MS", "50"),
        "MIN_EVENT_CHANNELS": e("MIN_EVENT_CHANNELS", "1"),
        "THREADS": e("THREADS", "0"),
        "LABEL": e("LABEL", ""),
    }
    p = dict(raw)
    if p["DETECTOR"] not in ("janca", "barkmeier"):
        raise ValueError(f"DETECTOR must be janca or barkmeier (got {raw['DETECTOR']!r})")
    if p["MONTAGE"] not in ("bipolar", "referential"):
        raise ValueError(f"MONTAGE must be bipolar or referential (got {raw['MONTAGE']!r})")
    p["k1"] = float(raw["JANCA_K1"])
    if p["k1"] <= 0:
        raise ValueError("JANCA_K1 must be > 0")
    p["exclude_seizure"] = _bool("EXCLUDE_SEIZURE", raw["EXCLUDE_SEIZURE"])
    p["margin"] = float(raw["SEIZURE_MARGIN_S"])
    if p["margin"] < 0:
        raise ValueError("SEIZURE_MARGIN_S must be >= 0")
    p["window"] = float(raw["WINDOW_S"])
    if not 60 <= p["window"] <= 570:
        # epycom 0.3's Janca detector fails on reads of >= 600 s (window + 2 x PAD_S of context):
        # its multi-buffer branch indexes a field "pos" that does not exist.
        raise ValueError("WINDOW_S must be between 60 and 570")
    p["group_window"] = float(raw["GROUP_WINDOW_MS"]) / 1000.0
    if p["group_window"] <= 0:
        raise ValueError("GROUP_WINDOW_MS must be > 0")
    p["min_channels"] = int(raw["MIN_EVENT_CHANNELS"])
    if p["min_channels"] < 1:
        raise ValueError("MIN_EVENT_CHANNELS must be >= 1")
    threads = int(raw["THREADS"])
    if threads < 0:
        raise ValueError("THREADS must be >= 0 (0 = every vCPU)")
    p["threads_requested"] = threads
    p["threads"], p["threads_why"] = R.resolve_workers(threads, WORKER_MB, MAIN_MB)
    if p["LABEL"] and not all(c.isalnum() or c in "-_" for c in p["LABEL"]):
        raise ValueError("LABEL may hold letters, digits, - and _ only")
    return p


class CsvOut:
    def __init__(self, path: Path, header: list[str]):
        self.fh = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.fh, fieldnames=header, extrasaction="ignore", lineterminator="\n")
        self.w.writeheader()

    def rows(self, rows):
        self.w.writerows(rows)
        self.fh.flush()

    def close(self):
        self.fh.close()


# --- montage -----------------------------------------------------------------------

def analysis_channels(rec: E.Recording, chans: list[E.Channel], montage: str) -> tuple[list[dict], str, list[str]]:
    """
    The channels to analyse: [{label, contacts, hemisphere, region, src: (raw index,) or (a, b)}],
    the montage used (`bipolar` | `referential`) and notes. Referential data become the
    longitudinal bipolar montage when montage=bipolar (pairs whose two electrodes are good).
    """
    good = [c for c in chans if c.status == "good"]
    notes: list[str] = []
    if not good:
        return [], montage, notes
    if E.is_bipolar_recording(rec, good):
        if montage == "referential":
            notes.append("MONTAGE=referential, but this is a bipolar-only recording: analysed on its pairs as recorded")
        out = [{"label": c.label, "contacts": c.contacts, "hemisphere": c.hemisphere, "region": c.region,
                "src": (c.name,)} for c in good if c.bipolar]
        return sorted(out, key=lambda d: (MT.chain_order(d["label"]), d["label"])), "bipolar", notes
    ref = [c for c in good if not c.bipolar]
    if montage == "bipolar":
        by_label = {c.label: c for c in ref}
        out, missing = [], []
        for a, b in MT.BIPOLAR_PAIRS:
            if a in by_label and b in by_label:
                out.append({"label": f"{a}-{b}", "contacts": (a, b), "hemisphere": MT.hemisphere((a, b)),
                            "region": MT.region((a, b)), "src": (by_label[a].name, by_label[b].name)})
            else:
                missing.append(f"{a}-{b}")
        if len(out) >= 2:
            if missing:
                notes.append(f"bipolar pairs without both electrodes good, not analysed: {', '.join(missing)}")
            return out, "bipolar", notes
        notes.append("fewer than 2 longitudinal bipolar pairs can be formed from the good electrodes; "
                     "analysed referentially")
    out = [{"label": c.label, "contacts": c.contacts, "hemisphere": c.hemisphere, "region": c.region,
            "src": (c.name,)} for c in ref]
    return out, "referential", notes


def derive(data: np.ndarray, pos: dict[str, int], chans: list[dict]) -> list[np.ndarray]:
    out = []
    for c in chans:
        s = c["src"]
        out.append(data[pos[s[0]]] if len(s) == 1 else data[pos[s[0]]] - data[pos[s[1]]])
    return out


# --- one recording ----------------------------------------------------------------

def process_recording(rec: E.Recording, params: dict, parallel, hypnograms: dict, fig_dir: Path) -> dict:
    from joblib import delayed

    t0 = time.monotonic()
    raw = E.open_raw(rec.path)
    fs = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    duration = n_times / fs
    if fs < 100:
        raise ValueError(f"sampling rate {fs:g} Hz is too low for spike detection (needs >= 100 Hz)")
    chans = E.channels(rec, list(raw.ch_names), list(raw.get_channel_types()))
    achans, montage, mnotes = analysis_channels(rec, chans, params["MONTAGE"])
    rec.notes.extend(mnotes)
    if not achans:
        raise ValueError(f"no good scalp EEG channel among {len(chans)}")
    names = sorted({n for c in achans for n in c["src"]}, key=list(raw.ch_names).index)
    picks = [list(raw.ch_names).index(n) for n in names]
    pos = {n: i for i, n in enumerate(names)}
    plan = E.analysis_plan(rec, duration, params["exclude_seizure"], params["margin"])
    evs = E.events(rec)
    hyp = hypnograms.get(rec.recording_id) or S.from_annotations(evs)
    line_hz = E.line_freq(rec)
    det = params["DETECTOR"]

    mask_iv = E.merge(plan.masks)
    mask_starts = [m[0] for m in mask_iv]

    def masked(t: float) -> bool:
        i = bisect.bisect_right(mask_starts, t) - 1
        return i >= 0 and t < mask_iv[i][1]

    labels = [c["label"] for c in achans]
    analysed: dict[str, list[tuple[float, float]]] = {l: [] for l in labels}
    found: dict[str, list[tuple[float, float, float]]] = {l: [] for l in labels}
    errors: dict[str, str] = {}
    windows = 0
    t_read = t_detect = 0.0
    last_log = time.monotonic()
    total_core = sum(max(0.0, b - a - 2 * EDGE_S) for a, b in plan.segments)
    done_core = 0.0
    for c0, c1, r0, r1 in E.chunks(plan.segments, params["window"], PAD_S, D.MIN_SECONDS, EDGE_S):
        if r0 is None:
            rec.notes.append(f"segment {c0:.1f}-{c1:.1f} s is shorter than {D.MIN_SECONDS:g} s; not analysed")
            continue
        s, e = max(0, int(round(r0 * fs))), min(n_times, int(round(r1 * fs)))
        ta = time.monotonic()
        data = E.read_window(raw, picks, s, e)
        xs = derive(data, pos, achans)
        del data
        tb = time.monotonic()
        ratios = [1.0] * len(achans)
        if det == "barkmeier":
            levels = np.array(parallel(delayed(D.broadband_level)(x, fs) for x in xs))
            ref = np.nanmedian(levels) if np.any(np.isfinite(levels)) else np.nan
            ratios = [float(v / ref) if np.isfinite(ref) and ref > 0 else 1.0 for v in levels]
        results = parallel(delayed(D.detect_channel)(x, fs, det, params["k1"], s / fs, (c0, c1), ratios[i], line_hz)
                           for i, x in enumerate(xs))
        del xs
        t_read += tb - ta
        t_detect += time.monotonic() - tb
        windows += 1
        core_ok = E.subtract([(c0, c1)], [m for m in mask_iv if m[1] > c0 and m[0] < c1])
        for lab, (dets, err) in zip(labels, results):
            if err:
                errors.setdefault(lab, f"{c0:.0f}-{c1:.0f} s: {err}")
                continue
            analysed[lab].extend(core_ok)
            found[lab].extend(d for d in dets if not masked(d[0]))
        done_core += c1 - c0
        if time.monotonic() - last_log >= PROGRESS_S:
            log.info("%s: %.0f%% (%.1f of %.1f min analysed so far)", rec.recording_id,
                     100 * done_core / max(total_core, 1e-9), done_core / 60, total_core / 60)
            last_log = time.monotonic()

    minutes = {l: sum(b - a for a, b in analysed[l]) / 60.0 for l in labels}
    info = {c["label"]: c for c in achans}
    spikes = []
    for lab in labels:
        for t, amp, dur in sorted(found[lab]):
            spikes.append({"recording_id": rec.recording_id, "channel": lab, "time_s": f"{t:.4f}",
                           "amplitude_uv": f"{amp:.2f}", "duration_ms": "" if not np.isfinite(dur) else f"{dur:.1f}",
                           "detector": det, "sleep_stage": hyp.stage_at(t) if hyp else "", "event_id": "",
                           "_t": t, "_a": amp})
    events = []
    if spikes:
        grouped = D.group_events([(s["_t"], s["channel"], s["_a"]) for s in spikes], params["group_window"])
        for k, ev in enumerate(grouped, start=1):
            for i in ev["members"]:
                spikes[i]["event_id"] = k
            lead = info[ev["leader"]]
            events.append({"recording_id": rec.recording_id, "event_id": k, "time_s": f"{ev['time_s']:.4f}",
                           "n_channels": ev["n_channels"], "channels": "|".join(ev["channels"]),
                           "leading_channel": ev["leader"], "hemisphere": lead["hemisphere"],
                           "region": lead["region"], "detector": det,
                           "sleep_stage": hyp.stage_at(ev["time_s"]) if hyp else "", "_t": ev["time_s"]})
    spikes.sort(key=lambda s: (s["_t"], s["channel"]))

    analysed_labels = [l for l in labels if minutes[l] > 0]
    n_by = Counter(s["channel"] for s in spikes)
    rate = {l: n_by[l] / minutes[l] for l in analysed_labels}
    rates = [{"recording_id": rec.recording_id, "channel": l, "n_spikes": n_by[l], "minutes": f"{minutes[l]:.3f}",
              "rate_per_min": f"{rate[l]:.4f}", "detector": det} for l in analysed_labels]
    features = [{"recording_id": rec.recording_id, "channel": l, "feature": f"spikerate_{det}",
                 "value": f"{rate[l]:.4f}"} for l in analysed_labels]

    # lateralisation: spike events (spanning >= MIN_EVENT_CHANNELS channels) by the hemisphere of their
    # leading channel
    total_min = max(minutes.values()) if minutes else 0.0
    counted = [e for e in events if e["n_channels"] >= params["min_channels"]]
    side = Counter(e["hemisphere"] for e in counted)
    nl, nr, nz = side.get("L", 0), side.get("R", 0), side.get("Z", 0)
    li = (nl - nr) / (nl + nr) if nl + nr else None
    reg = Counter(f"{MT.side_word(e['hemisphere'])} {e['region']}".strip() if e["hemisphere"] != "Z" else "midline"
                  for e in counted)
    dom_region = reg.most_common(1)[0][0] if reg else ""
    leaders = Counter(e["leading_channel"] for e in counted)
    dom_channel = leaders.most_common(1)[0][0] if leaders else ""
    mean_rate = lambda h: (float(np.mean([rate[l] for l in analysed_labels if info[l]["hemisphere"] == h]))  # noqa: E731
                           if any(info[l]["hemisphere"] == h for l in analysed_labels) else None)
    rl, rr = mean_rate("L"), mean_rate("R")
    rli = (rl - rr) / (rl + rr) if rl is not None and rr is not None and rl + rr > 0 else None
    lat = {"recording_id": rec.recording_id, "n_left": nl, "n_right": nr, "n_midline": nz,
           "laterality_index": "" if li is None else f"{li:.4f}", "dominant_region": dom_region,
           "n_events": len(counted), "minutes": f"{total_min:.3f}",
           "rate_left_per_min": "" if rl is None else f"{rl:.4f}", "rate_right_per_min": "" if rr is None else f"{rr:.4f}",
           "rate_laterality_index": "" if rli is None else f"{rli:.4f}", "dominant_channel": dom_channel,
           "montage": montage, "detector": det}

    # rates per sleep stage
    stage_rows: list[dict] = []
    by_stage: dict[str, dict] = {}
    if hyp:
        for lab in analysed_labels:
            mins = hyp.minutes(analysed[lab])
            counts = Counter(s["sleep_stage"] for s in spikes if s["channel"] == lab and s["sleep_stage"])
            for st in S.STAGES + ["sleep"]:
                if mins.get(st, 0) > 0:
                    stage_rows.append({"recording_id": rec.recording_id, "channel": lab, "stage": st,
                                       "n_spikes": counts.get(st, 0), "minutes": f"{mins[st]:.3f}",
                                       "rate_per_min": f"{counts.get(st, 0) / mins[st]:.4f}"})
        ref_lab = max(analysed_labels, key=lambda l: minutes[l]) if analysed_labels else None
        if ref_lab:
            mins = hyp.minutes(analysed[ref_lab])
            ev_counts = Counter(e["sleep_stage"] for e in counted if e["sleep_stage"])
            by_stage = {st: {"minutes": mins.get(st, 0.0), "events": ev_counts.get(st, 0)}
                        for st in S.STAGES + ["sleep"] if mins.get(st, 0) > 0}

    # figures
    figures = []
    if analysed_labels:
        from processor import figures as F

        f1 = fig_dir / f"{rec.recording_id}_spike_rates.png"
        try:
            F.rates_figure(f1, rec.recording_id, rate, {l: info[l]["contacts"] for l in analysed_labels},
                           {l: info[l]["hemisphere"] for l in analysed_labels}, montage, det)
            figures.append(("rates", f1))
        except Exception as exc:  # noqa: BLE001 - a figure must not fail the recording
            rec.notes.append(f"spike-rate figure not made: {type(exc).__name__}: {exc}")
        if by_stage:
            f2 = fig_dir / f"{rec.recording_id}_spikes_by_stage.png"
            try:
                if F.stage_figure(f2, rec.recording_id, by_stage, hyp.source):
                    figures.append(("stages", f2))
            except Exception as exc:  # noqa: BLE001
                rec.notes.append(f"stage figure not made: {type(exc).__name__}: {exc}")
        if dom_channel:
            f3 = fig_dir / f"{rec.recording_id}_example_spikes.png"
            try:
                snips = example_snippets(raw, counted, spikes, dom_channel, achans, names, picks, fs, n_times)
                if F.examples_figure(f3, rec.recording_id, snips, labels, dom_channel, montage):
                    figures.append(("examples", f3))
            except Exception as exc:  # noqa: BLE001
                rec.notes.append(f"example figure not made: {type(exc).__name__}: {exc}")

    n_ev_min = len(counted) / total_min if total_min > 0 else None
    top = max(analysed_labels, key=lambda l: rate[l]) if analysed_labels else ""
    headline = {
        "minutes_analysed": round(total_min, 2), "n_channels": len(analysed_labels), "montage": montage,
        "n_spikes": len(spikes), "n_events": len(counted), "n_events_all": len(events),
        "events_per_min": None if n_ev_min is None else round(n_ev_min, 3),
        "top_channel": top, "top_channel_rate_per_min": round(rate[top], 3) if top else None,
        "laterality_index": None if li is None else round(li, 3), "n_left": nl, "n_right": nr, "n_midline": nz,
        "dominant_region": dom_region, "dominant_channel": dom_channel,
    }
    if by_stage:
        ev_rate = {st: round(v["events"] / v["minutes"], 3) for st, v in by_stage.items()}
        headline["events_per_min_by_stage"] = ev_rate
        nrem_m = sum(by_stage[s]["minutes"] for s in S.NREM if s in by_stage)
        nrem_e = sum(by_stage[s]["events"] for s in S.NREM if s in by_stage)
        if nrem_m > 0:
            headline["events_per_min_nrem"] = round(nrem_e / nrem_m, 3)
        if nrem_m > 0 and "W" in by_stage and by_stage["W"]["events"] > 0:
            headline["nrem_wake_ratio"] = round((nrem_e / nrem_m) / (by_stage["W"]["events"] / by_stage["W"]["minutes"]), 3)
        headline["stage_source"] = hyp.source

    summary = {
        "sfreq": fs, "duration_s": round(duration, 3), "montage": montage, "n_channels_in_file": len(chans),
        "n_analysed": len(analysed_labels), "analysed": plan.description,
        "excluded": plan.excluded, "minutes": round(total_min, 3), "windows": windows, "n_spikes": len(spikes),
        "n_events": len(events), "stages": hyp.source if hyp else None, "line_freq_notched": line_hz,
        "channels_not_analysed": {c.name: c.reason for c in chans if c.status != "good"},
        "channel_errors": errors, "notes": rec.notes,
        "seconds": {"read": round(t_read, 2), "detect": round(t_detect, 2), "total": round(time.monotonic() - t0, 2)},
    }
    return {"spikes": spikes, "rates": rates, "events": events, "lat": lat, "stage_rows": stage_rows,
            "features": features, "summary": summary, "headline": headline, "figures": figures,
            "by_stage": by_stage}


def example_snippets(raw, events, spikes, leader, achans, names, picks, fs, n_times, k=3) -> list[dict]:
    """Up to k spike events led by `leader`: the ones at the 50th, 75th and 90th percentile of |amplitude|."""
    from scipy.signal import butter, sosfiltfilt

    amp = {(s["channel"], s["event_id"]): abs(s["_a"]) for s in spikes}
    cand = [e for e in events if e["leading_channel"] == leader]
    if not cand:
        return []
    cand.sort(key=lambda e: amp.get((leader, e["event_id"]), 0.0))
    idx = sorted({min(len(cand) - 1, int(round(q * (len(cand) - 1)))) for q in (0.5, 0.75, 0.9)})
    hi = min(70.0, fs / 2 - 1)
    sos = butter(2, [0.5, hi], "bandpass", fs=fs, output="sos")
    pos = {n: i for i, n in enumerate(names)}
    out = []
    for i in idx[:k]:
        t = cand[i]["_t"]
        s, e = max(0, int((t - 2.0) * fs)), min(n_times, int((t + 2.0) * fs))
        if e - s < int(fs):
            continue
        data = E.read_window(raw, picks, s, e)
        xs = np.vstack(derive(data, pos, achans)).astype(np.float64)
        xs = sosfiltfilt(sos, xs, axis=1)
        a, b = int((t - 0.75) * fs) - s, int((t + 0.75) * fs) - s
        a, b = max(0, a), min(xs.shape[1], b)
        out.append({"time_s": t, "data": xs[:, a:b], "fs": fs, "start_s": (s + a) / fs})
    return out


# --- captions ---------------------------------------------------------------------

def caption(kind: str, rid: str, montage: str, det: str, by_stage: dict | None = None) -> str:
    det_name = "Janca et al. 2015" if det == "janca" else "Barkmeier et al. 2012"
    if kind == "rates":
        return (f"{rid}: interictal spike rate ({det_name} detector, validated on intracranial EEG; run here on scalp "
                f"EEG, {montage} montage). Left: rate per electrode"
                + (" (for bipolar pairs, the mean of the pairs an electrode belongs to)" if montage == "bipolar" else "")
                + "; right: spikes per minute on every channel, blue left hemisphere, orange right. Look for one "
                  "region with consistently higher rates; isolated high rates on frontopolar (Fp1/Fp2) or temporal "
                  "edge channels are often eye-movement or muscle artefact. A research output, to be reviewed "
                  "against the EEG before any clinical use.")
    if kind == "stages":
        return (f"{rid}: spike events per minute in each sleep stage (minutes analysed above each bar). Interictal "
                "spikes typically increase in NREM sleep (deepest in N3) and fall in REM (Malow et al. 1998); a "
                "flat profile or more spikes in wake can mean artefact-driven detections. Research output.")
    return (f"{rid}: three spike events led by the most frequent leading channel (orange trace), at the median, "
            f"75th and 90th percentile of its amplitude, all channels ({montage} montage, 0.5–70 Hz) from 0.75 s "
            "before to 0.75 s after the peak. Check that these look like epileptiform discharges (sharp, "
            "disrupting the background, with an after-going slow wave and a field over neighbouring channels) "
            "rather than artefact.")


# --- main ---------------------------------------------------------------------------

def main() -> int:
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    try:
        params = env_params()
    except ValueError as exc:
        log.error("bad parameter: %s", exc)
        return 2
    out_dir = output_dir / APP_NAME / params["LABEL"] if params["LABEL"] else output_dir / APP_NAME
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    recs, seen, problems = E.find_recordings(input_dir)
    hypnograms, hyp_files = S.load_hypnograms(input_dir)
    log.info("input: %d recording(s), %d file(s), %d hypnogram(s) for %d recording(s); detector %s, montage %s, "
             "seizures %s", len(recs), len(seen), len(hyp_files), len(hypnograms), params["DETECTOR"],
             params["MONTAGE"], f"excluded (margin {params['margin']:g} s)" if params["exclude_seizure"] else "kept")
    log.info("parallel workers: %d (%s)", params["threads"], params["threads_why"])
    for p in problems:
        log.warning("%s: %s", p["input"], p["reason"])

    outs = {"spikes": CsvOut(out_dir / "spikes.csv", SPIKES_HEADER),
            "rates": CsvOut(out_dir / "spike_rates.csv", RATES_HEADER),
            "events": CsvOut(out_dir / "spike_events.csv", EVENTS_HEADER),
            "lat": CsvOut(out_dir / "lateralisation.csv", LAT_HEADER),
            "features": CsvOut(out_dir / "features.csv", FEATURES_HEADER)}
    stage_out = None
    prov_recs: dict[str, dict] = {}
    for p in problems:
        prov_recs[p["recording_id"] or p["input"]] = {"status": p.get("status", "failed"), "input": p["input"],
                                                      "error": p["reason"], "seconds": 0.0, "montage": None}
    n_failed = sum(1 for p in problems if p.get("status", "failed") == "failed")
    n_ok = 0
    headlines: dict[str, dict] = {}
    figures: list[dict] = []
    warnings: list[str] = []
    lat_rows: list[dict] = []

    from joblib import Parallel

    with Parallel(n_jobs=params["threads"], backend="loky", max_nbytes=None) as parallel:
        for rec in recs:
            try:
                res = process_recording(rec, params, parallel, hypnograms, fig_dir)
            except Exception as exc:  # noqa: BLE001 - one recording's failure must not sink the run
                log.error("%s: FAILED: %s\n%s", rec.recording_id, exc, traceback.format_exc())
                prov_recs[rec.recording_id] = {"status": "failed", "input": rec.rel, "seconds": 0.0, "montage": None,
                                               "error": f"{type(exc).__name__}: {exc}"}
                warnings.append(f"{rec.recording_id}: failed ({type(exc).__name__}: {exc})")
                n_failed += 1
                continue
            s = res["summary"]
            if s["n_analysed"] == 0:
                status = "skipped: no analysable time" + (f" ({'; '.join(s['notes'])})" if s["notes"] else
                                                          f" ({s['analysed']})")
                log.warning("%s: %s", rec.recording_id, status)
                prov_recs[rec.recording_id] = {"status": status, "input": rec.rel, "seconds": s["seconds"]["total"],
                                               "montage": s["montage"], **s}
                warnings.append(f"{rec.recording_id}: {status}")
                continue
            outs["spikes"].rows(res["spikes"])
            outs["rates"].rows(res["rates"])
            outs["events"].rows(res["events"])
            outs["lat"].rows([res["lat"]])
            outs["features"].rows(res["features"])
            if res["stage_rows"]:
                if stage_out is None:
                    stage_out = CsvOut(out_dir / "rates_by_stage.csv", STAGE_HEADER)
                stage_out.rows(res["stage_rows"])
            n_ok += 1
            for ch, err in s["channel_errors"].items():
                log.warning("%s: channel %s: %s", rec.recording_id, ch, err)
            for note in s["notes"]:
                log.warning("%s: %s", rec.recording_id, note)
            h = res["headline"]
            headlines[rec.recording_id] = h
            lat_rows.append({"recording_id": rec.recording_id, "laterality_index": h["laterality_index"],
                             "n_events": h["n_events"]})
            for kind, path in res["figures"]:
                figures.append({"path": str(path.relative_to(out_dir)),
                                "caption": caption(kind, rec.recording_id, s["montage"], params["DETECTOR"])})
            log.info("%s: %d channels in file, %d analysed (%s), %.1f Hz, %.2f min analysed (%s), %d spikes, "
                     "%d events (%.2f/min), LI %s, stages %s, %.1f s (read %.1f s, detect %.1f s)",
                     rec.recording_id, s["n_channels_in_file"], s["n_analysed"], s["montage"], s["sfreq"],
                     s["minutes"], s["analysed"], s["n_spikes"], s["n_events"], h["events_per_min"] or 0.0,
                     h["laterality_index"], s["stages"] or "none", s["seconds"]["total"], s["seconds"]["read"],
                     s["seconds"]["detect"])
            prov_recs[rec.recording_id] = {"status": "ok", "input": rec.rel, "seconds": s["seconds"]["total"],
                                           "montage": s["montage"], **s}
    for f in list(outs.values()) + ([stage_out] if stage_out else []):
        f.close()

    if len(lat_rows) > 1:
        from processor import figures as F

        p = fig_dir / "laterality_index.png"
        try:
            if F.laterality_figure(p, lat_rows):
                figures.insert(0, {"path": str(p.relative_to(out_dir)), "caption": (
                    "Laterality index of the spike events of every recording: (left − right) / (left + right) "
                    "events, by the hemisphere of each event's leading channel; +1 = all left, −1 = all right; dot "
                    "size grows with the number of events. Consistent lateralisation across a patient's recordings "
                    "is more meaningful than one value; a few events make an extreme index. Research output.")})
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"laterality figure not made: {type(exc).__name__}: {exc}")

    status = "ok"
    if not recs and not problems:
        log.error("no recording found under INPUT_DIR (looked for recordings.csv and %s files); it holds %d file(s): %s",
                  "/".join(E.SIGNAL_EXTENSIONS), len(seen), ", ".join(seen[:40]) + (" …" if len(seen) > 40 else ""))
        status = "failed: no input"
    elif n_ok == 0 and n_failed > 0:
        status = "failed: every recording failed"
    elif n_ok == 0:
        status = "ok: nothing analysable"
    if hypnograms and not any(r in hypnograms for r in headlines):
        warnings.append("a hypnogram.csv was given, but none of its recording_ids matches a recording analysed here")

    finished = datetime.now(timezone.utc)
    tables = [{"path": "spike_rates.csv", "caption": "Spike rate per channel (spikes per minute analysed)."},
              {"path": "spike_events.csv", "caption": "Spikes within 50 ms across channels merged into events, with "
                                                      "the leading (earliest) channel and its hemisphere."},
              {"path": "lateralisation.csv", "caption": "Left, right and midline-led spike events per recording and "
                                                        "the laterality index (L − R) / (L + R)."},
              {"path": "spikes.csv", "caption": "Every detected spike."}]
    if stage_out:
        tables.insert(1, {"path": "rates_by_stage.csv", "caption": "Spike rate per channel in each sleep stage."})
    summary = {"app": APP_NAME, "version": version_of(REPO_ROOT), "recordings": headlines, "figures": figures,
               "tables": tables, "warnings": warnings,
               "notice": ("Spikes are detected with detectors developed and validated on intracranial EEG (Janca "
                          "et al. 2015; Barkmeier et al. 2012), run here on scalp EEG. Rates and lateralisation "
                          "are research outputs to be reviewed against the EEG, not clinical findings.")}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    prov = {
        "app": APP_NAME, "version": version_of(REPO_ROOT), "variant": None, "tools": tool_versions(),
        "parameters": {k: params[k] for k in PARAM_NAMES},
        "inputs": seen,
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2),
        "peak_rss_mb": peak_rss_mb(),
        "threads": {"requested": params["threads_requested"], "used": params["threads"], "why": params["threads_why"]},
        "recordings": prov_recs,
        "citations": CITATIONS,
        "licence_notice": None,
        "status": status,
    }
    with open(out_dir / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
    log.info("done: %d recording(s) ok, %d failed, %d skipped, %.1f s, peak RSS %.0f MB -> %s", n_ok, n_failed,
             len(prov_recs) - n_ok - n_failed, prov["wall_seconds"], prov["peak_rss_mb"], out_dir)
    return 1 if status.startswith("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
