"""
Sleep stages for the spikes: eeg-sleep's `hypnogram.csv` (30 s epochs; the optional `sleep`
input), else the recording's annotated `sleep` / `wake` events (docs/conventions.md, "Sleep
stages shared between steps").
"""
from __future__ import annotations

import bisect
import csv
import logging
from pathlib import Path

from processor import eeg as E

log = logging.getLogger(__name__)

STAGES = ["W", "N1", "N2", "N3", "R"]
NREM = ("N1", "N2", "N3")
_MAP = {"W": "W", "WAKE": "W", "0": "W", "N1": "N1", "1": "N1", "S1": "N1", "N2": "N2", "2": "N2", "S2": "N2",
        "N3": "N3", "3": "N3", "S3": "N3", "S4": "N3", "N4": "N3", "R": "R", "REM": "R", "4": "R",
        "SLEEP": "sleep"}


def canonical_stage(value: str) -> str | None:
    """W/N1/N2/N3/R (YASA letters, integers 0-4, R&K S1-S4, `sleep`), or None (unscored, artefact)."""
    return _MAP.get((value or "").strip().upper())


class Hypnogram:
    """Stage intervals of one recording, sorted: (start_s, end_s, stage)."""

    def __init__(self, intervals: list[tuple[float, float, str]], source: str):
        self.intervals = sorted(intervals)
        self.starts = [a for a, _, _ in self.intervals]
        self.source = source

    def stage_at(self, t: float) -> str:
        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0:
            a, b, s = self.intervals[i]
            if a <= t < b:
                return s
        return ""

    def minutes(self, analysed: list[tuple[float, float]]) -> dict[str, float]:
        """Minutes of the analysed intervals in each stage."""
        out: dict[str, float] = {}
        iv = E.merge(analysed)
        starts = [x[0] for x in iv]
        for a, b, s in self.intervals:
            k = max(0, bisect.bisect_right(starts, a) - 1)
            m = 0.0
            while k < len(iv) and iv[k][0] < b:
                lo, hi = max(a, iv[k][0]), min(b, iv[k][1])
                if hi > lo:
                    m += hi - lo
                k += 1
            if m > 0:
                out[s] = out.get(s, 0.0) + m / 60.0
        return out


def load_hypnograms(input_dir: Path) -> tuple[dict[str, Hypnogram], list[str]]:
    """Every `hypnogram.csv` under input_dir, by recording_id. Returns (hypnograms, files read)."""
    rows_by_rec: dict[str, list[tuple[float, float, str]]] = {}
    files = []
    for p in E.walk_files(input_dir):
        if p.name != "hypnogram.csv":
            continue
        try:
            rows = E.read_csv(p)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            log.warning("%s unreadable: %s", p, exc)
            continue
        files.append(str(p))
        for r in rows:
            rid = r.get("recording_id", "")
            a, d = E._float(r.get("start_s")), E._float(r.get("duration_s"))
            st = canonical_stage(r.get("stage", ""))
            if not rid or a is None or d is None or d <= 0 or st is None:
                continue
            rows_by_rec.setdefault(rid, []).append((a, a + d, st))
    return {rid: Hypnogram(iv, "hypnogram.csv (eeg-sleep)") for rid, iv in rows_by_rec.items()}, files


def from_annotations(evs: list[dict]) -> Hypnogram | None:
    iv = E.sleep_annotations(evs)
    return Hypnogram(iv, "annotated sleep/wake events") if iv else None
