"""
Interictal epileptiform discharge (spike) detectors from epycom 0.3 (Apache-2.0; FNUSA-ICRC
Brno), and the cross-channel grouping of their detections.

Both detectors were developed and validated on **intracranial** EEG (Janca et al. 2015;
Barkmeier et al. 2012). This app runs them on scalp EEG and reports its own scalp
validation (README, "Validation").

Both run on the channel resampled to 200 Hz (the rate Janca's detector is designed for and
resamples to internally). We resample with a polyphase anti-aliasing filter ourselves because
epycom's internal decimation uses an integer factor while labelling the result 200 Hz (1024 Hz
-> factor 5 -> 204.8 Hz treated as 200 Hz, a 2.4 % timing error). Adapted from
processor-ieeg-spikes.
"""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

DETECTOR_FS = 200
# Janca needs >= 20 s per call (5 s windows, 1 s steps, a 5-point smoothing filtfilt of the
# window statistics); we ask for a little more.
MIN_SECONDS = 30.0
# A detection is moved to the largest |deflection| of the 1-70 Hz signal within this many
# seconds (Janca marks the envelope maximum, which can sit a few ms off the sharp peak).
PEAK_SEARCH_S = 0.04
# Base-to-base duration: the turning points on either side of the peak, at most this far.
BASE_SEARCH_S = 0.15


def to_detector_rate(x: np.ndarray, fs: float) -> np.ndarray:
    """Resample one channel to 200 Hz (polyphase FIR); unchanged when already at 200 Hz."""
    x = np.asarray(x, dtype=np.float64)
    fs_int = int(round(fs))
    if fs_int == DETECTOR_FS:
        return x
    frac = Fraction(DETECTOR_FS, fs_int)
    return resample_poly(x, frac.numerator, frac.denominator)


def notch(x200: np.ndarray, line_hz: float | None) -> np.ndarray:
    """Notch 60 Hz mains (Janca notches 50 Hz itself, whatever the mains frequency)."""
    if line_hz is None or line_hz != 60.0:
        return x200
    b, a = iirnotch(60.0, 30.0, fs=DETECTOR_FS)
    return filtfilt(b, a, x200)


def broad(x200: np.ndarray) -> np.ndarray:
    """The 1-70 Hz signal (2nd-order Butterworth high-pass, 4th-order low-pass, zero phase)."""
    fs = DETECTOR_FS
    bh, ah = butter(2, 1 / (fs / 2), "highpass")
    bl, al = butter(4, 70 / (fs / 2), "lowpass")
    return filtfilt(bl, al, filtfilt(bh, ah, x200))


def _barkmeier_broad(x200: np.ndarray) -> np.ndarray:
    """The 1-80 Hz signal Barkmeier's detector works on (its own filters)."""
    fs = DETECTOR_FS
    bh, ah = butter(2, 1 / (fs / 2), "highpass")
    bl, al = butter(4, 80 / (fs / 2), "lowpass")
    return filtfilt(bl, al, filtfilt(bh, ah, x200))


def broadband_level(x: np.ndarray, fs: float) -> float:
    """Mean |1-80 Hz signal| of one channel (µV), for Barkmeier's recording-level scaling."""
    try:
        x200 = to_detector_rate(x, fs)
        return float(np.mean(np.abs(_barkmeier_broad(x200)))) if np.all(np.isfinite(x200)) else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def janca(x200: np.ndarray, k1: float = 3.65, bandwidth=(10, 60)) -> list[float]:
    """Janca et al. 2015 detector (epycom.detect_spikes_janca) on a 200 Hz signal: detection times (s)."""
    from epycom.event_detection.spike.janca_detector import detect_spikes_janca

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # log(0) on flat stretches
        out = detect_spikes_janca(x200, fs=DETECTOR_FS, k1=k1, k2=k1, bandwidth=tuple(bandwidth))
    return [float(peak) / DETECTOR_FS for peak, _env, _amp in out]


def barkmeier(x200: np.ndarray, level_ratio: float = 1.0, union_s: float = 0.12) -> list[float]:
    """
    Barkmeier et al. 2012 detector (epycom.detect_spikes_barkmeier, default thresholds) on a
    200 Hz signal: detection times (s).

    The detector rescales the data to a 70 µV reference (epycom: scale / median(mean|x|), a
    median over channels in the paper's multi-channel algorithm) and applies fixed amplitude
    and slope thresholds. Given one channel, epycom normalises each channel to itself; we pass
    scale = 70 × (this channel's level / the recording's median level in the window), which
    restores the recording-level scaling, and merge detections closer than union_s (epycom can
    report one discharge at neighbouring samples). As in processor-ieeg-spikes.
    """
    from epycom.event_detection.spike.barkmeier_detector import detect_spikes_barkmeier

    fs = DETECTOR_FS
    scale = 70.0 * level_ratio if np.isfinite(level_ratio) and level_ratio > 0 else 70.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = detect_spikes_barkmeier(x200, fs=fs, scale=scale)
    if not out:
        return []
    b = _barkmeier_broad(x200)
    merged: list[int] = []
    for p in sorted(int(o[0]) for o in out):
        if merged and p - merged[-1] <= union_s * fs:
            if abs(b[p]) > abs(b[merged[-1]]):
                merged[-1] = p
            continue
        merged.append(p)
    return [p / fs for p in merged]


def describe(sig: np.ndarray, t: float) -> tuple[float, float, float]:
    """
    (peak time s, signed amplitude µV, base-to-base duration ms) of the discharge detected at
    t in sig (the 1-70 Hz signal at 200 Hz): the largest |deflection| within ±40 ms, and the
    nearest turning points of the opposite direction on either side within 150 ms.
    """
    fs = DETECTOR_FS
    n = len(sig)
    i0 = int(round(t * fs))
    w = int(round(PEAK_SEARCH_S * fs))
    lo, hi = max(0, i0 - w), min(n, i0 + w + 1)
    if hi <= lo:
        return t, 0.0, float("nan")
    p = lo + int(np.argmax(np.abs(sig[lo:hi])))
    s = np.sign(sig[p]) * sig  # the peak points up
    m = int(round(BASE_SEARCH_S * fs))
    left = p
    while left > max(0, p - m) and s[left - 1] <= s[left]:
        left -= 1
    right = p
    while right < min(n - 1, p + m) and s[right + 1] <= s[right]:
        right += 1
    dur = (right - left) / fs * 1000.0 if right > left else float("nan")
    return p / fs, float(sig[p]), dur


def detect_channel(x: np.ndarray, fs: float, detector: str, k1: float, offset_s: float,
                   core: tuple[float, float], level_ratio: float = 1.0,
                   line_hz: float | None = None) -> tuple[list[tuple[float, float, float]], str | None]:
    """
    Run one detector over one channel window. x starts at offset_s (seconds from the
    recording start); only detections inside core = (start_s, end_s) are returned, as
    (time_s from the recording start, amplitude µV, duration ms). Returns (detections, error).
    """
    try:
        x200 = to_detector_rate(x, fs)
        if not np.all(np.isfinite(x200)):
            return [], "non-finite samples"
        if np.ptp(x200) == 0:
            return [], None  # flat channel: nothing to detect
        x200 = notch(x200, line_hz)
        found = janca(x200, k1=k1) if detector == "janca" else barkmeier(x200, level_ratio)
        sig = broad(x200)
    except Exception as exc:  # noqa: BLE001 - one channel's failure must not sink the recording
        return [], f"{type(exc).__name__}: {exc}"
    out = []
    last = -1.0
    for t in sorted(found):
        tp, amp, dur = describe(sig, t)
        t_abs = offset_s + tp
        if core[0] <= t_abs < core[1] and t_abs - last > 1e-6:
            out.append((round(t_abs, 4), round(amp, 2), round(dur, 1) if np.isfinite(dur) else float("nan")))
            last = t_abs
    return out, None


def group_events(spikes: list[tuple[float, str, float]], window_s: float) -> list[dict]:
    """
    Cross-channel grouping: spikes (time_s, channel, amplitude) sorted by time; an event starts
    at a spike and takes every spike within window_s of that first spike. The leading channel
    is the one that peaks first (ties within one 200 Hz sample: the larger |amplitude|).
    Returns one dict per event with time_s, leader, n_channels, channels (in firing order) and
    the index list of its spikes.
    """
    tie = 0.5 / DETECTOR_FS
    order = sorted(range(len(spikes)), key=lambda i: spikes[i][0])
    events: list[dict] = []
    cur: list[int] = []
    for i in order:
        if cur and spikes[i][0] - spikes[cur[0]][0] > window_s:
            events.append(_event(spikes, cur, tie))
            cur = []
        cur.append(i)
    if cur:
        events.append(_event(spikes, cur, tie))
    return events


def _event(spikes, idx: list[int], tie: float) -> dict:
    t0 = spikes[idx[0]][0]
    first = [i for i in idx if spikes[i][0] - t0 <= tie]
    lead = max(first, key=lambda i: abs(spikes[i][2]))
    chans: list[str] = [spikes[lead][1]]
    for i in idx:
        if spikes[i][1] not in chans:
            chans.append(spikes[i][1])
    return {"time_s": spikes[lead][0], "leader": spikes[lead][1], "n_channels": len(chans),
            "channels": chans, "members": idx}
