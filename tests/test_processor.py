"""
Tests on the synthetic fixture (tests/make_fixture.py); no Docker, no real data.

The planted spikes have known peak times and sides, so the end-to-end tests assert recovery:
event-level sensitivity and precision at ±50 ms, the hemisphere of the leading channel, seizure
and bad-segment exclusion, and rates per sleep stage from the linked hypnogram.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from processor import detect as D
from processor import eeg as E
from processor import main as M
from processor import montage as MT
from processor import stages as S
from tests import make_fixture

REC1 = "sub-01_ses-01_task-szMonitoring_run-01"   # referential (average), hypnogram linked
REC2 = "sub-02_ses-01_task-routine_run-01"        # bipolar only
REC3 = "sub-03_ses-01_task-szMonitoring_run-00"   # plain BIDS, Siena style
TOL = 0.05


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    return make_fixture.make(tmp_path_factory.mktemp("fixture"))


def run(monkeypatch, input_dir: Path, output_dir: Path, **env) -> int:
    for k in M.PARAM_NAMES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("THREADS", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return M.main()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def default_run(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture_dir / "linked", out) == 0
    finally:
        mp.undo()
    return out / "eeg-spikes"


def planted(fixture_dir: Path, rid: str, analysed_only: bool = True) -> list[dict]:
    tr = json.loads((fixture_dir / "truth.json").read_text())[rid]
    cut = []
    if "seizure" in tr:
        cut.append((tr["seizure"][0] - 60, tr["seizure"][1] + 60))
    if "bad_segment" in tr:
        cut.append(tuple(tr["bad_segment"]))
    return [s for s in tr["spikes"] if not analysed_only or (
        1.5 < s["time_s"] < tr["duration"] - 1.5 and not any(a <= s["time_s"] < b for a, b in cut))]


def match(fixture_dir, out: Path, rid: str):
    """(sensitivity, precision, matched (event, planted) pairs) of spike events vs planted spikes at ±50 ms."""
    truth = planted(fixture_dir, rid)
    tt = np.array([s["time_s"] for s in truth])
    ev = [e for e in read_csv(out / "spike_events.csv") if e["recording_id"] == rid]
    et = np.array([float(e["time_s"]) for e in ev])
    sens = np.mean([np.min(np.abs(et - t)) <= TOL for t in tt])
    prec = np.mean([np.min(np.abs(tt - t)) <= TOL for t in et])
    pairs = [(e, truth[int(np.argmin(np.abs(tt - float(e["time_s"]))))]) for e in ev
             if np.min(np.abs(tt - float(e["time_s"]))) <= TOL]
    return sens, prec, pairs


# --- end to end ---------------------------------------------------------------------

def test_outputs_columns_and_provenance(default_run):
    for name, header in (("spikes.csv", M.SPIKES_HEADER), ("spike_rates.csv", M.RATES_HEADER),
                         ("spike_events.csv", M.EVENTS_HEADER), ("lateralisation.csv", M.LAT_HEADER),
                         ("rates_by_stage.csv", M.STAGE_HEADER), ("features.csv", M.FEATURES_HEADER)):
        with open(default_run / name) as fh:
            assert next(csv.reader(fh)) == header
    prov = json.loads((default_run / "provenance.json").read_text())
    assert list(prov)[:13] == ["app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                               "wall_seconds", "peak_rss_mb", "threads", "recordings", "citations"]
    assert prov["app"] == "eeg-spikes" and prov["version"] == "0.1.0" and prov["status"] == "ok"
    assert {k: v["status"] for k, v in prov["recordings"].items()} == {REC1: "ok", REC2: "ok"}
    assert {k: v["montage"] for k, v in prov["recordings"].items()} == {REC1: "bipolar", REC2: "bipolar"}
    assert set(prov["threads"]) == {"requested", "used", "why"}
    assert "10.1007/s10548-014-0379-1" in prov["citations"]
    assert "epycom" in prov["tools"] and "mne" in prov["tools"]


@pytest.mark.parametrize("rid", [REC1, REC2])
def test_sensitivity_and_precision_at_50ms(default_run, fixture_dir, rid):
    sens, prec, pairs = match(fixture_dir, default_run, rid)
    assert sens >= 0.9, sens
    assert prec >= 0.9, prec
    # the leading channel's hemisphere is the side the spike was planted on
    assert np.mean([e["hemisphere"] == p["side"] for e, p in pairs]) >= 0.9


def test_lateralisation(default_run, fixture_dir):
    lat = {r["recording_id"]: r for r in read_csv(default_run / "lateralisation.csv")}
    truth = planted(fixture_dir, REC1)
    nl = sum(s["side"] == "L" for s in truth)
    nr = sum(s["side"] == "R" for s in truth)
    assert abs(int(lat[REC1]["n_left"]) - nl) <= 0.1 * nl + 1
    assert abs(int(lat[REC1]["n_right"]) - nr) <= 0.1 * nr + 1
    assert float(lat[REC1]["laterality_index"]) == pytest.approx((nl - nr) / (nl + nr), abs=0.1)
    assert lat[REC1]["dominant_region"] == "left temporal"
    assert float(lat[REC2]["laterality_index"]) > 0.9


def test_seizure_and_bad_segment_excluded(default_run):
    t = [float(s["time_s"]) for s in read_csv(default_run / "spikes.csv") if s["recording_id"] == REC1]
    assert not [x for x in t if 180 <= x <= 330]   # seizure 240-270 s, 60 s margins
    assert not [x for x in t if 100 <= x < 106]    # bad segment
    rates = {(r["recording_id"], r["channel"]): r for r in read_csv(default_run / "spike_rates.csv")}
    # 480 s - 150 s seizure+margins - 6 s bad - 2 x 1 s edges per analysed stretch
    assert float(rates[(REC1, "F7-T7")]["minutes"]) == pytest.approx((480 - 150 - 6 - 4) / 60, abs=0.02)


def test_spike_columns(default_run):
    sp = [s for s in read_csv(default_run / "spikes.csv") if s["recording_id"] == REC1 and s["channel"] == "T7-P7"]
    assert len(sp) > 40
    dur = np.array([float(s["duration_ms"]) for s in sp])
    assert 20 <= np.median(dur) <= 120  # the planted spike is ~70 ms base to base
    assert 40 <= np.median([abs(float(s["amplitude_uv"])) for s in sp]) <= 300
    assert all(s["event_id"] for s in sp)
    assert {s["sleep_stage"] for s in sp} <= {"W", "N2", "N3"}


def test_rates_by_stage(default_run):
    rows = [r for r in read_csv(default_run / "rates_by_stage.csv") if r["channel"] == "T7-P7"]
    rate = {r["stage"]: float(r["rate_per_min"]) for r in rows}
    assert set(rate) == {"W", "N2", "N3"}
    assert rate["N2"] > 1.5 * rate["W"] and rate["N3"] > 1.5 * rate["W"]  # planted at 2x in sleep
    assert all(r["recording_id"] == REC1 for r in read_csv(default_run / "rates_by_stage.csv"))
    summ = json.loads((default_run / "summary.json").read_text())
    assert summ["recordings"][REC1]["nrem_wake_ratio"] > 1.5
    assert "events_per_min_by_stage" not in summ["recordings"][REC2]


def test_summary_and_figures(default_run):
    summ = json.loads((default_run / "summary.json").read_text())
    assert summ["app"] == "eeg-spikes"
    assert set(summ["recordings"]) == {REC1, REC2}
    paths = [f["path"] for f in summ["figures"]]
    for p in paths:
        assert (default_run / p).is_file()
    assert f"figures/{REC1}_spike_rates.png" in paths
    assert f"figures/{REC1}_spikes_by_stage.png" in paths
    assert f"figures/{REC1}_example_spikes.png" in paths
    assert "figures/laterality_index.png" in paths
    assert all(len(f["caption"]) > 80 for f in summ["figures"])
    assert all((default_run / t["path"]).is_file() for t in summ["tables"])
    h = summ["recordings"][REC1]
    assert isinstance(h["events_per_min"], float) and isinstance(h["laterality_index"], float)


def test_features(default_run):
    f = read_csv(default_run / "features.csv")
    assert {r["feature"] for r in f} == {"spikerate_janca"}
    assert len([r for r in f if r["recording_id"] == REC1]) == 18


def test_referential_montage(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, MONTAGE="referential") == 0
    out = tmp_path / "eeg-spikes"
    chans = {r["channel"] for r in read_csv(out / "spike_rates.csv") if r["recording_id"] == REC1}
    assert chans == set(make_fixture.ELECTRODES)
    sens, prec, _ = match(fixture_dir, out, REC1)
    assert sens >= 0.9 and prec >= 0.8
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["recordings"][REC1]["montage"] == "referential"
    assert prov["recordings"][REC2]["montage"] == "bipolar"  # bipolar-only stays bipolar, with a note
    assert any("bipolar-only" in n for n in prov["recordings"][REC2]["notes"])


def test_plain_bids_siena_style(fixture_dir, tmp_path, monkeypatch):
    """`Fp1-Avg`, old names T3-T6, no channels.tsv, the seizure as a HED-SCORE eventType: same result."""
    assert run(monkeypatch, fixture_dir / "plain", tmp_path) == 0
    out = tmp_path / "eeg-spikes"
    rates = {r["channel"] for r in read_csv(out / "spike_rates.csv")}
    assert "F7-T7" in rates and "T8-P8" in rates and len(rates) == 18
    t = [float(s["time_s"]) for s in read_csv(out / "spikes.csv")]
    assert not [x for x in t if 180 <= x <= 330]  # sz_foc_ia read as a seizure
    sens, _, _ = match(fixture_dir, out, REC3)
    assert sens >= 0.9
    # without eeg-prep the artefact at 100-106 s is not marked bad: its sharp transients are detected
    # (this is what the bad_segment mask is for); outside it precision is as on the derivative
    tt = np.array([s["time_s"] for s in planted(fixture_dir, REC3)])
    et = [float(e["time_s"]) for e in read_csv(out / "spike_events.csv")]
    false = [t for t in et if np.min(np.abs(tt - t)) > TOL]
    assert len(false) > 5 and np.mean([99 <= t <= 107 for t in false]) >= 0.9
    assert not (out / "rates_by_stage.csv").exists()


def test_barkmeier(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, DETECTOR="barkmeier") == 0
    out = tmp_path / "eeg-spikes"
    sens, prec, _ = match(fixture_dir, out, REC1)
    assert sens >= 0.8 and prec >= 0.7
    assert {r["detector"] for r in read_csv(out / "spike_rates.csv")} == {"barkmeier"}


def test_min_event_channels_and_label(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, MIN_EVENT_CHANNELS="30", LABEL="strict") == 0
    out = tmp_path / "eeg-spikes" / "strict"
    lat = {r["recording_id"]: r for r in read_csv(out / "lateralisation.csv")}
    assert lat[REC1]["n_events"] == "0" and lat[REC1]["laterality_index"] == ""
    assert len(read_csv(out / "spike_events.csv")) > 50  # every event is still listed


def test_sleep_from_annotations(fixture_dir, tmp_path, monkeypatch):
    src = fixture_dir / "linked" / "eeg"
    shutil.copytree(src, tmp_path / "in")
    ev = next((tmp_path / "in" / "sub-01").rglob("*_events.tsv"))
    ev.write_text(ev.read_text() + "0\t330\twake\tannotation\t\n330\t150\tsleep\tannotation\t\n")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out") == 0
    rows = read_csv(tmp_path / "out" / "eeg-spikes" / "rates_by_stage.csv")
    assert {r["stage"] for r in rows} == {"W", "sleep"}


def test_no_input_fails(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "notes.txt").write_text("x")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out") == 1
    prov = json.loads((tmp_path / "out" / "eeg-spikes" / "provenance.json").read_text())
    assert prov["status"] == "failed: no input"


def test_bad_parameter(fixture_dir, tmp_path, monkeypatch):
    for bad in ({"DETECTOR": "magic"}, {"WINDOW_S": "10"}, {"MONTAGE": "laplacian"}, {"MIN_EVENT_CHANNELS": "0"}):
        assert run(monkeypatch, fixture_dir / "linked", tmp_path, **bad) == 2


def test_failed_recording_is_skipped(fixture_dir, tmp_path, monkeypatch):
    shutil.copytree(fixture_dir / "linked" / "eeg", tmp_path / "in")
    next((tmp_path / "in" / "sub-02").rglob("*.edf")).write_bytes(b"not an edf")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out") == 0
    prov = json.loads((tmp_path / "out" / "eeg-spikes" / "provenance.json").read_text())
    assert prov["recordings"][REC1]["status"] == "ok" and prov["recordings"][REC2]["status"] == "failed"


# --- units ------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("Fp1-Avg", ("Fp1", ("Fp1",))), ("EEG T3-REF", ("T7", ("T7",))), ("T5-Avg", ("P7", ("P7",))),
    ("FP1-F7", ("Fp1-F7", ("Fp1", "F7"))), ("T8-P8-0", ("T8-P8", ("T8", "P8"))), ("FZ-CZ", ("Fz-Cz", ("Fz", "Cz"))),
    ("C3-A2", ("C3", ("C3",))), ("ECG", None), ("EDF Annotations", None), ("Fp1", ("Fp1", ("Fp1",)))])
def test_channel_names(name, expected):
    assert MT.parse(name) == expected


def test_hemisphere_and_region():
    assert MT.hemisphere(("F7", "T7")) == "L" and MT.hemisphere(("Fp2",)) == "R" and MT.hemisphere(("Fz", "Cz")) == "Z"
    assert MT.hemisphere(("FT9", "FT10")) == "Z"
    assert MT.region(("F7", "T7")) == "temporal" and MT.region(("P7", "O1")) == "temporal"
    assert MT.region(("F3", "C3")) == "frontal" and MT.region(("C3", "P3")) == "central"
    assert MT.region(("Fz", "Cz")) == "midline" and MT.region(("O1",)) == "occipital"
    assert len(MT.BIPOLAR_PAIRS) == 18


def test_group_events_leader_and_window():
    spikes = [(1.000, "A", 100), (1.020, "B", 50), (1.040, "C", 80), (1.060, "D", 90), (2.0, "B", 10),
              (2.0, "A", 20)]
    ev = D.group_events(spikes, 0.05)
    assert [(e["leader"], e["n_channels"]) for e in ev] == [("A", 3), ("D", 1), ("A", 2)]  # tie: larger amplitude


def test_timing_after_resampling_and_duration():
    rng = np.random.default_rng(1)
    fs = 512.0
    x = make_fixture.pink(int(fs * 120), rng, 10.0)
    on = np.arange(5.0, 115.0, 3.7)
    for t in on:
        make_fixture.add(x, fs, t, make_fixture.spike_template(fs, 150.0))
    dets, err = D.detect_channel(x.astype(np.float32), fs, "janca", 3.65, 0.0, (0.0, 120.0))
    assert err is None
    t = np.array([d[0] for d in dets])
    peaks = on + make_fixture.PEAK_OFFSET
    assert np.mean([np.min(np.abs(t - p)) < 0.01 for p in peaks[peaks > 2]]) > 0.9
    assert all(d[1] < 0 for d in dets if np.min(np.abs(peaks - d[0])) < 0.01)  # surface-negative


def test_chbmit_summary(tmp_path):
    (tmp_path / "chb01-summary.txt").write_text(
        "File Name: chb01_03.edf\nFile Start Time: 13:43:04\nNumber of Seizures in File: 1\n"
        "Seizure Start Time: 2996 seconds\nSeizure End Time: 3036 seconds\n\n"
        "File Name: chb01_04.edf\nNumber of Seizures in File: 1\nSeizure 1 Start Time: 1467 seconds\n"
        "Seizure 1 End Time: 1494 seconds\n")
    assert E.chbmit_summary(tmp_path / "chb01-summary.txt") == {"chb01_03.edf": [(2996.0, 3036.0)],
                                                               "chb01_04.edf": [(1467.0, 1494.0)]}


def test_hypnogram_mapping_and_minutes():
    assert [S.canonical_stage(s) for s in ("W", "0", "N3", "S4", "REM", "4", "Art", "")] == \
        ["W", "W", "N3", "N3", "R", "R", None, None]
    h = S.Hypnogram([(0, 30, "W"), (30, 60, "N2"), (60, 90, "N2")], "test")
    assert h.stage_at(45) == "N2" and h.stage_at(95) == ""
    assert h.minutes([(15, 75)]) == {"W": 0.25, "N2": 0.75}


def test_plan_masks_and_unknown_end(tmp_path):
    ev = tmp_path / "r_events.tsv"
    ev.write_text("onset\tduration\ttrial_type\n100\t\tseizure\n10\t5\tbad_segment\n")
    rec = E.Recording("r", tmp_path / "r.edf", "r.edf", "scan", events_tsv=ev)
    plan = E.analysis_plan(rec, 300.0, True, 60.0)
    assert plan.segments == [(0.0, 40.0)] and plan.masks == [(10.0, 15.0)]
    assert "end unknown" in plan.excluded[0]["reason"]


def test_detector_failure_on_every_channel_fails_the_step(fixture_dir, tmp_path, monkeypatch):
    """A detector that raises on every channel must fail the recording (and the step), not skip it."""
    monkeypatch.setattr(D, "detect_channel", lambda *a, **k: ([], "ValueError: no field of name pos"))
    assert run(monkeypatch, fixture_dir / "linked", tmp_path) == 1
    prov = json.loads((tmp_path / "eeg-spikes" / "provenance.json").read_text())
    assert prov["status"] == "failed: every recording failed"
    assert all(r["status"] == "failed" and "every channel" in r["error"] for r in prov["recordings"].values())


def test_detector_failure_on_one_channel_is_reported(fixture_dir, tmp_path, monkeypatch):
    real = D.detect_channel

    calls = {"n": 0}

    def flaky(x, fs, *a, **k):  # fails on the first channel (Fp1-F7) of every 18-channel window
        calls["n"] += 1
        if (calls["n"] - 1) % 18 == 0:
            return [], "ValueError: boom"
        return real(x, fs, *a, **k)

    monkeypatch.setattr(D, "detect_channel", flaky)
    assert run(monkeypatch, fixture_dir / "linked", tmp_path) == 0
    out = tmp_path / "eeg-spikes"
    prov = json.loads((out / "provenance.json").read_text())
    r1 = prov["recordings"][REC1]
    assert r1["status"] == "ok"
    assert r1["channel_errors"]["Fp1-F7"]["windows_failed"] == r1["windows"]
    assert any("Fp1-F7" in n and "failed in all" in n for n in r1["notes"])
    assert (REC1, "Fp1-F7") not in {(r["recording_id"], r["channel"]) for r in read_csv(out / "spike_rates.csv")}
    assert any("Fp1-F7" in w for w in json.loads((out / "summary.json").read_text())["warnings"])
