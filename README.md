# eeg-spikes

This app packages the interictal spike detectors of **epycom** (ElectroPhYsiology COmputational
Module), developed by the Biomedical Engineering Department of the International Clinical Research
Center at St. Anne's University Hospital in Brno (FNUSA-ICRC; https://github.com/ICRC-BME/epycom,
Apache-2.0). The default detector is the envelope-distribution method of **Radek Janca and
colleagues** (Janca et al. 2015, *Brain Topography*, doi:10.1007/s10548-014-0379-1), reprogrammed in
Python for epycom by Daniel Uher and Jan Cimbalnik (per epycom's source); the alternative is the slope-and-amplitude
method of **Daniel Barkmeier, Jeffrey Loeb and colleagues** (Barkmeier et al. 2012, *Clinical
Neurophysiology*, doi:10.1016/j.clinph.2011.09.023). Recordings are read with **MNE-Python**
(Alexandre Gramfort and the MNE community; https://mne.tools; Gramfort et al. 2013). Pennsieve
packages these methods for scalp EEG and validated them on scalp data (below); it did not develop
them. The demo and validation data are the **Siena Scalp EEG Database** (Paolo Detti and colleagues,
University of Siena; BIDS conversion by Jonathan Dan and Paolo Detti, EPFL), **CHB-MIT** (Ali Shoeb,
MIT, with Children's Hospital Boston) and **vEpiSet** (Nan Lin, Qiang Lu and colleagues, Peking Union
Medical College Hospital).

> **Both detectors were developed and validated on intracranial EEG**, not on scalp EEG (Janca et
> al. 2015: "epileptic and non-epileptic intracranial recordings"; Barkmeier et al. 2012:
> "intracranial EEG"). Our own scalp validation, below, shows they find most expert-marked scalp
> spikes but also many non-epileptiform transients. Every rate, event and laterality index this app
> writes is a **research output to be reviewed against the EEG**, not a clinical finding.

A step of both scalp EEG workflows (`workflow-scalp-eeg`):

- **Scalp EEG Interictal Biomarkers:** `eeg-prep → {eeg-spectral, eeg-connectivity, eeg-sleep → eeg-spikes} → eeg-report`
- **Scalp EEG Seizure Detection & Monitoring:** `eeg-prep → {eeg-seizures, eeg-sleep → eeg-spikes} → eeg-report`

**For the epileptologist.** The app marks candidate interictal epileptiform discharges (spikes and
sharp waves) on the longitudinal bipolar montage ("double banana") it derives from the
referential recording, as you would read it. Spikes on different channels within 50 ms are merged
into one **spike event** with its **leading channel** (the one that peaks first) and that channel's
hemisphere. Per recording you get the spike rate of every channel, a **laterality index** of the
events (left vs right leading channel) and the dominant region. With eeg-sleep's hypnogram
linked, every spike gets the sleep stage of its epoch and rates are given per stage: interictal
spiking increases with sleep depth in temporal lobe epilepsy (Malow et al. 1998), and spike rates around sleep help localise the seizure-onset zone (Conrad et al. 2023,
an **intracranial** result). Seizures (with 60 s margins) and eeg-prep's bad segments are left
out. Look at the example-spike figure first: if the examples are not epileptiform, neither are the
rates.

**For the researcher.** epycom 0.3's `detect_spikes_janca` (10–60 Hz envelope, log-normal
background fitted in 5 s windows, threshold k1 = 3.65) or `detect_spikes_barkmeier` runs per
channel on the signal resampled to 200 Hz (polyphase FIR), in 5-minute windows with 10 s context,
channels in parallel. Each detection is refined to the largest deflection of the 1–70 Hz signal
within ±40 ms (amplitude, sign) and given a base-to-base duration. Outputs are long-format CSV keyed
by `recording_id` and `channel` (workflow-scalp-eeg `docs/conventions.md`), shaped like
processor-ieeg-spikes' tables so scalp and intracranial results sit side by side, plus
`features.csv` (`spikerate_<detector>`) for normative comparison.

## Ports

| Port | Direction | Placement | What |
|---|---|---|---|
| `eeg` | input | `INPUT_DIR/eeg/` | eeg-prep's BIDS-EEG derivative (`recordings.csv`, `*_desc-preproc_eeg.edf`, `_channels.tsv`, `_events.tsv`). Also any BIDS-EEG dataset (BIDS Siena's `Fp1-Avg` names, old T3–T6 names and HED-SCORE `eventType` seizure rows are understood) and loose EDF/BDF/BrainVision/FIF/EEGLAB files with 10-20 names, referential or bipolar (`FP1-F7`; a CHB-MIT `chbNN-summary.txt` gives their seizures). Found recursively, symlinks followed. |
| `sleep` | input, optional | `INPUT_DIR/sleep/` | eeg-sleep's `hypnogram.csv` (30 s epochs, stages W/N1/N2/N3/R), matched by `recording_id`. Without it, annotated `sleep`/`wake` events of the recording are used when present. |
| `spikes` | output | `OUTPUT_DIR/eeg-spikes/` (or `eeg-spikes/<LABEL>/`) | tables, `figures/`, `summary.json`, `provenance.json` (below) |

Files are found by name and content anywhere under `INPUT_DIR`, so the app also works when
everything is merged at its root.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `DETECTOR` | `janca` | `janca` (Janca et al. 2015) or `barkmeier` (Barkmeier et al. 2012). Both intracranial-validated; see Validation. |
| `JANCA_K1` | `3.65` | Janca's threshold factor (the published default). Higher: fewer, more obvious spikes. On vEpiSet, 5 cuts false detections in patients without IEDs from 18.7 to 8.0/min but finds 51 % instead of 75 % of the expert-marked spikes. |
| `MONTAGE` | `bipolar` | `bipolar`: the 18-pair longitudinal bipolar montage (Fp1-F7, F7-T7, T7-P7, P7-O1; Fp1-F3, F3-C3, C3-P3, P3-O1; Fz-Cz, Cz-Pz; and the right side), from pairs whose two electrodes are good. `referential`: the channels as stored (eeg-prep's average reference). Bipolar-only recordings are always analysed on their own pairs (with a note). |
| `EXCLUDE_SEIZURE` | `true` | Leave annotated seizures out, `SEIZURE_MARGIN_S` on both sides; a seizure without an end is cut to the end of the recording. `bad_segment` and `artifact` events are always masked: the detector runs through them (its background stays continuous) but what it finds there is dropped and their time is not counted. |
| `SEIZURE_MARGIN_S` | `60` | Seconds left out before each seizure and after it. |
| `WINDOW_S` | `300` | Windows the recording is read and analysed in (plus 10 s context each side). Bounds memory. |
| `GROUP_WINDOW_MS` | `50` | Spikes within this many ms of an event's first spike belong to that event. |
| `MIN_EVENT_CHANNELS` | `1` | Events spanning fewer channels stay in `spike_events.csv` but are not counted in `lateralisation.csv`, the event rates in `summary.json`, the rates per stage and the example figure. 2–3 asks for a field; on vEpiSet it trades sensitivity for specificity without improving their balance (Validation). |
| `THREADS` | `0` | Channels analysed in parallel (processes); `0` = one per vCPU of the container (cgroup CPU quota, else CPU affinity, else CPU count), capped so ~250 MB per worker + ~600 MB fit the memory limit. Logged and kept in `provenance.json`. |
| `LABEL` | (empty) | Write to `eeg-spikes/<LABEL>/` (a workflow running this step twice). |

## Outputs

Everything is under `OUTPUT_DIR/eeg-spikes/`. Times are seconds from the recording start.

**`spikes.csv`**: one row per detected spike: `recording_id`, `channel` (`F7-T7`, or `T7` when
referential), `time_s` (peak), `amplitude_uv` (signed peak of the 1–70 Hz signal; on bipolar pairs
the sign depends on which electrode of the pair is more negative), `duration_ms` (base to base:
the turning points on either side of the peak, within 150 ms; 5 ms resolution), `detector`,
`sleep_stage` (empty when no stages are known), `event_id`.

**`spike_rates.csv`**: one row per analysed channel, zeros included: `recording_id, channel,
n_spikes, minutes, rate_per_min, detector`. `minutes` is the time actually analysed (seizures with
margins, bad segments, and the first and last second of each analysed stretch excluded).

**`spike_events.csv`**: one row per multi-channel spike event: `recording_id, event_id, time_s`
(leading channel's peak), `n_channels, channels` (`|`-separated, firing order, leader first),
`leading_channel` (earliest peak; ties within one 5 ms sample go to the larger amplitude),
`hemisphere` (`L`/`R`/`Z`), `region`, `detector`, `sleep_stage`. Isolated spikes are events with
`n_channels = 1`.

**`lateralisation.csv`**: per recording `recording_id, n_left, n_right, n_midline` (counted events
by the hemisphere of their leading channel), `laterality_index` = (L − R)/(L + R) (empty with no
lateral events), `dominant_region` (most frequent side and region of the leaders, e.g. `left
temporal`), and `n_events, minutes, rate_left_per_min, rate_right_per_min` (mean per-channel rate
over left/right channels), `rate_laterality_index`, `dominant_channel`, `montage`, `detector`.

**`rates_by_stage.csv`** (when stages are known): `recording_id, channel, stage, n_spikes, minutes,
rate_per_min`, stages `W, N1, N2, N3, R` (or `W`/`sleep` from annotations).

**`features.csv`**: `recording_id, channel, feature, value` with `spikerate_janca` or
`spikerate_barkmeier`.

**`figures/`** (150 dpi PNG, captions in `summary.json`): `<recording_id>_spike_rates.png`
(topomap of the rate per electrode, for bipolar pairs the mean of the pairs an electrode belongs
to, next to every channel's rate), `<recording_id>_spikes_by_stage.png` (events/min per stage),
`<recording_id>_example_spikes.png` (three events of the most frequent leader, at the median, 75th
and 90th percentile of its amplitude, all channels stacked), `laterality_index.png` (all
recordings).

**`summary.json`**: headline numbers per recording (minutes analysed, spikes, events, events/min,
top channel, laterality index, dominant region and channel, events/min per stage, NREM/wake ratio),
the figures in report order with captions, the tables, warnings, and a notice that the detectors
are intracranial-validated. **`provenance.json`**: the keys of conventions.md (tools and versions,
parameters, inputs, times, peak RSS, threads, per recording status/seconds/montage plus analysed
and excluded intervals, channels not analysed and why, notes), `citations` (DOIs for the report).

## Validation

Run by us in November 2026; scripts in `validation/`. Numbers are for the default parameters
unless stated.

**(a) Synthetic fixture** (`tests/make_fixture.py`, 19-channel 10-20 EEG at 256 Hz in average
reference with posterior alpha, a 3 Hz spike-and-wave seizure, an artefact marked `bad_segment`,
sleep-like slow waves, and planted surface-negative spikes with a temporal field, 80 % left (T7),
20 % right (T8), twice as frequent in sleep; plus a bipolar-only copy). Event-level recovery at
±50 ms: **sensitivity 1.00, precision 1.00** on both the referential and the bipolar-only
recording (91 and 39 planted spikes), median timing error 1.8 ms, leading-channel hemisphere
correct for every event, laterality index 0.54 for a planted 0.54, no detections inside the seizure
± 60 s or the bad segment, rates per stage N2/N3 ≈ 2× wake as planted. The same signal given as
plain BIDS without the `bad_segment` annotation shows why the mask matters: the artefact's sharp
transients come out as 34 false events. `pytest` asserts all of this (38 tests). A clean synthetic
fixture is a test of the plumbing, not of scalp performance.

**(b) vEpiSet** (Lin et al. 2025; figshare doi:10.6084/m9.figshare.28069568.v2, CC BY 4.0): 30 of
the 84 continuous 20-minute recordings (20 randomly chosen of the patients with IEDs, 10 of those
without), 19 scalp electrodes at 500 Hz in average reference, 1,477 single IED marks by consensus of
at least three experts. Epochs cut as the dataset's own code cuts them (4 s; 900 IED epochs, 8,135
non-IED epochs). Epoch score = channel detections in the epoch; an epoch is "detected" when it holds
an event spanning ≥ m channels.

| Setting | AUROC | Expert marks with an event within ±0.25 s | Events/min in patients without IEDs | m=1: sens / spec / precision | m=2 | m=3 |
|---|---|---|---|---|---|---|
| **janca, k1 3.65, bipolar (default)** | **0.79** | 0.75 | 18.7 | 0.87 / 0.57 / 0.18 | 0.71 / 0.73 / 0.23 | 0.58 / 0.84 / 0.29 |
| janca, k1 5 | 0.73 | 0.51 | 8.0 | 0.64 / 0.80 / 0.26 | 0.50 / 0.88 / 0.31 | 0.38 / 0.93 / 0.37 |
| janca, k1 6 | 0.70 | 0.39 | 5.4 | 0.52 / 0.86 / 0.29 | 0.40 / 0.91 / 0.33 | 0.27 / 0.95 / 0.37 |
| janca, k1 8 | 0.63 | 0.22 | 3.0 | 0.35 / 0.91 / 0.30 | 0.23 / 0.95 / 0.33 | 0.13 / 0.97 / 0.33 |
| janca, k1 3.65, referential | 0.82 | 0.78 | 18.2 | 0.91 / 0.58 / 0.19 | 0.76 / 0.78 / 0.27 | 0.56 / 0.88 / 0.34 |
| barkmeier, bipolar | 0.73 | 0.52 | 22.2 | 0.70 / 0.70 / 0.20 | 0.49 / 0.87 / 0.30 | 0.35 / 0.93 / 0.36 |

For comparison, the dataset authors' deep-learning models reach 97 % specificity and 74 %
precision at 80 % sensitivity on held-out epochs (VGG; Lin et al. 2025, Table 2). The classical
detectors find most expert-marked spikes but at that sensitivity flag about 40 % of non-IED epochs:
**most detections on scalp EEG are not epileptiform discharges** (muscle, eye movements, vertex
waves, sharp background). Simple amplitude, duration and channel-count criteria did not improve
the balance (we tried them on these data and left them out rather than tune them on 30
recordings). The referential montage did slightly better than bipolar here; we keep bipolar as the
default because clinicians read spikes on it and the two are within noise of each other. vEpiSet's
wake/sleep annotations (used through the annotation fallback): the experts' IED rate was higher
in sleep than in wake in 8 of 9 patients with both states, the detector's in 6 of 9 (in wake,
artefact detections dominate). vEpiSet is not staged by the workflow's fetch script (16.9 GB); the 30 recordings were read out of the figshare zip
with HTTP range requests (`validation/vepiset_to_bids.py`, `validation/vepiset_eval.py`).

**(c) Siena**, per-patient laterality of spike events against the seizure lateralisation in
PhysioNet's `subject_info.csv` (13 patients with a left or right temporal focus, 9 left and 4
right; PN10, bilateral frontal, left out), all runs of a patient pooled
(`validation/siena_lateralisation.py`):

| Input (41 recordings, 141 h) | Events/min per recording, median (IQR) | Laterality agrees with the seizure side | Patients with a left-sided index |
|---|---|---|---|
| **eeg-prep derivative** (ICA + ICLabel, bad segments; `data/work/siena/eeg-prep/eeg`) | 16 (12–26) | **9 of 13** (sign test p = 0.27); 8–9 of 13 with `MIN_EVENT_CHANNELS` 2–3 | 11 of 13 |
| plain BIDS Siena (no artefact handling) | 40 (31–65) | 4 of 13 (p = 0.27); 6–8 of 13 with `MIN_EVENT_CHANNELS` 2–6 or the referential montage | 7 of 13 |

eeg-prep's artefact handling halves the detections and moves the agreement from below to above
chance, but **9 of 13 is what always answering "left" scores on this cohort** (9 of the 13 foci
are left): the index came out left-sided in 11 of 13 patients, including 3 of the 4 right-sided
ones. So on Siena the app's lateralisation does not reliably recover the seizure side, and a left
bias (artefact or montage asymmetry) cannot be excluded. Rates of 16 events/min are implausibly
high for interictal spiking on scalp EEG, which again says most detections are not IEDs. The
seizure side is also only a proxy: interictal spikes need not be on the seizure side. Siena is
low-passed at 30 Hz by its source.

**CHB-MIT** (chb01, 6 one-hour bipolar-only recordings, 4 with a seizure, read directly): runs as
recorded on its 22 distinct pairs, seizures from `chb01-summary.txt` left out with their margins;
median 27 events/min. No spike annotations exist for it.

**ds003555** (eeg-prep and eeg-sleep outputs, 3 children, 3 h of sleep EEG each; the workflow's
own path, output in `data/work/ds003555/eeg-spikes/`): spike events per minute in W / N1 / N2 / N3 /
R: sub-01 (bilateral ischaemic lesions) 88 / 82 / 83 / 103 / 81; sub-02 (genetic) 20 / 19 / 19 / 27
/ 13; sub-03 (suspected FCD) 68 / 79 / 128 / 136 / 95, i.e. NREM/wake 1.07, 0.95 and 1.83. The
dataset has no spike annotations to check these against; at these rates sleep transients (vertex
waves, K-complexes) are certainly among the detections.

## Runtime

`computeTypes: [standard]`, **2 vCPU / 3 GiB**, 2 h timeout. One recording at a time, one 5-minute
window of every channel at a time, channels in parallel; memory is bounded by the window, not by
the recording length.

Measured in Docker (linux/amd64 **emulated** on an M-series Mac, `--cpus 2 --memory 3g`):

| Input | Wall time | Peak memory (cgroup `memory.peak`, page cache included) |
|---|---|---|
| Siena sub-01 run-00 (plain BIDS): 13.5 h, 19 channels, 256 Hz (803 min analysed, 18 pairs) | 105 s | 1.23 GB (main process 0.70 GB RSS) |
| ds003555 eeg-prep derivative + eeg-sleep hypnograms: 3 × 3 h, 23 channels, 1024 Hz | 87 s | 1.60 GB (main process 0.61 GB RSS) |
| synthetic fixture (2 recordings, 9 min), `--read-only --user 65534` Lambda-style | 8 s | 0.85 GB |

About 8 s per recording-hour on 2 emulated vCPU (native arm64: all 141 h of Siena in 5.3 min on 4
workers). A 24 h recording takes ~3–4 min; the 2 h timeout leaves room for a few days of
recordings in one run. Image: 212 MB compressed.

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install --no-deps --only-binary=:all: -r requirements-dev.txt
make test PYTHON=.venv/bin/python     # pytest on the synthetic fixture
make run PYTHON=.venv/bin/python      # Docker over the fixture -> data/output/
make run-demo DEMO=../workflow-scalp-eeg/data/work/siena/eeg-prep/eeg   # an eeg-prep output
```

## Limitations

- **Intracranial methods on scalp EEG.** Janca's and Barkmeier's detectors and their thresholds
  were developed on intracranial recordings, where spikes are large relative to the background and
  there is no scalp muscle or eye artefact. On scalp EEG they are sensitive but not specific
  (Validation b): read rates as "candidate discharges", review the example figure, and compare
  recordings with the same montage and preprocessing, not across datasets.
- **No artefact classifier.** Muscle, eye movements (frontopolar and F7/F8 channels), electrode
  pops and sleep transients (vertex waves, K-complexes, POSTS) are detected as spikes. eeg-prep's ICA
  and bad segments remove some of them; nothing in this app classifies morphology.
- **Lateralisation did not reliably agree with the seizure side on Siena** (9 of 13 on eeg-prep
  output, no better than always answering "left"; Validation c). Do not read a laterality index
  as a lateralising finding.
- **Rates per sleep stage** inherit the detector's false positives (more artefact in wake) and
  eeg-sleep's staging (EEG-only). The increase of spiking with sleep depth is long described
  (Malow et al. 1998); the use of spike patterns around sleep for localisation (Conrad et al. 2023)
  is an intracranial result.
- **Timing and leaders.** On a bipolar montage a discharge under one electrode appears on two pairs
  at the same time with opposite sign (phase reversal); the leading channel is then decided by
  amplitude. Leader and hemisphere are timing labels, not source localisation.
- Janca's implementation notches 50 Hz internally; 60 Hz mains is notched by this app when the
  sidecar says 60 Hz (eeg-prep removes it anyway). Siena is low-passed at 30 Hz at source, which
  narrows Janca's 10–60 Hz band to 10–30 Hz.
- Stretches shorter than 30 s between excluded intervals are not analysed (Janca needs ≥ 20 s).
- vEpiSet validation used 30 of 84 recordings from one centre; no full scalp benchmark with a
  permissive licence exists for classical detectors.

## Licences

Apache-2.0 for this app (`LICENSE`). The image bundles epycom (Apache-2.0), MNE-Python
(BSD-3-Clause) and their dependencies (`THIRD_PARTY.md`); no model weights. The demo and validation
data are not in the image or the repository.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in `CITATION.cff`.

Spike detectors (both validated on intracranial EEG):

> Janca R, Jezdik P, Cmejla R, Tomasek M, Worrell GA, Stead M, Wagenaar J, Jefferys JGR, Krsek P,
> Komarek V, Jiruska P, Marusic P. **Detection of Interictal Epileptiform Discharges Using Signal
> Envelope Distribution Modelling: Application to Epileptic and Non-Epileptic Intracranial
> Recordings.** Brain Topography 28(1):172–183 (2015). doi:10.1007/s10548-014-0379-1, PMID 24970691.

> Barkmeier DT, Shah AK, Flanagan D, Atkinson MD, Agarwal R, Fuerst DR, Jafari-Khouzani K, Loeb JA.
> **High inter-reviewer variability of spike detection on intracranial EEG addressed by an automated
> multi-channel algorithm.** Clinical Neurophysiology 123(6):1088–1095 (2012).
> doi:10.1016/j.clinph.2011.09.023, PMID 22033028.

Implementation: epycom 0.3, Biomedical Engineering, FNUSA-ICRC Brno,
https://github.com/ICRC-BME/epycom (Apache-2.0).

Reading the recordings:

> Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T,
> Parkkonen L, Hämäläinen M. **MEG and EEG data analysis with MNE-Python.** Frontiers in
> Neuroscience 7:267 (2013). doi:10.3389/fnins.2013.00267, PMID 24431986.

Sleep and spikes:

> Malow BA, Lin X, Kushwaha R, Aldrich MS. **Interictal Spiking Increases with Sleep Depth in
> Temporal Lobe Epilepsy.** Epilepsia 39(12):1309–1316 (1998). doi:10.1111/j.1528-1157.1998.tb01329.x,
> PMID 9860066.

> Conrad EC, Revell AY, Greenblatt AS, et al. **Spike patterns surrounding sleep and seizures
> localize the seizure-onset zone in focal epilepsy.** Epilepsia 64(3):754–768 (2023).
> doi:10.1111/epi.17482, PMID 36484572. (Intracranial EEG.)

Validation and demo data:

> Lin N, Zheng M, Li L, et al. **An EEG dataset for interictal epileptiform discharge with spatial
> distribution information.** Scientific Data 12:229 (2025). doi:10.1038/s41597-025-04572-1,
> PMID 39920162. Data: figshare, doi:10.6084/m9.figshare.28069568.v2 (CC BY 4.0).

> Detti P. **Siena Scalp EEG Database** (v1.0.0). PhysioNet (2020). doi:10.13026/5d4a-j060. And
> Detti P, Vatti G, Zabalo Manrique de Lara G. **EEG Synchronization Analysis for Seizure Prediction:
> A Study on Data of Noninvasive Recordings.** Processes 8(7):846 (2020). doi:10.3390/pr8070846.
> BIDS conversion: Dan J, Detti P. **BIDS Siena Scalp EEG Database.** EPFL, Zenodo (2024).
> doi:10.5281/zenodo.10640762 (CC BY 4.0).

> Shoeb A. **CHB-MIT Scalp EEG Database.** PhysioNet (2010). doi:10.13026/C2K01R (ODC-By 1.0).

> Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng
> C-K, Stanley HE. **PhysioBank, PhysioToolkit, and PhysioNet.** Circulation 101(23):e215 (2000).
> doi:10.1161/01.CIR.101.23.e215, PMID 10851218.

## Acknowledgements

epycom is developed by the Biomedical Engineering Department of the International Clinical
Research Center, St. Anne's University Hospital Brno. MNE-Python is developed by the MNE community.
The Siena Scalp EEG Database and CHB-MIT are distributed by PhysioNet, which asks that its
databases be cited with Goldberger et al. 2000. vEpiSet was collected at the Epilepsy Center of
Peking Union Medical College Hospital (funding listed in Lin et al. 2025).
