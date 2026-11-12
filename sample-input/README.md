# sample-input

Real input for this app is the `eeg` output of **eeg-prep**: a BIDS-EEG derivative with
`recordings.csv`, and per recording `*_desc-preproc_eeg.edf`, `_channels.tsv` and `_events.tsv`
(workflow-scalp-eeg `docs/conventions.md`), optionally with eeg-sleep's `hypnogram.csv` on the
`sleep` port. Any BIDS-EEG dataset or loose EDF files with 10-20 channel names work too.

The workflows' demo data (`workflow-scalp-eeg/scripts/fetch-sample-data.sh`):

- **BIDS Siena Scalp EEG Database** (Dan J, Detti P; Zenodo doi:10.5281/zenodo.10640762, CC BY 4.0;
  from PhysioNet doi:10.13026/5d4a-j060): 14 adults with focal epilepsy, 41 runs, 19 channels in
  average reference at 256 Hz, seizures as HED-SCORE events. Runs directly as a plain BIDS input:

  ```sh
  docker run --rm --platform linux/amd64 --cpus 2 --memory 3g --env-file dev.env \
    -v "$PWD/../workflow-scalp-eeg/data/siena/BIDS_Siena":/data/input/eeg:ro \
    -v "$PWD/data/output-siena":/data/output eeg-spikes:dev
  ```

  The seizure lateralisation of each patient is in PhysioNet's `subject_info.csv`
  (https://physionet.org/files/siena-scalp-eeg/1.0.0/subject_info.csv), which
  `validation/siena_lateralisation.py` compares with the app's output.
- **CHB-MIT** (Shoeb A; PhysioNet doi:10.13026/C2K01R, ODC-By 1.0), `data/chbmit/chb01/`:
  bipolar-only one-hour EDFs; mount the folder (with `chb01-summary.txt`, which gives the seizures).
- After eeg-prep (and eeg-sleep): `make run-demo DEMO=../workflow-scalp-eeg/data/work/siena/eeg-prep/eeg`,
  or mount `data/work/<demo>/eeg-prep/eeg` at `/data/input/eeg` and `data/work/<demo>/eeg-sleep` at
  `/data/input/sleep`.

For spike-level validation, **vEpiSet** (Lin N et al.; figshare doi:10.6084/m9.figshare.28069568.v2,
CC BY 4.0, 16.9 GB zip) has 84 continuous 20-minute scalp recordings with expert IED marks:
`validation/vepiset_to_bids.py` turns its `MAT_Files/*.mat` into plain BIDS EDF and
`validation/vepiset_eval.py` scores the app's output against the marks.

For tests and `make run`, `tests/make_fixture.py` writes a small synthetic input instead
(`make fixture` -> `data/fixture/`): a referential recording with planted spikes (left and right
temporal), a seizure, a bad segment and sleep; a bipolar-only recording; a hypnogram; the
referential recording again as plain BIDS Siena would store it; `truth.json` with the planted
spike times.
