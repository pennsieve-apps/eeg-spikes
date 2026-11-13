FROM --platform=linux/amd64 python:3.12.16-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# Pinned Python dependencies, binary wheels only, no dependency resolution at build time
# (requirements.txt lists every package). Test folders and MNE's report templates are removed.
# No model weights: the spike detectors are signal-processing methods.
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --root-user-action=ignore -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12 -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -depth -type d \( -name tests -o -name testing \) -path "*/mne/*" -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/pandas/tests /usr/local/lib/python3.12/site-packages/scipy/*/tests \
              /usr/local/lib/python3.12/site-packages/numpy/*/tests /usr/local/lib/python3.12/site-packages/matplotlib/tests \
              /usr/local/lib/python3.12/site-packages/mne/html_templates/report \
    && python3 -c "import epycom.event_detection.spike.janca_detector, mne, matplotlib; mne.channels.make_standard_montage('standard_1005'); print('mne', mne.__version__)" \
    && mkdir -p /opt/eeg-spikes/mpl \
    && MPLCONFIGDIR=/opt/eeg-spikes/mpl python3 -c "import matplotlib.font_manager" \
    && ls /opt/eeg-spikes/mpl/fontlist-*.json \
    && chmod -R a+rX /opt/eeg-spikes \
    && useradd --uid 1000 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin processor

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

# The Matplotlib font cache is built above into a read-only path; at start the app copies it to
# $TMPDIR/mpl (MPLCONFIGDIR) and points the numba and MNE caches at $TMPDIR, because HOME and
# /tmp are replaced at run time on Lambda. BLAS threads are set to 1 by the app before numpy
# loads (channels run in parallel processes instead).
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MNE_DONTWRITE_HOME=true \
    MNE_USE_CUDA=false

# Runs as uid 1000 on ECS and as an arbitrary unprivileged uid on Lambda; nothing here needs root.
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
