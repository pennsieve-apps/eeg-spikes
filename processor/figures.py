"""
Figures (PNG, 150 dpi, `<recording_id>_<what>.png`, the recording id in the title):

- spike rates: a topomap of spike rate per electrode, next to the rate of every channel in
  double-banana order;
- rates per sleep stage (when stages are known);
- example spikes: the spike events of the most frequent leading channel, all channels stacked;
- one cohort figure: the laterality index of every recording.
"""
from __future__ import annotations

import numpy as np

from processor import montage as MT
from processor import stages as S

DPI = 150
LEFT, RIGHT, MID = "#2a78d6", "#eb6834", "#8a8984"  # categorical slots 1 and 2, neutral for midline
INK, MUTED = "#0b0b0b", "#52514e"
SIDE_COLOR = {"L": LEFT, "R": RIGHT, "Z": MID}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.titlesize": 10})
    return plt


def electrode_values(rates: dict[str, float], contacts: dict[str, tuple[str, ...]]) -> dict[str, float]:
    """
    Rate per electrode: a referential channel's own rate; for bipolar pairs, the mean rate of the
    pairs an electrode belongs to (an electrode at the centre of a discharge is in two pairs
    that both see it, so it stands out).
    """
    acc: dict[str, list[float]] = {}
    for label, r in rates.items():
        for c in contacts.get(label, ()):
            if MT.electrode(c):
                acc.setdefault(c, []).append(r)
    return {e: float(np.mean(v)) for e, v in acc.items()}


def rates_figure(path, rid: str, rates: dict[str, float], contacts: dict[str, tuple[str, ...]],
                 hemis: dict[str, str], montage: str, detector: str) -> bool:
    import mne

    plt = _plt()
    elec = electrode_values(rates, contacts)
    names = [e for e in elec if e in set(mne.channels.make_standard_montage("standard_1005").ch_names)]
    labels = sorted(rates, key=lambda l: (MT.chain_order(l), l))
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(10, max(4.0, 0.22 * len(labels) + 1.2)),
                                         gridspec_kw={"width_ratios": [1, 1.2]})
    if len(names) >= 4:
        info = mne.create_info(names, 256.0, "eeg")
        info.set_montage("standard_1005", on_missing="ignore", verbose="error")
        vals = np.array([elec[n] for n in names])
        vmax = max(float(vals.max()), 1e-6)
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax_map, show=False, cmap="Purples", vlim=(0, vmax),
                                     names=names, sensors=True, contours=0)
        cb = fig.colorbar(im, ax=ax_map, shrink=0.7)
        cb.set_label("spikes / min" + (" (mean of its pairs)" if montage == "bipolar" else ""))
        ax_map.set_title("Spike rate per electrode")
    else:
        ax_map.axis("off")
        ax_map.text(0.5, 0.5, "too few 10-20 electrodes for a map", ha="center", va="center", color=MUTED)
    y = np.arange(len(labels))[::-1]
    ax_bar.barh(y, [rates[l] for l in labels], color=[SIDE_COLOR.get(hemis.get(l, "Z"), MID) for l in labels],
                height=0.75, edgecolor="white", linewidth=1)
    ax_bar.set_yticks(y, labels, fontsize=7)
    ax_bar.set_xlabel(f"spikes / min ({detector})")
    ax_bar.set_title(f"Per channel ({montage} montage)")
    from matplotlib.patches import Patch

    ax_bar.legend(handles=[Patch(color=LEFT, label="left"), Patch(color=RIGHT, label="right"),
                           Patch(color=MID, label="midline")], frameon=False, loc="lower right", fontsize=7)
    fig.suptitle(f"{rid}: interictal spike rates (research output, review before clinical use)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


def stage_figure(path, rid: str, by_stage: dict[str, dict], source: str) -> bool:
    """by_stage: stage -> {"events": n, "minutes": m}."""
    plt = _plt()
    stages = [s for s in S.STAGES + ["sleep"] if by_stage.get(s, {}).get("minutes", 0) > 0]
    if not stages:
        return False
    rates = [by_stage[s]["events"] / by_stage[s]["minutes"] for s in stages]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = ["#e3a33a" if s == "W" else ("#6250d6" if s == "R" else "#2a78d6") for s in stages]
    ax.bar(stages, rates, color=colors, width=0.6, edgecolor="white", linewidth=2)
    for i, s in enumerate(stages):
        ax.text(i, rates[i], f"{by_stage[s]['minutes']:.0f} min", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_ylabel("spike events / min")
    ax.set_xlabel(f"stage ({source})")
    ax.margins(y=0.15)
    ax.set_title(f"{rid}: spike events per sleep stage", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


def examples_figure(path, rid: str, snippets: list[dict], labels: list[str], leader: str, montage: str) -> bool:
    """
    snippets: [{"time_s", "data": (channels, samples) µV, "fs", "start_s"}] around spike events,
    channels in `labels` order. Stacked traces, the event time marked.
    """
    if not snippets:
        return False
    plt = _plt()
    n = len(snippets)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 0.8, max(4.0, 0.25 * len(labels) + 1.4)), sharey=True)
    axes = np.atleast_1d(axes)
    sd = float(np.median([np.median(np.std(s["data"], axis=1)) for s in snippets]))
    spacing = max(round(5 * sd / 5) * 5, 10.0)  # like a fixed display sensitivity: typical background fits
    for ax, s in zip(axes, snippets):
        d = s["data"]
        t = s["start_s"] + np.arange(d.shape[1]) / s["fs"] - s["time_s"]
        for i, lab in enumerate(labels):
            off = -i * spacing
            col = INK if lab != leader else RIGHT
            ax.plot(t, d[i] + off, color=col, linewidth=0.7 if lab != leader else 1.0)
        ax.axvline(0, color=LEFT, linewidth=0.8, alpha=0.6)
        ax.set_title(f"t = {s['time_s']:.1f} s", fontsize=8)
        ax.set_xlabel("s from the spike")
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks([-i * spacing for i in range(len(labels))], labels, fontsize=7)
    axes[0].tick_params(axis="y", length=0)
    # scale bar
    ax0 = axes[-1]
    x0 = ax0.get_xlim()[1]
    ax0.plot([x0, x0], [0, -spacing], color=INK, linewidth=1.5, clip_on=False)
    ax0.text(x0, -spacing / 2, f" {spacing:.0f} µV", va="center", fontsize=7, clip_on=False)
    fig.suptitle(f"{rid}: example spikes led by {leader} ({montage} montage, positive up)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


def laterality_figure(path, rows: list[dict]) -> bool:
    """rows: [{"recording_id", "laterality_index", "n_events", "expected" (optional L/R)}]."""
    rows = [r for r in rows if r.get("laterality_index") is not None]
    if not rows:
        return False
    plt = _plt()
    rows = sorted(rows, key=lambda r: r["recording_id"])
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.25 * len(rows) + 1.2)))
    y = np.arange(len(rows))[::-1]
    li = [r["laterality_index"] for r in rows]
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.scatter(li, y, s=[12 + 3 * np.sqrt(r["n_events"]) for r in rows],
               c=[LEFT if v > 0 else (RIGHT if v < 0 else MID) for v in li], edgecolors="white", linewidths=1,
               zorder=3)
    ax.set_yticks(y, [r["recording_id"] for r in rows], fontsize=7)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("laterality index (L − R) / (L + R) of spike events   ← right | left →")
    ax.set_title("Lateralisation of spike events per recording (dot size: number of events)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True
