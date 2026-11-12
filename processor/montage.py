"""
Scalp EEG channel names, hemispheres, regions and the longitudinal bipolar montage.

Channel names are canonicalised to the 10-20/10-10 spelling MNE uses (`standard_1005`):
`FP1` -> `Fp1`, the old temporal names `T3/T4/T5/T6` -> `T7/T8/P7/P8`, `EEG Fp1-Ref` and
`Fp1-Avg` -> `Fp1` (a reference suffix, not a second electrode), `FP1-F7` -> the bipolar pair
`Fp1-F7`, and CHB-MIT's duplicated `T8-P8-0` / `T8-P8-1` -> `T8-P8`.

The longitudinal bipolar montage ("double banana") is the one clinicians read spikes on:
four chains per side (temporal and parasagittal) plus the midline.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Old 10-20 names (Jasper 1958) and their 10-10 equivalents (MNE's standard_1005 spelling).
OLD_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Tokens after a dash that name a reference, not a second electrode ("Fp1-Avg", "C3-REF").
REFERENCE_TOKENS = {"AVG", "AV", "AVE", "AVERAGE", "REF", "CAR", "AR", "LE", "LER", "RE", "COM", "COMMON",
                    "A1A2", "M1M2", "LINKED", "G2", "N"}
# Ear and mastoid electrodes: a channel "Fp1-A1" is referential (to the ear), not a pair.
EAR_ELECTRODES = {"A1", "A2", "M1", "M2"}

# The longitudinal bipolar montage (double banana), 10-10 names, in reading order.
CHAINS: list[tuple[str, list[tuple[str, str]]]] = [
    ("left temporal", [("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1")]),
    ("left parasagittal", [("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1")]),
    ("midline", [("Fz", "Cz"), ("Cz", "Pz")]),
    ("right parasagittal", [("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")]),
    ("right temporal", [("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2")]),
]
BIPOLAR_PAIRS = [p for _, chain in CHAINS for p in chain]
CHAIN_OF = {f"{a}-{b}": name for name, chain in CHAINS for a, b in chain}


@lru_cache(maxsize=1)
def _known() -> dict[str, str]:
    """Upper-case -> canonical spelling of every standard_1005 electrode (and the ear electrodes)."""
    try:
        import mne

        names = list(mne.channels.make_standard_montage("standard_1005").ch_names)
    except Exception:  # noqa: BLE001 - MNE is always there in the image; keep the module importable without it
        names = []
    names += ["Fp1", "Fp2", "Fpz", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz",
              "P4", "P8", "O1", "Oz", "O2", "A1", "A2", "M1", "M2", "FT9", "FT10", "T9", "T10", "TP9", "TP10"]
    return {n.upper(): n for n in names}


def electrode(token: str) -> str | None:
    """The canonical 10-10 name of one electrode token, or None when it is not one."""
    t = token.strip()
    if not t:
        return None
    up = t.upper()
    if up in OLD_NAMES:
        return OLD_NAMES[up]
    return _known().get(up)


_PREFIX = re.compile(r"^(EEG|EEG-|POL|EEG_)\s*", re.IGNORECASE)
_DUP = re.compile(r"^(.+)-(\d)$")


def parse(name: str) -> tuple[str, tuple[str, ...]] | None:
    """
    (canonical label, contacts) of a channel name, or None when it is not a scalp EEG
    electrode or pair. Referential: ("Fp1", ("Fp1",)); bipolar: ("Fp1-F7", ("Fp1", "F7")).
    """
    n = _PREFIX.sub("", name.strip())
    m = _DUP.match(n)
    if m and "-" in m.group(1):  # CHB-MIT "T8-P8-0": a duplicated pair
        n = m.group(1)
    parts = [p for p in re.split(r"\s*-\s*", n) if p != ""]
    if len(parts) == 1:
        e = electrode(parts[0])
        return (e, (e,)) if e and e not in EAR_ELECTRODES else None
    if len(parts) == 2:
        a, b = electrode(parts[0]), electrode(parts[1])
        if a is None or a in EAR_ELECTRODES:
            return None
        if parts[1].upper() in REFERENCE_TOKENS or b in EAR_ELECTRODES:
            return a, (a,)
        if b is None:
            return None
        return f"{a}-{b}", (a, b)
    return None


def hemisphere_of_electrode(e: str) -> str:
    """L (odd number), R (even number), Z (midline, `z`)."""
    if e.lower().endswith("z"):
        return "Z"
    m = re.search(r"(\d+)h?$", e)
    if not m:
        return "Z"
    return "L" if int(m.group(1)) % 2 == 1 else "R"


def region_of_electrode(e: str) -> str:
    """frontal | temporal | central | parietal | occipital | midline, from the 10-10 name."""
    up = e.upper()
    if up.endswith("Z"):
        return "midline"
    num = re.search(r"(\d+)", up)
    k = int(num.group(1)) if num else 0
    letters = re.sub(r"[\dH]+$", "", up)
    if letters in ("T", "TP", "FT") or (letters in ("F", "P") and k in (7, 8, 9, 10)) or letters in ("A", "M"):
        return "temporal"
    if letters in ("FP", "AF", "F", "AFF", "AFP"):
        return "frontal"
    if letters in ("C", "FC", "FCC"):
        return "central"
    if letters in ("P", "CP", "CPP"):
        return "parietal"
    if letters in ("O", "PO", "I", "POO", "OI"):
        return "occipital"
    return "central"


def hemisphere(contacts: tuple[str, ...]) -> str:
    """A channel's hemisphere: its electrode's, or for a pair the common one (Z when they cross)."""
    hs = {hemisphere_of_electrode(c) for c in contacts}
    if len(hs) == 1:
        return hs.pop()
    sides = hs - {"Z"}
    return sides.pop() if len(sides) == 1 else "Z"  # a midline-lateral pair: the lateral side


def region(contacts: tuple[str, ...]) -> str:
    """
    A channel's region. A pair: midline when both electrodes are; temporal when either is
    (the lateral chain is read as the temporal chain); else the region of its first electrode.
    """
    regs = [region_of_electrode(c) for c in contacts]
    if all(r == "midline" for r in regs):
        return "midline"
    if "temporal" in regs:
        return "temporal"
    return next(r for r in regs if r != "midline") if any(r != "midline" for r in regs) else "midline"


def side_word(h: str) -> str:
    return {"L": "left", "R": "right", "Z": "midline"}.get(h, "")


def chain_order(label: str) -> tuple[int, int]:
    """Sort key: double-banana reading order for its pairs, then everything else by name."""
    try:
        i = [f"{a}-{b}" for a, b in BIPOLAR_PAIRS].index(label)
        return 0, i
    except ValueError:
        return 1, 0
