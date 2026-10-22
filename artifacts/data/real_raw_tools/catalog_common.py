"""Shared helpers for building D:\\GAN\\artifacts\\data\\real_raw\\catalog.csv (data parsing only)."""
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/GAN/artifacts/data/real_raw")
COLUMNS = ["item_id", "source", "source_record_url", "file_path", "licence", "technique", "edge_or_wavelength",
           "x_kind", "material", "composition", "sample_type", "temperature_K", "notes", "citation"]

# tabulated edge energies (eV), from xraydb at import time (local library, not downloaded data)
try:
    import xraydb

    def edge_energy(el, edge):
        try:
            return float(xraydb.xray_edge(el, edge).energy)
        except Exception:  # noqa: BLE001
            return float("nan")
except ImportError:  # pragma: no cover
    def edge_energy(el, edge):
        return {"PtL3": 11564.0, "CoK": 7709.0, "NiK": 8333.0, "FeK": 7112.0, "CuK": 8979.0}.get(el + edge, float("nan"))

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
            "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm").split()


def rel(p):
    return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()


def read_text(p):
    b = Path(p).read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin-1", errors="replace")


NUM_RE = re.compile(r"^[\s,;]*[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?([\s,;\t]+[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?|[\s,;\t]+nan)*[\s,;]*$", re.I)


def numeric_block(text, delim=None, decimal_comma=False):
    """Return 2D float array of the numeric lines in text (lines with the modal column count)."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if decimal_comma:
            s = s.replace(",", ".")
        parts = re.split(r"[\s,;]+" if delim is None else re.escape(delim), s.strip(" ,;\t"))
        try:
            vals = [float(x.replace("D", "E").replace("d", "e")) for x in parts if x != ""]
        except ValueError:
            continue
        if vals:
            rows.append(vals)
    if not rows:
        return np.zeros((0, 0))
    from collections import Counter
    ncol = Counter(len(r) for r in rows).most_common(1)[0][0]
    arr = np.array([r for r in rows if len(r) == ncol], dtype=float)
    return arr


def kmax_from_energy(e0, emax):
    if not (e0 and emax) or math.isnan(e0) or emax <= e0:
        return None
    return math.sqrt(0.262468 * (emax - e0))


def edge_label(el, edge):
    edge = (edge or "").strip().upper().replace("-EDGE", "").replace("EDGE", "").strip()
    edge = {"L21": "L2", "LIII": "L3", "LII": "L2", "LI": "L1"}.get(edge, edge)
    return f"{el} {edge}".strip()


def infer_edge_from_range(emin, emax, candidates=None):
    """Guess absorption edge from scan range: the tabulated K/L3/L2/L1 edge inside the range whose energy is
    closest to emin + ~0.15*(range) (typical pre-edge of 100-250 eV)."""
    best = None
    target = emin + min(250.0, max(30.0, 0.15 * (emax - emin)))
    els = candidates or ELEMENTS
    for el in els:
        for ed in ("K", "L3", "L2", "L1"):
            e = edge_energy(el, ed)
            if math.isnan(e) or not (emin + 10 <= e <= emin + 0.6 * (emax - emin)):
                continue
            d = abs(e - target)
            if best is None or d < best[0]:
                best = (d, el, ed, e)
    return (best[1], best[2], best[3]) if best else (None, None, None)


def edge_from_data(e, arr, ecol):
    """Locate the absorption jump from the data (largest step in log of any positive counter column or of
    log(col_a/col_b) ratios) and match it to a tabulated K/L3 (then L2/L1) edge within +-30 eV."""
    e = np.asarray(e, float)
    order = np.argsort(e)
    e = e[order]
    best = (0.0, None)
    ncol = arr.shape[1]
    sigs = []
    for j in range(ncol):
        if j == ecol:
            continue
        c = np.asarray(arr[order, j], float)
        if np.all(np.isfinite(c)) and np.all(c > 0):
            sigs.append(np.log(c))
        elif np.all(np.isfinite(c)) and np.ptp(c) > 0:
            sigs.append(c / (np.ptp(c) or 1))
    for s in sigs:
        if len(s) < 20:
            continue
        k = max(3, len(s) // 100)
        sm = np.convolve(s, np.ones(k) / k, mode="same")
        step = np.abs(sm[2 * k:] - sm[:-2 * k]) if len(sm) > 4 * k else np.abs(np.diff(sm))
        if step.size == 0:
            continue
        i = int(np.argmax(step[k:-k])) + k if step.size > 2 * k else int(np.argmax(step))
        val = float(step[i]) / (np.std(np.diff(sm)) + 1e-12)
        if val > best[0]:
            best = (val, float(e[min(len(e) - 1, i + k)]))
    ej = best[1]
    if ej is None:
        return None, None, None
    for edges in (("K", "L3"), ("L2", "L1")):
        cand = []
        for el in ELEMENTS:
            for ed in edges:
                en = edge_energy(el, ed)
                if not math.isnan(en) and abs(en - ej) <= 30:
                    cand.append((abs(en - ej), el, ed, en))
        if cand:
            cand.sort()
            return cand[0][1], cand[0][2], ej
    return None, None, ej


METALS = set("Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Cs Ba "
             "La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Th U".split())


def formula_elements(formula):
    return re.findall(r"[A-Z][a-z]?", formula or "")


def guess_sample_type(name, formula="", extra=""):
    s = f"{name} {extra}".lower()
    nmet = len({e for e in formula_elements(formula) if e in METALS})
    if nmet >= 2 and re.search(r"\bfoil\b|alloy", s):
        return "alloy"
    if re.search(r"\bfoil\b|metal foil|\bfoils\b|reference foil|ref(erence)? metal|metal_rt|_metal|\bmetal\b|pure metal", s):
        return "foil"
    if re.search(r"nano|\bnps?\b|nanoparticle|cluster|colloid", s):
        return "nanoparticle"
    if re.search(r"catalyst|/c\b|/al2o3|/sio2|/tio2|/ceo2|/zro2|supported|zeolite|zsm|/kb|/rgo|vulcan|ketjen|on carbon|pt/|_c$", s):
        return "catalyst"
    if re.search(r"alloy|intermetallic|steel|sus\d|nichrome|inconel|brass|bronze|solder", s):
        return "alloy"
    els = formula_elements(formula)
    if els and all(e in METALS for e in els):
        return "foil" if len(set(els)) == 1 else "alloy"
    if formula or name:
        return "compound"
    return "other"


def parse_temp(t):
    """Return (temperature_K_string, note)."""
    if t is None:
        return "", ""
    s = str(t).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return "", ""
    sl = s.lower().replace(" ", "")
    if sl in ("rt", "r.t.", "roomtemperature", "room", "ambient", "roomtemp", "roomtemp.", "rt."):
        return "298", "temperature given as RT (298 K assumed)"
    m = re.match(r"^([-+]?\d+(\.\d+)?)\s*(k|kelvin)$", sl)
    if m:
        return m.group(1), ""
    m = re.match(r"^([-+]?\d+(\.\d+)?)\s*(c|degc|°c|celsius|oc|ºc)$", sl)
    if m:
        return f"{float(m.group(1)) + 273.15:.2f}".rstrip("0").rstrip("."), f"T={s}"
    m = re.match(r"^([-+]?\d+(\.\d+)?)$", sl)
    if m:
        v = float(m.group(1))
        if v > 200:
            return m.group(1), "temperature unit not stated (assumed K)"
        return "", f"temperature '{s}' (unit not stated)"
    return "", f"temperature '{s}'"


def parse_xdi(path):
    """Minimal XDI/1.0 reader: returns (meta dict, column names list, data array)."""
    text = read_text(path)
    meta, cols, labels = {}, {}, []
    lines = text.splitlines()
    for ln in lines:
        if not ln.startswith("#"):
            continue
        s = ln[1:].strip()
        m = re.match(r"^([A-Za-z][\w]*\.[\w.\-]+)\s*:\s*(.*)$", s)
        if m:
            k, v = m.group(1), m.group(2).strip()
            meta[k] = v
            mc = re.match(r"^Column\.(\d+)$", k, re.I)
            if mc:
                cols[int(mc.group(1))] = v
        elif s and not s.startswith("/") and not s.startswith("-"):
            labels = s.split()
    data = numeric_block(text)
    names = [cols.get(i + 1, labels[i] if i < len(labels) else f"col{i+1}") for i in range(data.shape[1] if data.size else 0)]
    return meta, names, data


def mode_from_names(names, extra=""):
    s = (" ".join(names) + " " + extra).lower()
    if "herfd" in s:
        return "HERFD"
    if re.search(r"fluo|ifluor|\bif\b|mufluor|\bxmu_?f|pips|sdd|vortex|lytle", s):
        if re.search(r"itrans|mutrans|\bit\b|\bi1\b", s):
            return "transmission+fluorescence"
        return "fluorescence"
    if re.search(r"tey|electron yield|total electron", s):
        return "electron yield"
    if re.search(r"trans|itrans|mutrans|\bit\b|\bi1\b|ln\(i0/it\)", s):
        return "transmission"
    return ""


def fmt(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return ""


def xas_notes(mode, energies, el, edge, extra=""):
    parts = []
    if mode:
        parts.append(f"mode={mode}")
    if energies is not None and len(energies):
        emin, emax = float(np.nanmin(energies)), float(np.nanmax(energies))
        parts.append(f"npts={len(energies)}")
        parts.append(f"E={emin:.1f}-{emax:.1f} eV")
        e0 = edge_energy(el, edge) if el and edge else float("nan")
        k = kmax_from_energy(e0, emax)
        if k:
            parts.append(f"kmax~{k:.1f} A-1")
    if extra:
        parts.append(extra)
    return "; ".join(parts)


def write_two_col(dest, x, ys, header_lines, colnames):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for h in header_lines:
            fh.write(f"# {h}\n")
        fh.write("# columns: " + " ".join(colnames) + "\n")
        for i in range(len(x)):
            vals = [x[i]] + [y[i] for y in ys]
            fh.write(" ".join("nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.10g}" for v in vals) + "\n")
