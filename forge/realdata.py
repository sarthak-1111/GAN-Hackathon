"""Phase 3: real reference spectra -> model grids, hold-out splits and physics twins.

Inputs: artifacts/data/real_raw/catalog.csv (open licences only; source, ID and licence of every item recorded).
Outputs: artifacts/data/real/<technique>.npz (see forge.realism for the fields) and <technique>_items.csv.

Split: per technique, 20 % of *materials* (all spectra of a material together, so repeated measurements of one sample
cannot leak) are held out for evaluation and never used for training or model selection; 15 % of the remaining
materials are the validation set used for early stopping.
"""
import csv
import hashlib
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .paths import DATA, ROOT

RAW = DATA / "real_raw"
OUT = DATA / "real"
EDGE_E0 = {"Pt L3": 11564.0, "Co K": 7709.0, "Ni K": 8333.0, "Fe K": 7112.0, "Cu K": 8979.0}
EDGE_EL = {"Pt L3": "Pt", "Co K": "Co", "Ni K": "Ni", "Fe K": "Fe", "Cu K": "Cu"}
EDGES5 = ["Pt", "Co", "Ni", "Fe", "Cu"]
K_STORE = np.round(np.arange(321) * 0.05, 4)
K_MODEL = slice(40, 296)
S02 = {"Pt": 0.85, "Co": 0.80, "Ni": 0.80, "Fe": 0.80, "Cu": 0.85}


def stable_hash(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def split_by_group(groups, seed, eval_frac=0.2, val_frac=0.15):
    """groups: list of group keys (one per item) -> list of 'train' / 'val' / 'eval'."""
    uniq = sorted(set(groups), key=lambda g: stable_hash(f"{seed}:{g}"))
    n = len(uniq)
    n_eval = max(1, int(round(eval_frac * n)))
    n_val = max(1, int(round(val_frac * (n - n_eval))))
    lab = {}
    for i, g in enumerate(uniq):
        lab[g] = "eval" if i < n_eval else ("val" if i < n_eval + n_val else "train")
    return [lab[g] for g in groups]


# --------------------------------------------------------------------------------------------- XAS parsing
def _numeric_table(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.strip()
            if not t or t[0] in "#;!%" or t[0].isalpha():
                continue
            parts = re.split(r"[\s,]+", t)
            try:
                vals = [float(x) for x in parts if x]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals)
    if not rows:
        return None
    n = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    return np.array([r[:n] for r in rows if len(r) >= n])


def read_xas(path, notes=""):
    """(energy_eV, mu) from XDI or column text files (mu, mu*t, or I0 / It / If counters)."""
    path = Path(path)
    txt = path.read_text(encoding="utf-8", errors="replace")
    cols = {}
    for m in re.finditer(r"#\s*Column\.(\d+)\s*:\s*([^\s\n]+)", txt):
        cols[int(m.group(1)) - 1] = m.group(2).lower()
    a = _numeric_table(path)
    if a is None or a.shape[0] < 50:
        return None
    e = a[:, 0]
    if cols:
        name = {v: k for k, v in cols.items()}
        for key in ("mutrans", "mu", "xmu", "mufluor", "norm", "mu_trans", "mu_fluor"):
            if key in name and name[key] < a.shape[1]:
                return e, a[:, name[key]]
        if "i0" in name and "itrans" in name and max(name["i0"], name["itrans"]) < a.shape[1]:
            with np.errstate(divide="ignore", invalid="ignore"):
                return e, np.log(np.abs(a[:, name["i0"]]) / np.abs(a[:, name["itrans"]]))
        for f in ("ifluor", "ifl", "if"):
            if "i0" in name and f in name and max(name["i0"], name[f]) < a.shape[1]:
                return e, a[:, name[f]] / a[:, name["i0"]]
        return None
    # plain tables: energy first, mu second (the converted files of the catalog are documented this way)
    return e, a[:, 1]


def extract_chi(e, mu, e0_nominal):
    """Larch normalization + AUTOBK -> (chi on K_STORE, kmax of data, edge step, e0) or None."""
    from larch import Group
    from larch.xafs import autobk, pre_edge
    e = np.asarray(e, float)
    mu = np.asarray(mu, float)
    ok = np.isfinite(e) & np.isfinite(mu)
    e, mu = e[ok], mu[ok]
    o = np.argsort(e)
    e, mu = e[o], mu[o]
    keep = np.concatenate([[True], np.diff(e) > 1e-3])
    e, mu = e[keep], mu[keep]
    if e.max() < 300:  # relative energies: shift to the nominal edge
        e = e + e0_nominal
    if not (e.min() < e0_nominal - 20 and e.max() > e0_nominal + 400):
        return None
    g = Group(energy=e, mu=mu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre_edge(g, e0=None)
        if abs(g.e0 - e0_nominal) > 60:
            pre_edge(g, e0=e0_nominal)
        if not (g.edge_step > 0) or not np.isfinite(g.edge_step):
            return None
        kmax_data = math.sqrt(max(e.max() - g.e0, 0) / 3.80998)
        autobk(g, rbkg=1.0, kweight=2, kmax=min(kmax_data, 16.0))
    chi = np.interp(K_STORE, g.k, g.chi, left=0.0, right=0.0)
    chi[K_STORE > kmax_data] = 0.0
    return chi, kmax_data, float(g.edge_step), float(g.e0)


def comp6_from_formula(formula):
    """[Pt, Co, Ni, Fe, Cu, other] atomic fractions and an oxide flag from a formula string (None if unparsable)."""
    from pymatgen.core import Composition
    try:
        c = Composition(re.sub(r"[^A-Za-z0-9().]", "", str(formula))).fractional_composition
    except Exception:
        return None, None
    v = np.zeros(6, np.float32)
    for el, x in c.items():
        s = el.symbol
        if s in EDGES5:
            v[EDGES5.index(s)] += x
        else:
            v[5] += x
    return v, float("O" in [e.symbol for e in c.elements])


def _xas_item(row):
    try:
        r = read_xas(RAW / row["file_path"], row.get("notes", ""))
        if r is None:
            return row["item_id"], None, "unreadable"
        out = extract_chi(r[0], r[1], EDGE_E0[row["edge_or_wavelength"]])
        if out is None:
            return row["item_id"], None, "range/normalization"
        chi, kmax, step, e0 = out
        if kmax < 14.75:
            return row["item_id"], None, f"kmax {kmax:.1f} < 14.75"
        y = chi * K_STORE ** 2
        if not np.all(np.isfinite(y)) or np.abs(y[K_MODEL]).max() > 8:
            return row["item_id"], None, "chi out of range"
        return row["item_id"], dict(chi=chi.astype(np.float32), kmax=kmax, e0=e0, step=step), "ok"
    except Exception as e:
        return row["item_id"], None, f"error {type(e).__name__}: {str(e)[:80]}"


def load_catalog(technique):
    rows = list(csv.DictReader(open(RAW / "catalog.csv", encoding="utf-8")))
    return [r for r in rows if r["technique"] == technique and r["licence"].strip() and "unclear" not in r["licence"].lower()]


def process_xas(workers=24):
    rows = [r for r in load_catalog("xas") if r["edge_or_wavelength"] in EDGE_E0 and "matrix" not in r["notes"].lower()
            and "XANES range only" not in r["notes"] and "XANES-only" not in r["notes"]]
    print(f"XAS candidates at target edges: {len(rows)}", flush=True)
    res, reasons = {}, defaultdict(int)
    with ProcessPoolExecutor(workers) as ex:
        for item_id, out, why in ex.map(_xas_item, rows, chunksize=4):
            reasons[why.split(" ")[0] if not why.startswith("kmax") else "kmax<14.75"] += 1
            if out is not None:
                res[item_id] = out
    print("XAS processing:", dict(reasons), flush=True)
    return rows, res


# --------------------------------------------------------------------------------------------- XAS twins
CAT_COMP = [("Co", "Pt3Co"), ("Ni", "Pt3Ni"), ("Fe", "Pt3Fe"), ("Cu", "PtCu")]


def nanoparticle_class(row):
    """Concord Forge composition for Pt / Pt-alloy catalysts and nanoparticles (None if not representable)."""
    txt = f"{row['material']} {row['composition']}"
    if "Pt" not in txt or row["sample_type"] not in ("catalyst", "alloy", "nanoparticle", "other"):
        return None
    if re.search(r"Sn|Pd|Ru|Au|Ir|Rh|Re|Mo|W|S\b|Cl|zeolite|adsorbed|fluid|crust|goethite|hematite", txt):
        return None
    for el, comp in CAT_COMP:
        if re.search(rf"{el}(?![a-z])", txt):
            return comp
    return "Pt"


def _feff_twin_job(args):
    key, formula, absorber, tag = args
    from .twins import cod_structure, feff_bulk_chi
    s, cid = cod_structure(formula)
    if s is None:
        return key, None, cid
    try:
        U = {absorber: 0.005}
        for el in s.composition.elements:
            U.setdefault(el.symbol, 0.005 if el.symbol in EDGES5 else 0.008)
        chi = feff_bulk_chi(s, absorber, U, delta2=3.0, floor=0.35, s02=S02[absorber], tag=tag)
        if chi is None:
            return key, None, "absorber not in structure"
        return key, chi.astype(np.float32), f"COD {cid}"
    except Exception as e:
        return key, None, f"FEFF error {str(e)[:80]}"


def xas_twins(rows, res, workers=24, seed=0):
    """Twin chi(k) for each processed real XAS item: FEFF bulk (foils / compounds) or a nanoparticle simulation."""
    from .data import Dataset
    item_twin, why = {}, {}
    jobs = {}
    for r in rows:
        if r["item_id"] not in res:
            continue
        X = EDGE_EL[r["edge_or_wavelength"]]
        npc = nanoparticle_class(r)
        if npc is not None:
            item_twin[r["item_id"]] = ("np", npc, X)
            continue
        if r["sample_type"] in ("foil", "compound"):
            formula = X if r["sample_type"] == "foil" else re.sub(r"\s+", "", str(r["composition"]))
            key = f"{formula}|{X}"
            jobs[key] = (key, formula, X, re.sub(r"[^A-Za-z0-9]", "", key)[:40])
            item_twin[r["item_id"]] = ("feff", key, X)
        else:
            why[r["item_id"]] = f"no twin model for sample type '{r['sample_type']}'"
    print(f"twins: {len(jobs)} FEFF materials, {sum(1 for v in item_twin.values() if v[0] == 'np')} nanoparticle items",
          flush=True)
    feff_chi = {}
    with ProcessPoolExecutor(workers) as ex:
        for key, chi, msg in ex.map(_feff_twin_job, list(jobs.values())):
            feff_chi[key] = (chi, msg)
    # nanoparticle twins: pilot simulations of the same composition (random size), same edge
    sim = Dataset(ROOT / "artifacts/data/pilot", parts=[("clean", "train"), ("clean", "val"), ("clean", "test")])
    comps = np.array(sim.a["composition_id"])
    from .build import COMPOSITIONS
    rng = np.random.default_rng(seed)
    twins = {}
    for iid, (kind, key, X) in item_twin.items():
        if kind == "feff":
            chi, msg = feff_chi.get(key, (None, "missing"))
            if chi is None:
                why[iid] = f"FEFF twin unavailable: {msg}"
                continue
            twins[iid] = (chi, msg)
        else:
            cand = np.flatnonzero(comps == COMPOSITIONS.index(key))
            j = int(rng.choice(cand))
            chi = sim.a["chi_pt"][j] if X == "Pt" else sim.a["chi_m"][j]
            if not np.all(np.isfinite(chi)):
                why[iid] = "nanoparticle twin without this edge"
                continue
            twins[iid] = (chi.astype(np.float32), f"Concord Forge {sim.a['sample_id'][j]} ({key})")
    return twins, why


def build_exafs(workers=24):
    from .realism import cond_exafs
    rows, res = process_xas(workers)
    twins, why = xas_twins(rows, res, workers)
    items = []
    for r in rows:
        iid = r["item_id"]
        if iid not in res:
            continue
        comp, oxide = comp6_from_formula(r["composition"])
        if comp is None:
            comp, oxide = np.zeros(6, np.float32), 0.0
            comp[EDGES5.index(EDGE_EL[r["edge_or_wavelength"]])] = 1.0
        items.append(dict(row=r, chi=res[iid]["chi"], twin=twins.get(iid, (None, why.get(iid, "")))[0],
                          twin_src=twins.get(iid, (None, why.get(iid, "no twin")))[1], comp=comp, oxide=oxide))
    groups = [f"{it['row']['edge_or_wavelength']}|{re.sub(r'[^a-z0-9]', '', it['row']['material'].lower())[:40]}" for it in items]
    split = split_by_group(groups, seed=20261022)
    has_twin = np.array([it["twin"] is not None for it in items])
    k2 = K_STORE ** 2
    x = np.stack([(it["chi"] * k2)[K_MODEL] for it in items]).astype(np.float32)
    tw = np.stack([((it["twin"] if it["twin"] is not None else np.zeros_like(it["chi"])) * k2)[K_MODEL] for it in items]).astype(np.float32)
    edge_idx = np.array([EDGES5.index(EDGE_EL[it["row"]["edge_or_wavelength"]]) for it in items])
    cond = cond_exafs(edge_idx, np.stack([it["comp"] for it in items]), np.array([it["oxide"] for it in items]))
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.array(split)
    # items without a twin cannot be used for GAN training (no matched fake input); they stay in the evaluation pool
    sp_gan = np.where(has_twin | (sp == "eval"), sp, "unused")
    np.savez(OUT / "exafs.npz", x=x, twin=tw, has_twin=has_twin, cond=cond.astype(np.float32), split=sp_gan,
             item_id=np.array([it["row"]["item_id"] for it in items]), material=np.array([it["row"]["material"] for it in items]),
             source=np.array([it["row"]["source"] for it in items]), licence=np.array([it["row"]["licence"] for it in items]),
             edge=np.array([it["row"]["edge_or_wavelength"] for it in items]),
             sample_type=np.array([it["row"]["sample_type"] for it in items]),
             twin_src=np.array([str(it["twin_src"]) for it in items]))
    with open(OUT / "exafs_items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "source", "licence", "edge", "material", "sample_type", "split", "twin"])
        for it, s_ in zip(items, sp_gan):
            r = it["row"]
            w.writerow([r["item_id"], r["source"], r["licence"], r["edge_or_wavelength"], r["material"], r["sample_type"], s_, it["twin_src"]])
    from collections import Counter
    print("EXAFS real items:", len(items), "by edge:", dict(Counter(it["row"]["edge_or_wavelength"] for it in items)),
          "split:", dict(Counter(sp_gan)), "with twin:", int(has_twin.sum()), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "exafs":
        build_exafs()


def calibrate_exafs_twins(min_corr=0.5):
    """Align each XAS twin to its real spectrum with the two conventions that are not labels: the E0 reference (one
    deltaE0 in -12..12 eV, the usual free EXAFS fit parameter) and one global amplitude factor (0.6..1.4). Twins that
    still correlate below min_corr are marked unused for GAN training / C2ST."""
    z = dict(np.load(OUT / "exafs.npz", allow_pickle=False))
    k = np.linspace(2.0, 14.75, 256)
    x, tw = z["x"], z["twin"].copy()
    de0s = np.arange(-12.0, 12.01, 0.5)
    best_e0, best_a, corr = np.zeros(len(x)), np.ones(len(x)), np.full(len(x), np.nan)
    for i in range(len(x)):
        if not z["has_twin"][i]:
            continue
        chi_t = tw[i] / k ** 2
        best = (-2.0, 0.0, None)
        for de in de0s:
            kk = np.sqrt(np.clip(k ** 2 + de * 0.262468, 0, None))   # twin evaluated at the shifted wavenumber
            y = np.interp(kk, k, chi_t, left=0.0, right=0.0) * k ** 2
            c = np.corrcoef(x[i], y)[0, 1] if y.std() > 0 else -1
            if c > best[0]:
                best = (c, de, y)
        c, de, y = best
        a = float(np.clip(np.dot(x[i], y) / max(np.dot(y, y), 1e-12), 0.6, 1.4))
        tw[i] = (a * y).astype(np.float32)
        best_e0[i], best_a[i], corr[i] = de, a, c
    split = z["split"].copy()
    bad = z["has_twin"] & (corr < min_corr)
    split = np.where(bad & (split != "eval"), "unused", split)
    z["twin"], z["split"] = tw, split
    z["twin_de0"], z["twin_amp"], z["twin_corr"] = best_e0, best_a, corr
    z["twin_ok"] = z["has_twin"] & ~bad
    np.savez(OUT / "exafs.npz", **z)
    from collections import Counter
    print("calibrated twins: median corr", float(np.nanmedian(corr)), "median |dE0|", float(np.nanmedian(np.abs(best_e0))),
          "split:", dict(Counter(split)), "eval with usable twin:", int(((split == "eval") & z["twin_ok"]).sum()), flush=True)
