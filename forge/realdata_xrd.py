"""Phase 3 (XRD): real laboratory patterns -> Cu K-alpha model window, hold-out split and physics twins.

1) Pt / Pt-alloy catalysts and nanoparticles from open deposits (any laboratory wavelength; mapped to the Cu K-alpha
   2theta scale through Q). Twin: a Concord Forge simulation of the same composition class (size bins 0-2).
2) opXRD laboratory Cu K-alpha patterns with a single-phase structure label covering 20-80 deg 2theta. Twin: pymatgen
   XRDCalculator powder pattern of the labelled structure (K-alpha1/2, pseudo-Voigt, labelled crystallite size).
Output: artifacts/data/real/xrd.npz (fields as for exafs.npz) and xrd_items.csv.
"""
import csv
import json
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .paths import ROOT
from .realdata import OUT, RAW, comp6_from_formula, load_catalog, nanoparticle_class, split_by_group

WL_CU = 1.540562


def parse_wavelength(s):
    s = str(s).replace(" ", "")
    if "CoKa" in s or "CoK" in s:
        return 1.78897
    if "CuKa" in s or "CuK" in s:
        return 1.5406
    m = re.search(r"([0-9]\.[0-9]+)", s)
    return float(m.group(1)) if m else None


def read_xy_any(path):
    """(x, intensity) from csv / xy / ras / txt exports: first long numeric block; the intensity column is the one
    named Intensity / Counts in the header line above the block (PANalytical CSV: Angle, TimePerStep, Intensity, ESD),
    otherwise the second column, and never a constant column."""
    rows, header, last_text = [], None, None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.strip()
            if not t or t[0] in "#*;[":
                if rows and t.startswith("*RAS_INT_END"):
                    break
                if t and t[0] not in "*;[":
                    last_text = t.lstrip("#").strip()
                continue
            parts = [p for p in re.split(r"[\s,;]+", t) if p]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                if len(rows) > 50:
                    break
                last_text = t
                continue
            if len(vals) >= 2:
                if not rows:
                    header = last_text
                rows.append(vals)
    if len(rows) < 100:
        return None
    n = min(len(r) for r in rows)
    a = np.array([r[:n] for r in rows])
    col = 1
    if header:
        names = [h.strip().lower() for h in re.split(r"[,;	]+|\s{2,}", header) if h.strip()]
        for j, nm in enumerate(names[:n]):
            if "intens" in nm or "count" in nm or nm in ("i", "int", "y"):
                col = j
                break
    if a[:, col].std() == 0:
        varying = [j for j in range(1, n) if a[:, j].std() > 0]
        if not varying:
            return None
        col = varying[-1] if n > 2 else varying[0]
    return a[:, 0], a[:, col]


def to_cu_window(x, y, x_kind, wavelength):
    """Pattern -> intensity on the model grid (Cu K-alpha 2theta 20-80 deg), or None if not covered."""
    from .data import TTH_MODEL
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    if x_kind == "q":
        q = x
    else:
        if wavelength is None:
            return None
        q = 4 * math.pi * np.sin(np.radians(x / 2)) / wavelength
    tth_cu = np.degrees(2 * np.arcsin(np.clip(q * WL_CU / (4 * math.pi), 0, 1)))
    if tth_cu.min() > 20.5 or tth_cu.max() < 79.5:
        return None
    out = np.interp(TTH_MODEL, tth_cu, y)
    if not np.all(np.isfinite(out)) or out.max() <= 0:
        return None
    return out


def opxrd_structure(label_json):
    """pymatgen Structure of a single-phase opXRD label, or (None, None)."""
    from pymatgen.core import Lattice, Structure
    lab = json.loads(label_json) if isinstance(label_json, str) else label_json
    phases = lab.get("phases") or []
    if len(phases) != 1:
        return None, None
    ph = json.loads(phases[0]) if isinstance(phases[0], str) else phases[0]
    lat = [float(v) for v in re.findall(r"[-0-9.]+", ph["lattice"])][:6]
    basis = json.loads(ph["basis"]) if isinstance(ph["basis"], str) else ph["basis"]
    species, coords = [], []
    for b in basis:
        b = json.loads(b) if isinstance(b, str) else b
        el = re.sub(r"[0-9+\-]", "", b["symbol"])
        species.append({el: float(b.get("occupancy", 1) or 1)})
        coords.append([b["x"], b["y"], b["z"]])
    return Structure(Lattice.from_parameters(*lat), species, coords), lab.get("crystallite_size_nm")


def _opxrd_job(rel):
    from .data import TTH_MODEL
    from .twins import xrd_twin
    try:
        with open(RAW / "opxrd" / "extracted" / rel, encoding="utf-8") as fh:
            d = json.load(fh)
        y = to_cu_window(d["two_theta_values"], d["intensities"], "two_theta_deg", WL_CU)
        if y is None:
            return rel, None, "coverage"
        s, size = opxrd_structure(d["label"])
        if s is None:
            return rel, None, "multi-phase or no label"
        tw = xrd_twin(s, TTH_MODEL, crystallite_nm=float(size) if size else 50.0)
        if not np.all(np.isfinite(tw)) or tw.max() <= 0:
            return rel, None, "twin failed"
        return rel, dict(y=y.astype(np.float32), twin=tw.astype(np.float32), formula=s.composition.reduced_formula), "ok"
    except Exception as e:
        return rel, None, f"error {type(e).__name__}"


def build_xrd(max_opxrd=1500, workers=24, seed=0):
    from .build import COMPOSITIONS
    from .data import Dataset, view
    from .realism import cond_xrd
    rng = np.random.default_rng(seed)
    items = []
    cat = [r for r in load_catalog("xrd") if r["source"] != "opxrd"]
    sim = Dataset(ROOT / "artifacts/data/pilot", parts=[("clean", "train"), ("clean", "val"), ("clean", "test")])
    comps = np.array(sim.a["composition_id"])
    xr = view("xrd", sim.a["xrd_cu"])
    why = Counter()
    for r in cat:
        npc = nanoparticle_class(dict(r, sample_type="catalyst" if r["sample_type"] in ("compound", "") else r["sample_type"]))
        if npc is None:
            why["not Pt / Pt-alloy nanoparticle"] += 1
            continue
        xy = read_xy_any(RAW / r["file_path"])
        if xy is None:
            why["unreadable"] += 1
            continue
        y = to_cu_window(xy[0], xy[1], r["x_kind"], parse_wavelength(r["edge_or_wavelength"]))
        if y is None:
            why["coverage or unknown wavelength"] += 1
            continue
        if np.std(y / max(y.max(), 1e-12)) < 1e-3:
            why["degenerate (constant) pattern"] += 1
            continue
        cand = np.flatnonzero((comps == COMPOSITIONS.index(npc)) & (sim.a["size_bin"] <= 2))
        j = int(rng.choice(cand))
        comp, _ = comp6_from_formula(r["composition"] or npc)
        if comp is None:
            comp, _ = comp6_from_formula(npc)
        items.append(dict(item_id=r["item_id"], source=r["source"], licence=r["licence"], material=r["material"],
                          group=re.sub(r"\s*\[.*\]", "", r["material"]), y=y, twin=xr[j],
                          twin_src=f"Concord Forge {sim.a['sample_id'][j]} ({npc})", comp=comp))
        why["ok"] += 1
    print("XRD Pt-related:", dict(why), flush=True)
    ix = list(csv.DictReader(open(RAW / "opxrd" / "_opxrd_index.csv", encoding="utf-8")))
    cat_ids = {r["file_path"].replace("opxrd/extracted/", ""): r for r in load_catalog("xrd") if r["source"] == "opxrd"}
    ok = [r for r in ix if r["wavelength1"] and 1.538 <= float(r["wavelength1"]) <= 1.545 and r["n_phases"] == "1"
          and float(r["tt_min"]) <= 20 and float(r["tt_max"]) >= 80 and r["rel_path"] in cat_ids]
    ok = [ok[i] for i in rng.permutation(len(ok))[:max_opxrd]]
    print(f"opXRD candidates: {len(ok)}", flush=True)
    why2 = Counter()
    with ProcessPoolExecutor(workers) as ex:
        for rel, out, w in ex.map(_opxrd_job, [r["rel_path"] for r in ok], chunksize=4):
            why2[w.split(" ")[0]] += 1
            if out is None:
                continue
            c = cat_ids[rel]
            comp, _ = comp6_from_formula(out["formula"])
            if comp is None:
                continue
            items.append(dict(item_id=c["item_id"], source="opxrd", licence=c["licence"], material=out["formula"],
                              group=out["formula"], y=out["y"], twin=out["twin"],
                              twin_src="pymatgen XRDCalculator (opXRD label)", comp=comp))
    print("opXRD:", dict(why2), flush=True)
    split = split_by_group([f"{it['source'] == 'opxrd'}|{it['group']}" for it in items], seed=20261022)
    x = view("xrd", np.stack([it["y"] for it in items]))
    tw = view("xrd", np.stack([it["twin"] for it in items]))
    cond = cond_xrd(np.stack([it["comp"] for it in items]))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "xrd.npz", x=x.astype(np.float32), twin=tw.astype(np.float32), has_twin=np.ones(len(items), bool),
             cond=cond.astype(np.float32), split=np.array(split), item_id=np.array([it["item_id"] for it in items]),
             material=np.array([it["material"] for it in items]), source=np.array([it["source"] for it in items]),
             licence=np.array([it["licence"] for it in items]), twin_src=np.array([it["twin_src"] for it in items]))
    with open(OUT / "xrd_items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "source", "licence", "material", "split", "twin"])
        for it, s_ in zip(items, split):
            w.writerow([it["item_id"], it["source"], it["licence"], it["material"], s_, it["twin_src"]])
    print("XRD real items:", len(items), "split:", dict(Counter(split)),
          "Pt-related by split:", dict(Counter(s for s, it in zip(split, items) if it["source"] != "opxrd")), flush=True)


if __name__ == "__main__":
    build_xrd()
