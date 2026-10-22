"""Phase 3 (PDF): real nanoparticle G(r) -> model window (r 0.5-30.08 A), hold-out split and physics twins.

Real data: x-ray G(r) of metal nanoparticles / clusters with open licences (colloidal Au nanoparticles, Au144 clusters,
Pt nanoparticles). Bulk standards, neutron data, difference PDFs and oxide / molecular clusters are not used (they are
outside the simulated domain or not comparable).
Twin: Debye simulation (forge.simulate) of a particle of the same element; the twin library spans shapes and sizes
and the twin of each real G(r) is the library member with the highest correlation (r = 1.5-30 A), simulated with the
real data's Qmax (from the file metadata when stated, otherwise 24 A^-1) and Qdamp 0.04.
"""
import csv
import math
import re
from collections import Counter

import numpy as np

from .paths import CONFIGS, ROOT, load_yaml
from .realdata import OUT, RAW, comp6_from_formula, load_catalog, split_by_group

LATTICE = {"Au": 4.0782, "Pt": 3.9218}
U_ISO = {"Au": 0.0085, "Pt": 0.0050}


def element_of(row):
    t = f"{row['material']} {row['composition']}"
    if re.search(r"\bAu|Au1|gold", t, re.I):
        return "Au"
    if re.search(r"\bPt|platinum", t, re.I):
        return "Pt"
    return None


def read_gr(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            t = line.strip()
            if not t or t[0] in "#[;":
                continue
            parts = re.split(r"[\s,]+", t)
            try:
                vals = [float(p) for p in parts if p]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals[:2])
    if len(rows) < 100:
        return None
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def twin_library(device="cuda:0"):
    """{(element, shape, n_atoms): (G on the stored r grid for qmax in QMAXES)} built with the dataset simulator."""
    from ase.cluster import Decahedron, Icosahedron, Octahedron
    from .simulate import GpuSim
    inst = load_yaml(CONFIGS / "instruments.yaml")
    sim = GpuSim(inst, device)
    lib = []
    for el in ("Au", "Pt"):
        a = LATTICE[el]
        shapes = []
        for c in range(1, 12):
            shapes.append(("cuboctahedron", Octahedron(el, 2 * c + 1, c, latticeconstant=a).get_positions()))
        for n in range(2, 12):
            shapes.append(("icosahedron", Icosahedron(el, n, latticeconstant=a).get_positions()))
        for p in range(2, 9):
            shapes.append(("decahedron", Decahedron(el, p, p, 1, latticeconstant=a).get_positions()))
        for pos_name, pos in shapes:
            if len(pos) > 12000:
                continue
            types = np.zeros(len(pos), int)
            H, nb = sim.histograms(pos, types, 1)
            U = {el: U_ISO[el]}
            for qmax in (20.0, 22.0, 24.0, 26.0, 28.0):
                G, _, _ = sim.pdf(H, nb, [el], np.array([float(len(pos))]), U, 2.5, 0.35,
                                  dict(qmin=0.8, qmax=qmax, qdamp=0.04, qbroad=0.0))
                lib.append(dict(el=el, shape=pos_name, n=len(pos), qmax=qmax, G=G.astype(np.float32)))
    return lib, sim.g.r_pdf


def build_pdf():
    from .data import view
    from .realism import cond_pdf
    rows = [r for r in load_catalog("pdf") if r["x_kind"] == "r_A"]
    lib, r_store = twin_library()
    items, why = [], Counter()
    for r in rows:
        el = element_of(r)
        notes = r["notes"]
        if el is None or "neutron" in notes.lower() or "d-PDF" in r["material"] or "difference" in notes.lower():
            why["not an x-ray Au / Pt nanoparticle G(r)"] += 1
            continue
        g = read_gr(RAW / r["file_path"])
        if g is None:
            why["unreadable"] += 1
            continue
        rr, G = g
        if rr.max() < 29.9 or rr.min() > 1.0:
            why["r range"] += 1
            continue
        y = np.interp(r_store, rr, G, left=0.0, right=0.0)
        m = re.search(r"qmax=([0-9.]+)", notes)
        qmax = float(m.group(1)) if m else 24.0
        # twin: best-correlated library member of the same element with the nearest Qmax
        qs = min((20.0, 22.0, 24.0, 26.0, 28.0), key=lambda q: abs(q - qmax))
        cand = [e for e in lib if e["el"] == el and e["qmax"] == qs]
        sel = (r_store >= 1.5) & (r_store <= 30.0)
        yy = y[sel] - y[sel].mean()
        corr = lambda e: float(np.dot(yy, e["G"][sel] - e["G"][sel].mean()) /
                               (np.linalg.norm(yy) * np.linalg.norm(e["G"][sel] - e["G"][sel].mean()) + 1e-12))
        best = max(cand, key=corr)
        comp, _ = comp6_from_formula(el)
        items.append(dict(item_id=r["item_id"], source=r["source"], licence=r["licence"], material=r["material"],
                          group=f"{r['source']}|{re.sub(r'[^a-z0-9]', '', r['material'].lower())[:30]}",
                          y=y, twin=best["G"], twin_src=f"Debye {el} {best['shape']} N={best['n']} Qmax={qs}",
                          comp=comp, qmax=qmax, corr=corr(best)))
        why["ok"] += 1
    print("PDF real:", dict(why), flush=True)
    # AuNP files come from one self-driving-lab campaign (one particle per experiment): split by file, not by source
    groups = [it["item_id"] if it["source"] == "github_scatterlab_AuNP" else it["group"] for it in items]
    split = split_by_group(groups, seed=20261022)
    # a real G(r) whose best twin correlates < 0.5 is either a failed measurement or outside the simulated domain
    split = [s_ if it["corr"] >= 0.5 else "unused" for s_, it in zip(split, items)]
    x = view("pdf", np.stack([it["y"] for it in items]))
    tw = view("pdf", np.stack([it["twin"] for it in items]))
    cond = cond_pdf(np.stack([it["comp"] for it in items]), np.array([it["qmax"] for it in items]),
                    np.full(len(items), 0.04))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pdf.npz", x=x.astype(np.float32), twin=tw.astype(np.float32), has_twin=np.ones(len(items), bool),
             cond=cond.astype(np.float32), split=np.array(split), item_id=np.array([it["item_id"] for it in items]),
             material=np.array([it["material"] for it in items]), source=np.array([it["source"] for it in items]),
             licence=np.array([it["licence"] for it in items]), twin_src=np.array([it["twin_src"] for it in items]))
    with open(OUT / "pdf_items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "source", "licence", "material", "split", "twin", "twin_corr"])
        for it, s_ in zip(items, split):
            w.writerow([it["item_id"], it["source"], it["licence"], it["material"], s_, it["twin_src"], f"{it['corr']:.3f}"])
    print("PDF real items:", len(items), "split:", dict(Counter(split)),
          "by source:", dict(Counter(it["source"] for it in items)), flush=True)


if __name__ == "__main__":
    build_pdf()
