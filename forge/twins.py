"""Physics "twins" of real reference spectra: the same material simulated without instrument artifacts.

Twins give the realism GAN fake batches whose physical content matches the real batch, so the discriminator has to
judge artifacts (background, gain, noise, glitches) rather than structure.

XAS   bulk crystals (foils, oxides, simple compounds): COD structure for the formula, FEFF6L (RMAX 6 A, NLEG 4, no path
      filter) for every symmetry-inequivalent absorber site, chi(k) with the dataset's thermal model, averaged with
      site multiplicities. Catalysts / nanoparticles / Pt alloys: a Concord Forge simulation of the same composition.
XRD   single-phase opXRD patterns with a structure label: pymatgen XRDCalculator (Cu K-alpha1 and K-alpha2), pseudo-Voigt
      profiles from the labelled crystallite size (default 50 nm) and a typical laboratory resolution.
PDF   handled in forge.realdata (Debye simulation of the reported nanoparticle / bulk material).
"""
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import requests

from .feff import FeffWork, chi_from_paths, run_feff, write_feff_inp
from .paths import FEFF_DIR, STRUCT_DIR

COD_CACHE = STRUCT_DIR / "twin_cod"
K_STORE = np.round(np.arange(321) * 0.05, 4)
U_DEFAULT = {"O": 0.008, "H": 0.02, "C": 0.008, "N": 0.008, "S": 0.006, "Cl": 0.008}
EDGE_ELEMENTS = {"Pt": "L3", "Co": "K", "Ni": "K", "Fe": "K", "Cu": "K"}


def hill_formula(formula):
    """'Co3O4' -> 'Co3 O4' (COD Hill notation: C, H first if carbon present, then alphabetical)."""
    from pymatgen.core import Composition
    c = Composition(formula).reduced_composition
    els = sorted(c.elements, key=lambda e: e.symbol)
    if any(e.symbol == "C" for e in els):
        els = [e for e in els if e.symbol == "C"] + [e for e in els if e.symbol == "H"] + \
              [e for e in els if e.symbol not in ("C", "H")]
    parts = []
    for e in els:
        n = c[e]
        parts.append(e.symbol + ("" if abs(n - 1) < 1e-6 else (str(int(round(n))) if abs(n - round(n)) < 1e-6 else f"{n:g}")))
    return " ".join(parts)


PREFERRED_SG = {"Co": "P63/mmc", "Fe": "Im-3m", "Ni": "Fm-3m", "Cu": "Fm-3m", "Pt": "Fm-3m"}


def cod_structure(formula):
    """(pymatgen Structure, COD id) for a formula from COD (cached), or (None, reason)."""
    from pymatgen.io.cif import CifParser
    COD_CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9]", "_", formula)
    meta_f = COD_CACHE / f"{key}.json"
    if meta_f.exists():
        meta = json.loads(meta_f.read_text())
    else:
        try:
            hf = hill_formula(formula)
        except Exception as e:
            return None, f"unparsable formula: {e}"
        hits, err = None, None
        for attempt in range(5):  # the COD server drops some TLS connections; retry with backoff
            try:
                r = requests.get("https://www.crystallography.net/cod/result",
                                 params={"formula": hf, "format": "json"}, timeout=120)
                hits = r.json() if r.ok else []
                break
            except Exception as e:
                err = e
                time.sleep(3 * (attempt + 1))
        if hits is None:
            return None, f"COD search failed: {err}"
        meta = [dict(file=str(h["file"]), sg=(h.get("sg") or "").replace(" ", ""), title=h.get("title") or "",
                     cellpressure=h.get("cellpressure"), celltemp=h.get("celltemp"), year=h.get("year")) for h in hits]
        meta_f.write_text(json.dumps(meta))
    if not meta:
        return None, "no COD entry"
    bad = re.compile(r"pressure|compress|temperat|expansion|theory", re.I)
    cands = [m for m in meta if not bad.search(m["title"]) and not m.get("cellpressure")]
    if not cands:
        cands = meta
    pref = PREFERRED_SG.get(formula)
    if pref:
        cands = sorted(cands, key=lambda m: m["sg"] != pref)
    else:
        # most common space group among candidates first
        from collections import Counter
        cnt = Counter(m["sg"] for m in cands)
        cands = sorted(cands, key=lambda m: -cnt[m["sg"]])
    for m in cands[:6]:
        f = COD_CACHE / f"{m['file']}.cif"
        if not f.exists():
            for attempt in range(4):
                try:
                    rr = requests.get(f"https://www.crystallography.net/cod/{m['file']}.cif", timeout=120)
                    rr.raise_for_status()
                    f.write_bytes(rr.content)
                    break
                except Exception:
                    time.sleep(2 * (attempt + 1))
        if not f.exists():
            continue
        try:
            s = CifParser(str(f), occupancy_tolerance=1.05).parse_structures(primitive=False)[0]
        except Exception:
            continue
        if s.is_ordered:
            return s, m["file"]
    return None, "no parsable ordered COD structure"


def feff_bulk_chi(structure, absorber, U, delta2=3.0, floor=0.35, s02=1.0, rmax=6.0, nleg=4, tag="twin"):
    """chi(k) on K_STORE averaged over inequivalent absorber sites (FEFF6L, thermal model as in the dataset)."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sym = SpacegroupAnalyzer(structure, symprec=0.05).get_symmetrized_structure()
    groups = [g for g in sym.equivalent_sites if g[0].specie.symbol == absorber]
    if not groups:
        return None
    tot = sum(len(g) for g in groups)
    chi = np.zeros_like(K_STORE)
    for gi, g in enumerate(groups):
        site = g[0]
        nb = [n for n in structure.get_neighbors(site, 8.0) if n.specie.symbol != "H"]  # H omitted (standard practice)
        pos = np.array([[0.0, 0.0, 0.0]] + [n.coords - site.coords for n in nb])
        sp = [absorber] + [n.specie.symbol for n in nb]
        if len(set(sp[1:])) > 6:  # FEFF6: at most 7 unique potentials including the absorber
            return None
        if len(pos) > 900:  # FEFF6 cluster limit: keep the nearest atoms
            o = np.argsort(np.linalg.norm(pos, axis=1))[:900]
            pos, sp = pos[o], [sp[i] for i in o]
        with FeffWork(FEFF_DIR / "tmp", f"{tag}_{absorber}_{gi}") as fd:
            write_feff_inp(fd, pos, sp, 0, rmax=rmax, nleg=nleg, title=tag)
            paths = run_feff(fd)

        def s2(p):
            us = np.mean([U.get(a, U_DEFAULT.get(a, 0.006)) for a in p["atoms"][1:]])
            return (U.get(absorber, 0.005) + us) * max(floor, 1.0 - delta2 / p["reff"] ** 2)

        chi += len(g) / tot * chi_from_paths(paths, K_STORE, s2, s02=s02)
    return chi


def xrd_twin(structure, tth, crystallite_nm=50.0, inst_fwhm=0.08, ka2_ratio=0.5, eta=0.5):
    """Cu K-alpha1/2 powder pattern of a crystal on the 2theta grid (pymatgen structure factors, pseudo-Voigt)."""
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    y = np.zeros_like(tth, dtype=float)
    for lam, w in ((1.540562, 1.0), (1.544390, ka2_ratio)):
        pat = XRDCalculator(wavelength=lam).get_pattern(structure, two_theta_range=(float(tth.min()) - 2, float(tth.max()) + 2))
        for x0, I in zip(pat.x, pat.y):
            th = math.radians(x0 / 2)
            size_fwhm = math.degrees(0.9 * lam / (crystallite_nm * 10.0 * math.cos(th)))
            fw = math.sqrt(inst_fwhm ** 2 + size_fwhm ** 2)
            d = (tth - x0) / fw
            y += w * I * (eta / (1 + 4 * d ** 2) + (1 - eta) * np.exp(-4 * math.log(2) * d ** 2))
    return y
