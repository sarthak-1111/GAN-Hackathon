"""FEFF6L (bundled with xraylarch) runner, path reader and a vectorized chi(k) path sum.

The chi(k) formula reproduces larch.xafs.feffdat.FeffPathGroup._calc_chi (xraylarch 2026.3.1) with e0 = ei = 0 and no
third / fourth cumulants:
    p   = rep + i / lam,   pp = p^2
    chi = Im{ degen * s02 * amp / (k (reff + dR)^2) *
              exp(-2 reff Im(p) - 2 pp sigma2 + i (2 k reff + pha + 2 p (dR - 2 sigma2 / reff))) }
"""
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import larch
from larch.xafs.feffdat import FeffDatFile

FEFF6L = Path(larch.__file__).parent / "bin" / "win64" / "feff6l.exe"
class _Z(dict):
    """Atomic numbers for any element symbol (pymatgen periodic table)."""

    def __missing__(self, el):
        from pymatgen.core import Element
        z = Element(el).Z
        self[el] = z
        return z


Z = _Z(O=8, Fe=26, Co=27, Ni=28, Cu=29, Pt=78)
HOLE = {"K": 1, "L1": 2, "L2": 3, "L3": 4}
EDGE_OF = {"Pt": "L3", "Co": "K", "Ni": "K", "Fe": "K", "Cu": "K"}


def write_feff_inp(folder, pos, species, absorber_index, rmax, nleg, title="Concord Forge", criteria=(0.0, 0.0)):
    """Write feff.inp with the absorber at the origin; one unique potential per scatterer element.

    Path filtering is off by default (CRITERIA 0 0): FEFF's importance filter is relative to the strongest path
    including degeneracy, so it would keep weak multiple-scattering paths in a symmetric bulk cluster but drop the
    same paths in a low-symmetry nanoparticle cluster, where every path has degeneracy ~1.
    """
    pos = np.asarray(pos, float) - np.asarray(pos[absorber_index], float)
    X = species[absorber_index]
    # unique potentials only for elements present as scatterers (FEFF rejects potentials without atoms)
    els = sorted({s for i, s in enumerate(species) if i != absorber_index})
    ipot = {el: i + 1 for i, el in enumerate(els)}
    lines = [f"TITLE {title}", f"HOLE {HOLE[EDGE_OF[X]]} 1.0", "CONTROL 1 1 1 1", "PRINT 0 0 0 0",
             f"RMAX {rmax:.3f}", f"NLEG {nleg}", f"CRITERIA {criteria[0]} {criteria[1]}", "POTENTIALS",
             f"   0  {Z[X]:3d}  {X}"]
    for el in els:
        lines.append(f"   {ipot[el]}  {Z[el]:3d}  {el}")
    lines.append("ATOMS")
    order = np.argsort(np.linalg.norm(pos, axis=1))
    for i in order:
        p = 0 if i == absorber_index else ipot[species[i]]
        lines.append(f"  {pos[i, 0]:10.5f} {pos[i, 1]:10.5f} {pos[i, 2]:10.5f}  {p}  {species[i]}")
    lines.append("END")
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "feff.inp").write_text("\n".join(lines) + "\n")


def run_feff(folder, timeout=1800):
    """Run feff6l.exe in folder; returns list of path dicts (arrays + geometry)."""
    folder = Path(folder)
    r = subprocess.run([str(FEFF6L)], cwd=str(folder), capture_output=True, text=True, timeout=timeout)
    files = sorted(folder.glob("feff[0-9][0-9][0-9][0-9].dat"))
    if not files:
        raise RuntimeError(f"FEFF produced no paths in {folder}: {r.stdout[-500:]} {r.stderr[-500:]}")
    return [read_path(f) for f in files]


def read_path(f):
    d = FeffDatFile(str(f))
    geom = [(g[0], np.array([float(v) for v in g[4:7]])) for g in d.geom]  # (label, xyz) absorber first
    return dict(file=Path(f).name, nleg=int(d.nleg), degen=float(d.degen), reff=float(d.reff),
                k=np.asarray(d.k, float), amp=np.asarray(d.amp, float), pha=np.asarray(d.pha, float),
                rep=np.asarray(d.rep, float), lam=np.asarray(d.lam, float),
                atoms=[g[0] for g in geom], coords=np.array([g[1] for g in geom]))


def path_chi(k, amp, pha, rep, lam, reff, dr, sigma2, degen=1.0, s02=1.0, xp=np):
    """Vectorized larch path equation. Arrays broadcast; amp/pha/rep/lam already interpolated onto k."""
    p = rep + 1j / lam
    pp = p * p
    cchi = xp.exp(-2 * reff * p.imag - 2 * pp * sigma2 + 1j * (2 * k * reff + pha + 2 * p * (dr - 2 * sigma2 / reff)))
    return (degen * s02 * amp * cchi / (k * (reff + dr) ** 2)).imag


def interp_path(path, k):
    return {q: np.interp(k, path["k"], path[q]) for q in ("amp", "pha", "rep", "lam")}


def chi_from_paths(paths, k, sigma2_fn, s02=1.0):
    """Sum of chi(k) over FEFF paths with sigma2_fn(path) -> sigma^2 (no deltaR)."""
    k = np.asarray(k, float)
    ks = np.where(k == 0, 1e-6, k)
    chi = np.zeros_like(ks)
    for p in paths:
        ip = interp_path(p, ks)
        chi += path_chi(ks, ip["amp"], ip["pha"], ip["rep"], ip["lam"], p["reff"], 0.0, sigma2_fn(p), p["degen"], s02)
    chi[k == 0] = 0.0
    return chi


class FeffWork:
    """Temporary FEFF working folder that is removed afterwards unless keep=True."""

    def __init__(self, base, name, keep=False):
        self.dir = Path(base) / name
        self.keep = keep

    def __enter__(self):
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        return self.dir

    def __exit__(self, *exc):
        if not self.keep:
            shutil.rmtree(self.dir, ignore_errors=True)
