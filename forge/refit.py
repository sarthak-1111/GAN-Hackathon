"""Label recovery with standard tools (label-preservation diagnostic of the realism models).

XRD   lmfit: (111) + (200) pseudo-Voigt peaks with Cu K-alpha2 companions and a shared fcc lattice parameter,
      linear background, 33-52 deg 2theta -> lattice parameter, Scherrer size from the (111) FWHM.
PDF   diffpy.pdffit2: fcc model (mixed Pt / M occupancy for alloys), refining lattice parameter, scale, delta2, Uiso
      and the spherical particle diameter (spdiameter) with the sample's Qmax / Qdamp, r = 1.5-25 A.
EXAFS xraylarch feffit: one first-shell X-Pt path (FEFF6L), refining amplitude (N S0^2), deltaE0, deltaR, sigma^2 on
      k = 3-12 A^-1 (k^2, Hanning), R = 1.8-3.2 A.
"""
import io
import math
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from .paths import FEFF_DIR

LAM1, LAM2 = 1.540562, 1.544390


# ------------------------------------------------------------------------------------------------- XRD
def _pv(x, x0, fwhm, eta):
    g = np.exp(-4 * np.log(2) * ((x - x0) / fwhm) ** 2)
    l = 1 / (1 + 4 * ((x - x0) / fwhm) ** 2)
    return eta * l + (1 - eta) * g


def fit_xrd(tth, y, ka2_ratio=0.5, a0=3.9):
    import lmfit
    m = (tth >= 33) & (tth <= 52)
    x, yy = tth[m], y[m] / y[m].max()

    def model(p, x):
        a = p["a"].value
        out = p["b0"].value + p["b1"].value * (x - 42)
        for hkl, A, w in (((1, 1, 1), p["A1"].value, p["w1"].value), ((2, 0, 0), p["A2"].value, p["w2"].value)):
            d = a / math.sqrt(sum(h * h for h in hkl))
            t1 = 2 * math.degrees(math.asin(min(LAM1 / (2 * d), 0.999)))
            t2 = 2 * math.degrees(math.asin(min(LAM2 / (2 * d), 0.999)))
            out = out + A * (_pv(x, t1, w, p["eta"].value) + ka2_ratio * _pv(x, t2, w, p["eta"].value))
        return out

    p = lmfit.Parameters()
    p.add("a", a0, min=3.3, max=4.2)
    p.add("A1", 1.0, min=0)
    p.add("A2", 0.5, min=0)
    p.add("w1", 1.5, min=0.05, max=15)
    p.add("w2", 1.5, min=0.05, max=15)
    p.add("eta", 0.5, min=0, max=1)
    p.add("b0", 0.02)
    p.add("b1", 0.0)
    best = None
    for a_start in (a0 - 0.08, a0, a0 + 0.08):
        p["a"].value = a_start
        r = lmfit.minimize(lambda q: model(q, x) - yy, p, method="leastsq")
        if best is None or r.chisqr < best.chisqr:
            best = r
    a = best.params["a"].value
    th = math.asin(LAM1 * math.sqrt(3) / (2 * a))
    beta = math.radians(best.params["w1"].value)
    size_nm = 0.9 * LAM1 / (beta * math.cos(th)) / 10.0
    return dict(a=a, size_nm=size_nm, redchi=float(best.redchi), ok=bool(best.success))


# ------------------------------------------------------------------------------------------------- PDF
def fit_pdf(r, G, qmax, qdamp, comp, a0, d0, rmin=1.5, rmax=25.0):
    from diffpy.pdffit2 import PdfFit, redirect_stdout as pf_redirect
    pf_redirect(open(os.devnull, "w"))
    pf = PdfFit()
    els = [(el, x) for el, x in comp.items() if x > 1e-6 and el != "O"]
    lines = ["title fcc", "format pdffit", "scale 1.000000", "sharp 0.000000, 0.000000, 1.000000, 0.000000",
             "spcgr Fm-3m", f"cell {a0:.6f}, {a0:.6f}, {a0:.6f}, 90.000000, 90.000000, 90.000000",
             "dcell 0.0, 0.0, 0.0, 0.0, 0.0, 0.0", f"ncell 1, 1, 1, {4 * len(els)}", "atoms"]
    for (fx, fy, fz) in ((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)):
        for el, x in els:
            lines += [f"{el.upper():4s} {fx:.6f} {fy:.6f} {fz:.6f} {x:.4f}", "    0.0 0.0 0.0 0.0",
                      "    0.005 0.005 0.005", "    0.0 0.0 0.0", "    0.0 0.0 0.0", "    0.0 0.0 0.0"]
    pf.read_struct_string("\n".join(lines))
    m = (r >= rmin) & (r <= rmax)
    pf.read_data_lists("X", qmax, qdamp, list(r[m]), list(G[m]))
    pf.pdfrange(1, rmin, rmax)
    pf.constrain(pf.lat(1), "@1")
    pf.constrain(pf.lat(2), "@1")
    pf.constrain(pf.lat(3), "@1")
    pf.setpar(1, a0)
    pf.constrain(pf.pscale(), "@2")
    pf.setpar(2, 1.0)
    pf.constrain(pf.delta2(), "@3")
    pf.setpar(3, 2.0)
    for i in range(1, pf.num_atoms() + 1):
        for f in (pf.u11, pf.u22, pf.u33):
            pf.constrain(f(i), "@4")
    pf.setpar(4, 0.005)
    pf.constrain(pf.spdiameter(), "@5")
    pf.setpar(5, max(d0, 10.0))
    try:
        with redirect_stdout(io.StringIO()):
            pf.refine()
        return dict(a=pf.getpar(1), d_nm=pf.getpar(5) / 10.0, rw=pf.getrw(), ok=True)
    except Exception as e:
        return dict(a=np.nan, d_nm=np.nan, rw=np.nan, ok=False, error=str(e)[:200])


# ------------------------------------------------------------------------------------------------- EXAFS
_FIT_PATH_DIR = FEFF_DIR / "fit_paths"


def fit_path_file(X):
    """feff0001.dat of the first X-Pt shell (absorber X in an fcc Pt host), generated once and kept."""
    from .feff import run_feff, write_feff_inp
    f = _FIT_PATH_DIR / f"{X}_Pt" / "feff0001.dat"
    if f.exists():
        return f
    a = 3.9218
    g = np.arange(-4, 5)
    U = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    U = U[(U.sum(1) % 2) == 0]
    pos = U * a / 2
    pos = pos[np.linalg.norm(pos, axis=1) <= 6.0]
    sp = ["Pt"] * len(pos)
    i0 = int(np.argmin(np.linalg.norm(pos, axis=1)))
    sp[i0] = X
    write_feff_inp(f.parent, pos, sp, i0, rmax=2.9, nleg=2, title=f"fit path {X}-Pt")
    run_feff(f.parent)
    return f


def fit_exafs(k, chi, X, s02, kmin=3.0, kmax=12.0, rmin=1.8, rmax=3.2):
    from larch import Group
    from larch.fitting import guess, param_group
    from larch.xafs import feffit, feffit_dataset, feffit_transform, feffpath
    pars = param_group(amp=guess(8.0 * s02), del_e0=guess(0.0), del_r=guess(0.0), sig2=guess(0.005))
    path = feffpath(str(fit_path_file(X)), degen=1, s02="amp", e0="del_e0", deltar="del_r", sigma2="sig2")
    trans = feffit_transform(kmin=kmin, kmax=kmax, kweight=2, dk=1, window="hanning", rmin=rmin, rmax=rmax)
    data = Group(k=np.asarray(k, float), chi=np.asarray(chi, float))
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
            out = feffit(pars, feffit_dataset(data=data, paths={"p1": path}, transform=trans))
        P = out.params
        return dict(N=P["amp"].value / s02, R=path.reff + P["del_r"].value, sig2=P["sig2"].value,
                    e0=P["del_e0"].value, rfactor=out.rfactor, ok=True)
    except Exception as e:
        return dict(N=np.nan, R=np.nan, sig2=np.nan, e0=np.nan, rfactor=np.nan, ok=False, error=str(e)[:200])
