"""Label-preservation tests for the realism layer (hard rule 1: the GAN may only add artifacts).

The checks run on the shipped realism generators when they exist (artifacts/models/realism_<t>/generator_ema.pt) and
on freshly initialized generators otherwise, so the guarantees of the architecture itself are always tested.

    .venv\\Scripts\\python -m pytest tests -q
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from forge.data import Dataset, TTH_MODEL  # noqa: E402
from forge.gan import FirstShellFilter, Generator  # noqa: E402
from forge.paths import MODELS  # noqa: E402
from forge.realism import COND_DIM, LENGTH, load_realism, sim_conditions, sim_views  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA = ROOT / "artifacts" / "data" if (ROOT / "artifacts" / "data" / "splits.json").exists() else ROOT / "artifacts" / "data" / "pilot"


def generators():
    out = []
    for t in ("xrd", "pdf", "exafs"):
        m = load_realism(MODELS, t, DEV)
        if m is not None:
            out.append((t, "trained", m.G))
        torch.manual_seed(0)
        out.append((t, "untrained", Generator(LENGTH[t], COND_DIM[t], glitches=(t == "exafs")).to(DEV).eval()))
    return out


GENS = generators()


@pytest.fixture(scope="module")
def sims():
    ds = Dataset(DATA, parts=[("clean", "test")], max_n=64)
    out = {}
    for t in ("xrd", "pdf"):
        x, _ = sim_views(ds, t)
        out[t] = (x, sim_conditions(ds, t))
    x, _ = sim_views(ds, "exafs", "pt")
    out["exafs"] = (x, sim_conditions(ds, "exafs", "pt"))
    return out


def run(G, x, c, seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    with torch.no_grad():
        y, parts = G(torch.as_tensor(x, device=DEV), torch.as_tensor(c, device=DEV), gen=g, return_parts=True)
    return y.cpu().numpy(), {k: v.cpu().numpy() for k, v in parts.items() if v.dim() == 2}


@pytest.mark.parametrize("t,kind,G", GENS, ids=[f"{t}-{k}" for t, k, _ in GENS])
def test_gain_bounded(t, kind, G, sims):
    x, c = sims[t]
    _, parts = run(G, x, c)
    assert parts["gain"].min() >= 0.9 - 1e-6 and parts["gain"].max() <= 1.1 + 1e-6


# The first version of this check (power fraction above a cut-off in a periodic FFT of the background) was invalid:
# FFT wrap-around leakage makes any background whose two ends differ fail it, a straight line included, and it only
# passed for untrained generators because their background is exactly zero. It was replaced by the two tests below.

@pytest.mark.parametrize("t,kind,G", GENS, ids=[f"{t}-{k}" for t, k, _ in GENS])
def test_background_in_spline_span(t, kind, G, sims):
    """The background is exactly a combination of the generator's 12 cubic B-spline functions (the structural
    guarantee; test_background_basis_resolution checks what that span can form)."""
    x, c = sims[t]
    _, parts = run(G, x, c)
    B = G.B_bg.detach().cpu().numpy().astype(np.float64)
    bg = parts["bg"].astype(np.float64)
    proj = bg @ np.linalg.pinv(B).T @ B.T
    resid = np.sqrt(((bg - proj) ** 2).mean(1)) / (np.sqrt((bg ** 2).mean(1)) + 1e-12)
    assert np.all(resid < 1e-4), f"background leaves the B-spline span (relative residual up to {resid.max():.1e})"


GRID = {"xrd": TTH_MODEL, "pdf": 0.5 + 0.02 * np.arange(LENGTH["pdf"]), "exafs": np.linspace(2.0, 14.75, LENGTH["exafs"])}
# Narrowest label-carrying feature per technique (FWHM, grid units):
#   xrd    1.1 deg: Scherrer width of the (111) reflection of the largest particles (8 nm Pt, Cu K-alpha)
#   pdf    0.3 A: nearest-neighbour peak (sigma^2 <= 0.01 A^2 plus Qmax termination broadening)
#   exafs  0.57 A^-1: one lobe (half period, pi / 2R) of the first-shell oscillation for R = 2.77 A
NARROWEST = {"xrd": 1.1, "pdf": 0.3, "exafs": math.pi / (2 * 2.77)}


@pytest.mark.parametrize("t", [
    "xrd", "pdf",
    pytest.param("exafs", marks=pytest.mark.xfail(strict=True, reason=(
        "12 cubic B-spline coefficients over k = 2-14.75 A^-1 can form bumps of 1.15 A^-1 FWHM, comparable to the "
        "first-shell period, so for EXAFS the basis alone does not guarantee first-shell preservation; that rests on "
        "the first-shell preservation loss and test_exafs_first_shell_phase_unchanged")))])
def test_background_basis_resolution(t):
    """The narrowest bump the background span can form (FWHM of the projection kernel onto the 12-function basis,
    away from the grid ends) is at least 3 x the narrowest label-carrying feature, so the background cannot create,
    remove or reshape peaks."""
    from forge.gan import bspline_basis
    x = GRID[t]
    B = bspline_basis(len(x), 12).numpy().astype(np.float64)
    P = B @ np.linalg.pinv(B)
    widths = []
    for j in range(len(x) // 8, len(x) - len(x) // 8):
        col = P[:, j]
        above = np.flatnonzero(col >= col.max() / 2)
        widths.append(x[above[-1]] - x[above[0]])
    assert min(widths) >= 3 * NARROWEST[t], f"narrowest background bump {min(widths):.3f} < 3 x {NARROWEST[t]:.3f}"


def peak_positions(y, grid, windows):
    out = []
    for lo, hi in windows:
        m = (grid >= lo) & (grid <= hi)
        out.append(grid[m][np.argmax(y[m])])
    return np.array(out)


@pytest.mark.parametrize("t,kind,G", [g for g in GENS if g[0] == "xrd"], ids=lambda v: str(v) if isinstance(v, str) else "")
def test_xrd_peak_positions_unchanged(t, kind, G, sims):
    """Main fcc reflections ((111), (200), (220)) stay at the same 2theta within 0.15 deg after the realism layer
    (the smoothed generated pattern is compared with the input; noise is averaged over 8 latent draws)."""
    x, c = sims["xrd"]
    ys = np.mean([run(G, x, c, seed=s)[0] for s in range(8)], axis=0)
    from scipy.ndimage import gaussian_filter1d
    windows = [(36.5, 42.5), (43.5, 49.0), (64.0, 71.0)]
    shifts = []
    for i in range(len(x)):
        p0 = peak_positions(gaussian_filter1d(x[i], 4), TTH_MODEL, windows)
        p1 = peak_positions(gaussian_filter1d(ys[i], 4), TTH_MODEL, windows)
        shifts.append(np.abs(p1 - p0))
    shifts = np.array(shifts)
    assert np.median(shifts) <= 0.05 and np.percentile(shifts, 95) <= 0.15, f"95th pct shift {np.percentile(shifts, 95):.3f} deg"


@pytest.mark.parametrize("t,kind,G", [g for g in GENS if g[0] == "pdf"], ids=lambda v: str(v) if isinstance(v, str) else "")
def test_pdf_first_peak_unchanged(t, kind, G, sims):
    x, c = sims["pdf"]
    ys = np.mean([run(G, x, c, seed=s)[0] for s in range(8)], axis=0)
    r = 0.5 + 0.02 * np.arange(x.shape[1])
    m = (r > 2.2) & (r < 3.2)
    d = np.abs(r[m][np.argmax(ys[:, m], 1)] - r[m][np.argmax(x[:, m], 1)])
    assert np.percentile(d, 95) <= 0.02 + 1e-6, f"first-peak shift up to {d.max():.3f} A"  # one 0.02 A grid step


@pytest.mark.parametrize("t,kind,G", [g for g in GENS if g[0] == "exafs"], ids=lambda v: str(v) if isinstance(v, str) else "")
def test_exafs_first_shell_phase_unchanged(t, kind, G, sims):
    """Fourier-filtered first shell (R 1.8-3.2 A): amplitude-weighted phase difference < 0.1 rad and first-shell |chi(R)|
    peak position unchanged within 0.02 A (averaged over 8 latent draws)."""
    x, c = sims["exafs"]
    ys = np.mean([run(G, x, c, seed=s)[0] for s in range(8)], axis=0)
    k = np.linspace(2.0, 14.75, 256)
    filt = FirstShellFilter(k)
    a = filt(torch.as_tensor(x)).numpy()
    b = filt(torch.as_tensor(ys)).numpy()
    w = np.abs(a) / np.abs(a).mean(1, keepdims=True)
    dphi = np.abs(np.angle(b * np.conj(a)))
    assert np.median((w * dphi).mean(1)) < 0.1
    rg = np.linspace(1.8, 3.2, 141)
    win = np.hanning(256)

    def ftpk(y):
        ft = ((y * win)[:, None, :] * np.exp(2j * k[None, None, :] * rg[None, :, None])).sum(-1)
        return rg[np.argmax(np.abs(ft), 1)]

    assert np.percentile(np.abs(ftpk(ys) - ftpk(x)), 95) <= 0.02 + 1e-9
