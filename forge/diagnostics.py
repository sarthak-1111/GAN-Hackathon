"""Phase 5 GAN diagnostics -> artifacts/reports/gan_diagnostics.md (+ .json).

Per technique, on the held-out real spectra (split 'eval', never used for training or model selection):
  realism     classifier two-sample test (C2ST, 5-fold, gradient-boosted trees on spectrum / derivative / spectral
              features): real vs (a) raw twins, (b) classically augmented twins, (c) GAN(twins). 50 % = indistinguishable.
              Frechet distance in the embedding space of that technique's encoder from predictor (a), seed 1 (never saw
              GAN output), after PCA to d = min(32, n/3) dimensions.
  diversity   for 64 fixed inputs and 16 latent draws each: mean pairwise L2 distance between outputs and the variance
              of each learned artifact field (background, gain, noise scale) across draws (mode-collapse check).
  label preservation   standard-tool re-fits of 500 Concord Forge test spectra before and after the realism layer:
              XRD peak fit (lattice parameter), PDF diffpy fit (lattice parameter, particle diameter), EXAFS larch first
              shell fit (N, R). A spectrum fails if the realism layer moves a fitted label by more than the tolerance.
"""
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import augment
from .data import Dataset, view
from .paths import MODELS, REPORTS, ROOT
from .realism import LENGTH, load_real, load_realism, sim_conditions, sim_views

TOL = {"xrd": dict(a=0.005), "pdf": dict(a=0.005, d_rel=0.10), "exafs": dict(N_rel=0.10, R=0.01)}


def features(x):
    x = np.asarray(x, np.float32)
    step = max(1, x.shape[1] // 192)
    d = np.diff(x, axis=1)
    spec = np.log(np.abs(np.fft.rfft(d, axis=1))[:, 1:97] + 1e-6)
    return np.concatenate([x[:, ::step], d[:, ::step] * 20, spec, x.std(1, keepdims=True), np.abs(d).mean(1, keepdims=True)], 1)


def c2st(real, fake, seed=0):
    """5-fold classifier two-sample test accuracy (gradient-boosted trees). Returns (mean, std over folds).
    The minimum leaf size scales with the training-fold size (default 20, at least 2): with the default, trees cannot
    split at all for the 10 held-out real PDFs, and the test would return exactly 50 % regardless of the data."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    n = min(len(real), len(fake))
    if n < 6:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    real = real[rng.permutation(len(real))[:n]]
    fake = fake[rng.permutation(len(fake))[:n]]
    X = np.concatenate([features(real), features(fake)])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                             min_samples_leaf=max(2, min(20, len(tr) // 10)), random_state=seed)
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def frechet(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)  # scipy >= 1.18: no 'disp' argument
    covmean = np.real(covmean)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


@torch.no_grad()
def encoder_embed(predictor, view_name, x, dev, bs=256, dim=32):
    from .predictor import with_derivative
    out = []
    for s in range(0, len(x), bs):
        xb = torch.as_tensor(np.asarray(x[s:s + bs], np.float32), device=dev)
        out.append(predictor.enc[view_name](with_derivative(xb)).float().cpu().numpy())
    e = np.concatenate(out)
    # PCA to `dim` dimensions for a stable covariance with few real samples
    e = e - e.mean(0)
    u, s_, vt = np.linalg.svd(e, full_matrices=False)
    return e, vt[:dim]


@torch.no_grad()
def apply_gan(model, x, cond, dev, seed=0, z=None):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    out = []
    for s in range(0, len(x), 256):
        xb = torch.as_tensor(np.asarray(x[s:s + 256], np.float32), device=dev)
        cb = torch.as_tensor(np.asarray(cond[s:s + 256], np.float32), device=dev)
        out.append(model.G(xb, cb, gen=g).float().cpu().numpy())
    return np.concatenate(out)


def classical(x, view_name, dev, seed=0):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    return augment.classical(torch.as_tensor(np.asarray(x, np.float32), device=dev), view_name, g, "train").cpu().numpy()


def _refit(args):
    from .refit import fit_exafs, fit_pdf, fit_xrd
    kind, payload = args
    try:
        if kind == "xrd":
            tth, y, a0 = payload
            return fit_xrd(tth, y, a0=a0)
        if kind == "pdf":
            r, G, qmax, qdamp, comp, a0, d0 = payload
            return fit_pdf(r, G, qmax, qdamp, comp, a0, d0, rmax=25.0)
        k, chi, X = payload
        return fit_exafs(k, chi, X, 0.85 if X == "Pt" else 0.8)
    except Exception as e:
        return dict(ok=False, error=str(e)[:100])


def label_preservation(technique, model, dev, n=500, seed=0):
    from .data import TTH_MODEL
    ds = Dataset(ROOT / "artifacts" / "data", parts=[("clean", "test")])
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ds.n, size=min(n, ds.n), replace=False))
    if technique == "exafs":
        x, _ = sim_views(ds, "exafs", "pt")
        cond = sim_conditions(ds, "exafs", "pt")
    else:
        x, _ = sim_views(ds, technique)
        cond = sim_conditions(ds, technique)
    x, cond = x[idx], cond[idx]
    y = apply_gan(model, x, cond, dev, seed)
    jobs = []
    for j, i in enumerate(idx):
        pm = json.loads(json.dumps(ds.meta[i]["particle"]))
        sp = ds.meta[i]["spec"]
        for arr in (x[j], y[j]):
            if technique == "xrd":
                jobs.append(("xrd", (TTH_MODEL, arr, pm["lattice_parameter"])))
            elif technique == "pdf":
                r = 0.5 + 0.02 * np.arange(len(arr))
                jobs.append(("pdf", (r, arr, sp["instruments"]["pdf"]["qmax"], sp["instruments"]["pdf"]["qdamp"],
                                     pm["composition_true"], pm["lattice_parameter"], 10 * pm["diameter_nm"])))
            else:
                k = np.linspace(2.0, 14.75, 256)
                kk = np.arange(321) * 0.05
                chi = np.interp(kk, k, arr / k ** 2, left=0.0, right=0.0)
                jobs.append(("exafs", (kk, chi, "Pt")))
    with ProcessPoolExecutor(24) as ex:
        fits = list(ex.map(_refit, jobs, chunksize=4))
    rows = []
    for j, i in enumerate(idx):
        f0, f1 = fits[2 * j], fits[2 * j + 1]
        pm = ds.meta[i]["particle"]
        r = dict(sample_id=str(ds.a["sample_id"][i]), ok=bool(f0.get("ok") and f1.get("ok")),
                 fit_ok_clean=bool(f0.get("ok")), fit_ok_gan=bool(f1.get("ok")))
        if technique == "xrd":
            r.update(a_clean=f0.get("a"), a_gan=f1.get("a"), a_label=pm["lattice_parameter"])
            r["fail"] = (not r["ok"]) or abs(r["a_gan"] - r["a_clean"]) > TOL["xrd"]["a"]
            r["recover_clean"] = r["ok"] and abs(r["a_clean"] - r["a_label"]) < 0.02
            r["recover_gan"] = r["ok"] and abs(r["a_gan"] - r["a_label"]) < 0.02
        elif technique == "pdf":
            r.update(a_clean=f0.get("a"), a_gan=f1.get("a"), d_clean=f0.get("d_nm"), d_gan=f1.get("d_nm"),
                     a_label=pm["lattice_parameter"], d_label=pm["diameter_nm"])
            r["fail"] = (not r["ok"]) or abs(r["a_gan"] - r["a_clean"]) > TOL["pdf"]["a"] or \
                abs(r["d_gan"] - r["d_clean"]) > TOL["pdf"]["d_rel"] * r["d_clean"]
            r["recover_clean"] = r["ok"] and abs(r["a_clean"] - r["a_label"]) < 0.02
            r["recover_gan"] = r["ok"] and abs(r["a_gan"] - r["a_label"]) < 0.02
        else:
            sh = [s for s in pm["shells"] if s["absorber"] == "Pt" and s["shell"] == 1 and s["neighbor"] != "O"]
            n1 = sum(s["N"] for s in sh)
            r1 = sum(s["N"] * s["R"] for s in sh) / max(n1, 1e-9)
            r.update(N_clean=f0.get("N"), N_gan=f1.get("N"), R_clean=f0.get("R"), R_gan=f1.get("R"), N_label=n1, R_label=r1)
            r["fail"] = (not r["ok"]) or abs(r["N_gan"] - r["N_clean"]) > TOL["exafs"]["N_rel"] * abs(r["N_clean"]) or \
                abs(r["R_gan"] - r["R_clean"]) > TOL["exafs"]["R"]
            r["recover_clean"] = r["ok"] and abs(r["R_clean"] - r1) < 0.03
            r["recover_gan"] = r["ok"] and abs(r["R_gan"] - r1) < 0.03
        rows.append(r)
    return rows


def diversity(model, x, cond, dev, n_in=64, n_z=16, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(n_in, len(x)), replace=False)
    pd_, var = [], {"bg": [], "gain": [], "nscale": []}
    with torch.no_grad():
        for i in idx:
            xb = torch.as_tensor(np.repeat(x[i:i + 1], n_z, 0), device=dev)
            cb = torch.as_tensor(np.repeat(cond[i:i + 1], n_z, 0), device=dev)
            g = torch.Generator(device=dev)
            g.manual_seed(int(i))
            y, parts = model.G(xb, cb, gen=g, return_parts=True)
            y = y.float().cpu().numpy()
            d = np.linalg.norm(y[:, None] - y[None], axis=-1)
            pd_.append(d[np.triu_indices(n_z, 1)].mean() / (np.linalg.norm(x[i]) + 1e-9))
            for k in var:
                var[k].append(float(parts[k].float().var(0).mean().cpu()))
    return dict(rel_pairwise_distance=float(np.mean(pd_)), **{f"var_{k}": float(np.mean(v)) for k, v in var.items()})


GRIDS_MODEL = {"xrd": ("2θ (°, Cu Kα)", lambda L: np.linspace(20, 80, L)), "pdf": ("r (Å)", lambda L: 0.5 + 0.02 * np.arange(L)),
               "exafs": ("k (Å⁻¹)", lambda L: np.linspace(2.0, 14.75, L))}


@torch.no_grad()
def example_figure(technique, model, R, ev, dev, n=3, seed=0):
    """reports/figures/gan_examples_<t>.png: held-out real spectra, their physics twins and GAN(twin), plus the learned
    background and gain fields."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    pick = ev[np.sort(rng.choice(len(ev), size=min(n, len(ev)), replace=False))]
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x = torch.as_tensor(R["twin"][pick], device=dev)
    c = torch.as_tensor(R["cond"][pick], device=dev)
    y, parts = model.G(x, c, gen=g, return_parts=True)
    y = y.float().cpu().numpy()
    bg, gain = parts["bg"].float().cpu().numpy(), parts["gain"].float().cpu().numpy()
    xl, gridf = GRIDS_MODEL[technique]
    grid = gridf(y.shape[1])
    fig, axes = plt.subplots(len(pick), 2, figsize=(12, 2.8 * len(pick)), gridspec_kw=dict(width_ratios=[3, 1]), squeeze=False)
    for row, i in enumerate(pick):
        ax = axes[row, 0]
        ax.plot(grid, R["x"][i], color="black", lw=1.0, label="real (held out)")
        ax.plot(grid, R["twin"][i], color="tab:blue", lw=0.9, alpha=0.8, label="physics twin")
        ax.plot(grid, y[row], color="tab:red", lw=0.8, alpha=0.8, label="GAN(twin)")
        ax.set_title(f"{str(R['material'][i])[:70]}  [{str(R['source'][i])[:35]}]", fontsize=8)
        ax.set_xlabel(xl, fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(fontsize=7, loc="best")
        ax2 = axes[row, 1]
        ax2.plot(grid, bg[row], color="tab:green", lw=1, label="background")
        ax2.set_xlabel(xl, fontsize=8)
        ax2.tick_params(labelsize=7)
        ax3 = ax2.twinx()
        ax3.plot(grid, gain[row], color="tab:purple", lw=1, label="gain")
        ax3.set_ylim(0.88, 1.12)
        ax3.tick_params(labelsize=7)
        if row == 0:
            ax2.set_title("learned background (green) and gain (purple)", fontsize=8)
    fig.tight_layout()
    path = REPORTS / "figures" / f"gan_examples_{technique}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return f"figures/gan_examples_{technique}.png"


def run(technique, predictor_ckpt=None, device="cuda:0"):
    dev = torch.device(device)
    model = load_realism(MODELS, technique, dev)
    if model is None:
        return dict(technique=technique, shipped=False)
    R = load_real(technique)
    ok = R["twin_ok"] if "twin_ok" in R else np.ones(len(R["x"]), bool)
    ev = np.flatnonzero((R["split"] == "eval") & ok)
    real, twin, cond = R["x"][ev], R["twin"][ev], R["cond"][ev]
    vname = {"xrd": "xrd", "pdf": "pdf", "exafs": "exafs_pt"}[technique]
    gan = apply_gan(model, twin, cond, dev)
    cla = classical(twin, vname, dev)
    out = dict(technique=technique, n_eval_real=int(len(ev)), c2st={}, frechet={})
    out["figure"] = example_figure(technique, model, R, ev, dev)
    for name, fake in (("raw_twin", twin), ("classical", cla), ("gan", gan)):
        m, s = c2st(real, fake)
        out["c2st"][name] = dict(acc=m, fold_std=s)
    if predictor_ckpt is not None:
        from .predictor import Predictor
        ck = torch.load(predictor_ckpt, map_location=dev, weights_only=False)
        P = Predictor().to(dev)
        P.load_state_dict(ck["model"])
        P.eval()
        allx = np.concatenate([real, twin, cla, gan])
        out["frechet_dim"] = int(min(32, max(2, len(real) // 3)))  # n >= 3 x dim samples per covariance
        out["frechet_encoder"] = str(Path(predictor_ckpt).relative_to(ROOT)) if Path(predictor_ckpt).is_relative_to(ROOT) else str(predictor_ckpt)
        e, basis = encoder_embed(P, vname, allx, dev, dim=out["frechet_dim"])
        z = e @ basis.T
        n = len(real)
        zr = z[:n]
        for j, name in enumerate(("raw_twin", "classical", "gan")):
            out["frechet"][name] = frechet(zr, z[n * (j + 1): n * (j + 2)])
    out["diversity"] = diversity(model, twin, cond, dev)
    rows = label_preservation(technique, model, dev)
    ok = [r for r in rows if r["ok"]]
    out["label_preservation"] = dict(
        n=len(rows), n_fit_ok=len(ok), n_fit_failed_clean=int(sum(not r["fit_ok_clean"] for r in rows)),
        n_fit_failed_gan=int(sum(r["fit_ok_clean"] and not r["fit_ok_gan"] for r in rows)),
        # conservative: a failed fit of either spectrum counts as a failure
        failure_rate=float(np.mean([r["fail"] for r in rows])),
        # labels moved beyond tolerance, among spectra where both fits converged
        failure_rate_fit_ok=float(np.mean([r["fail"] for r in ok])) if ok else float("nan"),
        recover_clean=float(np.mean([r["recover_clean"] for r in rows])),
        recover_gan=float(np.mean([r["recover_gan"] for r in rows])), tolerance=TOL[technique])
    (REPORTS / f"gan_diag_{technique}.json").write_text(json.dumps(dict(out, rows=rows), indent=1, default=float))
    return out


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), indent=1, default=float))
