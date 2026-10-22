"""Realism GAN training, inference and diagnostics (one model per technique: xrd, pdf, exafs).

Data contract (written by forge.realdata):
  artifacts/data/real/<tech>.npz
    x       (n, L) real spectra in the model view (forge.data.view conventions, model grid)
    twin    (n, L) simulated "twin" of each real spectrum (same material / edge / instrument, physics only)
    cond    (n, C) condition vectors (see COND below)
    split   (n,)   'train' | 'val' | 'eval'   (eval = 20 % hold-out, never used for training or model selection)
    item_id, material, source, licence  (n,) strings
The generator is trained on twins -> real (unpaired adversarial loss; twins keep the content of the fake batch close
to the real batch so the discriminator has to look at artifacts). Preservation and magnitude losses are also applied
to Concord Forge nanoparticle simulations, the domain the model is used on.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_LIBUV", "0")

import numpy as np
import torch
import torch.nn.functional as F

from .data import EDGES, Dataset, view
from .gan import (Discriminator, FirstShellFilter, Generator, d_hinge, exafs_preservation, g_hinge, magnitude_penalty,
                  r1_penalty, xcorr_preservation)
from .paths import DATA, MODELS, ROOT, load_yaml, seed_everything, sha256_file

REAL = DATA / "real"
ELS = ["Pt", "Co", "Ni", "Fe", "Cu"]
COND_DIM = {"xrd": 6, "pdf": 8, "exafs": 12}
LENGTH = {"xrd": 1536, "pdf": 1480, "exafs": 256}
MAG_SCALE = {"xrd": 0.05, "pdf": 0.05, "exafs": 0.1}


# ------------------------------------------------------------------------------------------- conditions
def cond_xrd(comp6):
    """[Pt, Co, Ni, Fe, Cu, other] atomic fractions."""
    return np.asarray(comp6, np.float32)


def cond_pdf(comp6, qmax, qdamp):
    c = np.asarray(comp6, np.float32)
    return np.concatenate([c, ((np.asarray(qmax, np.float32) - 25.0) / 5.0)[:, None],
                           ((np.asarray(qdamp, np.float32) - 0.04) / 0.02)[:, None]], 1)


def cond_exafs(edge_idx, comp6, oxide):
    e = np.eye(len(EDGES), dtype=np.float32)[np.asarray(edge_idx)]
    return np.concatenate([e, np.asarray(comp6, np.float32), np.asarray(oxide, np.float32)[:, None]], 1)


def sim_conditions(ds, technique, edge="pt"):
    """Condition vectors for Concord Forge simulations (Dataset)."""
    comp5 = ds.composition_fractions()
    comp6 = np.concatenate([comp5, np.zeros((ds.n, 1), np.float32)], 1)
    if technique == "xrd":
        return cond_xrd(comp6)
    if technique == "pdf":
        return cond_pdf(comp6, ds.a["inst_pdf"][:, 1], ds.a["inst_pdf"][:, 2])
    if technique == "exafs":
        oxide = np.array([1.0 if (m["spec"]["fault"] in ("pt_oxide_shell", "mixed_oxidation")) else 0.0 for m in ds.meta])
        idx = np.zeros(ds.n, int) if edge == "pt" else ds.m_edge()
        return cond_exafs(idx, comp6, oxide)
    raise ValueError(technique)


def sim_views(ds, technique, edge="pt"):
    S, P = ds.spectra()
    if technique == "xrd":
        return S["xrd"], P["xrd"]
    if technique == "pdf":
        return S["pdf"], P["pdf"]
    return (S["exafs_pt"], P["exafs_pt"]) if edge == "pt" else (S["exafs_m"], P["exafs_m"])


# ------------------------------------------------------------------------------------------- model wrapper
class RealismModel:
    """EMA generator of one technique, applied to model-view batches."""

    def __init__(self, technique, G, device):
        self.t, self.G, self.dev = technique, G.eval().to(device), device

    @torch.no_grad()
    def apply(self, x, view_name, ctx, g):
        """x: (B, L) model-view tensor of technique view_name; ctx: batch context dict from the predictor trainer."""
        B = x.shape[0]
        comp = ctx["comp"].float()
        comp6 = torch.cat([comp, torch.zeros(B, 1, device=x.device)], 1)
        if self.t == "xrd":
            c = comp6
        elif self.t == "pdf":
            ip = ctx["inst_pdf"].float()
            c = torch.cat([comp6, ((ip[:, 1:2] - 25.0) / 5.0), ((ip[:, 2:3] - 0.04) / 0.02)], 1)
        else:
            idx = torch.zeros(B, dtype=torch.long, device=x.device) if view_name == "exafs_pt" else ctx["medge"].long()
            c = torch.cat([F.one_hot(idx, len(EDGES)).float(), comp6, torch.zeros(B, 1, device=x.device)], 1)
        z = torch.randn(B, self.G.z_dim, device=x.device, generator=g)
        y = self.G(x, c, z=z, gen=g)
        if self.t == "xrd":
            y = y / torch.clamp(y.max(1, keepdim=True).values, min=1e-6)
        elif self.t == "pdf":
            y = y / torch.clamp(y.abs().max(1, keepdim=True).values, min=1e-6)
        return y


def load_realism(model_dir, technique, device):
    d = Path(model_dir) / f"realism_{technique}"
    f = d / "generator_ema.pt"
    if not f.exists():
        return None
    ck = torch.load(f, map_location=device, weights_only=False)
    G = Generator(LENGTH[technique], COND_DIM[technique], glitches=(technique == "exafs"))
    G.load_state_dict(ck["state_dict"])
    return RealismModel(technique, G, device)


# ------------------------------------------------------------------------------------------- training
def load_real(technique):
    z = np.load(REAL / f"{technique}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def ema_update(ema, model, decay):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.lerp_(p, 1 - decay)
        for be, b in zip(ema.buffers(), model.buffers()):
            be.copy_(b)


def train(technique, cfg_path):
    cfg = load_yaml(cfg_path)
    seed_everything(cfg["seed"])
    dev = torch.device("cuda", cfg["gpu"])
    torch.cuda.set_device(dev)
    if cfg.get("gpu_mem_gib"):
        torch.cuda.set_per_process_memory_fraction(
            min(0.95, cfg["gpu_mem_gib"] * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory), dev)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_name = cfg.get("out_name", f"realism_{technique}")  # alternative runs of one technique get their own directory
    out = MODELS / run_name
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(cfg_path, out / "config.yaml")
    R = load_real(technique)
    ok = R["twin_ok"] if "twin_ok" in R else np.ones(len(R["x"]), bool)
    tr = np.flatnonzero((R["split"] == "train") & ok)
    va = np.flatnonzero((R["split"] == "val") & ok)
    print(f"[realism {technique}] real train {len(tr)}, val {len(va)}", flush=True)
    L, C = LENGTH[technique], COND_DIM[technique]
    x_real = torch.as_tensor(R["x"], device=dev)
    x_twin = torch.as_tensor(R["twin"], device=dev)
    c_real = torch.as_tensor(R["cond"], device=dev)
    # Concord Forge simulations (the target domain) for preservation / magnitude regularization
    sim = Dataset(ROOT / cfg["sim_data"], parts=[("clean", "train")])
    if technique == "exafs":
        xs_pt, _ = sim_views(sim, "exafs", "pt")
        xs_m, pm = sim_views(sim, "exafs", "m")
        cs = np.concatenate([sim_conditions(sim, "exafs", "pt"), sim_conditions(sim, "exafs", "m")[pm]])
        xs = np.concatenate([xs_pt, xs_m[pm]])
    else:
        xs, _ = sim_views(sim, technique)
        cs = sim_conditions(sim, technique)
    x_sim = torch.as_tensor(xs, device=dev)
    c_sim = torch.as_tensor(cs, device=dev)
    G = Generator(L, C, glitches=(technique == "exafs")).to(dev)
    D = Discriminator(C).to(dev)
    G_ema = Generator(L, C, glitches=(technique == "exafs")).to(dev)
    G_ema.load_state_dict(G.state_dict())
    oG = torch.optim.Adam(G.parameters(), cfg["lr_g"], betas=(0.0, 0.99))
    oD = torch.optim.Adam(D.parameters(), cfg["lr_d"], betas=(0.0, 0.99))
    filt = FirstShellFilter(np.linspace(2.0, 14.75, 256)).to(dev) if technique == "exafs" else None
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(ROOT / "runs" / run_name))
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["batch"]
    t0 = time.time()
    history, best = [], None
    patience = 0
    start = 0
    if (out / "last_full.pt").exists() and cfg.get("resume", True):
        ck = torch.load(out / "last_full.pt", map_location=dev, weights_only=False)
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
        G_ema.load_state_dict(ck["G_ema"])
        oG.load_state_dict(ck["oG"])
        oD.load_state_dict(ck["oD"])
        start = ck["step"]
        if (out / "history.json").exists():
            history = json.loads((out / "history.json").read_text())
            scores = [(abs(h["c2st_acc"] - 0.5) + 0.05 * h.get("feat_fd_ratio", 0.0), h["step"]) for h in history
                      if h["c2st_acc"] == h["c2st_acc"]]
            if scores:
                best = min(scores)
                patience = sum(1 for h in history if h["step"] > best[1])
        rng = np.random.default_rng(cfg["seed"] + start)
        print(f"[realism {technique}] resumed at step {start}", flush=True)
    for step in range(start, cfg["max_steps"]):
        ir = torch.as_tensor(rng.choice(tr, B), device=dev)
        it = torch.as_tensor(rng.choice(tr, B), device=dev)
        isim = torch.as_tensor(rng.integers(0, len(x_sim), cfg.get("sim_batch", 64)), device=dev)
        real, creal = x_real[ir], c_real[ir]
        twin, ctwin = x_twin[it], c_real[it]
        # ---- discriminator
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                fake = G(twin, ctwin, gen=gen)
            ld = d_hinge(D(real, creal), D(fake.float(), ctwin))
        loss_d = ld
        if step % cfg["r1_every"] == 0:
            loss_d = loss_d + cfg["r1_gamma"] / 2 * r1_penalty(D, real, creal) * cfg["r1_every"]
        oD.zero_grad(set_to_none=True)
        loss_d.backward()
        oD.step()
        # ---- generator
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fake, parts = G(twin, ctwin, gen=gen, return_parts=True)
            lg = g_hinge(D(fake.float(), ctwin))
            fs, ps = G(x_sim[isim], c_sim[isim], gen=gen, return_parts=True)
        fake, fs = fake.float(), fs.float()
        parts = {k: v.float() for k, v in parts.items()}
        ps = {k: v.float() for k, v in ps.items()}
        if technique == "exafs":
            lp = exafs_preservation(filt, twin, fake) + exafs_preservation(filt, x_sim[isim], fs)
        else:
            lp = xcorr_preservation(twin, fake) + xcorr_preservation(x_sim[isim], fs)
        lm = magnitude_penalty(parts, MAG_SCALE[technique]) + magnitude_penalty(ps, MAG_SCALE[technique])
        loss_g = lg + cfg["w_pres"] * lp + cfg["w_mag"] * lm
        oG.zero_grad(set_to_none=True)
        loss_g.backward()
        oG.step()
        ema_update(G_ema, G, cfg["ema"])
        if step % 100 == 0:
            tb.add_scalar("loss/d", float(ld), step)
            tb.add_scalar("loss/g_adv", float(lg), step)
            tb.add_scalar("loss/preserve", float(lp), step)
            tb.add_scalar("loss/magnitude", float(lm), step)
        if (step + 1) % cfg["ckpt_every"] == 0 or step == cfg["max_steps"] - 1:
            diag = quick_diagnostics(G_ema, R, va, technique, dev, gen)
            diag["step"] = step + 1
            history.append(diag)
            for k, v in diag.items():
                tb.add_scalar(f"diag/{k}", v, step)
            torch.save(dict(state_dict=G_ema.state_dict(), step=step + 1, cfg=cfg), out / f"ema_step{step + 1:06d}.pt")
            tmp = out / "last_full.tmp"
            torch.save(dict(G=G.state_dict(), D=D.state_dict(), G_ema=G_ema.state_dict(), oG=oG.state_dict(),
                            oD=oD.state_dict(), step=step + 1), tmp)
            os.replace(tmp, out / "last_full.pt")
            # checkpoint score: C2ST distance from chance, with the artifact-feature distance (relative to the raw
            # twins) as a smoother tiebreaker when the C2ST saturates
            score = abs(diag["c2st_acc"] - 0.5) + 0.05 * diag.get("feat_fd_ratio", 0.0)
            print(f"[realism {technique}] step {step + 1} D {float(ld):.3f} G {float(lg):.3f} pres {float(lp):.4f} "
                  f"mag {float(lm):.3f} | val c2st {diag['c2st_acc']:.3f} (twin {diag.get('c2st_twin_acc', float('nan')):.3f}) "
                  f"featFD ratio {diag.get('feat_fd_ratio', float('nan')):.3f} ({time.time() - t0:.0f}s)", flush=True)
            if best is None or score < best[0] - 0.005:
                best = (score, step + 1)
                torch.save(dict(state_dict=G_ema.state_dict(), step=step + 1, cfg=cfg, diag=diag), out / "generator_ema.pt")
                patience = 0
            else:
                patience += 1
            (out / "history.json").write_text(json.dumps(history, indent=1))
            if patience >= cfg["patience"]:
                print(f"[realism {technique}] early stop at step {step + 1}; best step {best[1]}", flush=True)
                break
    tb.close()
    (out / "generator_ema.sha256").write_text(sha256_file(out / "generator_ema.pt"))
    return best


@torch.no_grad()
def generate(G, x, c, dev, gen, bs=256):
    out = []
    for s in range(0, len(x), bs):
        xb = torch.as_tensor(x[s:s + bs], device=dev)
        cb = torch.as_tensor(c[s:s + bs], device=dev)
        out.append(G(xb, cb, gen=gen).float().cpu().numpy())
    return np.concatenate(out)


def c2st(a, b, seed=0):
    """Classifier two-sample test accuracy (5-fold logistic regression on spectrum + derivative features)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    def feats(x):
        x = np.asarray(x, np.float32)
        step = max(1, x.shape[1] // 256)
        xd = np.diff(x, axis=1)
        return np.concatenate([x[:, ::step], xd[:, ::step] * 10, np.log(np.abs(np.fft.rfft(xd, axis=1))[:, 1:65] + 1e-6)], 1)

    X = np.concatenate([feats(a), feats(b)])
    y = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    if min(len(a), len(b)) < 5:
        return float("nan")
    accs = []
    skf = StratifiedKFold(n_splits=min(5, min(len(a), len(b))), shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
    return float(np.mean(accs))


def artifact_features(x):
    """Artifact-sensitive features: noise residual level and spectral slope, derivative statistics, low-frequency
    background (mean of the smoothed spectrum in 8 segments)."""
    from scipy.ndimage import gaussian_filter1d
    x = np.asarray(x, np.float64)
    sm = gaussian_filter1d(x, 6, axis=1)
    res = x - sm
    p = np.abs(np.fft.rfft(res, axis=1)) ** 2
    nb = p.shape[1]
    bands = [p[:, int(nb * a):int(nb * b)].mean(1) for a, b in ((0.05, 0.15), (0.15, 0.4), (0.4, 1.0))]
    seg = np.array_split(gaussian_filter1d(x, 40, axis=1), 8, axis=1)
    d = np.diff(x, axis=1)
    feats = [np.log(res.std(1) + 1e-9)] + [np.log(b + 1e-12) for b in bands] +             [np.log(np.abs(d).mean(1) + 1e-9), np.log(d.std(1) + 1e-9)] + [s_.mean(1) for s_ in seg]
    return np.stack(feats, 1)


def feature_fd(a, b):
    from .diagnostics import frechet
    fa, fb = artifact_features(a), artifact_features(b)
    mu, sd = fa.mean(0), fa.std(0) + 1e-9
    return frechet((fa - mu) / sd, (fb - mu) / sd)


def quick_diagnostics(G, R, idx, technique, dev, gen):
    if len(idx) < 4:
        return dict(c2st_acc=float("nan"))
    fake = generate(G, R["twin"][idx], R["cond"][idx], dev, gen)
    real = R["x"][idx]
    fd_g, fd_t = feature_fd(real, fake), feature_fd(real, R["twin"][idx])
    return dict(c2st_acc=c2st(real, fake), c2st_twin_acc=c2st(real, R["twin"][idx]), feat_fd=fd_g, feat_fd_twin=fd_t,
                feat_fd_ratio=fd_g / max(fd_t, 1e-9))


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
