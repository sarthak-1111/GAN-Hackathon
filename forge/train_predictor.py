"""Train the multi-technique predictor with (a) classical augmentation or (b) GAN realism augmentation.

    python -m forge.train_predictor configs/predictor_a.yaml

Variant (a): forge.augment.classical(...) on every technique.
Variant (b): the trained realism generators (EMA weights) add learned artifacts to XRD (Cu), PDF and EXAFS; techniques
without a usable realism model fall back to the classical pipeline. Both variants share everything else (data,
model, schedule, seeds), so the comparison isolates the augmentation.
Single process per GPU; with several GPUs, gradients are averaged through shared host memory (forge.dist).
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
import torch.multiprocessing as mp

from . import augment
from .data import TECHNIQUES, Dataset
from .dist import SharedReducer, barrier, broadcast_module, init, sync_grads
from .paths import ROOT, load_yaml, seed_everything, sha256_file, sha256_json
from .predictor import REG_TARGETS, Predictor, decode, losses, technique_dropout


def to_dev(d, dev):
    return {k: torch.as_tensor(v, device=dev) for k, v in d.items()}


class RealismAugmenter:
    """Applies trained realism generators (forge.realism) to a batch; classical fallback per technique."""

    def __init__(self, model_dir, techniques, device):
        from .realism import load_realism
        self.models = {}
        for t in techniques:
            m = load_realism(Path(model_dir), t, device)
            if m is not None:
                self.models[t] = m

    KEY = {"xrd": "xrd", "pdf": "pdf", "exafs_pt": "exafs", "exafs_m": "exafs"}

    def has(self, t):
        return self.KEY.get(t) in self.models

    def __call__(self, t, x, cond_ctx, g):
        return self.models[self.KEY[t]].apply(x, t, cond_ctx, g)


def common_aug(x, t, g):
    """Augmentations shared by both variants (not artifacts): Gaussian broadening and normalization jitter."""
    B = x.shape[0]
    dev = x.device
    if t in ("xrd", "xrd_sync", "pdf"):
        x = augment.gaussian_broaden(x, torch.rand(B, 1, generator=g, device=dev) * (1.5 if t != "pdf" else 1.0))
    return x * (0.9 + 0.2 * torch.rand(B, 1, generator=g, device=dev))


def batch_augment(specs, variant, g, realism, ctx):
    """(a): classical pipeline; (b): shared non-artifact augmentation + learned artifacts where a realism model
    exists, otherwise exactly the classical pipeline of (a)."""
    out = {}
    for t, x in specs.items():
        if variant == "b" and realism is not None and realism.has(t):
            out[t] = realism(t, common_aug(x, t, g), ctx, g)
        else:
            out[t] = augment.classical(x, t, g, "train")
    return out


def worker(rank, world, port, cfg_path, reducer):
    cfg = load_yaml(cfg_path)
    gpus = cfg["gpus"]
    torch.cuda.set_device(gpus[rank])
    dev = torch.device("cuda", gpus[rank])
    if cfg.get("gpu_mem_gib"):
        frac = cfg["gpu_mem_gib"][gpus[rank]] * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(frac, 0.95), dev)
    if world > 1:
        init(rank, world, port)
    seed_everything(cfg["seed"] + rank)
    torch.backends.cudnn.benchmark = True
    out_dir = Path(cfg["out_dir"])
    data_dir = ROOT / cfg["data_dir"]
    tr = Dataset(data_dir, parts=[("clean", "train"), ("fault", "train")])
    va = Dataset(data_dir, parts=[("clean", "val"), ("fault", "val")])
    S, P = tr.spectra()
    T = tr.targets()
    meta = tr.metadata_vectors()
    medge = tr.m_edge()
    stats = {k: (float(np.nanmean(T[k])), float(np.nanstd(T[k]) + 1e-6)) for k in REG_TARGETS}
    ctx_all = dict(comp=tr.composition_fractions(), medge=medge, inst_pdf=tr.a["inst_pdf"], inst_cu=tr.a["inst_xrd_cu"],
                   inst_exafs=tr.a["inst_exafs"])
    model = Predictor().to(dev)
    broadcast_module(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["steps"]
    warm = cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    realism = RealismAugmenter(ROOT / cfg["realism_dir"], ["xrd", "pdf", "exafs"], dev) if cfg["variant"] == "b" else None
    g = torch.Generator(device=dev)
    g.manual_seed(cfg["seed"] * 1000 + rank)
    rng = np.random.default_rng(cfg["seed"] * 1000 + rank)
    params = [p for p in model.parameters()]
    tb = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        out_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(str(ROOT / "runs" / cfg["name"]))
        (out_dir / "config.json").write_text(json.dumps(dict(cfg=cfg, stats=stats, data_sha=[s["sha256"] for s in json.loads((data_dir / "splits.json").read_text())["shards"]]), indent=1))
    B = cfg["batch_per_gpu"]
    t0 = time.time()
    best = None
    for step in range(steps):
        idx = rng.integers(0, tr.n, B)
        specs = {t: torch.as_tensor(S[t][idx], device=dev) for t in TECHNIQUES}
        pres = {t: torch.as_tensor(P[t][idx], device=dev) for t in TECHNIQUES}
        ctx = {k: torch.as_tensor(v[idx], device=dev) for k, v in ctx_all.items()}
        with torch.no_grad():
            specs = batch_augment(specs, cfg["variant"], g, realism, ctx)
        pres = technique_dropout(pres, g, p=cfg["tech_dropout"])
        mpres = torch.rand(B, generator=g, device=dev) >= cfg["meta_dropout"]
        tgt = {k: torch.as_tensor(v[idx], device=dev) for k, v in T.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(specs, pres, torch.as_tensor(meta[idx], device=dev), mpres, torch.as_tensor(medge[idx], device=dev))
        out = {k: v.float() for k, v in out.items()}
        loss, parts = losses(out, tgt, stats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync_grads(params, reducer, rank)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        if rank == 0 and step % 50 == 0:
            tb.add_scalar("loss/total", float(loss), step)
            for k, v in parts.items():
                tb.add_scalar(f"loss/{k}", v, step)
        if rank == 0 and ((step + 1) % cfg["eval_every"] == 0 or step == steps - 1):
            m = evaluate(model, va, stats, dev, cfg, g, realism)
            for k, v in m.items():
                tb.add_scalar(f"val/{k}", v, step)
            score = m["score"]
            print(f"[{cfg['name']}] step {step + 1} loss {float(loss):.4f} val score {score:.4f} "
                  f"comp_acc {m['comp_acc']:.3f} lat_mae {m['lattice_mae']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            torch.save(dict(model=model.state_dict(), stats=stats, step=step + 1, cfg=cfg), out_dir / "last.pt")
            if best is None or score < best:
                best = score
                torch.save(dict(model=model.state_dict(), stats=stats, step=step + 1, cfg=cfg, val=m), out_dir / "best.pt")
        if world > 1 and (step + 1) % cfg["eval_every"] == 0:
            barrier()
    if rank == 0:
        (out_dir / "best.sha256").write_text(sha256_file(out_dir / "best.pt"))
        tb.close()


@torch.no_grad()
def evaluate(model, ds, stats, dev, cfg, g, realism, bs=256):
    """Validation on augmented validation spectra (same augmentation family as training). Composition and the
    regression targets are scored from the spectra alone (no metadata token); fault flags (which include metadata
    faults) are scored with the metadata token."""
    m_spec = _evaluate(model, ds, stats, dev, cfg, g, realism, bs, use_meta=False)
    m_meta = _evaluate(model, ds, stats, dev, cfg, g, realism, bs, use_meta=True)
    m = dict(m_spec)
    m["fault_auc_mean"] = m_meta["fault_auc_mean"]
    m["comp_acc_with_meta"] = m_meta["comp_acc"]
    m["score"] = (1 - m["comp_acc"]) + (1 - m["ordered_acc"]) + m["size_mae"] / 2.0 + m["lattice_mae"] / 0.02 +         m["n1_pt_mae"] / 2.0 + m["r1_pt_mae"] / 0.02 + (1 - m["fault_auc_mean"])
    return m


@torch.no_grad()
def _evaluate(model, ds, stats, dev, cfg, g, realism, bs=256, use_meta=True):
    model.eval()
    S, P = ds.spectra()
    T = ds.targets()
    meta = ds.metadata_vectors()
    medge = ds.m_edge()
    ctx_all = dict(comp=ds.composition_fractions(), medge=medge, inst_pdf=ds.a["inst_pdf"], inst_cu=ds.a["inst_xrd_cu"],
                   inst_exafs=ds.a["inst_exafs"])
    preds = []
    for s in range(0, ds.n, bs):
        sl = slice(s, min(ds.n, s + bs))
        specs = {t: torch.as_tensor(S[t][sl], device=dev) for t in TECHNIQUES}
        pres = {t: torch.as_tensor(P[t][sl], device=dev) for t in TECHNIQUES}
        ctx = {k: torch.as_tensor(v[sl], device=dev) for k, v in ctx_all.items()}
        specs = batch_augment(specs, cfg["variant"], g, realism, ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(specs, pres, torch.as_tensor(meta[sl], device=dev) if use_meta else None, None,
                        torch.as_tensor(medge[sl], device=dev))
        preds.append({k: v.float().cpu() for k, v in decode({k: v.float() for k, v in out.items()}, stats).items()})
    model.train()
    pr = {k: torch.cat([p[k] for p in preds]).numpy() for k in preds[0]}
    return metrics(pr, T)


def metrics(pr, T):
    m = {}
    m["comp_acc"] = float((pr["composition"] == T["composition"]).mean())
    om = T["ordered"] >= 0
    m["ordered_acc"] = float(((pr["ordered_prob"][om] > 0.5) == (T["ordered"][om] > 0.5)).mean())
    for k in REG_TARGETS:
        y = T[k]
        mk = np.isfinite(y)
        m[f"{k}_mae"] = float(np.abs(pr[k][mk] - y[mk]).mean())
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j in range(T["faults"].shape[1]):
        yj = T["faults"][:, j]
        if 0 < yj.sum() < len(yj):
            aucs.append(roc_auc_score(yj, pr["fault_prob"][:, j]))
    m["fault_auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
    # scalar model-selection score: classification errors + relative regression errors
    m["score"] = (1 - m["comp_acc"]) + (1 - m["ordered_acc"]) + m["size_mae"] / 2.0 + m["lattice_mae"] / 0.02 + \
        m["n1_pt_mae"] / 2.0 + m["r1_pt_mae"] / 0.02 + (1 - m["fault_auc_mean"])
    m["lattice_mae"] = m["lattice_mae"]
    return m


def main(cfg_path):
    cfg = load_yaml(cfg_path)
    world = len(cfg["gpus"])
    if world == 1:
        worker(0, 1, 0, cfg_path, None)
        return
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    n = sum(p.numel() for p in Predictor().parameters())
    reducer = SharedReducer(n, world)
    mp.spawn(worker, args=(world, port, cfg_path, reducer), nprocs=world, join=True)


if __name__ == "__main__":
    main(sys.argv[1])
