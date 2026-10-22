#!/usr/bin/env python
"""Apply the realism models to simulated spectra (sim spectra in, realistic spectra out).

    python artifacts/scripts/realism_infer.py <shard.h5 | sample_dir | samples_root> --out <dir>
                                             [--techniques xrd,pdf,exafs] [--seed 0] [--n-draws 1]

shard.h5      a Concord Forge shard: writes <out>/<shard>_realism.h5 with the model-view spectra (xrd on 2theta 20-80 deg,
              pdf on r 0.5-30.08 A, exafs k^2 chi on k 2-14.75 A^-1 for the Pt L3 and M K edges), the inputs, the
              learned artifact fields (background, gain, noise scale) and sample ids.
sample dir    Concord data contract directory (or a directory of them): writes a copy under <out>/samples/<id>/ with
              the realism layer applied to xrd.xy, pdf.gr and xafs_chi.k (and the M-edge chi), sigma columns set to the
              learned local noise scale, and sample.json origin = "simulated+realism".
The realism layer only adds smooth background, a gain field within 0.9-1.1, coloured noise and (EXAFS) glitches;
peak positions and oscillation frequencies are left unchanged by construction.
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import torch  # noqa: E402

from forge.data import EDGES, TTH_MODEL, view  # noqa: E402
from forge.paths import CONFIGS, load_yaml  # noqa: E402
from forge.realism import COND_DIM, cond_exafs, cond_pdf, cond_xrd, load_realism  # noqa: E402
from forge.simulate import Grids  # noqa: E402

GRIDS = Grids(load_yaml(CONFIGS / "instruments.yaml"))
K_MODEL = np.linspace(2.0, 14.75, 256)
COMP_X = {"Pt": 0, "Pt3Co": .25, "PtCo": .5, "Pt3Ni": .25, "PtNi": .5, "Pt3Fe": .25, "PtCu": .5}
M_EL = {"Pt3Co": "Co", "PtCo": "Co", "Pt3Ni": "Ni", "PtNi": "Ni", "Pt3Fe": "Fe", "PtCu": "Cu"}


def comp6(name):
    c = np.zeros(6, np.float32)
    x = COMP_X.get(name, 0.0)
    c[0] = 1 - x
    if x:
        c[1 + ["Co", "Ni", "Fe", "Cu"].index(M_EL[name])] = x
    return c


@torch.no_grad()
def apply(model, x, cond, seed, dev, return_parts=False):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    xt = torch.as_tensor(np.asarray(x, np.float32), device=dev)
    ct = torch.as_tensor(np.asarray(cond, np.float32), device=dev)
    z = torch.randn(len(xt), model.G.z_dim, device=dev, generator=g)
    y, parts = model.G(xt, ct, z=z, gen=g, return_parts=True)
    out = y.float().cpu().numpy()
    if return_parts:
        return out, {k: v.float().cpu().numpy() for k, v in parts.items() if v.dim() == 2}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=str(ROOT / "artifacts" / "models"))
    ap.add_argument("--techniques", default="xrd,pdf,exafs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    models = {t: load_realism(args.models, t, dev) for t in args.techniques.split(",")}
    missing = [t for t, m in models.items() if m is None]
    if missing:
        print(f"no shipped realism model for: {missing} (not applied)", file=sys.stderr)
    models = {t: m for t, m in models.items() if m is not None}
    inp, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if inp.suffix in (".h5", ".hdf5"):
        with h5py.File(inp, "r") as h:
            ids = h["sample_id"][:]
            comp_id = h["composition_id"][:]
            names = json.loads(h.attrs["compositions"])
            comps = np.stack([comp6(names[c]) for c in comp_id])
            res = {}
            if "xrd" in models:
                x = view("xrd", h["xrd_cu"][:])
                res["xrd"] = (x, apply(models["xrd"], x, cond_xrd(comps), args.seed, dev, True))
            if "pdf" in models:
                x = view("pdf", h["pdf_gr"][:])
                ip = h["inst_pdf"][:]
                res["pdf"] = (x, apply(models["pdf"], x, cond_pdf(comps, ip[:, 1], ip[:, 2]), args.seed, dev, True))
            if "exafs" in models:
                x = view("exafs_pt", h["chi_pt"][:])
                res["exafs_pt"] = (x, apply(models["exafs"], x, cond_exafs(np.zeros(len(x), int), comps, np.zeros(len(x))),
                                            args.seed, dev, True))
                has_m = comp_id != 0
                if has_m.any():
                    xm = view("exafs_m", np.nan_to_num(h["chi_m"][:]))[has_m]
                    edge = np.array([EDGES.index(M_EL[names[c]]) for c in comp_id[has_m]])
                    ym = apply(models["exafs"], xm, cond_exafs(edge, comps[has_m], np.zeros(len(xm))), args.seed + 1, dev, True)
                    res["exafs_m"] = (xm, ym, has_m)
        f = out / f"{inp.stem}_realism.h5"
        with h5py.File(f, "w") as o:
            o.attrs["origin"] = "simulated+realism"
            o.attrs["source_shard"] = str(inp)
            o.create_dataset("sample_id", data=ids)
            o.create_dataset("grid/xrd_two_theta", data=TTH_MODEL.astype(np.float32))
            o.create_dataset("grid/pdf_r", data=GRIDS.r_pdf[:1480].astype(np.float32))
            o.create_dataset("grid/exafs_k", data=K_MODEL.astype(np.float32))
            for t, v in res.items():
                o.create_dataset(f"{t}/input", data=v[0].astype(np.float32))
                o.create_dataset(f"{t}/realistic", data=v[1][0].astype(np.float32))
                for k, a in v[1][1].items():
                    o.create_dataset(f"{t}/{k}", data=a.astype(np.float32))
                if len(v) == 3:
                    o.create_dataset(f"{t}/has_edge", data=v[2])
        print(f"wrote {f} ({len(ids)} samples, techniques {sorted(res)})")
        return
    dirs = [inp] if (inp / "sample.json").exists() else sorted(p.parent for p in inp.glob("*/sample.json"))
    for d in dirs:
        meta = json.loads((d / "sample.json").read_text())
        dst = out / "samples" / d.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(d, dst)
        truth = meta.get("ground_truth", {}).get("composition", meta.get("composition"))
        c = comp6(truth)[None]
        if "xrd" in models and (d / "xrd.xy").exists():
            a = np.loadtxt(d / "xrd.xy")
            x = np.interp(TTH_MODEL, a[:, 0], a[:, 1])
            scale = x.max()
            y, parts = apply(models["xrd"], view("xrd", x[None]), cond_xrd(c), args.seed, dev, True)
            np.savetxt(dst / "xrd.xy", np.column_stack([TTH_MODEL, y[0] * scale, parts["nscale"][0] * scale]), fmt="%.6g",
                       header="2theta_deg intensity sigma  (simulated + realism layer; sigma = learned noise scale)")
        if "pdf" in models and (d / "pdf.gr").exists():
            a = np.loadtxt(d / "pdf.gr")
            x = np.interp(GRIDS.r_pdf[:1480], a[:, 0], a[:, 1], left=0, right=0)
            scale = np.abs(x).max()
            ins = meta.get("instrument", {}).get("pdf", {})
            cp = cond_pdf(c, [ins.get("qmax_inv_A", 25.0)], [ins.get("qdamp", 0.04)])
            y, parts = apply(models["pdf"], view("pdf", x[None]), cp, args.seed, dev, True)
            np.savetxt(dst / "pdf.gr", np.column_stack([GRIDS.r_pdf[:1480], y[0] * scale, parts["nscale"][0] * scale]), fmt="%.6g",
                       header="r_A G sigma  (simulated + realism layer)")
        if "exafs" in models:
            files = [("xafs_chi.k", "Pt")] + [(f, f.split("_")[2]) for f in meta.get("extra_files", [])
                                               if f.startswith("xafs_chi_")]
            for fn, el in files:
                if not (d / fn).exists():
                    continue
                a = np.loadtxt(d / fn)
                chi = np.interp(np.arange(321) * 0.05, a[:, 0], a[:, 1], left=0, right=0)
                x = view("exafs_pt", chi[None])
                y = apply(models["exafs"], x, cond_exafs([EDGES.index(el)], c, [0.0]), args.seed, dev)
                np.savetxt(dst / fn, np.column_stack([K_MODEL, y[0] / K_MODEL ** 2]), fmt="%.6g",
                           header=f"k_inv_A chi  (simulated + realism layer, {el} edge)")
        meta["origin"] = "simulated+realism"
        (dst / "sample.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(dirs)} sample directories to {out / 'samples'}")


if __name__ == "__main__":
    main()
