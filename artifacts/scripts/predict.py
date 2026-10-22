#!/usr/bin/env python
"""Predict nanoparticle properties from spectra with the shipped multi-technique predictor.

    python artifacts/scripts/predict.py <sample_dir | shard.h5 | samples_root> [--out predictions.json]
                                       [--model artifacts/models/predictor] [--techniques xrd,pdf,exafs_pt,...]

Input
  sample directory  Concord data contract: sample.json, xrd.xy, pdf.gr, xafs_chi.k (+ optional extra files
                    xrd_sync.xy, xafs_chi_<M>_K.k listed in sample.json "extra_files"). Any subset of techniques works.
  directory of sample directories (e.g. artifacts/data/contract_export/samples)
  HDF5 shard        a Concord Forge shard (all samples, all techniques)
By default the nominal composition and candidate phases supplied with the input are used (metadata token); with
--no-metadata the prediction uses the spectra only.
Output (JSON list, one record per sample)
  composition (class + probabilities), ordered probability (alloys), diameter_nm, lattice_parameter_A,
  first-shell N and R for the Pt and M absorbers, fault probabilities.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from forge.build import COMPOSITIONS, FAULTS, PHASES  # noqa: E402
from forge.data import EDGES, TECHNIQUES, TTH_MODEL, view  # noqa: E402
from forge.predictor import Predictor, decode  # noqa: E402
from forge.paths import CONFIGS, load_yaml  # noqa: E402
from forge.simulate import Grids  # noqa: E402

GRIDS = Grids(load_yaml(CONFIGS / "instruments.yaml"))
K_STORE = np.arange(321) * 0.05


def _load_cols(path):
    return np.loadtxt(path, comments="#")


def _interp(x_new, x, y):
    y = np.interp(x_new, x, y, left=np.nan, right=np.nan)
    if np.isnan(y).all():
        return None
    # edges outside the measured range: hold the last measured value
    good = np.flatnonzero(~np.isnan(y))
    y[: good[0]] = y[good[0]]
    y[good[-1] + 1:] = y[good[-1]]
    return y


def read_sample_dir(d):
    d = Path(d)
    meta = json.loads((d / "sample.json").read_text())
    specs, present = {}, {}
    if (d / "xrd.xy").exists():
        a = _load_cols(d / "xrd.xy")
        y = _interp(TTH_MODEL, a[:, 0], a[:, 1])
        if y is not None:
            specs["xrd"] = view("xrd", y[None])[0]
    if (d / "xrd_sync.xy").exists() and meta.get("instrument", {}).get("xrd_sync", {}).get("wavelength_A"):
        a = _load_cols(d / "xrd_sync.xy")
        lam = meta["instrument"]["xrd_sync"]["wavelength_A"]
        q = 4 * math.pi * np.sin(np.radians(a[:, 0] / 2)) / lam
        y = _interp(GRIDS.q_sync, q, a[:, 1])
        if y is not None:
            specs["xrd_sync"] = view("xrd_sync", y[None])[0]
    if (d / "pdf.gr").exists():
        a = _load_cols(d / "pdf.gr")
        y = np.interp(GRIDS.r_pdf, a[:, 0], a[:, 1], left=0.0, right=0.0)
        specs["pdf"] = view("pdf", y[None])[0]
    edge_m = None
    for f, t in (("xafs_chi.k", "exafs_pt"),):
        if (d / f).exists():
            a = _load_cols(d / f)
            chi = np.interp(K_STORE, a[:, 0], a[:, 1], left=0.0, right=0.0)
            specs[t] = view(t, chi[None])[0]
    for f in meta.get("extra_files", []):
        if f.startswith("xafs_chi_") and f.endswith("_K.k"):
            el = f.split("_")[2]
            a = _load_cols(d / f)
            chi = np.interp(K_STORE, a[:, 0], a[:, 1], left=0.0, right=0.0)
            specs["exafs_m"] = view("exafs_m", chi[None])[0]
            edge_m = EDGES.index(el)
    md = np.zeros(len(COMPOSITIONS) + len(PHASES), np.float32)
    has_meta = False
    if meta.get("composition") in COMPOSITIONS:
        md[COMPOSITIONS.index(meta["composition"])] = 1
        has_meta = True
    for p in meta.get("candidate_phases", []):
        if p in PHASES:
            md[len(COMPOSITIONS) + PHASES.index(p)] = 1
    return meta.get("sample_id", d.name), specs, md, has_meta, edge_m


def load_model(model_dir, device):
    ck = torch.load(Path(model_dir) / "model.pt", map_location=device, weights_only=False)
    m = Predictor().to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["stats"]


def records(pred, ids, has_m):
    out = []
    for i, sid in enumerate(ids):
        r = dict(sample_id=sid, composition=COMPOSITIONS[int(pred["composition"][i])],
                 composition_prob={c: round(float(p), 4) for c, p in zip(COMPOSITIONS, pred["composition_prob"][i])},
                 ordered_prob=round(float(pred["ordered_prob"][i]), 4), diameter_nm=round(float(pred["size"][i]), 3),
                 lattice_parameter_A=round(float(pred["lattice"][i]), 4),
                 first_shell_Pt=dict(N=round(float(pred["n1_pt"][i]), 2), R_A=round(float(pred["r1_pt"][i]), 4)),
                 fault_prob={f: round(float(p), 4) for f, p in zip(FAULTS, pred["fault_prob"][i])})
        if has_m[i]:
            r["first_shell_M"] = dict(N=round(float(pred["n1_m"][i]), 2), R_A=round(float(pred["r1_m"][i]), 4))
        if r["composition"] == "Pt":
            r["ordered_prob"] = None
        out.append(r)
    return out


@torch.no_grad()
def run(model, stats, batch_specs, present, meta, meta_present, medge, device):
    sp = {t: torch.as_tensor(np.asarray(v, np.float32), device=device) for t, v in batch_specs.items()}
    pr = {t: torch.as_tensor(np.asarray(v), device=device) for t, v in present.items()}
    out = model(sp, pr, torch.as_tensor(meta, device=device), torch.as_tensor(meta_present, device=device),
                torch.as_tensor(medge, device=device))
    return {k: v.float().cpu().numpy() for k, v in decode({k: v.float() for k, v in out.items()}, stats).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default=str(ROOT / "artifacts" / "models" / "predictor"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--techniques", default=None, help="comma-separated subset of " + ",".join(TECHNIQUES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-metadata", action="store_true",
                    help="ignore the nominal composition and candidate phases given with the input (spectra only)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model, stats = load_model(args.model, dev)
    use = set(args.techniques.split(",")) if args.techniques else set(TECHNIQUES)
    inp = Path(args.input)
    results = []
    if inp.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(inp, "r") as h:
            ids = [x.decode() for x in h["sample_id"][:]]
            comp = h["composition_id"][:]
            raw = dict(xrd=view("xrd", h["xrd_cu"][:]), xrd_sync=view("xrd_sync", h["xrd_sync"][:]),
                       pdf=view("pdf", h["pdf_gr"][:]), exafs_pt=view("exafs_pt", h["chi_pt"][:]),
                       exafs_m=view("exafs_m", np.nan_to_num(h["chi_m"][:])))
            metas = [json.loads(m) for m in h["meta_json"][:]]
        n = len(ids)
        md = np.zeros((n, len(COMPOSITIONS) + len(PHASES)), np.float32)
        medge = np.zeros(n, np.int64)
        elmap = {"Pt3Co": "Co", "PtCo": "Co", "Pt3Ni": "Ni", "PtNi": "Ni", "Pt3Fe": "Fe", "PtCu": "Cu"}
        for i, m in enumerate(metas):
            mm = m["spec"]["metadata"]
            md[i, COMPOSITIONS.index(mm["nominal_composition"])] = 1
            for p in mm["candidate_phases"]:
                md[i, len(COMPOSITIONS) + PHASES.index(p)] = 1
            medge[i] = EDGES.index(elmap.get(COMPOSITIONS[comp[i]], "Pt"))
        for s in range(0, n, 256):
            sl = slice(s, min(n, s + 256))
            specs = {t: raw[t][sl] for t in TECHNIQUES if t in use}
            present = {t: np.ones(sl.stop - sl.start, bool) for t in specs}
            if "exafs_m" in present:
                present["exafs_m"] = comp[sl] != 0
            meta_present = np.full(sl.stop - sl.start, not args.no_metadata)
            pred = run(model, stats, specs, present, md[sl], meta_present, medge[sl], dev)
            results += records(pred, ids[sl], comp[sl] != 0)
    else:
        dirs = [inp] if (inp / "sample.json").exists() else sorted(p.parent for p in inp.glob("*/sample.json"))
        for d in dirs:
            sid, specs, md, has_meta, edge_m = read_sample_dir(d)
            specs = {t: v for t, v in specs.items() if t in use}
            if not specs:
                print(f"{sid}: no usable spectra", file=sys.stderr)
                continue
            batch = {t: v[None] for t, v in specs.items()}
            present = {t: np.ones(1, bool) for t in batch}
            pred = run(model, stats, batch, present, md[None], np.array([has_meta and not args.no_metadata]),
                       np.array([edge_m if edge_m is not None else 0]), dev)
            results += records(pred, [sid], [edge_m is not None])
    txt = json.dumps(results, indent=1)
    if args.out:
        Path(args.out).write_text(txt)
        print(f"wrote {len(results)} predictions to {args.out}")
    else:
        print(txt)


if __name__ == "__main__":
    main()
