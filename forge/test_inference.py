"""Run the shipped inference scripts on the test split and summarize them (Phase 5 deliverable check).

    python -m forge.test_inference

1. artifacts/scripts/predict.py on every test shard (with and without the metadata token) and on the Concord contract
   export of the test split; predictions are compared with the ground truth.
2. artifacts/scripts/realism_infer.py on every test shard and on contract sample directories (the whole export if
   the realism layer is shipped with the predictor, otherwise the first 200 samples). Label preservation is checked on
   every test spectrum: XRD (111) peak position, PDF first-peak position, EXAFS first-shell |chi(R)| peak position.
Outputs: artifacts/data/test_outputs/, artifacts/reports/test_split_inference.md (+ .json).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .build import COMPOSITIONS, FAULTS
from .paths import DATA, MODELS, REPORTS, ROOT

OUT = DATA / "test_outputs"
PY = sys.executable
EXPORT = DATA / "contract_export" / "samples"


def _run(args, log):
    t0 = time.time()
    r = subprocess.run([PY] + [str(a) for a in args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    dt = time.time() - t0
    rel = [str(Path(a).relative_to(ROOT)) if Path(str(a)).is_absolute() and Path(str(a)).is_relative_to(ROOT) else str(a)
           for a in args]
    log.append(dict(command="python " + " ".join(rel), exit_code=r.returncode, seconds=round(dt, 1),
                    stdout_tail=r.stdout.strip().splitlines()[-1:] if r.stdout.strip() else []))
    print(f"[{r.returncode}] {dt:7.1f}s  python {' '.join(rel)}", flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True)
    return r.returncode


def _truth_from_shards(shards):
    T = {}
    for sh in shards:
        with h5py.File(sh, "r") as h:
            ids = [x.decode() for x in h["sample_id"][:]]
            fs = h["first_shell"][:]
            for i, sid in enumerate(ids):
                T[sid] = dict(comp=int(h["composition_id"][i]), lattice=float(h["lattice_parameter"][i]),
                              size=float(h["diameter_nm"][i]), n1_pt=float(fs[i, 0, 0]), r1_pt=float(fs[i, 0, 1]),
                              n1_m=float(fs[i, 1, 0]), r1_m=float(fs[i, 1, 1]), ordered=int(h["ordered"][i]),
                              faults=h["faults"][i].astype(int).tolist())
    return T


def _pred_metrics(preds, T):
    from sklearn.metrics import roc_auc_score
    P = {p["sample_id"]: p for p in preds}
    ids = [s for s in T if s in P]
    m = dict(n=len(ids))
    m["composition_acc"] = float(np.mean([COMPOSITIONS.index(P[s]["composition"]) == T[s]["comp"] for s in ids]))
    alloy = [s for s in ids if T[s]["ordered"] >= 0 and P[s]["ordered_prob"] is not None]
    if alloy:
        m["ordered_acc"] = float(np.mean([(P[s]["ordered_prob"] > 0.5) == (T[s]["ordered"] == 1) for s in alloy]))
    m["size_mae_nm"] = float(np.mean([abs(P[s]["diameter_nm"] - T[s]["size"]) for s in ids]))
    m["lattice_mae_A"] = float(np.mean([abs(P[s]["lattice_parameter_A"] - T[s]["lattice"]) for s in ids]))
    m["n1_pt_mae"] = float(np.mean([abs(P[s]["first_shell_Pt"]["N"] - T[s]["n1_pt"]) for s in ids]))
    m["r1_pt_mae_A"] = float(np.mean([abs(P[s]["first_shell_Pt"]["R_A"] - T[s]["r1_pt"]) for s in ids]))
    al = [s for s in ids if "first_shell_M" in P[s] and np.isfinite(T[s]["r1_m"])]
    if al:
        m["n1_m_mae"] = float(np.mean([abs(P[s]["first_shell_M"]["N"] - T[s]["n1_m"]) for s in al]))
        m["r1_m_mae_A"] = float(np.mean([abs(P[s]["first_shell_M"]["R_A"] - T[s]["r1_m"]) for s in al]))
    y = np.array([T[s]["faults"] for s in ids])
    if y.sum():
        pr = np.array([[P[s]["fault_prob"][f] for f in FAULTS] for s in ids])
        m["fault_auroc"] = {f: float(roc_auc_score(y[:, j], pr[:, j])) for j, f in enumerate(FAULTS) if 0 < y[:, j].sum() < len(y)}
    return m


def _curve(y, grid, lo, hi, smooth=0.0):
    msk = (grid >= lo) & (grid <= hi)
    yy = gaussian_filter1d(y, smooth, axis=1) if smooth else y
    return grid[msk], yy[:, msk]


def _ft_curve(y, k, rlo=1.8, rhi=3.2):
    rg = np.linspace(rlo, rhi, 141)
    win = np.hanning(len(k))
    return rg, np.abs(((y * win)[:, None, :] * np.exp(2j * k[None, None, :] * rg[None, :, None])).sum(-1))


def _peak(y, grid, lo, hi, smooth=0.0):
    g, c = _curve(y, grid, lo, hi, smooth)
    return g[np.argmax(c, axis=1)]


def _ft_peak(y, k, rlo=1.8, rhi=3.2):
    rg, c = _ft_curve(y, k, rlo, rhi)
    return rg[np.argmax(c, axis=1)]


def _second_max_ratio(c):
    """Height of the second-highest local maximum relative to the highest, per row (0 if there is one maximum)."""
    out = np.zeros(len(c))
    for i, row in enumerate(c):
        lm = [row[j] for j in range(1, len(row) - 1) if row[j] >= row[j - 1] and row[j] >= row[j + 1]]
        lm = sorted(lm + [row[0], row[-1]], reverse=True)
        out[i] = lm[1] / lm[0] if len(lm) > 1 and lm[0] > 0 else 0.0
    return out


def _realism_checks(files):
    acc = {"xrd": [], "pdf": [], "exafs_pt": [], "exafs_m": []}
    mag = {k: [] for k in acc}
    flips = {k: [] for k in acc}
    step = {"xrd": 60.0 / 1535, "pdf": 0.02, "exafs_pt": 0.01, "exafs_m": 0.01}
    for f in files:
        with h5py.File(f, "r") as h:
            tth, r, k = h["grid/xrd_two_theta"][:], h["grid/pdf_r"][:], h["grid/exafs_k"][:]
            for t in acc:
                if t not in h:
                    continue
                x, y = h[f"{t}/input"][:], h[f"{t}/realistic"][:]
                if t == "xrd":
                    d = np.abs(_peak(y, tth, 36.5, 42.5, 4) - _peak(x, tth, 36.5, 42.5, 4))
                    _, cx = _curve(x, tth, 36.5, 42.5, 4)
                elif t == "pdf":
                    d = np.abs(_peak(y, r, 2.2, 3.2) - _peak(x, r, 2.2, 3.2))
                    _, cx = _curve(x, r, 2.2, 3.2)
                else:
                    d = np.abs(_ft_peak(y, k) - _ft_peak(x, k))
                    _, cx = _ft_curve(x, k)
                acc[t].append(d)
                ids = [s.decode() for s in h["sample_id"][:]]
                if t == "exafs_m":
                    ids = [i for i, m in zip(ids, h["exafs_m/has_edge"][:]) if m]
                big = np.flatnonzero(d > 1.5 * step[t])
                if len(big):
                    ratio = _second_max_ratio(cx[big])
                    src = Path(h.attrs["source_shard"])
                    with h5py.File(src if src.is_absolute() else ROOT / src, "r") as sh:
                        sid = [v.decode() for v in sh["sample_id"][:]]
                        info = {}
                        for i in big:
                            m = json.loads(sh["meta_json"][sid.index(ids[i])])
                            info[i] = (f"{m['spec']['composition']} {m['spec']['shape']}, "
                                       f"{m['particle']['diameter_nm']:.1f} nm, {m['particle']['n_atoms']} atoms")
                    flips[t] += [dict(sample_id=ids[i], shift=float(d[i]), second_max_ratio=float(q), particle=info[i])
                                 for i, q in zip(big, ratio)]
                scale = np.abs(x).max(1)
                mag[t].append(np.stack([np.sqrt((h[f"{t}/bg"][:] ** 2).mean(1)) / scale,
                                        np.sqrt(((h[f"{t}/gain"][:] - 1) ** 2).mean(1)),
                                        h[f"{t}/nscale"][:].mean(1) / scale], 1))
    out = {}
    unit = {"xrd": "deg 2theta", "pdf": "A", "exafs_pt": "A", "exafs_m": "A"}
    for t in acc:
        if not acc[t]:
            continue
        d, mg = np.concatenate(acc[t]), np.concatenate(mag[t])
        out[t] = dict(n=int(len(d)), unit=unit[t], shift_median=float(np.median(d)), shift_p95=float(np.percentile(d, 95)),
                      shift_max=float(d.max()), bg_rms_rel=float(mg[:, 0].mean()), gain_rms_dev=float(mg[:, 1].mean()),
                      noise_scale_rel=float(mg[:, 2].mean()), above_one_step=flips[t])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = json.loads((DATA / "splits.json").read_text())
    shards = [DATA / "shards" / s["shard"] for s in splits["shards"] if s["split"] == "test"]
    shipped = json.loads((MODELS / "predictor" / "SHIPPED.json").read_text())
    log = []
    pdir = OUT / "predictions"
    pdir.mkdir(exist_ok=True)
    # 1. predictions
    for sh in shards:
        _run(["artifacts/scripts/predict.py", sh, "--out", pdir / f"{sh.stem}.json"], log)
        _run(["artifacts/scripts/predict.py", sh, "--out", pdir / f"{sh.stem}_spectra_only.json", "--no-metadata"], log)
    _run(["artifacts/scripts/predict.py", EXPORT, "--out", pdir / "contract_export.json"], log)
    T = _truth_from_shards(shards)
    res = dict(shipped_variant=shipped["variant"], shipped_seed=shipped["seed"], commands=log)
    for name, suffix in (("shards_with_metadata", ""), ("shards_spectra_only", "_spectra_only")):
        preds = sum((json.loads((pdir / f"{sh.stem}{suffix}.json").read_text()) for sh in shards), [])
        res[name] = _pred_metrics(preds, T)
    res["contract_export_with_metadata"] = _pred_metrics(json.loads((pdir / "contract_export.json").read_text()), T)
    # 2. realism layer
    rdir = OUT / "realism"
    for sh in shards:
        _run(["artifacts/scripts/realism_infer.py", sh, "--out", rdir], log)
    if shipped["variant"] == "b":
        src, dst, note = EXPORT, DATA / "contract_export_realism", "the whole test export (the realism layer is shipped)"
    else:
        src = OUT / "contract_sample_input"
        if src.exists():
            shutil.rmtree(src)
        src.mkdir(parents=True)
        for d in sorted(p for p in EXPORT.iterdir() if p.is_dir())[:200]:
            shutil.copytree(d, src / d.name)
        dst, note = OUT / "realism_contract_sample", "the first 200 test samples (the realism layer is not shipped)"
    _run(["artifacts/scripts/realism_infer.py", src, "--out", dst], log)
    res["realism_contract_scope"] = note
    res["realism_contract_n"] = len(list((dst / "samples").glob("*/sample.json")))
    res["realism"] = _realism_checks(sorted(rdir.glob("*_realism.h5")))
    (REPORTS / "test_split_inference.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(res)
    return res


def write_md(res):
    def row(name, m):
        fa = m.get("fault_auroc", {})
        return (f"| {name} | {m['n']} | {m['composition_acc']:.4f} | {m.get('ordered_acc', float('nan')):.4f} | "
                f"{m['size_mae_nm']:.3f} | {m['lattice_mae_A']:.4f} | {m['n1_pt_mae']:.3f} | {m['r1_pt_mae_A']:.4f} | "
                f"{np.mean(list(fa.values())) if fa else float('nan'):.3f} |")
    L = ["# Test-split inference with the shipped scripts", "",
         f"Shipped predictor: variant ({res['shipped_variant']}), seed {res['shipped_seed']}. Every command below exited "
         f"with code {'0' if all(c['exit_code'] == 0 for c in res['commands']) else 'NON-ZERO (see below)'}.", "",
         "## Commands", "", "| Command | exit | seconds |", "|---|---|---|",
         *[f"| `{c['command']}` | {c['exit_code']} | {c['seconds']} |" for c in res["commands"]], "",
         "## predict.py against the ground truth", "",
         "These are the stored simulated test spectra, without the harsh noise of the downstream sets B and C. "
         "\"With metadata\" means the nominal composition and candidate phases supplied with each sample are used, as "
         "Concord would call the predictor. They are wrong for the mislabeled-metadata faults. \"Spectra only\" is "
         "`--no-metadata`.", "",
         "| Input | samples | composition acc. | ordered acc. | size MAE (nm) | lattice MAE (Å) | Pt N1 MAE | Pt R1 MAE (Å) | "
         "mean fault AUROC |", "|---|---|---|---|---|---|---|---|---|",
         row("test shards, with metadata", res["shards_with_metadata"]),
         row("test shards, spectra only", res["shards_spectra_only"]),
         row("contract export (text files, counting noise), with metadata", res["contract_export_with_metadata"]), "",
         "Fault AUROC per type (test shards, with metadata): " + ", ".join(
             f"{k} {v:.3f}" for k, v in res["shards_with_metadata"].get("fault_auroc", {}).items()) + ".", "",
         "## realism_infer.py on the test split", "",
         f"The realism layer was applied to every test spectrum in the shards, and to {res['realism_contract_n']} "
         f"contract sample directories ({res['realism_contract_scope']}; `origin` = `simulated+realism`). The table "
         "shows the label-preservation check on every spectrum: peak position before and after the layer.", "",
         "| Technique | spectra | feature | median shift | 95th pct | max | background RMS / max | gain RMS dev. | "
         "noise scale / max |", "|---|---|---|---|---|---|---|---|---|"]
    feat = {"xrd": "(111) peak, 2θ (smoothed)", "pdf": "first G(r) peak, r", "exafs_pt": "Pt L3 first-shell |χ(R)| peak",
            "exafs_m": "M K first-shell |χ(R)| peak"}
    for t, v in res["realism"].items():
        L.append(f"| {t} | {v['n']} | {feat[t]} ({v['unit']}) | {v['shift_median']:.4f} | {v['shift_p95']:.4f} | "
                 f"{v['shift_max']:.4f} | {v['bg_rms_rel']:.4f} | {v['gain_rms_dev']:.4f} | {v['noise_scale_rel']:.4f} |")
    L += ["", "Grid steps: XRD 0.039° 2θ, PDF 0.02 Å; the |χ(R)| peak is located on a 0.01 Å grid.", ""]
    for t, v in res["realism"].items():
        ab = v.get("above_one_step", [])
        ties = [a for a in ab if a["second_max_ratio"] > 0.98]
        L.append(f"- **{t}:** {len(ab)} of {v['n']} spectra moved by more than one grid step"
                 + (f". {len(ties)} of them are near-ties: the clean spectrum has two maxima in the window within 2 % "
                    "of each other, so a sub-percent change flips the argmax rather than moving a peak ("
                    + "; ".join(f"{a['sample_id']}: {a['shift']:.2f}, second maximum at {100 * a['second_max_ratio']:.1f} %"
                                + f", {a.get('particle', '')}" for a in ties[:5]) + ")" if ties else "")
                 + ((". Others: " if ties else ": ") + "; ".join(f"{a['sample_id']} {a['shift']:.3f} ({a.get('particle', '')})"
                                             for a in ab if a not in ties) if len(ties) < len(ab) else "") + ".")
    L.append("")
    (REPORTS / "test_split_inference.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def resummarize():
    """Recompute the realism checks from the existing outputs and rewrite the report (no scripts re-run)."""
    res = json.loads((REPORTS / "test_split_inference.json").read_text())
    res["realism"] = _realism_checks(sorted((OUT / "realism").glob("*_realism.h5")))
    (REPORTS / "test_split_inference.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(res)


if __name__ == "__main__":
    resummarize() if sys.argv[1:] == ["resummarize"] else main()
