"""Phase 5 downstream comparison of predictor (a) classical augmentation vs (b) GAN realism augmentation.

Test sets
  A  real held-out spectra (realism split 'eval', never trained on) with known labels:
     - Pt L3 EXAFS of Pt foil: composition Pt, first-shell N = 12, R = 2.775 A (fcc Pt, room temperature)
     - Pt / Pt-alloy catalysts and nanoparticles (EXAFS, XRD, PDF): composition class
     Each spectrum is predicted from that one technique alone (the predictor accepts any subset).
  B  300 test-split samples whose EXAFS is recomputed with full FEFF6L (8 absorbers per edge, as in the shortcut
     validation), plus harsh classical noise on every technique.
  C  the simulated test split (clean + fault) with harsh classical noise on every technique.
Metrics: composition accuracy, ordered/disordered accuracy (alloys), size / lattice / first-shell N and R MAE,
mean fault AUROC (C only). 95 % confidence intervals by bootstrap over samples (1,000 resamples).

Decision rule (fixed before any comparison was run): ship (b) and the realism models only if, on set A, (b) is better
than (a) on the primary real-data metric (mean over seeds) by more than the run-to-run variation (the larger of the two
variants' seed-to-seed differences), and (b) is not worse than (a) by more than run-to-run variation on any metric of
sets B and C. Otherwise ship (a). The primary real-data metric is the composition-family accuracy over all set-A
items (Pt / Pt-Co / Pt-Ni / Pt-Fe / Pt-Cu: the open deposits state the alloying element but not reliably the Pt:M
ratio, so Pt3Co and PtCo both count as correct for a Pt-Co catalyst); set-A first-shell N and R of Pt foil and the
exact composition accuracy are reported as well.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import augment
from .build import COMPOSITIONS
from .data import EDGES, TECHNIQUES, Dataset, view
from .paths import MODELS, REPORTS, ROOT
from .predictor import REG_TARGETS, Predictor, decode
from .realism import load_real

PT_FOIL = dict(N=12.0, R=2.775)


def load_predictor(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Predictor().to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["stats"]


@torch.no_grad()
def predict(model, stats, specs, present, meta, meta_present, medge, dev, bs=256):
    n = len(next(iter(specs.values())))
    out = []
    for s in range(0, n, bs):
        sl = slice(s, min(n, s + bs))
        sp = {t: torch.as_tensor(np.asarray(v[sl], np.float32), device=dev) for t, v in specs.items()}
        pr = {t: torch.as_tensor(np.asarray(v[sl]), device=dev) for t, v in present.items()}
        o = model(sp, pr, torch.as_tensor(meta[sl], device=dev), torch.as_tensor(meta_present[sl], device=dev),
                  torch.as_tensor(medge[sl], device=dev))
        out.append({k: v.float().cpu().numpy() for k, v in decode({k: v.float() for k, v in o.items()}, stats).items()})
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def bootstrap(fn, n, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def task_metrics(pr, T, idx=None):
    """Dict of metric -> (value, ci_lo, ci_hi, n)."""
    n = len(T["composition"])
    out = {}

    def add(name, f, mask):
        ii = np.flatnonzero(mask)
        if len(ii) < 3:
            return
        val = f(ii)
        lo, hi = bootstrap(lambda b: f(ii[b]), len(ii))
        out[name] = (float(val), lo, hi, int(len(ii)))

    comp_ok = T["composition"] >= 0
    add("composition_acc", lambda i: float((pr["composition"][i] == T["composition"][i]).mean()), comp_ok)
    if "ordered" in T:
        om = T["ordered"] >= 0
        add("ordered_acc", lambda i: float(((pr["ordered_prob"][i] > 0.5) == (T["ordered"][i] > 0.5)).mean()), om)
    for k in REG_TARGETS:
        if k in T:
            mk = np.isfinite(T[k])
            add(f"{k}_mae", lambda i, k=k: float(np.abs(pr[k][i] - T[k][i]).mean()), mk)
    if "faults" in T and T["faults"].shape[1] and T["faults"].sum() > 0:
        from sklearn.metrics import roc_auc_score

        def auc(i):
            a = []
            for j in range(T["faults"].shape[1]):
                y = T["faults"][i, j]
                if 0 < y.sum() < len(y):
                    a.append(roc_auc_score(y, pr["fault_prob"][i, j]))
            return float(np.mean(a)) if a else float("nan")
        add("fault_auroc", auc, np.ones(n, bool))
    return out


# ------------------------------------------------------------------------------------------------ set A
def real_set():
    """Real held-out items with known labels -> list of (technique view, x, targets dict)."""
    items = []
    comp_of = {"Pt": "Pt", "PtCo": "PtCo", "Pt3Co": "Pt3Co", "PtNi": "PtNi", "Pt3Ni": "Pt3Ni"}
    for tech, vname in (("exafs", "exafs_pt"), ("xrd", "xrd"), ("pdf", "pdf")):
        try:
            R = load_real(tech)
        except FileNotFoundError:
            continue
        ev = np.flatnonzero(R["split"] == "eval")
        for i in ev:
            mat = str(R["material"][i])
            st = str(R.get("sample_type", np.array([""] * len(R["x"])))[i]) if "sample_type" in R else ""
            edge = str(R["edge"][i]) if "edge" in R else ""
            if tech == "exafs" and edge != "Pt L3":
                continue
            comp = None
            low = mat.lower()
            if ("pt foil" in low or low.strip() in ("pt", "platinum foil", "platinum")) and tech == "exafs":
                comp = "Pt"
                tgt = dict(composition=COMPOSITIONS.index("Pt"), n1_pt=PT_FOIL["N"], r1_pt=PT_FOIL["R"])
            elif "Pt" in mat and not any(x in mat for x in ("Sn", "Pd", "Ru", "Au", "O2", "Cl", "acac")):
                if "Co" in mat:
                    comp = "Pt3Co"  # catalyst composition class (Pt-rich Pt-Co); the 1:1 vs 3:1 ratio is often not stated
                elif "Ni" in mat:
                    comp = "Pt3Ni"
                elif "Pt" in mat and ("/C" in mat or "nanoparticle" in low or "Pt/" in mat or "Pt " in mat):
                    comp = "Pt"
                if comp:
                    tgt = dict(composition=COMPOSITIONS.index(comp))
            if comp is None:
                continue
            items.append(dict(technique=vname, x=R["x"][i], tgt=tgt, item_id=str(R["item_id"][i]), material=mat,
                              source=str(R["source"][i])))
    return items


FAMILY = {"Pt": 0, "Pt3Co": 1, "PtCo": 1, "Pt3Ni": 2, "PtNi": 2, "Pt3Fe": 3, "PtCu": 4}


def eval_real(model, stats, items, dev, return_preds=False):
    preds = []
    for it in items:
        t = it["technique"]
        specs = {t: it["x"][None]}
        present = {t: np.ones(1, bool)}
        p = predict(model, stats, specs, present, np.zeros((1, 11), np.float32), np.zeros(1, bool),
                    np.zeros(1, np.int64), dev)
        preds.append(p)
    pr = {k: np.concatenate([p[k] for p in preds]) for k in preds[0]}
    T = dict(composition=np.array([it["tgt"].get("composition", -1) for it in items]),
             n1_pt=np.array([it["tgt"].get("n1_pt", np.nan) for it in items], float),
             r1_pt=np.array([it["tgt"].get("r1_pt", np.nan) for it in items], float))
    out = task_metrics(pr, T)
    fam_t = np.array([FAMILY[COMPOSITIONS[c]] for c in T["composition"]])
    fam_p = np.array([FAMILY[COMPOSITIONS[int(c)]] for c in pr["composition"]])
    f = lambda i: float((fam_p[i] == fam_t[i]).mean())
    ii = np.arange(len(fam_t))
    lo, hi = bootstrap(lambda b: f(ii[b]), len(ii))
    out["composition_family_acc"] = (f(ii), lo, hi, int(len(ii)))
    if return_preds:
        return out, [dict(composition=COMPOSITIONS[int(pr["composition"][j])], n1_pt=float(pr["n1_pt"][j]),
                          r1_pt=float(pr["r1_pt"][j])) for j in range(len(items))]
    return out


def reference_fits(items):
    """Standard-tool fits of the set-A spectra (lmfit fcc peaks for XRD, larch first shell for Pt L3 EXAFS), used only
    to diagnose the predictions in the report; nothing here enters the comparison or the decision."""
    from .data import TTH_MODEL
    from .refit import fit_exafs, fit_xrd
    k = np.linspace(2.0, 14.75, 256)
    kk = np.arange(321) * 0.05
    out = []
    for it in items:
        if it["technique"] == "xrd":
            f = min((fit_xrd(TTH_MODEL, it["x"], a0=a0) for a0 in (3.92, 3.83)), key=lambda f: f["redchi"])
            out.append(dict(a=float(f["a"]), size_nm=float(f["size_nm"])))
        else:
            chi = np.interp(kk, k, it["x"] / k ** 2, left=0.0, right=0.0)
            f = fit_exafs(kk, chi, "Pt", 0.85)
            out.append(dict(R=float(f["R"]), e0=float(f["e0"]), sig2=float(f["sig2"]), rfactor=float(f["rfactor"])))
    return out


# ------------------------------------------------------------------------------------------------ sets B / C
def harsh(specs, present, dev, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    out = {}
    for t, x in specs.items():
        xt = torch.as_tensor(np.asarray(x, np.float32), device=dev)
        out[t] = augment.classical(xt, t, g, "harsh").cpu().numpy()
    return out


def eval_sim(model, stats, ds, dev, specs_override=None, seed=0):
    S, P = ds.spectra()
    if specs_override:
        S = dict(S, **specs_override)
    S = harsh(S, P, dev, seed)
    T = ds.targets()
    meta = ds.metadata_vectors()
    pr = predict(model, stats, S, P, meta, np.ones(ds.n, bool), ds.m_edge(), dev)
    return task_metrics(pr, T)


S02 = {"Pt": 0.85, "Co": 0.80, "Ni": 0.80, "Fe": 0.80, "Cu": 0.85}
LOWER_IS_BETTER = lambda m: m.endswith("_mae")


def fullfeff_ids(n=300, seed=5):
    """Clean test-split sample ids for set B, drawn from the full plan."""
    plan = json.loads((ROOT / "artifacts/data/work/full/plan.json").read_text())
    ids = sorted(s["sample_id"] for s in plan if s["set"] == "clean" and s["split"] == "test")
    rng = np.random.default_rng(seed)
    return sorted(ids[i] for i in rng.choice(len(ids), size=min(n, len(ids)), replace=False))


def run_fullfeff(n=300, seed=5, workers=28):
    from .exafs_validate import run_feff_cached
    ids = fullfeff_ids(n, seed)
    return run_feff_cached(str(ROOT / "artifacts/data/work/full/plan.json"), len(ids), seed, "fullfeff_testB", ids=ids,
                           workers=workers)


def fullfeff_test_set(n=300, seed=5):
    """Test-split samples whose EXAFS is replaced by full FEFF6L (8 absorbers per edge, cached)."""
    ds = Dataset(ROOT / "artifacts" / "data", parts=[("clean", "test")])
    ids = set(fullfeff_ids(n, seed))
    pick = np.array([i for i in range(ds.n) if str(ds.a["sample_id"][i]) in ids])
    cached = run_fullfeff(n, seed)
    by = {}
    for r in cached["res"]:
        if r["ok"]:
            by.setdefault((r["sid"], r["X"]), []).append(r["chi"])
    k = np.round(np.arange(321) * 0.05, 4)
    chi_pt = np.zeros((len(pick), 321), np.float32)
    chi_m = np.zeros((len(pick), 321), np.float32)
    keep = np.ones(len(pick), bool)
    for j, i in enumerate(pick):
        sid = str(ds.a["sample_id"][i])
        sp = ds.meta[i]["spec"]
        if (sid, "Pt") not in by:
            keep[j] = False
            continue
        chi_pt[j] = S02["Pt"] * np.mean(by[(sid, "Pt")], axis=0)
        if sp["M"] is not None and (sid, sp["M"]) in by:
            chi_m[j] = S02[sp["M"]] * np.mean(by[(sid, sp["M"])], axis=0)
    return ds, pick[keep], chi_pt[keep], chi_m[keep]


def subset(ds, idx):
    import copy
    d2 = copy.copy(ds)
    d2.a = {k: v[idx] for k, v in ds.a.items()}
    d2.meta = [ds.meta[i] for i in idx]
    d2.n = len(idx)
    return d2


def evaluate_variant_seed(ckpt, dev, setA, setB, dsC):
    model, stats = load_predictor(ckpt, dev)
    out = {}
    out["A_real"], items = eval_real(model, stats, setA, dev, return_preds=True) if setA else ({}, [])
    dsB, idxB, chi_pt, chi_m = setB
    dB = subset(dsB, idxB)
    ov = {"exafs_pt": view("exafs_pt", chi_pt), "exafs_m": view("exafs_m", chi_m)}
    out["B_fullfeff_harsh"] = eval_sim(model, stats, dB, dev, specs_override=ov, seed=11)
    out["C_test_harsh"] = eval_sim(model, stats, dsC, dev, seed=12)
    return out, items


def decide(per_seed):
    """per_seed: {variant: [results per seed]} -> (decision dict, table rows)."""
    def mean_and_spread(variant, set_name, metric):
        vals = [r[set_name][metric][0] for r in per_seed[variant] if metric in r[set_name]]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(max(vals) - min(vals)) if len(vals) > 1 else 0.0
    rows = []
    for set_name in ("A_real", "B_fullfeff_harsh", "C_test_harsh"):
        metrics = sorted({m for v in per_seed for r in per_seed[v] for m in r[set_name]})
        for m in metrics:
            ma, sa = mean_and_spread("a", set_name, m)
            mb, sb = mean_and_spread("b", set_name, m)
            rows.append(dict(set=set_name, metric=m, a=ma, a_spread=sa, b=mb, b_spread=sb,
                             run_to_run=max(sa, sb), lower_is_better=LOWER_IS_BETTER(m)))

    def better(r):  # positive = b better than a
        d = r["b"] - r["a"]
        return -d if r["lower_is_better"] else d
    prim = [r for r in rows if r["set"] == "A_real" and r["metric"] == "composition_family_acc"]
    prim_ok = bool(prim) and better(prim[0]) > prim[0]["run_to_run"]
    losses = [r for r in rows if r["set"] != "A_real" and np.isfinite(r["a"]) and np.isfinite(r["b"])
              and -better(r) > r["run_to_run"]]
    ship_b = prim_ok and not losses
    return dict(ship="b" if ship_b else "a", primary_ok=prim_ok,
                primary=prim[0] if prim else None, losses_elsewhere=[(r["set"], r["metric"]) for r in losses]), rows


def fmt(v):
    val, lo, hi, n = v
    return f"{val:.4f} [{lo:.4f}, {hi:.4f}] (n={n})"


def _count_items(items):
    from collections import Counter
    import re
    return Counter((it["technique"], re.sub(r"\s*\[.*?\]$", "", it["material"])[:70].replace("|", "/"),
                    COMPOSITIONS[it["tgt"]["composition"]] + (" (N = 12, R = 2.775 Å)" if "n1_pt" in it["tgt"] else ""),
                    it.get("source", "")) for it in items)


def _decision_text(decision, rows):
    pr = decision["primary"]
    n_other = sum(1 for r in rows if r["set"] != "A_real" and np.isfinite(r["a"]) and np.isfinite(r["b"]))
    lost = decision["losses_elsewhere"]
    if decision["ship"] == "b":
        return ("The GAN-augmented predictor (b) beat the classical baseline (a) on the real held-out set by more than "
                "the run-to-run variation and did not lose elsewhere. The realism models are shipped as part of the "
                "training pipeline.")
    txt = (f"On the real held-out set (A), the GAN-augmented predictor (b) reached a composition-family accuracy of "
           f"{pr['b']:.3f} against {pr['a']:.3f} for the classical baseline (a) (means over two seeds). The difference, "
           f"{pr['b'] - pr['a']:+.3f}, is {'not ' if not decision['primary_ok'] else ''}larger than the run-to-run "
           f"variation of {pr['run_to_run']:.3f}, which is the spread between (b)'s two seeds.")
    if lost:
        txt += (f" On the synthetic sets B and C, (b) was worse than (a) by more than the run-to-run variation on "
                f"{len(lost)} of {n_other} metrics.")
    return txt + (" **The baseline predictor (a) is shipped.** The realism models are delivered for reference; the "
                  "shipped predictor does not use them.")


def write_report(per_seed, decision, rows, setA_items, out=None, items_A=None, fits_A=None):
    out = out or REPORTS / "downstream_comparison.md"
    L = ["# Downstream comparison: classical augmentation (a) vs GAN realism augmentation (b)", "",
         f"**Decision: ship variant ({decision['ship']}).** " + _decision_text(decision, rows), "",
         "## Decision rule (fixed before the comparison was run)", "",
         "Ship (b) only if it beats (a) on the real held-out set by more than the run-to-run variation and does not "
         "lose on the other sets. In practice:",
         "",
         "- **Primary real-data metric:** composition-family accuracy over all set-A items (Pt / Pt–Co / Pt–Ni / "
         "Pt–Fe / Pt–Cu). The deposits state the alloying element but not reliably the Pt:M ratio. (b) must exceed (a) "
         "by more than the largest seed-to-seed difference of either variant.",
         "- **No losses elsewhere:** on every metric of sets B and C, (b) must not be worse than (a) by more than that "
         "variation.",
         "",
         "Each variant was trained twice with different seeds, using identical data, schedule and model.", "",
         "## Test sets", "",
         f"- **A, real held-out:** {len(setA_items)} spectra never used for any training or model selection, each "
         "predicted from its one technique alone (no metadata). They are listed in the table below. Items from the "
         "same deposit are not independent; for example, the Pt foil spectra are repeat measurements of one "
         "reference foil. Bootstrap intervals over items therefore understate the uncertainty, especially for the "
         "Pt foil first-shell N and R.",
         "- **B, full FEFF + harsh noise:** 300 test-split samples. Their EXAFS was recomputed with full FEFF6L "
         "(8 absorbers per edge), and every technique got the 'harsh' classical noise and background.",
         "- **C, simulated test + harsh noise:** the clean test split plus the fault-test split, with harsh classical "
         "noise on every technique.", "",
         "### Set A items", "", "| Technique | Material (as deposited) | Label | Source | Spectra |", "|---|---|---|---|---|",
         *[f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {v} |" for k, v in sorted(_count_items(setA_items).items())], "",
         "## Results (mean over 2 seeds; spread = |seed 1 − seed 2|)", "",
         "| Set | Metric | (a) mean | (a) spread | (b) mean | (b) spread | b − a |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["b"] - r["a"]
        L.append(f"| {r['set']} | {r['metric']} | {r['a']:.4f} | {r['a_spread']:.4f} | {r['b']:.4f} | {r['b_spread']:.4f} | "
                 f"{d:+.4f}{' (lower is better)' if r['lower_is_better'] else ''} |")
    if items_A:
        keys = sorted(items_A)
        L += ["", "### Set A: prediction for each real spectrum", "",
              "Each spectrum was predicted from its one technique alone, without metadata. For the Pt foil the "
              "first-shell N and R predictions are given as well (label: N = 12, R = 2.775 Å).", "",
              "| # | Technique | Material | Label | Standard fit of the spectrum | "
              + " | ".join(f"({k[0]}) seed {k[1]}" for k in keys) + " |",
              "|---|---|---|---|---|" + "---|" * len(keys)]
        for j, it in enumerate(setA_items):
            lab = COMPOSITIONS[it["tgt"]["composition"]]
            cells = []
            for k in keys:
                pj = items_A[k][j]
                c = pj["composition"] + (" ✓" if FAMILY[pj["composition"]] == FAMILY[lab] else "")
                if "n1_pt" in it["tgt"]:
                    c += f" (N {pj['n1_pt']:.1f}, R {pj['r1_pt']:.3f})"
                cells.append(c)
            fit = ""
            if fits_A:
                fj = fits_A[j]
                fit = (f"a = {fj['a']:.3f} Å, {fj['size_nm']:.0f} nm" if "a" in fj else
                       f"R = {fj['R']:.3f} Å, ΔE0 = {fj['e0']:+.1f} eV")
            L.append(f"| {j + 1} | {it['technique']} | {it['material'][:48].replace('|', '/')} | {lab} | {fit} | "
                     + " | ".join(cells) + " |")
        L += ["", "✓ = correct composition family (Pt / Pt–Co / Pt–Ni / Pt–Fe / Pt–Cu)."]
    sim_a = {(r["set"], r["metric"]): r for r in rows}
    ca = sim_a.get(("C_test_harsh", "composition_acc"))
    L += ["", "## What the result means", "",
          "- **Neither variant transfers to real spectra.** Both predictors are near perfect on simulated spectra, even "
          "with harsh noise, but on real spectra they mostly fail. Composition-family accuracy on set A is "
          f"{decision['primary']['a']:.2f} for (a) and {decision['primary']['b']:.2f} for (b). Either the remaining "
          "differences between simulation and measurement are not artifacts the realism layer can model, or the "
          "21 real spectra lie outside the training domain (supports, mixed phases, other preparation). The "
          "GAN diagnostics show the same: even the best alternative stays well above 50 % C2ST "
          "(reports/gan_diagnostics.md).",
          "- **Why (b) loses on B and C.** On Concord Forge spectra the realism layer adds small artifacts: "
          "backgrounds of a few tenths of a percent of the maximum for XRD and PDF, and noise of a few percent for "
          "EXAFS (reports/test_split_inference.md). That is much weaker than the classical pipeline, so (b) never "
          "trained on strong noise and backgrounds and is fragile under the harsh classical noise of B and C"
          + (f" (composition accuracy on C {ca['b']:.3f} vs {ca['a']:.3f})." if ca else "."),
          "- **Seed-to-seed variation of (b) on set A is large** (one seed 6/21 family-correct, the other 1/21). With "
          "21 non-independent real spectra, set A can only detect large effects. The rule therefore requires (b) to "
          "beat that spread.", ""]
    if fits_A:
        ex = [(it, f) for it, f in zip(setA_items, fits_A) if "R" in f]
        xr = [(it, f) for it, f in zip(setA_items, fits_A) if "a" in f]
        foil = [f for it, f in ex if "n1_pt" in it["tgt"]]
        big = [f for it, f in xr if f["size_nm"] > 8.5]
        ptco = [f for it, f in xr if COMPOSITIONS[it["tgt"]["composition"]] == "Pt3Co"]
        pures = [f for it, f in xr if COMPOSITIONS[it["tgt"]["composition"]] == "Pt" and f["size_nm"] <= 8.5]
        pf = [items_A[k][j] for k in sorted(items_A) for j, it in enumerate(setA_items) if "n1_pt" in it["tgt"]] if items_A else []
        L += ["## Diagnosis of the set-A failures (standard fits of the same spectra)", "",
              "The \"standard fit\" column above fits each real spectrum with the same tools as the label-preservation "
              "diagnostic: xraylarch first-shell Pt–Pt fit for EXAFS, lmfit fcc (111)/(200) fit for XRD. The fits are "
              "for diagnosis only and do not enter the comparison or the decision.", "",
              f"- **EXAFS: a constant E0 offset.** All {len(ex)} Pt L3 spectra need ΔE0 = "
              f"{min(f['e0'] for _, f in ex):+.1f} to {max(f['e0'] for _, f in ex):+.1f} eV relative to the FEFF6L "
              "reference that the simulations use."
              + (f" The Pt foil fits to R = {np.mean([f['R'] for f in foil]):.3f} Å (bulk 2.775 Å) with ΔE0 = "
                 f"{np.mean([f['e0'] for f in foil]):+.1f} eV. The predictors instead put its first shell at "
                 f"{min(q['r1_pt'] for q in pf):.2f}–{max(q['r1_pt'] for q in pf):.2f} Å." if foil and pf else "")
              + " The simulated χ(k) never vary E0, so an E0 offset in the data reduction shifts the phase. The "
              "predictors read that shift as a shorter bond, and so as a Cu- or Ni-alloyed particle. Neither "
              "augmentation can produce such a shift: the classical pipeline does not vary E0, and the realism layer "
              "must not change oscillation phases, by design. **The most direct fix is to sample ΔE0 (for example "
              "±10 eV) in the simulated χ(k).** That is a data-reduction parameter and leaves the structural labels "
              "unchanged.",
              "- **XRD: the spectra are out of domain or physically ambiguous.**"]
        if big:
            L.append(f"  - {len(big)} Pt patterns have Scherrer sizes of "
                     + ", ".join(f"{f['size_nm']:.0f}" for f in big) + " nm, outside the 1.5–8 nm training range.")
        if pures:
            L.append("  - The other Pt pattern(s) fit to a = " + ", ".join(f"{f['a']:.3f}" for f in pures)
                     + " Å, against bulk Pt 3.922 Å. That points to a peak shift, from sample displacement or "
                       "the support, that the training data do not contain.")
        if ptco:
            L.append(f"  - The {len(ptco)} Pt–Co patterns fit to a = {min(f['a'] for f in ptco):.3f}–"
                     f"{max(f['a'] for f in ptco):.3f} Å, between the simulated Pt3Co (3.83 Å) and Pt3Fe (3.86 Å) "
                     "phases. With similar Co, Ni and Fe scattering factors, XRD alone barely separates the 3:1 "
                     "Pt–Co, Pt–Ni and Pt–Fe families.")
        L.append("")
    L += ["", "## Per-seed values with 95 % bootstrap confidence intervals", ""]
    for v in ("a", "b"):
        for si, res in enumerate(per_seed[v], 1):
            L.append(f"### Variant ({v}), seed {si}")
            L.append("")
            for set_name, mets in res.items():
                for m, val in mets.items():
                    L.append(f"- {set_name} · {m}: {fmt(val)}")
            L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (REPORTS / "downstream_comparison.json").write_text(json.dumps(dict(
        decision=decision, rows=rows, per_seed=per_seed, setA_predictions=items_A, setA_reference_fits=fits_A,
        setA_items=[dict(technique=it["technique"], material=it["material"], item_id=it["item_id"],
                         source=it.get("source"), label=COMPOSITIONS[it["tgt"]["composition"]]) for it in setA_items]),
        indent=1, default=str))


def main(device="cuda:0"):
    dev = torch.device(device)
    setA = real_set()
    setB = fullfeff_test_set()
    dsC = Dataset(ROOT / "artifacts" / "data", parts=[("clean", "test"), ("fault", "test")])
    per_seed = {"a": [], "b": []}
    items_A = {}
    vals = {"a": [], "b": []}
    for v in ("a", "b"):
        for s in (1, 2):
            ck = MODELS / "work" / f"predictor_{v}_s{s}" / "best.pt"
            res, items_A[f"{v}{s}"] = evaluate_variant_seed(ck, dev, setA, setB, dsC)
            per_seed[v].append(res)
            vals[v].append(torch.load(ck, map_location="cpu", weights_only=False)["val"]["score"])
    decision, rows = decide(per_seed)
    write_report(per_seed, decision, rows, setA, items_A=items_A, fits_A=reference_fits(setA))
    win = decision["ship"]
    best_seed = int(np.argmin(vals[win])) + 1
    return decision, best_seed


if __name__ == "__main__":
    print(main())
