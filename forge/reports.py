"""Markdown reports with figures (artifacts/reports/)."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import REPORTS

FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _ft(k, chi, rg):
    m = (k >= 3) & (k <= 12)
    y = chi[m] * k[m] ** 2 * np.hanning(m.sum())
    return np.abs((y[None, :] * np.exp(2j * k[m][None, :] * rg[:, None])).sum(1)) * (k[1] - k[0]) / np.sqrt(np.pi)


def exafs_validation_report(json_name="exafs_shortcut_validation", trials=None, history=None):
    d = json.loads((REPORTS / f"{json_name}.json").read_text())
    history_lines = ["| Run | Samples (seed) | (sample, edge) pairs | median R | 95th pct R | median amp. error | verdict |",
                     "|---|---|---|---|---|---|---|"]
    for name, fname, samp in (history or []):
        p = REPORTS / f"{fname}.json"
        if not p.exists():
            continue
        t = json.loads(p.read_text())
        tr = np.array([x["R"] for x in t["rows"]])
        ta = np.median(np.abs([x["amp_rel"] for x in t["rows"]]))
        tol_ = t["tolerances"]
        ok = np.median(tr) <= tol_["r_median"] and np.percentile(tr, 95) <= tol_["r_p95"] and ta <= tol_["amp_median"]
        history_lines.append(f"| {name} | {samp} | {len(tr)} | {np.median(tr):.4f} | {np.percentile(tr, 95):.4f} | "
                             f"{ta * 100:.1f} % | {'pass' if ok else 'FAIL'} |")
    rows, tol, k = d["rows"], d["tolerances"], np.array(d["k"])
    R = np.array([r["R"] for r in rows])
    amp = np.abs([r["amp_rel"] for r in rows])
    dr = np.abs([r["dR"] for r in rows])
    s = dict(n=len(rows), R_median=float(np.median(R)), R_p95=float(np.percentile(R, 95)), R_max=float(R.max()),
             amp_median=float(np.median(amp)), dR_median=float(np.median(dr)),
             R_ss_only_median=float(np.median([r["R_ss_only"] for r in rows])))
    passed = s["R_median"] <= tol["r_median"] and s["R_p95"] <= tol["r_p95"] and s["amp_median"] <= tol["amp_median"] \
        and s["dR_median"] <= tol["dr_median"]

    def group(key):
        g = defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        out = []
        for name, rs in sorted(g.items()):
            rr = np.array([x["R"] for x in rs])
            out.append(f"| {name} | {len(rs)} | {np.median(rr):.4f} | {np.percentile(rr, 95):.4f} | "
                       f"{np.median([abs(x['amp_rel']) for x in rs]) * 100:.1f} % | "
                       f"{np.median([x['R_part_ss'] for x in rs]):.4f} | {np.median([x['R_part_ms'] for x in rs]):.4f} |")
        return out

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(np.log10(R), bins=40, color="#4472c4")
    ax[0].axvline(np.log10(tol["r_p95"]), color="r", ls="--", label="p95 tolerance")
    ax[0].axvline(np.log10(tol["r_median"]), color="k", ls=":", label="median tolerance")
    ax[0].set_xlabel("log10 R-factor (k^2 chi, 3-12 1/A)")
    ax[0].legend(fontsize=8)
    ax[1].scatter([r["d_nm"] for r in rows], R, s=8, c=["#c00000" if r["edge"] != "Pt" else "#4472c4" for r in rows])
    ax[1].set_yscale("log")
    ax[1].set_xlabel("particle diameter (nm)")
    ax[1].set_ylabel("R-factor (blue Pt L3, red M K)")
    plt.tight_layout()
    plt.savefig(FIG / "exafs_validation_hist.png", dpi=110)
    plt.close()
    order = np.argsort(R)
    picks = [("best", order[0]), ("median", order[len(order) // 2]), ("95th pct", order[int(0.95 * (len(order) - 1))]),
             ("worst", order[-1])]
    fig, ax = plt.subplots(2, 4, figsize=(16, 6))
    rg = np.linspace(0, 6, 300)
    for j, (lab, i) in enumerate(picks):
        r = rows[i]
        cs, cf = np.array(r["chi_s"]), np.array(r["chi_f"])
        ax[0, j].plot(k, cf * k ** 2, label="full FEFF")
        ax[0, j].plot(k, cs * k ** 2, "--", label="shortcut")
        ax[0, j].set_title(f"{lab}: {r['sample_id']} {r['edge']} {r['composition']} {r['phase']} R={r['R']:.4f}", fontsize=8)
        ax[0, j].set_xlabel("k (1/A)")
        ax[1, j].plot(rg, _ft(k, cf, rg))
        ax[1, j].plot(rg, _ft(k, cs, rg), "--")
        ax[1, j].set_xlabel("R (A, no phase correction)")
    ax[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "exafs_validation_examples.png", dpi=100)
    plt.close()

    trial_lines = []
    if trials:
        trial_lines = ["", "### Development history (12-sample trial set, seed 1; not used for the final verdict)", "",
                       "| Revision | median R | 95th pct R | median first-shell amplitude error | verdict on trial set |",
                       "|---|---|---|---|---|"]
        for name, f in trials:
            p = REPORTS / f"{f}.json"
            if p.exists():
                t = json.loads(p.read_text())
                tr = np.array([x["R"] for x in t["rows"]])
                ta = np.median(np.abs([x["amp_rel"] for x in t["rows"]]))
                ok = np.median(tr) <= tol["r_median"] and np.percentile(tr, 95) <= tol["r_p95"] and ta <= tol["amp_median"]
                trial_lines.append(f"| {name} | {np.median(tr):.4f} | {np.percentile(tr, 95):.4f} | {ta * 100:.1f} % | "
                                   f"{'pass' if ok else 'fail'} |")

    md = [
        "# EXAFS shortcut validation", "",
        f"**Verdict: {'PASS' if passed else 'FAIL'}.** On {s['n']} (sample, edge) pairs from 200 random samples, the "
        f"shortcut χ(k) differs from full FEFF6L by a median R-factor of {s['R_median']:.4f} (tolerance "
        f"{tol['r_median']}), 95th percentile {s['R_p95']:.4f} (tolerance {tol['r_p95']}), worst {s['R_max']:.3f}; "
        f"median first-shell |χ(R)| peak error {s['amp_median'] * 100:.1f} % (tolerance {tol['amp_median'] * 100:.0f} %), "
        f"median peak-position error {s['dR_median']:.3f} Å (tolerance {tol['dr_median']} Å).", "",
        "## What is validated", "",
        "The dataset's χ(k) is not computed with FEFF per sample. It is synthesized from each particle's absorber-centred "
        "partial distance distributions (single scattering, r ≤ 6 Å) and from multiple-scattering paths of bulk FEFF "
        "runs of the same composition and phase, rescaled to the particle (`forge/exafs.py`):", "",
        "- **Single scattering:** every 0.004 Å distance bin is a path with the FEFF amplitude, phase, Re(p) and mean free "
        "path of the nearest reference shell from the bulk run of the same composition / phase (pair runs as fallback), "
        "placed at the bin distance through ΔR (larch path equation).",
        "- **Multiple scattering:** each bulk FEFF path (NLEG ≤ 4, half path ≤ 6 Å) is matched to a *motif*: species "
        "and neighbour shells of its 1–2 distinct scatterers and their mutual distance. Its degeneracy is scaled by "
        "the particle/bulk motif count. Its length is scaled by the particle/bulk mean motif length, and its σ² gets "
        "the particle's static motif variance. Paths with 3 distinct scatterers, about 5% of the MS amplitude, use a "
        "shell-occupancy rule.",
        "- **Thermal disorder** uses the same correlated model as the Debye calculations (σ² = (U_a + U_b)·max(0.35, 1 − δ2/R²)).", "",
        "## Protocol", "",
        "- 200 samples drawn at random (seed 8) from the pilot plan (all compositions, phases, shapes, sizes and faults); "
        "none of them was used during development.",
        "- Per sample and edge (Pt L3; M K for alloys) up to 8 random absorbers. Each absorber's cluster (atoms within "
        "7.5 Å) is run through FEFF6L (xraylarch 2026.3.1) with RMAX 6.0 Å, NLEG 4 and no path filter (CRITERIA 0 0). "
        "χ(k) is summed over all paths with the same thermal model and averaged over the absorbers. S0² = 1 on both sides.",
        "- The shortcut is evaluated on exactly the same absorbers and geometry (their own partial distributions and motif "
        "statistics). Clusters that hit FEFF6's path-hash collision were re-run with a ≤ 0.012 Å coordinate jitter; the "
        "shortcut used the same jittered geometry.",
        f"- {d['n_jobs']} FEFF runs on 28 CPU workers; {len(d['failed'])} failed runs.", "",
        "**Tolerances, fixed before the first trial run:** R-factor of k²χ(k) over k = 3–12 Å⁻¹, median ≤ 0.02 and "
        "95th percentile ≤ 0.05; first-shell Fourier peak (k², Hanning 3–12 Å⁻¹): median |relative amplitude error| ≤ 5 % "
        "and median |peak shift| ≤ 0.02 Å.", "",
        "## Results", "",
        "| Group | n | median R | 95th pct R | median amp. error | median SS part | median MS part |",
        "|---|---|---|---|---|---|---|",
        *group(lambda r: "all"),
        *group(lambda r: f"edge {r['edge']}"),
        *group(lambda r: f"phase {r['phase']}"),
        *group(lambda r: "d < 2.5 nm" if r["d_nm"] < 2.5 else ("2.5-4 nm" if r["d_nm"] < 4 else "d >= 4 nm")),
        *group(lambda r: f"fault {r['fault'] or 'none'}"), "",
        "SS / MS part: contribution of the single- / multiple-scattering difference to the R-factor (same denominator). "
        f"Without the MS term the median R-factor would be {s['R_ss_only_median']:.3f}, so the MS synthesis matters.", "",
        "![R-factor distribution](figures/exafs_validation_hist.png)", "",
        "![Examples](figures/exafs_validation_examples.png)", "",
        "## History: failures and fixes before scaling up", "",
        "The tolerances above were fixed before the first trial and never changed. Development used separate sample "
        "sets. The verdict at the top comes from 200 freshly drawn samples (seed 8) that were never used during "
        "development.", "",
        *history_lines, "",
        "**Fixes on the 12-sample trial set (seed 1)**, in this order:", "",
        "1. **Composition-specific single-scattering references.** Single-scattering amplitudes and phases come from "
        "the bulk run of the same composition and phase, using the nearest shell with ΔR. Before, pair references were "
        "interpolated between shells. This removed a 2–19 % first-shell amplitude bias.",
        "2. **Motif counting.** Multiple-scattering degeneracies are scaled by motif counts in the actual particle, "
        "computed on the GPU. Before, one coordination rule was applied to all paths.",
        "3. **Per-motif lengths and static variance.** Surface relaxation shortens chains near the surface; fixed "
        "lattice-scaled lengths gave phase errors on the strong focusing paths.",
        "4. **2π alignment of FEFF phases.** Phases are aligned before averaging the same shell across disorder "
        "configurations. FEFF writes each path's phase with an arbitrary 2π offset, and averaging them raw had shifted "
        "second-shell phases by about π/2.", "",
        "**Fixes after the failed first 200-sample run (seed 7)**, developed on an 80-sample set (seed 11):", "",
        "5. **Twin-boundary environments.** Decahedra and twinned particles get extra FEFF templates from fcc clusters "
        "with a {111} twin plane (absorbers on and next to the plane), and the motif shells include the hcp-stacking "
        "distances √(8/3)·r₁ and √(11/3)·r₁.",
        "6. **Icosahedral templates.** Mackay icosahedra are radially compressed fcc tetrahedra, so their templates "
        "come from interior absorbers of an 11-shell icosahedron of the same composition.",
        "7. **Correlated path-length variance.** The static variance of each multiple-scattering path length now "
        "includes the covariances between legs of the same atom pair (e.g. |XB| ≈ |XA| + |AB| on collinear chains). "
        "Treating legs as independent had doubled the static σ² of the strongest focusing paths.",
        "",
        "Turning off FEFF's path filter on both sides (CRITERIA 0 0) was also tested. It is fairer, because the filter "
        "depends on degeneracy, which differs between bulk and nanoparticle clusters, but it changed little; it is kept "
        "in all runs.",
        *trial_lines, "",
        "## Limitations", "",
        f"- **Icosahedra are the weak spot.** All of the worst cases are Mackay icosahedra (median R "
        f"{np.median([r['R'] for r in rows if r['shape'] == 'icosahedron']):.3f}, 95th percentile "
        f"{np.percentile([r['R'] for r in rows if r['shape'] == 'icosahedron'], 95):.3f}), while every other shape has a "
        f"95th percentile ≤ {max(np.percentile([r['R'] for r in rows if r['shape'] == sh], 95) for sh in ('cuboctahedron', 'truncated_octahedron', 'sphere', 'decahedron')):.3f}. "
        "The error is in multiple scattering on the strained five-fold geometry (single scattering stays below 0.01). "
        "Icosahedra are about 11 % of the dataset. Users who need the most accurate EXAFS can drop them; the "
        "shape label is stored with every sample.",
        f"- {100 * np.mean(R > 0.05):.1f} % of all (sample, edge) pairs exceed R = 0.05.",
        "- FEFF6L potentials are non-self-consistent (overlapped atoms) in both the reference runs and the full runs, so "
        "the validation measures the shortcut's error relative to FEFF6L, not FEFF's error relative to experiment.",
        "- The full calculation averages 8 absorbers per edge, and the shortcut is checked on the same 8. The dataset's "
        "χ(k) averages all absorbers.",
        "- XANES (μ(E) near the edge) is a simple arctan edge plus Lorentzian white line, not FEFF; only χ(k) is validated.",
    ]
    (REPORTS / "exafs_shortcut_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return passed, s


def _training_section():
    """Training runs and checkpoint selection, from artifacts/models/**/history.json and the selection records."""
    from .paths import MODELS, load_yaml
    lines = ["## Training runs and checkpoint selection", "",
             "Every 2,000 steps the EMA generator was scored on the **validation** split: score = |C2ST − 0.5| + 0.05 × "
             "(artifact-feature Fréchet distance of GAN output ÷ that of the raw twins). The C2ST here is the faster "
             "logistic-regression version used during training. The checkpoint with the lowest score was kept. "
             "Training stopped after 6 checkpoints without an improvement of at least 0.005. For PDF and EXAFS, the "
             "first run's discriminator won outright (hinge loss below 0.05), so a second run with stronger R1 "
             "regularization (γ = 10 instead of 1) was trained. The run with the better validation score was kept. "
             "The held-out evaluation split was not used for any of these choices.", "",
             "| Technique | Run | R1 γ | Best step | Stopped at | Validation C2ST at best | Feature-FD ratio at best | Shipped |",
             "|---|---|---|---|---|---|---|---|"]
    for t in ("xrd", "pdf", "exafs"):
        sel = REPORTS / f"realism_{t}_selection.json"
        if sel.exists():
            runs = json.loads(sel.read_text())["runs"]
        else:
            import torch
            # the shipped generator records its own step and validation diagnostics (training keeps a new best only if
            # the score improves by at least 0.005, so this can differ from the plain minimum over history.json)
            h = json.loads((MODELS / f"realism_{t}" / "history.json").read_text())
            ck = torch.load(MODELS / f"realism_{t}" / "generator_ema.pt", map_location="cpu", weights_only=False)
            cfg = load_yaml(MODELS / f"realism_{t}" / "config.yaml")
            runs = [dict(r1_gamma=cfg["r1_gamma"], best_step=ck["step"], last_step=h[-1]["step"],
                         best_val_c2st=ck["diag"]["c2st_acc"], best_val_feat_fd_ratio=ck["diag"].get("feat_fd_ratio"),
                         kept_as=f"realism_{t}")]
        for i, r in enumerate(runs, 1):
            shipped = r.get("kept_as") == f"realism_{t}"
            lines.append(f"| {t.upper()} | {i} | {r['r1_gamma']:g} | {r['best_step']} | {r['last_step']} | "
                         f"{r['best_val_c2st']:.3f} | {r['best_val_feat_fd_ratio']:.2f} | {'yes' if shipped else 'no'} |")
    lines += ["", "For PDF and EXAFS, the best checkpoint is an early one (step 2,000). Later in training the "
              "discriminator kept improving, and the generator's artifact statistics drifted away from the real "
              "ones; the feature-FD ratio grew by one to two orders of magnitude. Early stopping therefore selected "
              "generators that add modest, realistic-scale artifacts.", ""]
    return lines


LIMITATIONS = [
    "## Limitations", "",
    "- **The real reference data are mostly not Pt nanoparticles.**",
    "  - XRD: opXRD laboratory patterns of many inorganic phases, plus a few dozen Pt and Pt–Co catalyst patterns.",
    "  - EXAFS: reference compounds and catalysts from open XAS databases, at several edges.",
    "  - PDF: colloidal Au nanoparticles from one self-driving-lab campaign.",
    "",
    "  The realism models learn instrument and processing artifacts of these sources, not of Pt nanoparticles as such.",
    "- **The physics twins are approximations of the measured materials:** a labelled structure for opXRD, bulk FEFF "
    "for EXAFS, and a Debye-simulated particle for PDF. Content differences between twin and real spectrum (phase "
    "impurities, structure, disorder) cannot be removed by an artifacts-only generator, and they dominate the C2ST: "
    "even the best alternative stays well above 50 %.",
    "- **Small held-out sets:** 55 XRD, 10 PDF and 74 EXAFS spectra. Fold-to-fold standard deviations of the C2ST are "
    "4–10 percentage points, and the PDF numbers are indicative only.",
    "- **C2ST and Fréchet distance can disagree.** The C2ST sees the spectrum directly, including backgrounds and noise. "
    "The Fréchet distance sees only what the predictor's encoder represents, which is trained to ignore artifacts.",
    "",
]


def gan_diagnostics_report(results):
    """results: list of forge.diagnostics.run outputs (one per technique)."""
    lines = ["# GAN realism diagnostics", "",
             "All realism numbers are measured on real spectra held out from GAN training and checkpoint selection "
             "(the 20 % evaluation split of Phase 3, items with a usable physics twin). Each real spectrum is compared "
             "with three alternatives built from its physics twin: the twin alone, the twin with classical "
             "augmentation (the baseline, variant (a)), and the twin passed through the GAN realism layer.", "",
             "## Method", "",
             "- **C2ST:** a classifier two-sample test. Gradient-boosted trees (scikit-learn "
             "`HistGradientBoostingClassifier`, 150 iterations, 15 leaves) separate real from alternative spectra, "
             "using spectrum, derivative and log power-spectrum features, with 5-fold cross-validation. 50 % means "
             "indistinguishable; 100 % means perfectly separable. The minimum leaf size is max(2, min(20, training "
             "fold / 10)); with the library default of 20, the trees cannot split at all on the 10 held-out PDFs.",
             "- **Fréchet distance:** computed in the embedding space of the technique's encoder from predictor (a), "
             "seed 1, which never saw GAN output. Embeddings are projected onto their first d principal components, "
             "d = min(32, n/3), so each covariance has at least 3 samples per dimension. Lower is closer to real.",
             "- **Diversity:** 64 fixed inputs, 16 latent draws each. The mean pairwise L2 distance between draws "
             "(relative to the input norm) and the variance of each learned artifact field across draws.",
             "- **Label preservation:** 500 Concord Forge test spectra are fitted with standard tools before and after "
             "the realism layer, using the same tool and starting values:",
             "  - XRD: lmfit, fcc (111)+(200) with Kα2 → lattice parameter.",
             "  - PDF: diffpy.pdffit2, fcc with spherical envelope → lattice parameter and diameter.",
             "  - EXAFS: xraylarch feffit, first shell → N and R.",
             "",
             "  A spectrum **fails** if a fitted label moves by more than the tolerance between the two fits. "
             "\"Recovery\" is reported separately: how often each fit lands within an absolute tolerance of the "
             "simulation's label. It measures the fitting model itself, for example a single-path EXAFS fit applied "
             "to the mixed Pt–Pt/Pt–M first shell of an alloy.", "",
             "## Summary", "",
             "| Technique | held-out real | C2ST twin | C2ST classical | C2ST GAN | FD twin | FD classical | FD GAN | "
             "label-preservation failures |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r.get("shipped", True):
            lines.append(f"| {r['technique'].upper()} | – | – | – | – | – | – | – | no trained model |")
            continue
        c, fd, lp = r["c2st"], r.get("frechet", {}), r["label_preservation"]
        lines.append(f"| {r['technique'].upper()} | {r['n_eval_real']} | {c['raw_twin']['acc']:.3f} | "
                     f"{c['classical']['acc']:.3f} | {c['gan']['acc']:.3f} | {fd.get('raw_twin', float('nan')):.3g} | "
                     f"{fd.get('classical', float('nan')):.3g} | {fd.get('gan', float('nan')):.3g} | "
                     f"{100 * lp['failure_rate']:.1f} % ({int(round(lp['failure_rate'] * lp['n']))}/{lp['n']}) |")
    lines.append("")
    lines += _training_section()
    for r in results:
        t = r["technique"]
        lines += [f"## {t.upper()}", ""]
        if not r.get("shipped", True):
            lines += ["No trained realism model.", ""]
            continue
        c, fd = r["c2st"], r.get("frechet", {})
        n = r["n_eval_real"]
        lines += [f"Held-out real spectra: {n}. Fréchet dimension d = {r.get('frechet_dim', 32)}.", ""]
        if r.get("figure"):
            lines += [f"![{t.upper()} examples: held-out real, physics twin, GAN output]({r['figure']})", ""]
        if n < 30:
            lines += [f"**Low power:** with only {n} held-out real spectra, C2ST accuracies have fold standard deviations "
                      "of tens of percentage points, and the Fréchet distance uses very few dimensions. Treat these "
                      "numbers as indicative only.", ""]
        lines += ["| Alternative compared with real | C2ST accuracy | fold std | Fréchet distance |", "|---|---|---|---|"]
        for name, lab in (("raw_twin", "simulated twin, no artifacts"), ("classical", "twin + classical augmentation"),
                          ("gan", "twin + GAN realism layer")):
            lines.append(f"| {lab} | {c[name]['acc']:.3f} | {c[name]['fold_std']:.3f} | {fd.get(name, float('nan')):.3g} |")
        g, tw, cl = (abs(c[k]["acc"] - 0.5) for k in ("gan", "raw_twin", "classical"))
        fdv = ""
        if fd:
            fdv = (f" By Fréchet distance it is {'closer' if fd['gan'] < fd['raw_twin'] else 'not closer'} than the raw "
                   f"twins and {'closer' if fd['gan'] < fd['classical'] else 'not closer'} than classical augmentation.")
        lines += ["", f"**Realism:** by C2ST, the GAN output is {'closer' if g < tw else 'not closer'} to real than the "
                      f"raw twins and {'closer' if g < cl else 'not closer'} than classical augmentation.{fdv}", ""]
        d = r["diversity"]
        lp = r["label_preservation"]
        lines += [f"**Diversity:** 16 latent draws for the same input differ by {d['rel_pairwise_distance']:.4f} of the "
                  f"spectrum norm (mean pairwise distance). Variance across draws: background {d['var_bg']:.3g}, gain "
                  f"{d['var_gain']:.3g}, noise scale {d['var_nscale']:.3g} "
                  f"({'no sign of mode collapse' if d['rel_pairwise_distance'] > 1e-3 else 'possible mode collapse'}).", "",
                  f"**Label preservation** ({lp['n']} spectra; tolerance {lp['tolerance']}):",
                  "",
                  f"- **Failure rate:** {100 * lp['failure_rate']:.1f} %, counting a fit failure of either spectrum as a "
                  f"failure. Fits failed on {lp.get('n_fit_failed_clean', 'n/a')} clean spectra and on "
                  f"{lp.get('n_fit_failed_gan', 'n/a')} more after the realism layer.",
                  f"- **Among the {lp['n_fit_ok']} spectra where both fits converged:** "
                  f"{100 * lp.get('failure_rate_fit_ok', float('nan')):.1f} % moved a label beyond tolerance.",
                  f"- **Recovery of the simulation's label:** {100 * lp['recover_clean']:.1f} % of clean fits and "
                  f"{100 * lp['recover_gan']:.1f} % of fits after the realism layer.", ""]
    lines += LIMITATIONS
    (REPORTS / "gan_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
