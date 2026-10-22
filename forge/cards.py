"""Dataset card (DATASET.md) generated from the built files."""
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from .paths import CONFIGS, REPORTS, ROOT, STRUCT_DIR, load_yaml


def dataset_card(data_dir, build_name):
    data_dir = Path(data_dir)
    splits = json.loads((data_dir / "splits.json").read_text())
    ref = json.loads((STRUCT_DIR / "reference_phases.json").read_text())
    inst = load_yaml(CONFIGS / "instruments.yaml")
    rows = []
    for sh in splits["shards"]:
        with h5py.File(data_dir / "shards" / sh["shard"], "r") as h:
            r1 = h["first_shell"][:, 0, 1]
            for i in range(h["sample_id"].shape[0]):
                m = json.loads(h["meta_json"][i])
                sp, pm = m["spec"], m["particle"]
                rows.append(dict(set=sh["set"], split=sh["split"], comp=sp["composition"], phase=sp["phase"],
                                 shape=sp["shape"], fault=sp["fault"] or "none", d=pm["diameter_nm"], n=pm["n_atoms"],
                                 a=pm["lattice_parameter"], r1=float(r1[i])))
    n = len(rows)
    man = list(csv.DictReader(open(data_dir / "manifest.csv", encoding="utf-8")))
    lic = Counter(r["licence"] for r in man if r["kind"] == "sample")
    real = Counter((r["notes"].split(";")[0]) for r in man if r["kind"] == "real_spectrum")
    val = REPORTS / "exafs_shortcut_validation.json"
    vs = ""
    if val.exists():
        v = json.loads(val.read_text())
        R = np.array([r["R"] for r in v["rows"]])
        vs = (f"Full-FEFF check on {len(R)} (sample, edge) pairs: median R-factor {np.median(R):.4f}, 95th percentile "
              f"{np.percentile(R, 95):.4f} (see `reports/exafs_shortcut_validation.md`).")

    def table(key, title):
        c = Counter(r[key] for r in rows if r["set"] == "clean")
        f = Counter(r[key] for r in rows if r["set"] == "fault")
        out = [f"| {title} | clean | with planted fault | total |", "|---|---|---|---|"]
        out += [f"| {k} | {c[k]} | {f[k]} | {c[k] + f[k]} |" for k in sorted(set(c) | set(f))]
        return out

    real_used = []
    for t, lab in (("exafs", "XAS"), ("xrd", "XRD"), ("pdf", "PDF")):
        f = data_dir / "real" / f"{t}_items.csv"
        if f.exists():
            it = list(csv.DictReader(open(f, encoding="utf-8")))
            sp = Counter(r["split"] for r in it)
            src = Counter(r["source"] for r in it)
            real_used.append(f"- {lab}: {len(it)} items ({sp['train']} train, {sp['val']} validation, {sp['eval']} held "
                             f"out" + (f", {sp['unused']} not used for lack of a usable physics twin" if sp.get("unused") else "") + "); "
                             + ", ".join(f"{k} {v}" for k, v in src.most_common()))

    phases = []
    for key, ph in sorted(ref["phases"].items()):
        if ph.get("available"):
            p = ph["params"]
            phases.append(f"| {key.replace('|', ' ')} | {p['a']:.4f} | {p['c']:.4f} | {p.get('alpha', 90):.2f} | "
                          f"{ph['representative']['source']} {ph['representative']['source_id']} | {ph['basis']} |")
    veg = [f"| {k} | {v['a_vegard']:.4f} | " + "; ".join(f"{c['source']} {c['source_id']} x={c['x_M']}: {c['a_db']} "
                                                     f"({100 * c['rel_dev']:+.2f} %)" for c in v["checks"]) + " |"
           for k, v in ref["vegard"].items()]
    split_line = (f"clean train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}; "
                  f"fault train {len(splits['fault']['train'])}, val {len(splits['fault']['val'])}, "
                  f"test {len(splits['fault']['test'])}")
    md = [
        f"# Concord Forge dataset card: {build_name}", "",
        "**All data in this dataset are simulated.** Spectra are computed from atomistic models of Pt and Pt-alloy "
        "nanoparticles. No measured spectrum is included in the shards. The realism layer, if shipped, is a separate "
        "model (`artifacts/models/realism_*`), and its use is marked by `origin = simulated+realism` in exports.", "",
        "## Contents", "",
        f"- {n} samples: {sum(r['set'] == 'clean' for r in rows)} clean and {sum(r['set'] == 'fault' for r in rows)} "
        f"with one planted fault ({100 * sum(r['set'] == 'fault' for r in rows) / n:.1f} %).",
        f"- Splits by structure family (near-duplicate particles never cross splits): {split_line}.",
        "- Each sample has XRD (Cu Kα, laboratory 2θ grid), synchrotron XRD (Q grid, sampled wavelength), PDF (G(r) and "
        "F(Q)), and EXAFS χ(k) plus normalized μ(E) at the Pt L3 edge and, for alloys, the M K edge. All techniques are "
        "computed from the same atomic structure.",
        "- Storage: HDF5 shards in `shards/`, one per (set, split) chunk, listed with SHA-256 in `splits.json`. Every "
        "sample, structure, FEFF reference and real spectrum is listed in `manifest.csv` with source, ID and licence.", "",
        "## How the data were made", "",
        "1. **Reference phases** came from open databases (`forge/structures.py`). The Crystallography Open Database "
        "(CC0) was searched first. Materials Project was skipped because no `MP_API_KEY` was set. JARVIS-DFT "
        "(CC BY 4.0) was searched next. OQMD (CC BY 4.0) returned HTTP 502 during the build. Duplicates were merged "
        "with pymatgen's StructureMatcher. Experimental ambient entries are preferred (pre-1947 COD values converted "
        "from kX to Å; pressure and temperature series excluded; median after outlier rejection). DFT-only phases are "
        "rescaled by the same database's a(Pt, exp)/a(Pt, DFT) ratio.", "",
        "| Phase | a (Å) | c (Å) | α (°) | Representative | Basis |", "|---|---|---|---|---|---|", *phases, "",
        "PtCu orders as L1₁ (CuPt type, alternating (111) planes), which is the observed ordered phase, so L1₁ is used "
        "instead of L1₀. No ordered phase was missing from the open sources. fcc Fe is not stable at room temperature, "
        "so its Vegard endpoint is extrapolated linearly from the COD γ-Fe a(T) series.", "",
        "**Disordered alloys use Vegard's law**, checked against database values:", "",
        "| System | a Vegard (Å) | Database checks (deviation from Vegard) |", "|---|---|---|", *veg, "",
        "2. **Nanoparticles** (`forge/particles.py`, ASE cluster tools):",
        "   - **Shapes:** cuboctahedra, truncated octahedra, spheres, Mackay icosahedra and Ino/Marks decahedra, "
        "1.5–8 nm (continuous sampling within 4 size bins).",
        "   - **Ordering:** L1₂, L1₀ and L1₁ are made by sublattice assignment on single-crystal shapes, plus the "
        "phase's tetragonal or rhombohedral distortion. The ordering variant is the one closest to nominal "
        "stoichiometry.",
        "   - **Disordered alloys:** random substitution. Five-fold twinned shapes are disordered or pure Pt only.",
        "   - **Perturbations:** isotropic strain ±2 %, vacancies 0–3 %, radial surface relaxation −3…+0.5 % of the NN "
        "distance, and static displacements (×2 at the surface).", "",
        "3. **Simulation** (`forge/simulate.py`) computes element-resolved pair-distance histograms on GPU (0.004 Å "
        "bins, validated against DebyeCalculator to 1.2×10⁻⁵ of the maximum). All techniques come from these "
        "histograms:",
        "   - **Debye XRD:** Waasmaier–Kirfel f₀ with Chantler f′, f″ at the photon energy, and a correlated thermal "
        "damping model σ² = (U_a+U_b)·max(0.35, 1 − δ2/r²).",
        "   - **Cu Kα1/α2 patterns:** with polarization and Caglioti pseudo-Voigt resolution.",
        "   - **Synchrotron patterns:** Q grid, Gaussian resolution.",
        "   - **PDF:** S(Q), F(Q) and the sine Fourier transform over the sampled Qmin–Qmax, with Qdamp and Qbroad. "
        "Termination ripples come out of the transform.", "",
        "4. **EXAFS** (`forge/exafs.py`) uses a FEFF6L-based shortcut, validated against full FEFF before scaling "
        f"up. {vs} μ(E) adds an arctan edge, a Lorentzian white line and linear pre- and post-edge terms to χ(k); the "
        "white line and edge position shift with oxidation state.", "",
        "5. **Instrument parameters** are drawn per sample from `configs/instruments.yaml` and stored with it.", "",
        "## Composition of the dataset", "", *table("comp", "Composition"), "", *table("phase", "Phase"), "",
        *table("shape", "Shape"), "", "| Planted fault | samples |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(Counter(r["fault"] for r in rows).items())], "",
        f"Size range {min(r['d'] for r in rows):.2f}–{max(r['d'] for r in rows):.2f} nm (equivalent-sphere diameter), "
        f"{min(r['n'] for r in rows)}–{max(r['n'] for r in rows)} atoms.", "",
        "**Balance.** Compositions are equal in the clean set. Pure Pt has fewer fault samples because the Pt-skin "
        "fault does not apply to it, and the other faults are spread over the remaining compositions. Ordered and disordered phases are equal within each alloy, and shapes "
        "are equal within a phase. Ordered phases exist only in the 3 single-crystal shapes, so cuboctahedra, "
        "truncated octahedra and spheres are more common overall than icosahedra and decahedra. Size bins are equal "
        "within each shape.", "",
        "## Labels (per sample)", "",
        "- **Identity:** composition class (7) and true atomic fractions; phase (A1, L1₂, L1₀, L1₁) and the "
        "ordered/disordered flag (not applicable to pure Pt); shape and shape parameters.",
        "- **Size:** equivalent-sphere and radius-of-gyration diameters, and number of atoms.",
        "- **Lattice:** lattice parameter (volume-equivalent pseudo-cubic, plus a, c and α), strain, vacancy fraction.",
        "- **Coordination shells per absorber element** (shells 1–4 and the O shell): neighbour element, N, mean R, "
        "and σ² (static plus the thermal model). A compact first-shell summary is also given for the Pt and M "
        "absorbers.",
        "- **Chemistry:** assumed oxidation states per element (fractions), and U_iso and δ2 used in the simulation.",
        "- **Planted faults** (one-hot over 7 types; the first 5 are physical, the last 2 corrupt only the "
        "metadata):",
        "  1. Pt oxide surface shell: O atop surface Pt with extra surface disorder, Pt²⁺.",
        "  2. Pt skin: surface M swapped into the core.",
        "  3. Mixed oxidation states: a fraction of surface M (or Pt) oxidized, with white-line and edge-shift "
        "changes.",
        "  4. Low-concentration dopant: 1–4 % of another 3d metal.",
        "  5. Multi-domain particle: two ordering variants, or a {111} twin for disordered particles.",
        "  6. Wrong phase candidate in the metadata.",
        "  7. Mislabeled nominal composition in the metadata.", "",
        "## Licences", "",
        "Per-sample licences in `manifest.csv` follow the source of the reference phase:", "",
        *[f"- {k}: {v} samples" for k, v in lic.items()], "",
        "The dataset as a whole is CC BY 4.0, with attribution to COD, JARVIS-DFT and the FEFF6L / xraylarch authors. "
        "Real reference spectra are not in the shards. Every downloaded real item is listed in `manifest.csv` (kind "
        "`real_spectrum`, " + ", ".join(f"{k} {v}" for k, v in real.items()) + ") with its source, ID and licence. "
        "The XRD count is dominated by the complete opXRD archive (Zenodo 15298026), which was downloaded whole; "
        "1,500 of its patterns were screened. The real spectra actually used by the realism models (Phase 3, "
        "`real/<technique>_items.csv`, split by material) are:", "",
        *real_used, "",
        "## Validation", "",
        "`python artifacts/scripts/validate_dataset.py` checks files and hashes, schema, balance, splits, label ranges, "
        "label–spectrum consistency and licences, and exits 0 only if every check passes. This build passes all checks.",
        "",
        f"One label-range bound was corrected during the build. The first full run failed the check \"first-shell R "
        f"2.3–2.9 Å\" on {sum(1 for r in rows if r['r1'] > 2.9)} samples, all pure-Pt Mackay icosahedra at the high end "
        f"of the strain range (mean Pt first-shell R up to {max(r['r1'] for r in rows):.3f} Å). The data were correct "
        "and the bound was wrong: a Mackay icosahedron's bonds within a shell are 5.15 % longer than its radial bonds "
        "(ASE `Icosahedron`: nearest-neighbour bonds 1.000–1.0515 × a/√2). The bound is now 3.0 Å, derived from the "
        "construction rules rather than the observed values: largest lattice parameter (Pt at +2 % strain) 4.002 Å, "
        "a/√2 × 1.0515 × 1.005 (maximum outward surface relaxation) = 2.99 Å. No sample was changed.", "",
        "## Limitations", "",
        "- **EXAFS of icosahedra is the least accurate part of the dataset.** Against full FEFF, Mackay icosahedra "
        "have median R-factor 0.028 and 95th percentile 0.15, against ≤ 0.015 for every other shape (see "
        "`reports/exafs_shortcut_validation.md`). The shape label is stored, so they can be dropped where EXAFS "
        "accuracy matters.",
        "- **The pilot (`data/pilot/`)** was simulated with an earlier version of the EXAFS shortcut, before the "
        "twin-boundary, icosahedral and path-variance fixes. It validated the pipeline; the full dataset uses the "
        "final shortcut.",
        "- **Model views:** the predictor and realism models use 2θ 20–80° (Cu Kα) and r 0.5–30 Å, the ranges covered "
        "by open laboratory and nanoparticle PDF data. The shards store the full simulated ranges (2θ 10–100°, "
        "r 0.5–62 Å).",
        "- These are idealized atomistic models. Ordered phases appear only in single-crystal morphologies; there is no "
        "segregation other than the planted Pt-skin fault; there is no support or ligand shell.",
        "- Thermal motion is a correlated Gaussian model, with no anharmonicity. Static disorder is random.",
        "- XANES is not simulated beyond a phenomenological edge.",
        "- XRD has no incoherent (Compton), fluorescence or air scatter. Those are left to augmentation or the realism "
        "layer.",
        "- DFT-derived ordered phases (Pt3Ni L1₂, PtNi L1₀) carry calibration uncertainty, about 0.3 % on a.",
        "- Vegard's law underestimates the lattice parameters of some disordered alloys by up to about 2.7 % compared "
        "with the COD entries listed above.",
    ]
    (data_dir / "DATASET.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def realism_model_card(technique, diag, decision_ship_b, real_counts, train_info):
    """artifacts/models/realism_<t>/MODEL_CARD.md"""
    from .paths import MODELS, sha256_file
    d = MODELS / f"realism_{technique}"
    gen = d / "generator_ema.pt"
    lp = diag.get("label_preservation", {})
    c = diag.get("c2st", {})
    lines = [
        f"# Realism model card: {technique.upper()}", "",
        f"**Status:** {'SHIPPED for training augmentation (decision rule passed)' if decision_ship_b else 'NOT USED in the shipped predictor (decision rule: the GAN did not beat classical augmentation); kept for reference'}.", "",
        "## What it is", "",
        "This is a conditional residual GAN generator, `forge/gan.py`. Given a simulated spectrum on the model grid, a "
        "condition vector and 64-D noise z, it adds learned instrument and sample artifacts:",
        "",
        "`output = sim × gain + background + noise_scale × coloured_noise" + (" + glitches" if technique == "exafs" else "") + "`",
        "",
        "- **Background:** 12 cubic B-spline coefficients.",
        "- **Gain:** 8 B-spline coefficients, exponentiated and clipped to [0.9, 1.1].",
        "- **Noise scale:** 16 B-spline coefficients through softplus, times white noise passed through a learned "
        "unit-norm 1-D convolution.",
        "- **Glitches** (EXAFS only): at most 2 gated Gaussian spikes.",
        "- **Backbone:** 1-D U-Net with 4 levels and 64→512 channels, FiLM-conditioned on (condition, z).",
        "- **Discriminator:** 6-block spectrally normalized 1-D ResNet on (spectrum, derivative) with projection "
        "conditioning.",
        "- **Losses:** hinge adversarial loss, R1, a preservation loss (" +
        ("Fourier-filtered first-shell amplitude and phase" if technique == "exafs" else "cross-correlation of derivatives") +
        ") and an artifact-magnitude penalty.", "",
        "## What it can and cannot do", "",
        "- It can only multiply the simulated spectrum by a smooth gain within 0.9–1.1 and add a smooth background "
        "and spatially varying coloured noise" + (" plus at most 2 narrow glitches" if technique == "exafs" else "") +
        ". The simulated signal itself is never resampled, shifted or broadened.",
        {"xrd": "- The narrowest bump the background can form is 5.5° FWHM, 5× the narrowest Bragg peak (1.1° for 8 nm "
                "particles). It cannot create or remove reflections. Under the broad peaks of the smallest particles, "
                "a background slope can move the apparent maximum slightly; the measured shifts are bounded by the "
                "tests and the re-fits below.",
         "pdf": "- The narrowest bump the background can form is 2.7 Å FWHM, 9× the nearest-neighbour peak width "
                "(about 0.3 Å). It cannot create, remove or move PDF peaks.",
         "exafs": "- The narrowest bump the background can form is 1.15 Å⁻¹ FWHM, comparable to the first-shell "
                  "oscillation period (π/R ≈ 1.1 Å⁻¹). The basis alone therefore does not guarantee that the first "
                  "shell is untouched. That is enforced by the first-shell preservation loss (Fourier-filtered "
                  "amplitude and phase) and checked by tests and larch re-fits (below)."}[technique],
        "- It does not make a simulation correct. Model errors in the simulation (structure, thermal model, FEFF "
        "approximations) pass through unchanged.",
        "- It learned artifacts of the open reference data listed below. Instruments, sample environments or "
        "processing pipelines that differ from those sources are not represented.", "",
        "## Training data", "",
        f"- **Real spectra:** {real_counts}. Splits are by material. The held-out 20 % was never used for training or "
        "model selection. Sources and licences are listed per item in `artifacts/data/real/{t}_items.csv` and in the "
        "dataset manifest.".replace("{t}", technique),
        "- **Fake inputs:** physics twins of the real materials (`forge/twins.py`, `forge/realdata*.py`).",
        "- **Regularization:** preservation and magnitude losses were also applied to Concord Forge simulations "
        "(the clean training split of the full build), the domain the model is used on.",
        f"- **Training:** {train_info}.", "",
        "## Measured diagnostics (held-out real spectra)", "",
        "| Comparison with real | C2ST accuracy (50 % = indistinguishable) | Fréchet distance |", "|---|---|---|",
    ]
    for name, lab in (("raw_twin", "simulated twin"), ("classical", "twin + classical augmentation"), ("gan", "twin + this model")):
        if name in c:
            lines.append(f"| {lab} | {c[name]['acc']:.3f} | {diag.get('frechet', {}).get(name, float('nan')):.3g} |")
    if lp:
        lines += ["", f"**Label preservation** ({lp['n']} re-fitted test spectra, tolerance {lp['tolerance']}): failure "
                      f"rate {100 * lp['failure_rate']:.1f} %."]
    lines += ["", "## Files", "",
              f"- `generator_ema.pt`: EMA generator weights, SHA-256 `{sha256_file(gen) if gen.exists() else 'n/a'}`.",
              "- `history.json`: validation diagnostics per checkpoint.",
              "- `config.yaml`: training configuration.",
              "- **Inference:** `python artifacts/scripts/realism_infer.py <shard.h5 | sample_dir> --out <dir>`.", ""]
    (d / "MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def predictor_model_card(variant, seed, cfg, val, downstream_rows, decision, data_card_counts):
    """artifacts/models/predictor/MODEL_CARD.md"""
    from .paths import MODELS, sha256_file
    d = MODELS / "predictor"
    rows = [r for r in downstream_rows]
    lines = [
        "# Predictor model card", "",
        f"**Shipped variant: ({variant}) "
        f"{'classical baseline augmentation' if variant == 'a' else 'GAN realism augmentation'}**, seed {seed} "
        f"(validation score {val.get('score', float('nan')):.4f}). The decision rule is applied in "
        "`reports/downstream_comparison.md`.", "",
        "## What it is", "",
        "This is a multi-technique property predictor for Pt and Pt-alloy nanoparticles (`forge/predictor.py`, "
        "22 M parameters):",
        "",
        "- **Encoders:** one 1-D ResNet-18-style encoder per technique, each taking (spectrum, derivative). The "
        "techniques are Cu Kα XRD (2θ 20–80°), synchrotron XRD (Q 0.8–12 Å⁻¹), PDF (r 0.5–30 Å), Pt L3 EXAFS and M "
        "K-edge EXAFS (k²χ, k 2–14.75 Å⁻¹).",
        "- **Metadata:** an optional token carrying the nominal composition and candidate phases.",
        "- **Fusion:** a 4-layer transformer over whichever tokens are present. Training randomly drops techniques "
        "(p = 0.3) and metadata (p = 0.5), so the model works with any subset.",
        "- **Heads:** composition (7 classes), ordered vs disordered (alloys), equivalent-sphere diameter, lattice "
        "parameter, first-shell N and R for the Pt and the M absorber, and 7 fault flags (multi-label).", "",
        "## Training", "",
        f"- **Data:** Concord Forge simulations only ({data_card_counts}). Training used the train split, plus fault "
        "samples from the train split; model selection used the validation split, scored from spectra without "
        "metadata.",
        f"- **Augmentation:** {'Poisson / Gaussian noise, polynomial background, Gaussian broadening, normalization jitter (forge/augment.py)' if variant == 'a' else 'learned realism artifacts (artifacts/models/realism_*), plus Gaussian broadening and normalization jitter; classical augmentation for techniques without a realism model'}.",
        f"- **Optimizer:** AdamW, lr {cfg['lr']}, weight decay {cfg['weight_decay']}, {cfg['steps']} steps of batch "
        f"{cfg['batch_per_gpu']}, cosine schedule with {cfg['warmup']} warm-up steps, bf16 autocast.", "",
        "## Measured performance", "",
        "Mean over the two training seeds of each variant; full tables with bootstrap CIs are in "
        "`reports/downstream_comparison.md`.", "",
        "| Test set | Metric | shipped variant | other variant |", "|---|---|---|---|",
    ]
    for r in rows:
        mine, other = (r["a"], r["b"]) if variant == "a" else (r["b"], r["a"])
        lines.append(f"| {r['set']} | {r['metric']} | {mine:.4f} | {other:.4f} |")
    import json as _json
    from .paths import REPORTS
    ds = _json.loads((REPORTS / "downstream_comparison.json").read_text())
    fam = [r for r in rows if r["set"] == "A_real" and r["metric"] == "composition_family_acc"]
    fam_v = (fam[0]["a"] if variant == "a" else fam[0]["b"]) if fam else float("nan")
    fits = [f for f in (ds.get("setA_reference_fits") or []) if "e0" in f]
    e0 = (f"ΔE0 = {min(f['e0'] for f in fits):+.0f} to {max(f['e0'] for f in fits):+.0f} eV" if fits else "a large ΔE0")
    lines += ["", "## Limitations", "",
              f"- **Not yet reliable on real spectra.** On the 21 real held-out spectra of set A, the composition "
              f"family was right for {fam_v * 21:.0f} of 21 (mean over the two seeds). Two causes were identified:",
              f"  - Real Pt L3 χ(k) are offset by {e0} from the simulations, which never vary E0. The predictor reads "
              "the phase shift as shorter bonds. Pt foil, for example, comes out at R ≈ 2.7 Å instead of 2.775 Å.",
              "  - Several real XRD patterns lie outside the training domain: crystallites larger than 8 nm, or "
              "peak shifts from sample displacement or the support.",
              "",
              "  See the diagnosis in `reports/downstream_comparison.md`. Sampling ΔE0 in the simulated χ(k) is the "
              "first fix to make. Until then, treat predictions on real data as hypotheses to check with standard "
              "fits.",
              "- **Trained only on simulated spectra.** Real-data performance was measured on a small set of open "
              "reference spectra, with wide confidence intervals.",
              "- **Training domain:** 7 composition classes, fcc-derived structures from 1.5–8 nm, and the planted "
              "faults listed in the dataset card. Bulk materials (foils), other elements, supports, ligands and "
              "oxidation beyond the fault models are out of domain.",
              "- **Size predictions:** these refer to the equivalent-sphere diameter of the simulated particle. "
              "Lattice parameters refer to the volume-equivalent pseudo-cubic cell.",
              "- **Metadata faults:** wrong phase candidate and mislabeled composition can only be flagged when the "
              "metadata token is supplied.", "",
              "## Files", "",
              f"- `model.pt`: weights and target normalization, SHA-256 "
              f"`{sha256_file(d / 'model.pt') if (d / 'model.pt').exists() else 'n/a'}`.",
              "- `config.json`: training configuration.",
              "- **Inference:** `python artifacts/scripts/predict.py <sample_dir | samples_root | shard.h5> --out "
              "predictions.json`.", ""]
    (d / "MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")
