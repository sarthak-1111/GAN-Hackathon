# Prompt 2 – Concord Forge: Pt / Pt-alloy dataset + GAN realism model (4 × A6000)

Paste everything below the line into Claude Code on the GPU workstation. Optional: put `MP_API_KEY=<key>` in `.env` (gitignored) for extra structures.

---

You are building **Concord Forge**, the data and model factory for Concord (a multi-technique XRD + PDF + XAFS analysis agent). Everything runs locally on this machine. The deliverables are:

1. A large, labelled, physics-simulated dataset of Pt and Pt-alloy nanoparticles (XRD, PDF and EXAFS from the same structure).
2. A trained **conditional GAN realism model** per technique that adds learned instrument and sample artifacts to simulated spectra without changing any label-relevant feature.
3. A trained **multi-technique predictor** used to test whether the GAN actually helps, delivered with whichever training variant wins.
4. Inference scripts, model cards, a dataset card and validation reports.

No LLM is needed.

## Hard rules

1. **Physics labels are the truth.** The GAN only adds artifacts (background, gain, noise, glitches). It must never move peaks, change oscillation frequencies or alter anything a label depends on. Enforce this by design (residual generator) and by tests.
2. **Open data only.** Structures and reference spectra come from openly licensed sources. Record source, ID and licence of every item in `manifest.csv`.
3. **Verify before depending.** Check every library API and version against current docs or source. Pin versions.
4. **Reproducible.** Fixed seeds, a config file for every run, content hashes for every output, resumable jobs.
5. **Honest results.** The GAN is used only if it beats classical augmentation on the downstream test. Otherwise ship the baseline and say so. Never tune a test to make it pass.
6. **Secrets.** `.env` is gitignored; never log keys.

## Phase 0 – Hardware check, then use everything

- Confirm 4 GPUs are visible with ~48 GB each, the driver and GPU runtime versions, bf16 support, and that a 4-GPU all-reduce test passes. Record CPU cores, RAM and free disk.
- If all checks pass, use all four GPUs: distributed data-parallel training (one process per GPU) and a 4-GPU work queue for simulation. Use most CPU cores (leave 4 free) for FEFF and file I/O.
- If any check fails, report it and adapt (fewer GPUs, smaller batches) instead of stopping.
- Write `reports/hardware.md`.

## Phase 1 – Structures (Pt and Pt alloys only)

- **Systems:** Pt, Pt3Co, PtCo, Pt3Ni, PtNi, Pt3Fe, PtCu.
  - Ordered phases (L1₂ for Pt3M, L1₀ for PtM) from open databases where available.
  - Disordered fcc solid solutions generated from Pt by random substitution, with lattice parameters from Vegard's law, checked against database values where available.
- **Sources, in order:** Crystallography Open Database (no key), Materials Project if `MP_API_KEY` is set, then JARVIS and OQMD. Deduplicate with pymatgen's structure matcher. Skip and log any phase with no open structure.
- **Nanoparticles** (ASE cluster tools): cuboctahedra, icosahedra, decahedra, truncated octahedra and spherical cuts, 1.5–8 nm (continuous sampling), with lattice strain (±2%), vacancies (0–3%) and surface relaxation.

## Phase 2 – Simulation (one pair-distance calculation feeds all three techniques)

For each nanoparticle, compute element-resolved pair-distance histograms **once on GPU**, then derive all three techniques from them:

- **XRD:** Debye scattering equation from the histograms with Q-dependent atomic form factors; convert to 2θ for Cu Kα and for a synchrotron wavelength. Study DebyeCalculator for the GPU implementation.
- **PDF:** total scattering F(Q) from the same Debye calculation, then G(r) by sine Fourier transform with sampled Qmax (20–30 Å⁻¹), Qdamp and Qbroad. Termination ripples then appear naturally.
- **EXAFS:** run FEFF (via Larch) only on a small set of reference clusters per absorber–scatterer pair to get scattering amplitude and phase functions. Then synthesize χ(k) for every sample from its absorber-centred partial distance distribution, with single scattering up to about 3.5 Å and multiple-scattering paths taken from the bulk FEFF run, scaled by coordination. Edges: Pt L3, plus the alloying element's K edge (Co, Ni, Fe, Cu). Also produce μ(E) with a simple edge and background model.
- **Validate the EXAFS shortcut:** run full FEFF on 200 random samples and report the error of the synthesized χ(k). If the error is above tolerance, fix the method before scaling up.
- Instrument parameters (wavelength, resolution, k-range, noise levels) are sampled from `configs/instruments.yaml`.

### Labels per sample
Composition, ordered/disordered phase, lattice parameter, shape, size, strain, vacancy fraction, coordination shells per absorber (neighbour, N, R, σ²), assumed oxidation states.

### Planted faults (15% of samples)
Pt oxide surface shell (PtOx), core–shell segregation (Pt skin), mixed oxidation states, low-concentration dopant, wrong-phase candidate in the metadata, multi-domain particles, mislabeled metadata. Record each fault in the labels.

## Dataset size

**50,000 samples total**: 42,500 clean and 7,500 with planted faults, balanced across the seven compositions, ordered and disordered phases, shapes and size bins. Splits: 80% train, 10% validation, 10% test, plus the fault set. Split by structure family, so near-duplicate particles never cross splits.

Why 50,000: 1D convolutional models on spectra stop improving noticeably beyond a few tens of thousands of varied samples. This gives each composition × phase × shape combination several hundred examples across the size range, plus enough faults to measure detection, and it fits comfortably in disk and GPU time. The GAN's real limit is the amount of real reference data, not the simulated count.

**Build in two stages:** a 2,000-sample pilot first, fully validated, then the full 50,000.

## Phase 3 – Real reference data (for the realism model)

Find and download openly licensed experimental spectra relevant to Pt and Pt alloys, and record licences:
- **XAS:** open XAS libraries (XASLIB, MDR XAFS DB, RefXAS, XASDB and similar): Pt foil, Pt oxides and Pt-alloy references at the Pt L3 edge, plus Co, Ni, Fe and Cu K-edge foils and compounds.
- **XRD:** open experimental powder XRD collections (study opXRD) and published Pt-nanoparticle datasets with data deposits.
- **PDF:** published Pt-nanoparticle PDF datasets with open data deposits.

Interpolate onto the same grids as the simulations. Hold out 20% of real spectra per technique for evaluation and never train on them. If a technique has fewer than 50 usable real spectra, train its realism model with stronger regularization and report it as low-confidence, or skip it and document why.

## Phase 4 – Models

### 4a. GAN realism model (one per technique: XRD, PDF, EXAFS)

**Generator** (residual, label-preserving):
- Inputs: the simulated spectrum on a fixed grid, a condition vector (instrument parameters normalized, composition fractions, edge for EXAFS), and noise z (64 dims).
- Backbone: 1D U-Net, 4 levels, channels 64 → 512.
- Output heads, artifacts only:
  - background: 12 cubic B-spline coefficients
  - gain field: 8 B-spline coefficients, exponentiated and clipped to 0.9–1.1
  - noise scale field: 16 B-spline coefficients through softplus, multiplied by coloured noise (white noise through a learned 1D convolution)
  - glitches (EXAFS only): at most 2 Gaussian spikes with gated position and amplitude
- Output spectrum = `sim × gain + background + noise_scale × coloured_noise (+ glitches)`.

**Discriminator:**
- Input: the spectrum and its first derivative as two channels.
- 1D ResNet, 6 blocks, channels 64 → 512, spectral normalization on every layer, projection conditioning on the condition vector.

**Losses:**
- Hinge adversarial loss, plus R1 gradient penalty on real samples.
- Preservation losses:
  - XRD and PDF: peak positions unchanged (cross-correlation of derivatives).
  - EXAFS: Fourier-filtered first-shell amplitude and phase unchanged.
- Penalty on artifact magnitude, so the generator can't cover the signal.

**Training:**
- Distributed data-parallel on 4 GPUs, bf16 mixed precision.
- Batch 128 per GPU. Adam, learning rate 2e-4 for the generator and 4e-4 for the discriminator, betas (0.0, 0.99).
- EMA of generator weights (0.999). Checkpoints every 2,000 steps.
- Up to 60,000 steps per technique, with early stopping on the diagnostics below.
- TensorBoard logging.

### 4b. Multi-technique predictor (the downstream test)

- One 1D ResNet-18-style encoder per technique, then a small transformer that fuses whichever techniques are present. Random technique dropout during training, so it works with any subset.
- Heads: composition (7 classes), ordered/disordered, size, lattice parameter, first-shell N and R per absorber, fault flags (multi-label).
- Train two versions: (a) on baseline-augmented data, (b) on GAN-augmented data.

**Baseline augmentation:** Poisson and Gaussian noise, polynomial background, Gaussian broadening, normalization jitter. Study the aberration synthesis in XCA (`maffettone/xca`) and the simulation-to-real work in arXiv 2608.22400.

## Phase 5 – Diagnostics and decision

- **Realism:** classifier two-sample test (a classifier separating held-out real from generated; target accuracy close to 50%), plus a 1D Fréchet distance using the predictor's encoder.
- **Diversity:** variance and pairwise distance of generated artifacts, to catch mode collapse.
- **Label preservation:** re-fit 500 generated spectra with standard tools (diffpy for PDF, Larch for EXAFS, peak fitting for XRD) and confirm labels are recovered within tolerance. Report the failure rate.
- **Downstream:** compare (a) and (b) on the held-out real spectra that have known labels (e.g. Pt foil and reference compounds) and on a held-out set synthesized with full FEFF plus harsh classical noise. Report per-task metrics with bootstrap confidence intervals.
- **Decision rule:** ship (b) and the GAN only if it beats (a) by more than run-to-run variation on the real held-out set without losing on the others. Otherwise ship (a) and document the result.

## Outputs (written to `./artifacts/`)

```
artifacts/
  data/
    shards/               # HDF5 shards for training (all 50,000 samples, all fields)
    contract_export/      # test + fault splits in the Concord data contract (below)
    manifest.csv
    splits.json
    DATASET.md            # dataset card
  models/
    realism_xrd/  realism_pdf/  realism_exafs/   # EMA weights, config, MODEL_CARD.md
    predictor/                                   # winning variant, config, MODEL_CARD.md
  scripts/
    validate_dataset.py   # must exit 0
    realism_infer.py      # sim spectra in, realistic spectra out
    predict.py            # one sample directory or HDF5 file in, predictions out
  reports/
    hardware.md  exafs_shortcut_validation.md  gan_diagnostics.md  downstream_comparison.md
```

**Concord data contract** (for `contract_export/`, identical to the main project):

```
samples/<sample_id>/
  sample.json   # sample_id, composition, techniques, instrument, origin, ground_truth, faults
  xrd.xy        # columns: x (2theta_deg or q_inv_A), intensity, sigma
  pdf.gr        # columns: r_A, G, sigma
  xafs_mu.dat   # columns: energy_eV, mu
  xafs_chi.k    # columns: k_inv_A, chi
```

The `origin` field is `simulated` or `simulated+realism`. Model cards and the dataset card state clearly that the data is simulated, what the realism layer can and can't do, and the measured diagnostics.

## Time plan (32 hours)

| Hours | Work |
|---|---|
| 0–1 | Hardware check, environment, pinned dependencies |
| 1–6 | Structures, simulation pipeline, EXAFS shortcut validation, 2,000-sample pilot, validator |
| 6–16 | Full 50,000-sample build in the background; real reference data collection; baseline augmentation; GAN code tested on the pilot |
| 16–26 | GAN training (all 4 GPUs), predictor training (a) and (b) |
| 26–30 | Diagnostics, decision rule, model and dataset cards, contract export |
| 30–32 | Buffer; final validation |

Give me a short status update at each block boundary, with numbers. If the full build falls behind, ship a smaller balanced dataset (minimum 20,000) rather than an unbalanced large one.

---

## `/goal` lines (type one at a time in Claude Code)

```
/goal reports/hardware.md confirms the GPU setup and the 2,000-sample pilot passes python artifacts/scripts/validate_dataset.py, with the EXAFS shortcut validation report written
```
```
/goal the full dataset passes validate_dataset.py, and the realism models and predictors (a) and (b) have trained checkpoints
```
```
/goal reports/gan_diagnostics.md and reports/downstream_comparison.md contain measured results, the decision rule has been applied, and realism_infer.py and predict.py run on the test split
```
