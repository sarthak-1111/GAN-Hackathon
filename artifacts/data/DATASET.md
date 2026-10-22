# Concord Forge dataset card: full build, 50,000 samples

**All data in this dataset are simulated.** Spectra are computed from atomistic models of Pt and Pt-alloy nanoparticles. No measured spectrum is included in the shards. The realism layer, if shipped, is a separate model (`artifacts/models/realism_*`), and its use is marked by `origin = simulated+realism` in exports.

## Contents

- 50000 samples: 42500 clean and 7500 with one planted fault (15.0 %).
- Splits by structure family (near-duplicate particles never cross splits): clean train 34002, val 4198, test 4300; fault train 6006, val 743, test 751.
- Each sample has XRD (Cu Kα, laboratory 2θ grid), synchrotron XRD (Q grid, sampled wavelength), PDF (G(r) and F(Q)), and EXAFS χ(k) plus normalized μ(E) at the Pt L3 edge and, for alloys, the M K edge. All techniques are computed from the same atomic structure.
- Storage: HDF5 shards in `shards/`, one per (set, split) chunk, listed with SHA-256 in `splits.json`. Every sample, structure, FEFF reference and real spectrum is listed in `manifest.csv` with source, ID and licence.

## How the data were made

1. **Reference phases** came from open databases (`forge/structures.py`). The Crystallography Open Database (CC0) was searched first. Materials Project was skipped because no `MP_API_KEY` was set. JARVIS-DFT (CC BY 4.0) was searched next. OQMD (CC BY 4.0) returned HTTP 502 during the build. Duplicates were merged with pymatgen's StructureMatcher. Experimental ambient entries are preferred (pre-1947 COD values converted from kX to Å; pressure and temperature series excluded; median after outlier rejection). DFT-only phases are rescaled by the same database's a(Pt, exp)/a(Pt, DFT) ratio.

| Phase | a (Å) | c (Å) | α (°) | Representative | Basis |
|---|---|---|---|---|---|
| Co A1 | 3.5442 | 3.5442 | 90.00 | COD 9008466 | experimental (COD, ambient), median of 3 entries after outlier rejection |
| Cu A1 | 3.6150 | 3.6150 | 90.00 | COD 4105681 | experimental (COD, ambient), median of 7 entries after outlier rejection |
| Fe A1 | 3.5619 | 3.5619 | 90.00 | COD 9012706,9012707,9012708,9012709,9012710,9012711,9012712,9012713,9012714 | gamma-Fe a(T) series (9 COD entries, 1189-1661 K) linearly extrapolated to 293 K (slope 8.59e-05 A/K) |
| Ni A1 | 3.5282 | 3.5282 | 90.00 | COD 4320486 | experimental (COD, ambient), median of 14 entries after outlier rejection |
| Pt3Co L1_2 | 3.8310 | 3.8310 | 90.00 | COD 1524398 | experimental (COD, ambient), median of 1 entries after outlier rejection |
| Pt3Fe L1_2 | 3.8640 | 3.8640 | 90.00 | COD 9013737 | experimental (COD, ambient), median of 1 entries after outlier rejection |
| Pt3Ni L1_2 | 3.8277 | 3.8277 | 90.00 | JARVIS JVASP-128087 | JARVIS DFT, isotropically rescaled by a(Pt,exp)/a(Pt,JARVIS) = 0.9862 |
| PtCo L1_0 | 3.8056 | 3.6840 | 90.00 | COD 1525472 | experimental (COD, ambient), median of 1 entries after outlier rejection |
| PtCu L1_1 | 3.7778 | 3.7778 | 90.54 | COD 5910049 | experimental (COD, ambient), median of 1 entries after outlier rejection |
| PtNi L1_0 | 3.7934 | 3.5812 | 90.00 | JARVIS JVASP-78658 | JARVIS DFT, isotropically rescaled by a(Pt,exp)/a(Pt,JARVIS) = 0.9862 |
| Pt A1 | 3.9218 | 3.9218 | 90.00 | COD 1011107 | experimental (COD, ambient), median of 7 entries after outlier rejection |

PtCu orders as L1₁ (CuPt type, alternating (111) planes), which is the observed ordered phase, so L1₁ is used instead of L1₀. No ordered phase was missing from the open sources. fcc Fe is not stable at room temperature, so its Vegard endpoint is extrapolated linearly from the COD γ-Fe a(T) series.

**Disordered alloys use Vegard's law**, checked against database values:

| System | a Vegard (Å) | Database checks (deviation from Vegard) |
|---|---|---|
| Pt3Co | 3.8274 | COD 1524153 x=0.75: 3.668 (+0.81 %); COD 1524808 x=0.5: 3.835 (+2.73 %); ordered phase (volume-equivalent) 1524398 x=0.25: 3.831 (+0.09 %) |
| PtCo | 3.7330 | COD 1524153 x=0.75: 3.668 (+0.81 %); COD 1524808 x=0.5: 3.835 (+2.73 %); ordered phase (volume-equivalent) 1525472 x=0.5: 3.7647 (+0.85 %) |
| Pt3Ni | 3.8234 | COD 1523344 x=0.92: 3.75 (+5.35 %); COD 1523590 x=0.5: 3.7577 (+0.88 %); COD 1537974 x=0.92: 3.564 (+0.12 %); COD 1538610 x=0.5: 3.785 (+1.61 %); ordered phase (volume-equivalent) JVASP-128087 x=0.25: 3.8277 (+0.11 %) |
| PtNi | 3.7250 | COD 1523344 x=0.92: 3.75 (+5.35 %); COD 1523590 x=0.5: 3.7577 (+0.88 %); COD 1537974 x=0.92: 3.564 (+0.12 %); COD 1538610 x=0.5: 3.785 (+1.61 %); ordered phase (volume-equivalent) JVASP-78658 x=0.5: 3.7213 (-0.10 %) |
| Pt3Fe | 3.8318 | COD 1522533 x=0.75: 3.723 (+1.95 %); COD 1523354 x=0.65: 3.779 (+2.47 %); COD 1523420 x=0.14: 3.896 (+0.64 %); ordered phase (volume-equivalent) 9013737 x=0.25: 3.864 (+0.84 %) |
| PtCu | 3.7684 | ordered phase (volume-equivalent) 5910049 x=0.5: 3.7776 (+0.24 %) |

2. **Nanoparticles** (`forge/particles.py`, ASE cluster tools):
   - **Shapes:** cuboctahedra, truncated octahedra, spheres, Mackay icosahedra and Ino/Marks decahedra, 1.5–8 nm (continuous sampling within 4 size bins).
   - **Ordering:** L1₂, L1₀ and L1₁ are made by sublattice assignment on single-crystal shapes, plus the phase's tetragonal or rhombohedral distortion. The ordering variant is the one closest to nominal stoichiometry.
   - **Disordered alloys:** random substitution. Five-fold twinned shapes are disordered or pure Pt only.
   - **Perturbations:** isotropic strain ±2 %, vacancies 0–3 %, radial surface relaxation −3…+0.5 % of the NN distance, and static displacements (×2 at the surface).

3. **Simulation** (`forge/simulate.py`) computes element-resolved pair-distance histograms on GPU (0.004 Å bins, validated against DebyeCalculator to 1.2×10⁻⁵ of the maximum). All techniques come from these histograms:
   - **Debye XRD:** Waasmaier–Kirfel f₀ with Chantler f′, f″ at the photon energy, and a correlated thermal damping model σ² = (U_a+U_b)·max(0.35, 1 − δ2/r²).
   - **Cu Kα1/α2 patterns:** with polarization and Caglioti pseudo-Voigt resolution.
   - **Synchrotron patterns:** Q grid, Gaussian resolution.
   - **PDF:** S(Q), F(Q) and the sine Fourier transform over the sampled Qmin–Qmax, with Qdamp and Qbroad. Termination ripples come out of the transform.

4. **EXAFS** (`forge/exafs.py`) uses a FEFF6L-based shortcut, validated against full FEFF before scaling up. Full-FEFF check on 372 (sample, edge) pairs: median R-factor 0.0025, 95th percentile 0.0338 (see `reports/exafs_shortcut_validation.md`). μ(E) adds an arctan edge, a Lorentzian white line and linear pre- and post-edge terms to χ(k); the white line and edge position shift with oxidation state.

5. **Instrument parameters** are drawn per sample from `configs/instruments.yaml` and stored with it.

## Composition of the dataset

| Composition | clean | with planted fault | total |
|---|---|---|---|
| Pt | 6071 | 918 | 6989 |
| Pt3Co | 6072 | 1098 | 7170 |
| Pt3Fe | 6071 | 1098 | 7169 |
| Pt3Ni | 6071 | 1097 | 7168 |
| PtCo | 6072 | 1097 | 7169 |
| PtCu | 6071 | 1096 | 7167 |
| PtNi | 6072 | 1096 | 7168 |

| Phase | clean | with planted fault | total |
|---|---|---|---|
| A1 | 24286 | 4212 | 28498 |
| L1_0 | 6072 | 1098 | 7170 |
| L1_1 | 3035 | 547 | 3582 |
| L1_2 | 9107 | 1643 | 10750 |

| Shape | clean | with planted fault | total |
|---|---|---|---|
| cuboctahedron | 10929 | 2021 | 12950 |
| decahedron | 4856 | 723 | 5579 |
| icosahedron | 4858 | 723 | 5581 |
| sphere | 10928 | 2026 | 12954 |
| truncated_octahedron | 10929 | 2007 | 12936 |

| Planted fault | samples |
|---|---|
| low_dopant | 1072 |
| mislabeled_metadata | 1071 |
| mixed_oxidation | 1072 |
| multi_domain | 1072 |
| none | 42500 |
| pt_oxide_shell | 1071 |
| pt_skin | 1071 |
| wrong_phase_candidate | 1071 |

Size range 1.25–8.35 nm (equivalent-sphere diameter), 77–22143 atoms.

**Balance.** Compositions are equal in the clean set. Pure Pt has fewer fault samples because the Pt-skin fault does not apply to it, and the other faults are spread over the remaining compositions. Ordered and disordered phases are equal within each alloy, and shapes are equal within a phase. Ordered phases exist only in the 3 single-crystal shapes, so cuboctahedra, truncated octahedra and spheres are more common overall than icosahedra and decahedra. Size bins are equal within each shape.

## Labels (per sample)

- **Identity:** composition class (7) and true atomic fractions; phase (A1, L1₂, L1₀, L1₁) and the ordered/disordered flag (not applicable to pure Pt); shape and shape parameters.
- **Size:** equivalent-sphere and radius-of-gyration diameters, and number of atoms.
- **Lattice:** lattice parameter (volume-equivalent pseudo-cubic, plus a, c and α), strain, vacancy fraction.
- **Coordination shells per absorber element** (shells 1–4 and the O shell): neighbour element, N, mean R, and σ² (static plus the thermal model). A compact first-shell summary is also given for the Pt and M absorbers.
- **Chemistry:** assumed oxidation states per element (fractions), and U_iso and δ2 used in the simulation.
- **Planted faults** (one-hot over 7 types; the first 5 are physical, the last 2 corrupt only the metadata):
  1. Pt oxide surface shell: O atop surface Pt with extra surface disorder, Pt²⁺.
  2. Pt skin: surface M swapped into the core.
  3. Mixed oxidation states: a fraction of surface M (or Pt) oxidized, with white-line and edge-shift changes.
  4. Low-concentration dopant: 1–4 % of another 3d metal.
  5. Multi-domain particle: two ordering variants, or a {111} twin for disordered particles.
  6. Wrong phase candidate in the metadata.
  7. Mislabeled nominal composition in the metadata.

## Licences

Per-sample licences in `manifest.csv` follow the source of the reference phase:

- CC0-1.0: 42830 samples
- CC-BY-4.0: 7170 samples

The dataset as a whole is CC BY 4.0, with attribution to COD, JARVIS-DFT and the FEFF6L / xraylarch authors. Real reference spectra are not in the shards. Every downloaded real item is listed in `manifest.csv` (kind `real_spectrum`, xas 2469, xrd 92687, pdf 436) with its source, ID and licence. The XRD count is dominated by the complete opXRD archive (Zenodo 15298026), which was downloaded whole; 1,500 of its patterns were screened. The real spectra actually used by the realism models (Phase 3, `real/<technique>_items.csv`, split by material) are:

- XAS: 717 items (284 train, 74 validation, 130 held out, 229 not used for lack of a usable physics twin); mdr_xafs_db 326, cls_xasdb 171, bmm_standards_nist 56, zenodo_17131699_PtCo_PtNi_HTPEM 48, lisa_xasdb 42, xaslib 27, mendeley_9bhtr2w54w 21, cardiff_27052390 18, github_xraylarch_examples 8
- XRD: 326 items (223 train, 48 validation, 55 held out); opxrd 257, zenodo_17131699_PtCo_PtNi_HTPEM 54, cardiff_27052390 6, zenodo_15310405_PtAC 5, zenodo_14186519_GO_PtNP 3, zenodo_21028650_GOPt10 1
- PDF: 86 items (49 train, 7 validation, 10 held out, 20 not used for lack of a usable physics twin); github_scatterlab_AuNP 78, github_DeepStruc 4, zenodo_56682_ncomm_goldnp_2016 4

## Validation

`python artifacts/scripts/validate_dataset.py` checks files and hashes, schema, balance, splits, label ranges, label–spectrum consistency and licences, and exits 0 only if every check passes. This build passes all checks.

One label-range bound was corrected during the build. The first full run failed the check "first-shell R 2.3–2.9 Å" on 47 samples, all pure-Pt Mackay icosahedra at the high end of the strain range (mean Pt first-shell R up to 2.915 Å). The data were correct and the bound was wrong: a Mackay icosahedron's bonds within a shell are 5.15 % longer than its radial bonds (ASE `Icosahedron`: nearest-neighbour bonds 1.000–1.0515 × a/√2). The bound is now 3.0 Å, derived from the construction rules rather than the observed values: largest lattice parameter (Pt at +2 % strain) 4.002 Å, a/√2 × 1.0515 × 1.005 (maximum outward surface relaxation) = 2.99 Å. No sample was changed.

## Limitations

- **EXAFS of icosahedra is the least accurate part of the dataset.** Against full FEFF, Mackay icosahedra have median R-factor 0.028 and 95th percentile 0.15, against ≤ 0.015 for every other shape (see `reports/exafs_shortcut_validation.md`). The shape label is stored, so they can be dropped where EXAFS accuracy matters.
- **The pilot (`data/pilot/`)** was simulated with an earlier version of the EXAFS shortcut, before the twin-boundary, icosahedral and path-variance fixes. It validated the pipeline; the full dataset uses the final shortcut.
- **Model views:** the predictor and realism models use 2θ 20–80° (Cu Kα) and r 0.5–30 Å, the ranges covered by open laboratory and nanoparticle PDF data. The shards store the full simulated ranges (2θ 10–100°, r 0.5–62 Å).
- These are idealized atomistic models. Ordered phases appear only in single-crystal morphologies; there is no segregation other than the planted Pt-skin fault; there is no support or ligand shell.
- Thermal motion is a correlated Gaussian model, with no anharmonicity. Static disorder is random.
- XANES is not simulated beyond a phenomenological edge.
- XRD has no incoherent (Compton), fluorescence or air scatter. Those are left to augmentation or the realism layer.
- DFT-derived ordered phases (Pt3Ni L1₂, PtNi L1₀) carry calibration uncertainty, about 0.3 % on a.
- Vegard's law underestimates the lattice parameters of some disordered alloys by up to about 2.7 % compared with the COD entries listed above.
