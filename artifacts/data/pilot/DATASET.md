# Concord Forge dataset card (pilot (2,000 samples))

**All data in this dataset are simulated.** Spectra are computed from atomistic models of Pt and Pt-alloy nanoparticles. No measured spectrum is included in the shards. The realism layer, if shipped, is a separate model (`artifacts/models/realism_*`), and its use is marked by `origin = simulated+realism` in exports.

## Contents

- 2000 samples: 1700 clean and 300 with one planted fault (15.0 %).
- Splits by structure family (near-duplicate particles never cross splits): clean train 1355, val 176, test 169; fault train 233, val 37, test 30.
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

4. **EXAFS** (`forge/exafs.py`) uses a FEFF6L-based shortcut, validated against full FEFF before scaling up.  μ(E) adds an arctan edge, a Lorentzian white line and linear pre- and post-edge terms to χ(k); the white line and edge position shift with oxidation state.

5. **Instrument parameters** are drawn per sample from `configs/instruments.yaml` and stored with it.

## Composition of the dataset

| Composition | samples |
|---|---|
| Pt | 279 |
| Pt3Co | 286 |
| Pt3Fe | 287 |
| Pt3Ni | 287 |
| PtCo | 286 |
| PtCu | 286 |
| PtNi | 289 |

| Phase | samples |
|---|---|
| A1 | 1142 |
| L1_0 | 285 |
| L1_1 | 143 |
| L1_2 | 430 |

| Shape | samples |
|---|---|
| cuboctahedron | 519 |
| decahedron | 229 |
| icosahedron | 218 |
| sphere | 511 |
| truncated_octahedron | 523 |

| Planted fault | samples |
|---|---|
| low_dopant | 43 |
| mislabeled_metadata | 43 |
| mixed_oxidation | 43 |
| multi_domain | 43 |
| none | 1700 |
| pt_oxide_shell | 42 |
| pt_skin | 43 |
| wrong_phase_candidate | 43 |

Size range 1.32–8.09 nm (equivalent-sphere diameter), 78–20088 atoms.

**Balance.** Compositions are equal. Ordered and disordered phases are equal within each alloy, and shapes are equal within a phase. Ordered phases exist only in the 3 single-crystal shapes, so cuboctahedra, truncated octahedra and spheres are more common overall than icosahedra and decahedra. Size bins are equal within each shape.

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

- CC0-1.0: 1713 samples
- CC-BY-4.0: 287 samples

The dataset as a whole is CC BY 4.0, with attribution to COD, JARVIS-DFT and the FEFF6L / xraylarch authors. Real reference spectra are not in the shards. They are listed in `manifest.csv` (kind `real_spectrum`) with their own licences. Counts by technique:

- xas: 2469
- xrd: 92687
- pdf: 436

## Limitations

- These are idealized atomistic models. Ordered phases appear only in single-crystal morphologies; there is no segregation other than the planted Pt-skin fault; there is no support or ligand shell.
- Thermal motion is a correlated Gaussian model, with no anharmonicity. Static disorder is random.
- XANES is not simulated beyond a phenomenological edge.
- XRD has no incoherent (Compton), fluorescence or air scatter. Those are left to augmentation or the realism layer.
- DFT-derived ordered phases (Pt3Ni L1₂, PtNi L1₀) carry calibration uncertainty, about 0.3 % on a.
- Vegard's law underestimates the lattice parameters of some disordered alloys by up to about 2.7 % compared with the COD entries listed above.
