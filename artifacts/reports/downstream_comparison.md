# Downstream comparison: classical augmentation (a) vs GAN realism augmentation (b)

**Decision: ship variant (a).** On the real held-out set (A), the GAN-augmented predictor (b) reached a composition-family accuracy of 0.167 against 0.095 for the classical baseline (a) (means over two seeds). The difference, +0.071, is not larger than the run-to-run variation of 0.238, which is the spread between (b)'s two seeds. On the synthetic sets B and C, (b) was worse than (a) by more than the run-to-run variation on 17 of 17 metrics. **The baseline predictor (a) is shipped.** The realism models are delivered for reference; the shipped predictor does not use them.

## Decision rule (fixed before the comparison was run)

Ship (b) only if it beats (a) on the real held-out set by more than the run-to-run variation and does not lose on the other sets. In practice:

- **Primary real-data metric:** composition-family accuracy over all set-A items (Pt / Pt–Co / Pt–Ni / Pt–Fe / Pt–Cu). The deposits state the alloying element but not reliably the Pt:M ratio. (b) must exceed (a) by more than the largest seed-to-seed difference of either variant.
- **No losses elsewhere:** on every metric of sets B and C, (b) must not be worse than (a) by more than that variation.

Each variant was trained twice with different seeds, using identical data, schedule and model.

## Test sets

- **A, real held-out:** 21 spectra never used for any training or model selection, each predicted from its one technique alone (no metadata). They are listed in the table below. Items from the same deposit are not independent; for example, the Pt foil spectra are repeat measurements of one reference foil. Bootstrap intervals over items therefore understate the uncertainty, especially for the Pt foil first-shell N and R.
- **B, full FEFF + harsh noise:** 300 test-split samples. Their EXAFS was recomputed with full FEFF6L (8 absorbers per edge), and every technique got the 'harsh' classical noise and background.
- **C, simulated test + harsh noise:** the clean test split plus the fault-test split, with harsh classical noise on every technique.

### Set A items

| Technique | Material (as deposited) | Label | Source | Spectra |
|---|---|---|---|---|
| exafs_pt | Pt foil (reference channel measured simultaneously) | Pt (N = 12, R = 2.775 Å) | cardiff_27052390 | 4 |
| exafs_pt | Pt nanoparticles on Al2O3 (50C - reduced) | Pt | cardiff_27052390 | 1 |
| exafs_pt | Pt nanoparticles on Al2O3 (75C - fresh) | Pt | cardiff_27052390 | 1 |
| exafs_pt | Pt nanoparticles on Al2O3 (Pt75 used merge 18k) | Pt | cardiff_27052390 | 1 |
| exafs_pt | Pt-Co/Ketjen Black (intermetallic); leached 24 h in 98 wt% H3PO4 at 18 | Pt3Co | zenodo_17131699_PtCo_PtNi_HTPEM | 1 |
| exafs_pt | Pt-Ni/Ketjen Black (intermetallic); leached 1 h in 98 wt% H3PO4 at 180 | Pt3Ni | zenodo_17131699_PtCo_PtNi_HTPEM | 1 |
| exafs_pt | Pt/C (commercial); leached 1 h in 98 wt% H3PO4 at 180 C | Pt | zenodo_17131699_PtCo_PtNi_HTPEM | 1 |
| xrd | Pt nanoparticles on activated carbon (PtNP_AC-fresh) | Pt | zenodo_15310405_PtAC | 1 |
| xrd | Pt nanoparticles on graphene oxide, gamma-irradiated 10KGY | Pt | zenodo_14186519_GO_PtNP | 1 |
| xrd | Pt nanoparticles on graphene oxide, gamma-irradiated 1KGY | Pt | zenodo_14186519_GO_PtNP | 1 |
| xrd | Pt-Co/reduced graphene oxide | Pt3Co | zenodo_17131699_PtCo_PtNi_HTPEM | 8 |

## Results (mean over 2 seeds; spread = |seed 1 − seed 2|)

| Set | Metric | (a) mean | (a) spread | (b) mean | (b) spread | b − a |
|---|---|---|---|---|---|---|
| A_real | composition_acc | 0.0000 | 0.0000 | 0.1429 | 0.1905 | +0.1429 |
| A_real | composition_family_acc | 0.0952 | 0.0000 | 0.1667 | 0.2381 | +0.0714 |
| A_real | n1_pt_mae | 0.9838 | 0.3501 | 0.7245 | 0.3186 | -0.2593 (lower is better) |
| A_real | r1_pt_mae | 0.0838 | 0.0253 | 0.0832 | 0.0562 | -0.0006 (lower is better) |
| B_fullfeff_harsh | composition_acc | 0.9983 | 0.0033 | 0.8700 | 0.1133 | -0.1283 |
| B_fullfeff_harsh | lattice_mae | 0.0117 | 0.0002 | 0.0281 | 0.0056 | +0.0164 (lower is better) |
| B_fullfeff_harsh | n1_m_mae | 0.1962 | 0.0201 | 0.4101 | 0.0516 | +0.2139 (lower is better) |
| B_fullfeff_harsh | n1_pt_mae | 0.1639 | 0.0230 | 0.5535 | 0.0680 | +0.3896 (lower is better) |
| B_fullfeff_harsh | ordered_acc | 1.0000 | 0.0000 | 0.9781 | 0.0120 | -0.0219 |
| B_fullfeff_harsh | r1_m_mae | 0.0052 | 0.0001 | 0.0155 | 0.0012 | +0.0103 (lower is better) |
| B_fullfeff_harsh | r1_pt_mae | 0.0058 | 0.0004 | 0.0160 | 0.0031 | +0.0102 (lower is better) |
| B_fullfeff_harsh | size_mae | 0.2112 | 0.0055 | 0.5630 | 0.1945 | +0.3519 (lower is better) |
| C_test_harsh | composition_acc | 0.9980 | 0.0012 | 0.8422 | 0.1085 | -0.1558 |
| C_test_harsh | fault_auroc | 0.8876 | 0.0084 | 0.7328 | 0.0279 | -0.1548 |
| C_test_harsh | lattice_mae | 0.0066 | 0.0002 | 0.0248 | 0.0054 | +0.0182 (lower is better) |
| C_test_harsh | n1_m_mae | 0.1453 | 0.0055 | 0.3730 | 0.0371 | +0.2277 (lower is better) |
| C_test_harsh | n1_pt_mae | 0.1423 | 0.0158 | 0.5553 | 0.0677 | +0.4130 (lower is better) |
| C_test_harsh | ordered_acc | 1.0000 | 0.0000 | 0.9613 | 0.0310 | -0.0387 |
| C_test_harsh | r1_m_mae | 0.0027 | 0.0001 | 0.0142 | 0.0020 | +0.0115 (lower is better) |
| C_test_harsh | r1_pt_mae | 0.0032 | 0.0001 | 0.0147 | 0.0021 | +0.0115 (lower is better) |
| C_test_harsh | size_mae | 0.2061 | 0.0094 | 0.6148 | 0.1148 | +0.4088 (lower is better) |

### Set A: prediction for each real spectrum

Each spectrum was predicted from its one technique alone, without metadata. For the Pt foil the first-shell N and R predictions are given as well (label: N = 12, R = 2.775 Å).

| # | Technique | Material | Label | Standard fit of the spectrum | (a) seed 1 | (a) seed 2 | (b) seed 1 | (b) seed 2 |
|---|---|---|---|---|---|---|---|---|
| 1 | exafs_pt | Pt/C (commercial); leached 1 h in 98 wt% H3PO4 a | Pt | R = 2.756 Å, ΔE0 = +8.2 eV | PtCu | PtCu | Pt ✓ | Pt ✓ |
| 2 | exafs_pt | Pt-Co/Ketjen Black (intermetallic); leached 24 h | Pt3Co | R = 2.744 Å, ΔE0 = +6.5 eV | PtCu | PtCu | Pt | Pt |
| 3 | exafs_pt | Pt-Ni/Ketjen Black (intermetallic); leached 1 h  | Pt3Ni | R = 2.679 Å, ΔE0 = +3.3 eV | PtCu | PtNi ✓ | PtNi ✓ | PtCu |
| 4 | exafs_pt | Pt nanoparticles on Al2O3 (75C - fresh) | Pt | R = 2.747 Å, ΔE0 = +6.6 eV | PtCu | PtCu | PtNi | PtCu |
| 5 | exafs_pt | Pt nanoparticles on Al2O3 (50C - reduced) | Pt | R = 2.749 Å, ΔE0 = +7.3 eV | PtCu | PtCu | PtCu | PtCu |
| 6 | exafs_pt | Pt foil (reference channel measured simultaneous | Pt | R = 2.766 Å, ΔE0 = +7.5 eV | PtCu (N 10.9, R 2.678) | Pt3Ni (N 11.2, R 2.704) | Pt ✓ (N 11.1, R 2.718) | PtCu (N 11.4, R 2.663) |
| 7 | exafs_pt | Pt foil (reference channel measured simultaneous | Pt | R = 2.765 Å, ΔE0 = +7.4 eV | PtCu (N 10.8, R 2.679) | Pt3Ni (N 11.2, R 2.704) | Pt ✓ (N 11.1, R 2.724) | PtCu (N 11.4, R 2.665) |
| 8 | exafs_pt | Pt foil (reference channel measured simultaneous | Pt | R = 2.765 Å, ΔE0 = +7.6 eV | PtCu (N 10.8, R 2.679) | Pt3Ni (N 11.2, R 2.704) | Pt ✓ (N 11.1, R 2.720) | PtCu (N 11.4, R 2.663) |
| 9 | exafs_pt | Pt nanoparticles on Al2O3 (Pt75 used merge 18k) | Pt | R = 2.752 Å, ΔE0 = +7.1 eV | PtCu | PtCu | PtCu | PtCu |
| 10 | exafs_pt | Pt foil (reference channel measured simultaneous | Pt | R = 2.765 Å, ΔE0 = +7.5 eV | PtCu (N 10.8, R 2.678) | Pt3Ni (N 11.2, R 2.703) | Pt ✓ (N 11.1, R 2.718) | PtCu (N 11.4, R 2.663) |
| 11 | xrd | Pt-Co/reduced graphene oxide [Go-0] | Pt3Co | a = 3.840 Å, 3 nm | PtNi | PtNi | PtNi | PtNi |
| 12 | xrd | Pt-Co/reduced graphene oxide [Go-1] | Pt3Co | a = 3.867 Å, 3 nm | Pt3Ni | Pt3Fe | PtNi | PtNi |
| 13 | xrd | Pt-Co/reduced graphene oxide [Go-2] | Pt3Co | a = 3.873 Å, 3 nm | Pt3Ni | Pt3Fe | PtNi | PtNi |
| 14 | xrd | Pt-Co/reduced graphene oxide [rGO-1H] | Pt3Co | a = 3.858 Å, 3 nm | Pt3Fe | Pt3Fe | PtNi | PtNi |
| 15 | xrd | Pt-Co/reduced graphene oxide [rGO-24H] | Pt3Co | a = 3.848 Å, 3 nm | PtCo ✓ | Pt3Fe | PtNi | PtNi |
| 16 | xrd | Pt-Co/reduced graphene oxide [rGO-3H] | Pt3Co | a = 3.860 Å, 3 nm | Pt3Fe | Pt3Fe | PtNi | PtNi |
| 17 | xrd | Pt-Co/reduced graphene oxide [rGO-72H] | Pt3Co | a = 3.862 Å, 3 nm | Pt3Ni | Pt3Fe | PtNi | PtNi |
| 18 | xrd | Pt-Co/reduced graphene oxide [rGO-8H] | Pt3Co | a = 3.847 Å, 3 nm | PtCo ✓ | PtCo ✓ | PtNi | PtNi |
| 19 | xrd | Pt nanoparticles on activated carbon (PtNP_AC-fr | Pt | a = 3.962 Å, 8 nm | PtCu | PtCu | PtNi | PtCo |
| 20 | xrd | Pt nanoparticles on graphene oxide, gamma-irradi | Pt | a = 3.925 Å, 13 nm | PtCu | PtCu | PtCo | PtCo |
| 21 | xrd | Pt nanoparticles on graphene oxide, gamma-irradi | Pt | a = 3.927 Å, 21 nm | PtCu | PtCu | PtNi | PtCo |

✓ = correct composition family (Pt / Pt–Co / Pt–Ni / Pt–Fe / Pt–Cu).

## What the result means

- **Neither variant transfers to real spectra.** Both predictors are near perfect on simulated spectra, even with harsh noise, but on real spectra they mostly fail. Composition-family accuracy on set A is 0.10 for (a) and 0.17 for (b). Either the remaining differences between simulation and measurement are not artifacts the realism layer can model, or the 21 real spectra lie outside the training domain (supports, mixed phases, other preparation). The GAN diagnostics show the same: even the best alternative stays well above 50 % C2ST (reports/gan_diagnostics.md).
- **Why (b) loses on B and C.** On Concord Forge spectra the realism layer adds small artifacts: backgrounds of a few tenths of a percent of the maximum for XRD and PDF, and noise of a few percent for EXAFS (reports/test_split_inference.md). That is much weaker than the classical pipeline, so (b) never trained on strong noise and backgrounds and is fragile under the harsh classical noise of B and C (composition accuracy on C 0.842 vs 0.998).
- **Seed-to-seed variation of (b) on set A is large** (one seed 6/21 family-correct, the other 1/21). With 21 non-independent real spectra, set A can only detect large effects. The rule therefore requires (b) to beat that spread.

## Diagnosis of the set-A failures (standard fits of the same spectra)

The "standard fit" column above fits each real spectrum with the same tools as the label-preservation diagnostic: xraylarch first-shell Pt–Pt fit for EXAFS, lmfit fcc (111)/(200) fit for XRD. The fits are for diagnosis only and do not enter the comparison or the decision.

- **EXAFS: a constant E0 offset.** All 10 Pt L3 spectra need ΔE0 = +3.3 to +8.2 eV relative to the FEFF6L reference that the simulations use. The Pt foil fits to R = 2.765 Å (bulk 2.775 Å) with ΔE0 = +7.5 eV. The predictors instead put its first shell at 2.66–2.72 Å. The simulated χ(k) never vary E0, so an E0 offset in the data reduction shifts the phase. The predictors read that shift as a shorter bond, and so as a Cu- or Ni-alloyed particle. Neither augmentation can produce such a shift: the classical pipeline does not vary E0, and the realism layer must not change oscillation phases, by design. **The most direct fix is to sample ΔE0 (for example ±10 eV) in the simulated χ(k).** That is a data-reduction parameter and leaves the structural labels unchanged.
- **XRD: the spectra are out of domain or physically ambiguous.**
  - 2 Pt patterns have Scherrer sizes of 13, 21 nm, outside the 1.5–8 nm training range.
  - The other Pt pattern(s) fit to a = 3.962 Å, against bulk Pt 3.922 Å. That points to a peak shift, from sample displacement or the support, that the training data do not contain.
  - The 8 Pt–Co patterns fit to a = 3.840–3.873 Å, between the simulated Pt3Co (3.83 Å) and Pt3Fe (3.86 Å) phases. With similar Co, Ni and Fe scattering factors, XRD alone barely separates the 3:1 Pt–Co, Pt–Ni and Pt–Fe families.


## Per-seed values with 95 % bootstrap confidence intervals

### Variant (a), seed 1

- A_real · composition_acc: 0.0000 [0.0000, 0.0000] (n=21)
- A_real · n1_pt_mae: 1.1588 [1.1506, 1.1692] (n=4)
- A_real · r1_pt_mae: 0.0964 [0.0959, 0.0970] (n=4)
- A_real · composition_family_acc: 0.0952 [0.0000, 0.2381] (n=21)
- B_fullfeff_harsh · composition_acc: 0.9967 [0.9867, 1.0000] (n=300)
- B_fullfeff_harsh · ordered_acc: 1.0000 [1.0000, 1.0000] (n=251)
- B_fullfeff_harsh · size_mae: 0.2139 [0.1812, 0.2512] (n=300)
- B_fullfeff_harsh · lattice_mae: 0.0116 [0.0106, 0.0127] (n=300)
- B_fullfeff_harsh · n1_pt_mae: 0.1754 [0.1518, 0.2006] (n=300)
- B_fullfeff_harsh · r1_pt_mae: 0.0056 [0.0051, 0.0062] (n=300)
- B_fullfeff_harsh · n1_m_mae: 0.2063 [0.1764, 0.2395] (n=251)
- B_fullfeff_harsh · r1_m_mae: 0.0052 [0.0048, 0.0057] (n=251)
- C_test_harsh · composition_acc: 0.9974 [0.9960, 0.9988] (n=5051)
- C_test_harsh · ordered_acc: 1.0000 [1.0000, 1.0000] (n=4355)
- C_test_harsh · size_mae: 0.2108 [0.2024, 0.2197] (n=5051)
- C_test_harsh · lattice_mae: 0.0065 [0.0063, 0.0067] (n=5051)
- C_test_harsh · n1_pt_mae: 0.1503 [0.1455, 0.1551] (n=5051)
- C_test_harsh · r1_pt_mae: 0.0032 [0.0031, 0.0033] (n=5051)
- C_test_harsh · n1_m_mae: 0.1480 [0.1434, 0.1525] (n=4355)
- C_test_harsh · r1_m_mae: 0.0027 [0.0027, 0.0028] (n=4355)
- C_test_harsh · fault_auroc: 0.8918 [0.8802, 0.9033] (n=5051)

### Variant (a), seed 2

- A_real · composition_acc: 0.0000 [0.0000, 0.0000] (n=21)
- A_real · n1_pt_mae: 0.8087 [0.8050, 0.8143] (n=4)
- A_real · r1_pt_mae: 0.0711 [0.0707, 0.0716] (n=4)
- A_real · composition_family_acc: 0.0952 [0.0000, 0.2381] (n=21)
- B_fullfeff_harsh · composition_acc: 1.0000 [1.0000, 1.0000] (n=300)
- B_fullfeff_harsh · ordered_acc: 1.0000 [1.0000, 1.0000] (n=251)
- B_fullfeff_harsh · size_mae: 0.2084 [0.1821, 0.2377] (n=300)
- B_fullfeff_harsh · lattice_mae: 0.0118 [0.0107, 0.0130] (n=300)
- B_fullfeff_harsh · n1_pt_mae: 0.1524 [0.1361, 0.1701] (n=300)
- B_fullfeff_harsh · r1_pt_mae: 0.0060 [0.0055, 0.0065] (n=300)
- B_fullfeff_harsh · n1_m_mae: 0.1862 [0.1596, 0.2158] (n=251)
- B_fullfeff_harsh · r1_m_mae: 0.0051 [0.0047, 0.0056] (n=251)
- C_test_harsh · composition_acc: 0.9986 [0.9974, 0.9996] (n=5051)
- C_test_harsh · ordered_acc: 1.0000 [1.0000, 1.0000] (n=4355)
- C_test_harsh · size_mae: 0.2014 [0.1939, 0.2091] (n=5051)
- C_test_harsh · lattice_mae: 0.0067 [0.0065, 0.0069] (n=5051)
- C_test_harsh · n1_pt_mae: 0.1344 [0.1308, 0.1380] (n=5051)
- C_test_harsh · r1_pt_mae: 0.0033 [0.0032, 0.0034] (n=5051)
- C_test_harsh · n1_m_mae: 0.1425 [0.1385, 0.1467] (n=4355)
- C_test_harsh · r1_m_mae: 0.0026 [0.0026, 0.0027] (n=4355)
- C_test_harsh · fault_auroc: 0.8834 [0.8720, 0.8947] (n=5051)

### Variant (b), seed 1

- A_real · composition_acc: 0.2381 [0.0940, 0.4762] (n=21)
- A_real · n1_pt_mae: 0.8838 [0.8748, 0.8907] (n=4)
- A_real · r1_pt_mae: 0.0551 [0.0527, 0.0573] (n=4)
- A_real · composition_family_acc: 0.2857 [0.0952, 0.5238] (n=21)
- B_fullfeff_harsh · composition_acc: 0.8133 [0.7667, 0.8567] (n=300)
- B_fullfeff_harsh · ordered_acc: 0.9721 [0.9522, 0.9920] (n=251)
- B_fullfeff_harsh · size_mae: 0.4658 [0.4143, 0.5186] (n=300)
- B_fullfeff_harsh · lattice_mae: 0.0253 [0.0232, 0.0277] (n=300)
- B_fullfeff_harsh · n1_pt_mae: 0.5875 [0.5275, 0.6443] (n=300)
- B_fullfeff_harsh · r1_pt_mae: 0.0145 [0.0130, 0.0160] (n=300)
- B_fullfeff_harsh · n1_m_mae: 0.4359 [0.3808, 0.4941] (n=251)
- B_fullfeff_harsh · r1_m_mae: 0.0149 [0.0133, 0.0165] (n=251)
- C_test_harsh · composition_acc: 0.7880 [0.7771, 0.7987] (n=5051)
- C_test_harsh · ordered_acc: 0.9458 [0.9389, 0.9522] (n=4355)
- C_test_harsh · size_mae: 0.5574 [0.5409, 0.5747] (n=5051)
- C_test_harsh · lattice_mae: 0.0221 [0.0216, 0.0226] (n=5051)
- C_test_harsh · n1_pt_mae: 0.5892 [0.5761, 0.6013] (n=5051)
- C_test_harsh · r1_pt_mae: 0.0137 [0.0134, 0.0140] (n=5051)
- C_test_harsh · n1_m_mae: 0.3916 [0.3804, 0.4031] (n=4355)
- C_test_harsh · r1_m_mae: 0.0132 [0.0128, 0.0136] (n=4355)
- C_test_harsh · fault_auroc: 0.7188 [0.7008, 0.7357] (n=5051)

### Variant (b), seed 2

- A_real · composition_acc: 0.0476 [0.0000, 0.1429] (n=21)
- A_real · n1_pt_mae: 0.5652 [0.5625, 0.5683] (n=4)
- A_real · r1_pt_mae: 0.1113 [0.1106, 0.1117] (n=4)
- A_real · composition_family_acc: 0.0476 [0.0000, 0.1429] (n=21)
- B_fullfeff_harsh · composition_acc: 0.9267 [0.8967, 0.9533] (n=300)
- B_fullfeff_harsh · ordered_acc: 0.9841 [0.9681, 0.9960] (n=251)
- B_fullfeff_harsh · size_mae: 0.6603 [0.5907, 0.7352] (n=300)
- B_fullfeff_harsh · lattice_mae: 0.0309 [0.0283, 0.0337] (n=300)
- B_fullfeff_harsh · n1_pt_mae: 0.5195 [0.4671, 0.5692] (n=300)
- B_fullfeff_harsh · r1_pt_mae: 0.0176 [0.0158, 0.0195] (n=300)
- B_fullfeff_harsh · n1_m_mae: 0.3843 [0.3450, 0.4299] (n=251)
- B_fullfeff_harsh · r1_m_mae: 0.0161 [0.0141, 0.0179] (n=251)
- C_test_harsh · composition_acc: 0.8965 [0.8883, 0.9050] (n=5051)
- C_test_harsh · ordered_acc: 0.9768 [0.9722, 0.9812] (n=4355)
- C_test_harsh · size_mae: 0.6723 [0.6545, 0.6914] (n=5051)
- C_test_harsh · lattice_mae: 0.0275 [0.0269, 0.0282] (n=5051)
- C_test_harsh · n1_pt_mae: 0.5215 [0.5093, 0.5336] (n=5051)
- C_test_harsh · r1_pt_mae: 0.0158 [0.0154, 0.0162] (n=5051)
- C_test_harsh · n1_m_mae: 0.3545 [0.3462, 0.3638] (n=4355)
- C_test_harsh · r1_m_mae: 0.0152 [0.0147, 0.0156] (n=4355)
- C_test_harsh · fault_auroc: 0.7467 [0.7291, 0.7636] (n=5051)

