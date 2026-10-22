# Predictor model card

**Shipped variant: (a) classical baseline augmentation**, seed 1 (validation score 0.4862). The decision rule is applied in `reports/downstream_comparison.md`.

## What it is

This is a multi-technique property predictor for Pt and Pt-alloy nanoparticles (`forge/predictor.py`, 22 M parameters):

- **Encoders:** one 1-D ResNet-18-style encoder per technique, each taking (spectrum, derivative). The techniques are Cu Kα XRD (2θ 20–80°), synchrotron XRD (Q 0.8–12 Å⁻¹), PDF (r 0.5–30 Å), Pt L3 EXAFS and M K-edge EXAFS (k²χ, k 2–14.75 Å⁻¹).
- **Metadata:** an optional token carrying the nominal composition and candidate phases.
- **Fusion:** a 4-layer transformer over whichever tokens are present. Training randomly drops techniques (p = 0.3) and metadata (p = 0.5), so the model works with any subset.
- **Heads:** composition (7 classes), ordered vs disordered (alloys), equivalent-sphere diameter, lattice parameter, first-shell N and R for the Pt and the M absorber, and 7 fault flags (multi-label).

## Training

- **Data:** Concord Forge simulations only (34002 clean + 6006 fault training samples, 4198 + 743 validation). Training used the train split, plus fault samples from the train split; model selection used the validation split, scored from spectra without metadata.
- **Augmentation:** Poisson / Gaussian noise, polynomial background, Gaussian broadening, normalization jitter (forge/augment.py).
- **Optimizer:** AdamW, lr 0.001, weight decay 0.01, 12000 steps of batch 128, cosine schedule with 500 warm-up steps, bf16 autocast.

## Measured performance

Mean over the two training seeds of each variant; full tables with bootstrap CIs are in `reports/downstream_comparison.md`.

| Test set | Metric | shipped variant | other variant |
|---|---|---|---|
| A_real | composition_acc | 0.0000 | 0.1429 |
| A_real | composition_family_acc | 0.0952 | 0.1667 |
| A_real | n1_pt_mae | 0.9838 | 0.7245 |
| A_real | r1_pt_mae | 0.0838 | 0.0832 |
| B_fullfeff_harsh | composition_acc | 0.9983 | 0.8700 |
| B_fullfeff_harsh | lattice_mae | 0.0117 | 0.0281 |
| B_fullfeff_harsh | n1_m_mae | 0.1962 | 0.4101 |
| B_fullfeff_harsh | n1_pt_mae | 0.1639 | 0.5535 |
| B_fullfeff_harsh | ordered_acc | 1.0000 | 0.9781 |
| B_fullfeff_harsh | r1_m_mae | 0.0052 | 0.0155 |
| B_fullfeff_harsh | r1_pt_mae | 0.0058 | 0.0160 |
| B_fullfeff_harsh | size_mae | 0.2112 | 0.5630 |
| C_test_harsh | composition_acc | 0.9980 | 0.8422 |
| C_test_harsh | fault_auroc | 0.8876 | 0.7328 |
| C_test_harsh | lattice_mae | 0.0066 | 0.0248 |
| C_test_harsh | n1_m_mae | 0.1453 | 0.3730 |
| C_test_harsh | n1_pt_mae | 0.1423 | 0.5553 |
| C_test_harsh | ordered_acc | 1.0000 | 0.9613 |
| C_test_harsh | r1_m_mae | 0.0027 | 0.0142 |
| C_test_harsh | r1_pt_mae | 0.0032 | 0.0147 |
| C_test_harsh | size_mae | 0.2061 | 0.6148 |

## Limitations

- **Not yet reliable on real spectra.** On the 21 real held-out spectra of set A, the composition family was right for 2 of 21 (mean over the two seeds). Two causes were identified:
  - Real Pt L3 χ(k) are offset by ΔE0 = +3 to +8 eV from the simulations, which never vary E0. The predictor reads the phase shift as shorter bonds. Pt foil, for example, comes out at R ≈ 2.7 Å instead of 2.775 Å.
  - Several real XRD patterns lie outside the training domain: crystallites larger than 8 nm, or peak shifts from sample displacement or the support.

  See the diagnosis in `reports/downstream_comparison.md`. Sampling ΔE0 in the simulated χ(k) is the first fix to make. Until then, treat predictions on real data as hypotheses to check with standard fits.
- **Trained only on simulated spectra.** Real-data performance was measured on a small set of open reference spectra, with wide confidence intervals.
- **Training domain:** 7 composition classes, fcc-derived structures from 1.5–8 nm, and the planted faults listed in the dataset card. Bulk materials (foils), other elements, supports, ligands and oxidation beyond the fault models are out of domain.
- **Size predictions:** these refer to the equivalent-sphere diameter of the simulated particle. Lattice parameters refer to the volume-equivalent pseudo-cubic cell.
- **Metadata faults:** wrong phase candidate and mislabeled composition can only be flagged when the metadata token is supplied.

## Files

- `model.pt`: weights and target normalization, SHA-256 `f6df3d64121f3c094cf13671d11ee87c7de3104e055bb2ddf22cffd853de9ac3`.
- `config.json`: training configuration.
- **Inference:** `python artifacts/scripts/predict.py <sample_dir | samples_root | shard.h5> --out predictions.json`.
