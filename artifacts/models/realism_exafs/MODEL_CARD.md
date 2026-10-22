# Realism model card: EXAFS

**Status:** NOT USED in the shipped predictor (decision rule: the GAN did not beat classical augmentation); kept for reference.

## What it is

This is a conditional residual GAN generator, `forge/gan.py`. Given a simulated spectrum on the model grid, a condition vector and 64-D noise z, it adds learned instrument and sample artifacts:

`output = sim × gain + background + noise_scale × coloured_noise + glitches`

- **Background:** 12 cubic B-spline coefficients.
- **Gain:** 8 B-spline coefficients, exponentiated and clipped to [0.9, 1.1].
- **Noise scale:** 16 B-spline coefficients through softplus, times white noise passed through a learned unit-norm 1-D convolution.
- **Glitches** (EXAFS only): at most 2 gated Gaussian spikes.
- **Backbone:** 1-D U-Net with 4 levels and 64→512 channels, FiLM-conditioned on (condition, z).
- **Discriminator:** 6-block spectrally normalized 1-D ResNet on (spectrum, derivative) with projection conditioning.
- **Losses:** hinge adversarial loss, R1, a preservation loss (Fourier-filtered first-shell amplitude and phase) and an artifact-magnitude penalty.

## What it can and cannot do

- It can only multiply the simulated spectrum by a smooth gain within 0.9–1.1 and add a smooth background and spatially varying coloured noise plus at most 2 narrow glitches. The simulated signal itself is never resampled, shifted or broadened.
- The narrowest bump the background can form is 1.15 Å⁻¹ FWHM, comparable to the first-shell oscillation period (π/R ≈ 1.1 Å⁻¹). The basis alone therefore does not guarantee that the first shell is untouched. That is enforced by the first-shell preservation loss (Fourier-filtered amplitude and phase) and checked by tests and larch re-fits (below).
- It does not make a simulation correct. Model errors in the simulation (structure, thermal model, FEFF approximations) pass through unchanged.
- It learned artifacts of the open reference data listed below. Instruments, sample environments or processing pipelines that differ from those sources are not represented.

## Training data

- **Real spectra:** 251 train, 66 validation, 130 held out (74 with usable twins); sources: bmm_standards_nist, cardiff_27052390, cls_xasdb, github_xraylarch_examples, lisa_xasdb, mdr_xafs_db, mendeley_9bhtr2w54w, xaslib, zenodo_17131699_PtCo_PtNi_HTPEM. Splits are by material. The held-out 20 % was never used for training or model selection. Sources and licences are listed per item in `artifacts/data/real/exafs_items.csv` and in the dataset manifest.
- **Fake inputs:** physics twins of the real materials (`forge/twins.py`, `forge/realdata*.py`).
- **Regularization:** preservation and magnitude losses were also applied to Concord Forge simulations (the clean training split of the full build), the domain the model is used on.
- **Training:** selected checkpoint step 2000, last evaluated step 14000; batch 128, Adam 0.0002 / 0.0004, betas (0, 0.99), EMA 0.999, R1 gamma 10 (every 4 steps), preservation weight 10, magnitude weight 0.05, bf16; 2 training runs (run with R1 gamma 1: best validation score 1.731 at step 2000 (C2ST 0.917), stopped at 14000; run with R1 gamma 10: best validation score 0.742 at step 2000 (C2ST 0.932), stopped at 14000); selected the run with R1 gamma 10 on the validation score.

## Measured diagnostics (held-out real spectra)

| Comparison with real | C2ST accuracy (50 % = indistinguishable) | Fréchet distance |
|---|---|---|
| simulated twin | 0.946 | 82.3 |
| twin + classical augmentation | 0.919 | 167 |
| twin + this model | 0.906 | 82.9 |

**Label preservation** (500 re-fitted test spectra, tolerance {'N_rel': 0.1, 'R': 0.01}): failure rate 0.0 %.

## Files

- `generator_ema.pt`: EMA generator weights, SHA-256 `fa58f869bfdf73e6b4f9c3677e1369e165b7be037aa4ff1f92ea41cec327a14e`.
- `history.json`: validation diagnostics per checkpoint.
- `config.yaml`: training configuration.
- **Inference:** `python artifacts/scripts/realism_infer.py <shard.h5 | sample_dir> --out <dir>`.
