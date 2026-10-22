# Realism model card: PDF

**Status:** NOT USED in the shipped predictor (decision rule: the GAN did not beat classical augmentation); kept for reference.

## What it is

This is a conditional residual GAN generator, `forge/gan.py`. Given a simulated spectrum on the model grid, a condition vector and 64-D noise z, it adds learned instrument and sample artifacts:

`output = sim × gain + background + noise_scale × coloured_noise`

- **Background:** 12 cubic B-spline coefficients.
- **Gain:** 8 B-spline coefficients, exponentiated and clipped to [0.9, 1.1].
- **Noise scale:** 16 B-spline coefficients through softplus, times white noise passed through a learned unit-norm 1-D convolution.
- **Glitches** (EXAFS only): at most 2 gated Gaussian spikes.
- **Backbone:** 1-D U-Net with 4 levels and 64→512 channels, FiLM-conditioned on (condition, z).
- **Discriminator:** 6-block spectrally normalized 1-D ResNet on (spectrum, derivative) with projection conditioning.
- **Losses:** hinge adversarial loss, R1, a preservation loss (cross-correlation of derivatives) and an artifact-magnitude penalty.

## What it can and cannot do

- It can only multiply the simulated spectrum by a smooth gain within 0.9–1.1 and add a smooth background and spatially varying coloured noise. The simulated signal itself is never resampled, shifted or broadened.
- The narrowest bump the background can form is 2.7 Å FWHM, 9× the nearest-neighbour peak width (about 0.3 Å). It cannot create, remove or move PDF peaks.
- It does not make a simulation correct. Model errors in the simulation (structure, thermal model, FEFF approximations) pass through unchanged.
- It learned artifacts of the open reference data listed below. Instruments, sample environments or processing pipelines that differ from those sources are not represented.

## Training data

- **Real spectra:** 49 train, 7 validation, 10 held out (10 with usable twins); sources: github_DeepStruc, github_scatterlab_AuNP, zenodo_56682_ncomm_goldnp_2016. Splits are by material. The held-out 20 % was never used for training or model selection. Sources and licences are listed per item in `artifacts/data/real/pdf_items.csv` and in the dataset manifest.
- **Fake inputs:** physics twins of the real materials (`forge/twins.py`, `forge/realdata*.py`).
- **Regularization:** preservation and magnitude losses were also applied to Concord Forge simulations (the clean training split of the full build), the domain the model is used on.
- **Training:** selected checkpoint step 2000, last evaluated step 14000; batch 128, Adam 0.0002 / 0.0004, betas (0, 0.99), EMA 0.999, R1 gamma 10 (every 4 steps), preservation weight 10, magnitude weight 0.05, bf16; 2 training runs (run with R1 gamma 1: best validation score 1.144 at step 4000 (C2ST 1.000), stopped at 16000; run with R1 gamma 10: best validation score 0.618 at step 2000 (C2ST 1.000), stopped at 14000); selected the run with R1 gamma 10 on the validation score.

## Measured diagnostics (held-out real spectra)

| Comparison with real | C2ST accuracy (50 % = indistinguishable) | Fréchet distance |
|---|---|---|
| simulated twin | 0.950 | 201 |
| twin + classical augmentation | 0.800 | 195 |
| twin + this model | 0.950 | 200 |

**Label preservation** (500 re-fitted test spectra, tolerance {'a': 0.005, 'd_rel': 0.1}): failure rate 2.4 %.

## Files

- `generator_ema.pt`: EMA generator weights, SHA-256 `d49038f02a2bfb1366f3760a8fe35ad89c8e4875e87e7ee4b766b07a7f944582`.
- `history.json`: validation diagnostics per checkpoint.
- `config.yaml`: training configuration.
- **Inference:** `python artifacts/scripts/realism_infer.py <shard.h5 | sample_dir> --out <dir>`.
