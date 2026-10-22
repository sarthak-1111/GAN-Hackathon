# Realism model card: XRD

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
- The narrowest bump the background can form is 5.5° FWHM, 5× the narrowest Bragg peak (1.1° for 8 nm particles). It cannot create or remove reflections. Under the broad peaks of the smallest particles, a background slope can move the apparent maximum slightly; the measured shifts are bounded by the tests and the re-fits below.
- It does not make a simulation correct. Model errors in the simulation (structure, thermal model, FEFF approximations) pass through unchanged.
- It learned artifacts of the open reference data listed below. Instruments, sample environments or processing pipelines that differ from those sources are not represented.

## Training data

- **Real spectra:** 223 train, 48 validation, 55 held out (55 with usable twins); sources: cardiff_27052390, opxrd, zenodo_14186519_GO_PtNP, zenodo_15310405_PtAC, zenodo_17131699_PtCo_PtNi_HTPEM, zenodo_21028650_GOPt10. Splits are by material. The held-out 20 % was never used for training or model selection. Sources and licences are listed per item in `artifacts/data/real/xrd_items.csv` and in the dataset manifest.
- **Fake inputs:** physics twins of the real materials (`forge/twins.py`, `forge/realdata*.py`).
- **Regularization:** preservation and magnitude losses were also applied to Concord Forge simulations (the clean training split of the full build), the domain the model is used on.
- **Training:** selected checkpoint step 14000, last evaluated step 26000; batch 128, Adam 0.0002 / 0.0004, betas (0, 0.99), EMA 0.999, R1 gamma 1 (every 4 steps), preservation weight 10, magnitude weight 0.05, bf16.

## Measured diagnostics (held-out real spectra)

| Comparison with real | C2ST accuracy (50 % = indistinguishable) | Fréchet distance |
|---|---|---|
| simulated twin | 0.845 | 58.5 |
| twin + classical augmentation | 0.809 | 51.6 |
| twin + this model | 0.664 | 61 |

**Label preservation** (500 re-fitted test spectra, tolerance {'a': 0.005}): failure rate 1.4 %.

## Files

- `generator_ema.pt`: EMA generator weights, SHA-256 `f22b89923c5158643b83b73d1987bc148f89d983d5f18c6d87d5a8522d0a6216`.
- `history.json`: validation diagnostics per checkpoint.
- `config.yaml`: training configuration.
- **Inference:** `python artifacts/scripts/realism_infer.py <shard.h5 | sample_dir> --out <dir>`.
