# Test-split inference with the shipped scripts

Shipped predictor: variant (a), seed 1. Every command below exited with code 0.

## Commands

| Command | exit | seconds |
|---|---|---|
| `python artifacts/scripts/predict.py artifacts\data\shards\full_clean_test_000.h5 --out artifacts\data\test_outputs\predictions\full_clean_test_000.json` | 0 | 18.8 |
| `python artifacts/scripts/predict.py artifacts\data\shards\full_clean_test_000.h5 --out artifacts\data\test_outputs\predictions\full_clean_test_000_spectra_only.json --no-metadata` | 0 | 17.4 |
| `python artifacts/scripts/predict.py artifacts\data\shards\full_clean_test_001.h5 --out artifacts\data\test_outputs\predictions\full_clean_test_001.json` | 0 | 17.4 |
| `python artifacts/scripts/predict.py artifacts\data\shards\full_clean_test_001.h5 --out artifacts\data\test_outputs\predictions\full_clean_test_001_spectra_only.json --no-metadata` | 0 | 18.7 |
| `python artifacts/scripts/predict.py artifacts\data\shards\full_fault_test_000.h5 --out artifacts\data\test_outputs\predictions\full_fault_test_000.json` | 0 | 16.8 |
| `python artifacts/scripts/predict.py artifacts\data\shards\full_fault_test_000.h5 --out artifacts\data\test_outputs\predictions\full_fault_test_000_spectra_only.json --no-metadata` | 0 | 17.0 |
| `python artifacts/scripts/predict.py artifacts\data\contract_export\samples --out artifacts\data\test_outputs\predictions\contract_export.json` | 0 | 609.9 |
| `python artifacts/scripts/realism_infer.py artifacts\data\shards\full_clean_test_000.h5 --out artifacts\data\test_outputs\realism` | 0 | 20.5 |
| `python artifacts/scripts/realism_infer.py artifacts\data\shards\full_clean_test_001.h5 --out artifacts\data\test_outputs\realism` | 0 | 18.7 |
| `python artifacts/scripts/realism_infer.py artifacts\data\shards\full_fault_test_000.h5 --out artifacts\data\test_outputs\realism` | 0 | 16.5 |
| `python artifacts/scripts/realism_infer.py artifacts\data\test_outputs\contract_sample_input --out artifacts\data\test_outputs\realism_contract_sample` | 0 | 50.7 |

## predict.py against the ground truth

These are the stored simulated test spectra, without the harsh noise of the downstream sets B and C. "With metadata" means the nominal composition and candidate phases supplied with each sample are used, as Concord would call the predictor. They are wrong for the mislabeled-metadata faults. "Spectra only" is `--no-metadata`.

| Input | samples | composition acc. | ordered acc. | size MAE (nm) | lattice MAE (Å) | Pt N1 MAE | Pt R1 MAE (Å) | mean fault AUROC |
|---|---|---|---|---|---|---|---|---|
| test shards, with metadata | 5051 | 1.0000 | 1.0000 | 0.100 | 0.0042 | 0.100 | 0.0021 | 0.928 |
| test shards, spectra only | 5051 | 1.0000 | 1.0000 | 0.098 | 0.0041 | 0.100 | 0.0020 | 0.787 |
| contract export (text files, counting noise), with metadata | 5051 | 1.0000 | 1.0000 | 0.105 | 0.0044 | 0.105 | 0.0021 | 0.919 |

Fault AUROC per type (test shards, with metadata): pt_oxide_shell 0.999, pt_skin 1.000, mixed_oxidation 0.623, low_dopant 0.875, multi_domain 1.000, wrong_phase_candidate 1.000, mislabeled_metadata 1.000.

## realism_infer.py on the test split

The realism layer was applied to every test spectrum in the shards, and to 200 contract sample directories (the first 200 test samples (the realism layer is not shipped); `origin` = `simulated+realism`). The table shows the label-preservation check on every spectrum: peak position before and after the layer.

| Technique | spectra | feature | median shift | 95th pct | max | background RMS / max | gain RMS dev. | noise scale / max |
|---|---|---|---|---|---|---|---|---|
| xrd | 5051 | (111) peak, 2θ (smoothed) (deg 2theta) | 0.0000 | 0.0000 | 0.0782 | 0.0032 | 0.0020 | 0.0006 |
| pdf | 5051 | first G(r) peak, r (A) | 0.0000 | 0.0000 | 0.0200 | 0.0009 | 0.0009 | 0.0021 |
| exafs_pt | 5051 | Pt L3 first-shell |χ(R)| peak (A) | 0.0000 | 0.0000 | 0.6900 | 0.0040 | 0.0000 | 0.0158 |
| exafs_m | 4355 | M K first-shell |χ(R)| peak (A) | 0.0000 | 0.0000 | 0.0100 | 0.0035 | 0.0000 | 0.0118 |

Grid steps: XRD 0.039° 2θ, PDF 0.02 Å; the |χ(R)| peak is located on a 0.01 Å grid.

- **xrd:** 8 of 5051 spectra moved by more than one grid step: F017126 0.078 (Pt icosahedron, 2.1 nm, 303 atoms); F018425 0.078 (Pt icosahedron, 2.1 nm, 304 atoms); F021176 0.078 (Pt icosahedron, 2.1 nm, 307 atoms); F022518 0.078 (Pt icosahedron, 2.1 nm, 303 atoms); F032147 0.078 (Pt icosahedron, 2.1 nm, 306 atoms); F047901 0.078 (Pt icosahedron, 2.1 nm, 303 atoms); F048188 0.078 (Pt icosahedron, 2.1 nm, 305 atoms); F049627 0.078 (Pt icosahedron, 2.1 nm, 305 atoms).
- **pdf:** 0 of 5051 spectra moved by more than one grid step.
- **exafs_pt:** 1 of 5051 spectra moved by more than one grid step. 1 of them are near-ties: the clean spectrum has two maxima in the window within 2 % of each other, so a sub-percent change flips the argmax rather than moving a peak (F040786: 0.69, second maximum at 99.7 %, Pt icosahedron, 2.1 nm, 302 atoms).
- **exafs_m:** 0 of 4355 spectra moved by more than one grid step.

