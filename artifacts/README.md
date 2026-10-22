# Concord Forge artifacts

Concord Forge is the data and model factory for Concord, a multi-technique XRD + PDF + XAFS analysis agent. Everything
here was produced on this machine by the code in `forge/`. All spectra in the dataset are **simulated**.

## Results

- **Dataset.** 50,000 simulated Pt and Pt-alloy nanoparticles in 23 HDF5 shards: 42,500 clean and 7,500 with one
  planted fault. Each has XRD (Cu Kα and synchrotron), PDF, and EXAFS χ(k) plus μ(E). `validate_dataset.py` passes
  all 33 checks. See `data/DATASET.md`.
- **EXAFS shortcut.** Validated against full FEFF6L before scaling up: median R-factor 0.0025, 95th percentile 0.034
  on 372 (sample, edge) pairs. See `reports/exafs_shortcut_validation.md`.
- **Realism GANs.** On held-out real spectra, the C2ST accuracy (real vs alternative; 50 % = indistinguishable) is:

  | Technique | raw twin | GAN | classical augmentation |
  |---|---|---|---|
  | XRD | 0.845 | 0.664 | 0.809 |
  | EXAFS | 0.946 | 0.906 | 0.919 |
  | PDF (only 10 held-out spectra) | 0.950 | 0.950 | 0.800 |

  Re-fits with standard tools show that the GAN preserves labels: label failures are 1.4 % for XRD, 0 % for EXAFS,
  and 0 % among converged PDF fits. See `reports/gan_diagnostics.md`.
- **Decision rule: the classical baseline (a) is shipped.** The GAN-augmented predictor (b) did not beat (a) on the
  real held-out set by more than the seed-to-seed variation: composition-family accuracy 0.17 vs 0.10, with a seed
  spread of 0.24. It was also worse on all 17 metrics of the harsh-noise synthetic sets. `models/predictor` is
  variant (a), seed 1. The realism models are delivered for reference only. See `reports/downstream_comparison.md`.
- **Real-data caveat.** The shipped predictor is near perfect on simulated test spectra: composition 1.000, lattice
  MAE 0.004 Å, first-shell R MAE 0.002 Å (`reports/test_split_inference.md`). On real spectra it is not reliable
  yet: the composition family is right for 2 of 21. The main identified cause is an E0 offset of +3 to +8 eV between
  real Pt L3 χ(k) and the simulations, which never vary E0. Sampling ΔE0 in the simulated χ(k) is the recommended
  next step.
- **Test split.** `predict.py` and `realism_infer.py` run on all 5,051 test samples, both on the HDF5 shards and on
  the contract export. Outputs are in `data/test_outputs/`, and the label-preservation check covers every test
  spectrum.

## Layout

```
artifacts/
  data/
    shards/            HDF5 shards, all samples and fields (one file per set / split chunk)
    contract_export/   test + fault-test samples in the Concord data contract (samples/<id>/...)
    manifest.csv       every structure, FEFF reference, sample, shard and real spectrum, with source, ID and licence
    splits.json        split membership (by structure family) and shard hashes
    DATASET.md         dataset card
    pilot/             the 2,000-sample pilot (stage 1), same layout
    real/              real reference spectra on the model grids, with their physics twins (realism training / evaluation)
    work/              intermediate build products (plans, particles, per-sample simulations, FEFF caches)
  models/
    realism_xrd/  realism_pdf/  realism_exafs/   EMA generator, config, validation history, MODEL_CARD.md
    predictor/                                   shipped predictor, config, SHIPPED.json, MODEL_CARD.md
    work/                                        all predictor runs (a/b x 2 seeds), unselected GAN runs, GAN checkpoints
  scripts/
    validate_dataset.py   dataset checks; exits 0 only if every check passes
    realism_infer.py      simulated spectra in, realistic spectra out
    predict.py            one sample directory, a directory of samples, or an HDF5 shard in; predictions out
  reports/
    hardware.md  exafs_shortcut_validation.md  gan_diagnostics.md  downstream_comparison.md  (+ JSON, figures/)
```

`reports/` at the repository root is a link to `artifacts/reports/`.

## Quick start

```
.venv\Scripts\python artifacts\scripts\validate_dataset.py                          # full dataset (or the pilot if not built)
.venv\Scripts\python artifacts\scripts\predict.py artifacts\data\contract_export\samples\F000005
.venv\Scripts\python artifacts\scripts\predict.py artifacts\data\shards\full_clean_test_000.h5 --out predictions.json
.venv\Scripts\python artifacts\scripts\realism_infer.py artifacts\data\shards\full_clean_test_000.h5 --out out\
.venv\Scripts\python -m pytest tests -q -p no:cacheprovider                          # label-preservation and data tests
```

## Rebuilding

```
python -m forge.hw_check                                  # Phase 0
python -m forge.structures                                # Phase 1: reference phases from open databases
python -m forge.exafs_refs                                # FEFF6L references for the EXAFS shortcut
python -m forge.build configs\dataset_pilot.yaml all      # pilot: plan, particles, simulation, shards
python -m forge.exafs_validate <plan.json> <name> <n> <seed>   # shortcut vs full FEFF
python -m forge.build configs\dataset_full.yaml all       # full build
python -c "from forge.export import export; export('artifacts/data', 'artifacts/data/contract_export')"
python -m forge.realdata exafs ; python -m forge.realdata_xrd ; python -m forge.realdata_pdf   # Phase 3
python -m forge.realism <xrd|pdf|exafs> configs\realism_<t>.yaml                                 # Phase 4a, run 1
python -m forge.realism <pdf|exafs> configs\realism_<t>_r1g10.yaml                               # run 2 (R1 gamma 10)
python -m forge.finalize select                          # pick between realism runs on the validation score
python -m forge.train_predictor configs\predictor_<a|b>_s<1|2>.yaml                              # Phase 4b
python -c "from forge.downstream import run_fullfeff; run_fullfeff()"                             # set B full FEFF
python -m forge.finalize downstream                      # Phase 5: comparison, decision rule, ship the predictor
python -m forge.finalize diagnostics                     # Phase 5: GAN diagnostics
python -m forge.finalize cards                           # model cards and dataset card
python -m forge.finalize tidy                            # intermediate checkpoints -> artifacts/models/work/
```

`scripts/finish_b.sh` is the exact sequence used for the PDF run selection and the two predictor (b) runs.

Every run has a config file in `configs/`. Seeds are fixed, and every output has a SHA-256 hash in `manifest.csv`,
`splits.json` or the model directories. Every stage is resumable: per-sample outputs are written atomically and
skipped when present.
