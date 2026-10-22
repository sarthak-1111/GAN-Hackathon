"""Export test + fault-test samples in the Concord data contract.

samples/<sample_id>/
  sample.json   sample_id, composition (nominal, as reported), techniques, instrument, origin, ground_truth, faults
  xrd.xy        2theta_deg  intensity  sigma        (Cu K-alpha laboratory pattern)
  pdf.gr        r_A  G  sigma
  xafs_mu.dat   energy_eV  mu                       (Pt L3)
  xafs_chi.k    k_inv_A  chi                        (Pt L3)
Additional files listed under sample.json "extra_files" (ignored by readers that only know the contract):
  xrd_sync.xy (synchrotron pattern, 2theta for the sampled wavelength), xafs_mu_<M>_K.dat, xafs_chi_<M>_K.k

origin = "simulated": physics simulation plus counting / measurement noise drawn from the instrument parameters.
origin = "simulated+realism": the same with the shipped realism model applied (only if the GAN wins the decision rule).
"""
import json
import math
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from .build import COMPOSITIONS, FAULTS
from .paths import CONFIGS, load_yaml

E0 = {"Pt": 11564.0, "Co": 7709.0, "Ni": 8333.0, "Fe": 7112.0, "Cu": 8979.0}


def _write(path, cols, header):
    np.savetxt(path, np.column_stack(cols), fmt="%.6g", header=header, comments="# ")


def export(data_dir, out_dir, parts=(("clean", "test"), ("fault", "test")), realism=None, seed=0):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    root = out_dir / "samples"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    splits = json.loads((data_dir / "splits.json").read_text())
    rng = np.random.default_rng(seed)
    n = 0
    for sh in splits["shards"]:
        if (sh["set"], sh["split"]) not in set(parts):
            continue
        with h5py.File(data_dir / "shards" / sh["shard"], "r") as h:
            g = {k: h["grid/" + k][:] for k in h["grid"]}
            for i in range(h["sample_id"].shape[0]):
                meta = json.loads(h["meta_json"][i])
                sp, pm = meta["spec"], meta["particle"]
                sid = sp["sample_id"]
                d = root / sid
                d.mkdir()
                ins = sp["instruments"]
                # XRD (Cu K-alpha): Poisson counts at the sampled peak count level
                I = h["xrd_cu"][i].astype(float)
                cnt = rng.poisson(I / I.max() * ins["xrd_cu"]["counts_max"]).astype(float)
                _write(d / "xrd.xy", [g["xrd_cu_two_theta"], cnt, np.sqrt(np.maximum(cnt, 1.0))],
                       "2theta_deg intensity sigma  (Cu K-alpha1/2, counts)")
                # synchrotron pattern on 2theta for its wavelength
                lam = ins["xrd_sync"]["wavelength"]
                q = g["xrd_sync_q"]
                tth = np.degrees(2 * np.arcsin(np.clip(q * lam / (4 * math.pi), 0, 1)))
                Is = h["xrd_sync"][i].astype(float)
                cs = rng.poisson(Is / Is.max() * ins["xrd_sync"]["counts_max"]).astype(float)
                _write(d / "xrd_sync.xy", [tth, cs, np.sqrt(np.maximum(cs, 1.0))],
                       f"2theta_deg intensity sigma  (synchrotron, wavelength {lam:.5f} A, counts)")
                # PDF with Q-white noise (r-correlated) at the sampled relative level
                G = h["pdf_gr"][i].astype(float)
                sig = ins["pdf"]["noise_rel"] * np.abs(G).max()
                noise = np.convolve(rng.normal(size=len(G) + 8), np.ones(5) / np.sqrt(5), mode="same")[4:-4]
                _write(d / "pdf.gr", [g["pdf_r"], G + sig * noise, np.full(len(G), sig)], "r_A G sigma")
                # XAFS: chi(k) noise ~ constant in chi, mu(E) noise scaled to the edge step
                extra = ["xrd_sync.xy"]
                for edge, X in (("pt", "Pt"), ("m", sp["M"])):
                    if X is None:
                        continue
                    chi = h[f"chi_{edge}"][i].astype(float)
                    mu = h[f"mu_{edge}"][i].astype(float)
                    s = ins["exafs"]["noise_chi"]
                    chi_n = chi + rng.normal(size=len(chi)) * s
                    mu_n = mu + rng.normal(size=len(mu)) * s * 0.3
                    if X == "Pt":
                        fk, fm = "xafs_chi.k", "xafs_mu.dat"
                    else:
                        fk, fm = f"xafs_chi_{X}_K.k", f"xafs_mu_{X}_K.dat"
                        extra += [fm, fk]
                    _write(d / fk, [g["exafs_k"], chi_n], f"k_inv_A chi  ({X} {'L3' if X == 'Pt' else 'K'} edge)")
                    _write(d / fm, [E0[X] + g["mu_e_rel"], mu_n], f"energy_eV mu  ({X} {'L3' if X == 'Pt' else 'K'} edge, normalized)")
                gt = dict(composition=sp["composition"], composition_fractions=pm["composition_true"],
                          phase=sp["phase"], ordered=None if sp["M"] is None else sp["phase"] != "A1",
                          lattice_parameter_A=pm["lattice_parameter"], lattice_a_A=pm["lattice_a"], lattice_c_A=pm["lattice_c"],
                          shape=sp["shape"], diameter_nm=pm["diameter_nm"], n_atoms=pm["n_atoms"], strain=sp["strain"],
                          vacancy_fraction=sp["vacancy_fraction"], coordination_shells=pm["shells"],
                          oxidation_states=pm["oxidation_states"], u_iso_A2=pm["U"], delta2_A2=sp["delta2"])
                sample = dict(
                    sample_id=sid, composition=sp["metadata"]["nominal_composition"],
                    candidate_phases=sp["metadata"]["candidate_phases"],
                    techniques=["xrd", "pdf", "xafs"],
                    instrument=dict(
                        xrd=dict(radiation="Cu K-alpha1/2", wavelength_A=[ins["xrd_cu"]["wavelength_ka1"], ins["xrd_cu"]["wavelength_ka2"]],
                                 ka2_ratio=ins["xrd_cu"]["ka2_ratio"], monochromator=ins["xrd_cu"]["monochromator"],
                                 caglioti_UVW_deg2=[ins["xrd_cu"]["caglioti_U"], ins["xrd_cu"]["caglioti_V"], ins["xrd_cu"]["caglioti_W"]],
                                 x_unit="2theta_deg"),
                        xrd_sync=dict(wavelength_A=lam, x_unit="2theta_deg"),
                        pdf=dict(qmin_inv_A=ins["pdf"]["qmin"], qmax_inv_A=ins["pdf"]["qmax"], qdamp=ins["pdf"]["qdamp"],
                                 qbroad=ins["pdf"]["qbroad"]),
                        xafs=dict(edge="Pt L3", e0_eV=E0["Pt"], extra_edges=[f"{sp['M']} K"] if sp["M"] else [])),
                    origin="simulated" if realism is None else "simulated+realism",
                    ground_truth=gt, faults=[sp["fault"]] if sp["fault"] else [], extra_files=extra,
                    notes="Simulated data (Concord Forge). Noise drawn from the sample's instrument parameters.")
                (d / "sample.json").write_text(json.dumps(sample, indent=1, default=float))
                n += 1
    (out_dir / "README.md").write_text(
        "# Concord contract export\n\nTest split and fault-test split of the Concord Forge dataset in the Concord data "
        "contract (see artifacts/data/DATASET.md). All samples are simulated; `origin` says whether the realism layer "
        f"was applied.\n\n{n} samples.\n")
    print(f"exported {n} samples to {root}")
    return n


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
