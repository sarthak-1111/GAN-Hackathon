"""Shard loading and model views (fixed grids, normalization, label targets) shared by the GAN and the predictor."""
import json
from pathlib import Path

import h5py
import numpy as np

from .build import COMPOSITIONS, FAULTS, PHASES, SHAPE_LIST

# model grids
K_SLICE = slice(40, 296)          # stored k = 0..16 step 0.05 -> 2.00 .. 14.75 A^-1 (256 points)
TTH_STORE = np.linspace(10.0, 100.0, 2048)
TTH_MODEL = np.linspace(20.0, 80.0, 1536)   # Cu K-alpha window shared with most open laboratory patterns (opXRD)
TECHNIQUES = ["xrd", "xrd_sync", "pdf", "exafs_pt", "exafs_m"]
PDF_N = 1480                      # r = 0.50 ... 30.08 A (0.02 A steps): the range of open nanoparticle PDF data
LENGTHS = {"xrd": 1536, "xrd_sync": 2048, "pdf": PDF_N, "exafs_pt": 256, "exafs_m": 256}
EDGES = ["Pt", "Co", "Ni", "Fe", "Cu"]


def view(technique, arr, k=None):
    """Raw stored arrays -> model input (n, L) float32.

    xrd:            Cu K-alpha pattern resampled to 2theta = 20-80 deg (1536 points), intensity / max
    xrd_sync:       intensity / max on the stored Q grid (relative intensities; absolute scale is arbitrary)
    pdf:            G(r) on r = 0.5-30.08 A, / max|G| (scale factor is a refined quantity in experiments)
    exafs_*:        k^2 chi(k) on 2 .. 14.75 A^-1, absolute (edge-step normalized, carries coordination)
    """
    a = np.asarray(arr, np.float32)
    if technique == "xrd":
        if a.shape[1] != len(TTH_MODEL):
            a = np.stack([np.interp(TTH_MODEL, TTH_STORE, row) for row in a]).astype(np.float32)
        return a / np.maximum(a.max(axis=1, keepdims=True), 1e-12)
    if technique == "xrd_sync":
        return a / np.maximum(a.max(axis=1, keepdims=True), 1e-12)
    if technique == "pdf":
        a = a[:, :PDF_N]
        return a / np.maximum(np.abs(a).max(axis=1, keepdims=True), 1e-12)
    if technique.startswith("exafs"):
        kk = np.asarray(k if k is not None else np.arange(321) * 0.05, np.float32)
        return (a * kk ** 2)[:, K_SLICE]
    raise ValueError(technique)


class Dataset:
    """In-memory arrays for selected (set, split) shards."""

    def __init__(self, data_dir, parts=(("clean", "train"),), max_n=None):
        data_dir = Path(data_dir)
        splits = json.loads((data_dir / "splits.json").read_text())
        files = [s["shard"] for s in splits["shards"] if (s["set"], s["split"]) in set(parts)]
        acc = {}
        meta = []
        for f in sorted(files):
            with h5py.File(data_dir / "shards" / f, "r") as h:
                for key in ("xrd_cu", "xrd_sync", "pdf_gr", "chi_pt", "chi_m", "composition_id", "phase_id", "ordered",
                            "shape_id", "size_bin", "diameter_nm", "lattice_parameter", "strain", "vacancy_fraction",
                            "first_shell", "faults", "x_M_true", "inst_xrd_cu", "inst_xrd_sync", "inst_pdf", "inst_exafs"):
                    acc.setdefault(key, []).append(h[key][:])
                acc.setdefault("sample_id", []).append(np.array([x.decode() for x in h["sample_id"][:]]))
                meta += [json.loads(m) for m in h["meta_json"][:]]
                self.grids = {g: h["grid/" + g][:] for g in h["grid"]}
        self.a = {k: np.concatenate(v) for k, v in acc.items()}
        self.meta = meta
        if max_n is not None:
            for k in self.a:
                self.a[k] = self.a[k][:max_n]
            self.meta = self.meta[:max_n]
        self.n = len(self.a["sample_id"])

    # ------------------------------------------------------------------ inputs
    def spectra(self):
        a = self.a
        out = {"xrd": view("xrd", a["xrd_cu"]), "xrd_sync": view("xrd_sync", a["xrd_sync"]),
               "pdf": view("pdf", a["pdf_gr"]), "exafs_pt": view("exafs_pt", a["chi_pt"])}
        m = view("exafs_m", np.nan_to_num(a["chi_m"]))
        out["exafs_m"] = m
        present = {t: np.ones(self.n, bool) for t in out}
        present["exafs_m"] = a["composition_id"] != 0
        return out, present

    def m_edge(self):
        """Index into EDGES of the alloying element K edge (0 = none for pure Pt)."""
        comp = [COMPOSITIONS[c] for c in self.a["composition_id"]]
        el = {"Pt": "Pt", "Pt3Co": "Co", "PtCo": "Co", "Pt3Ni": "Ni", "PtNi": "Ni", "Pt3Fe": "Fe", "PtCu": "Cu"}
        return np.array([EDGES.index(el[c]) for c in comp])

    def composition_fractions(self):
        """Nominal (true) composition fractions over [Pt, Co, Ni, Fe, Cu]."""
        out = np.zeros((self.n, 5), np.float32)
        for i, c in enumerate(self.a["composition_id"]):
            name = COMPOSITIONS[c]
            x = {"Pt": 0, "Pt3Co": .25, "PtCo": .5, "Pt3Ni": .25, "PtNi": .5, "Pt3Fe": .25, "PtCu": .5}[name]
            out[i, 0] = 1 - x
            if x:
                out[i, EDGES.index(self.m_el(name))] = x
        return out

    @staticmethod
    def m_el(name):
        return {"Pt3Co": "Co", "PtCo": "Co", "Pt3Ni": "Ni", "PtNi": "Ni", "Pt3Fe": "Fe", "PtCu": "Cu"}.get(name)

    def metadata_vectors(self):
        """Experimenter metadata as model input: nominal composition one-hot (7) + candidate phase one-hot (4)."""
        out = np.zeros((self.n, len(COMPOSITIONS) + len(PHASES)), np.float32)
        for i, m in enumerate(self.meta):
            md = m["spec"]["metadata"]
            out[i, COMPOSITIONS.index(md["nominal_composition"])] = 1
            for p in md["candidate_phases"]:
                out[i, len(COMPOSITIONS) + PHASES.index(p)] = 1
        return out

    # ------------------------------------------------------------------ targets
    def targets(self):
        a = self.a
        fs = a["first_shell"]
        t = dict(composition=a["composition_id"].astype(np.int64),
                 ordered=a["ordered"].astype(np.float32),                       # -1 = not applicable (pure Pt)
                 size=a["diameter_nm"].astype(np.float32),
                 lattice=a["lattice_parameter"].astype(np.float32),
                 n1_pt=fs[:, 0, 0], r1_pt=fs[:, 0, 1], n1_m=fs[:, 1, 0], r1_m=fs[:, 1, 1],
                 faults=a["faults"].astype(np.float32))
        return t
