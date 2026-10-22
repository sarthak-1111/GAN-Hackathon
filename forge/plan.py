"""Dataset plan: balanced sample specifications, planted faults and structure-family splits.

Balancing is hierarchical: equal counts per composition (7); within an alloy equal ordered / disordered; within a
phase equal counts per allowed shape (disordered and pure Pt: 5 shapes; ordered: the 3 single-crystal shapes);
within a shape equal counts per size bin. Faults (15%) are spread equally over the 7 fault types and, within a
type, over the compositions / phases / shapes / size bins the fault applies to.

A structure family is (composition, phase, shape, discrete shape parameters) - for spheres the diameter rounded to
family_sphere_bin_nm. All samples of one family (clean or faulted) go to the same split.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .particles import ALL_FAULTS, METADATA_FAULTS, SHAPES, SINGLE_CRYSTAL, choose_shape_params, shape_tables
from .paths import CONFIGS, STRUCT_DIR, load_yaml
from .structures import SYSTEMS

COMPOSITIONS = list(SYSTEMS)


def phase_params(ref, comp, phase):
    if phase == "A1":
        a = ref["endpoints"]["Pt"] if SYSTEMS[comp]["M"] is None else ref["vegard"][comp]["a_vegard"]
        return dict(a=a, c=a, a_mean=a, alpha=90.0)
    return ref["phases"][f"{comp}|{phase}"]["params"]


def _unused_strata_weights():
    """(composition, phase, shape) -> weight, summing to 1."""
    w = {}
    for comp in COMPOSITIONS:
        spec = SYSTEMS[comp]
        phases = ["A1"] if spec["M"] is None else ["A1", spec["ordered"]]
        for phase in phases:
            shapes = SHAPES if phase == "A1" else list(SINGLE_CRYSTAL)
            for shape in shapes:
                w[(comp, phase, shape)] = (1 / len(COMPOSITIONS)) * (1 / len(phases)) * (1 / len(shapes))
    return w


def largest_remainder(n, weights, rng):
    """Integer allocation of n proportional to weights; ties in the remainder are broken at random."""
    keys = list(weights)
    raw = np.array([weights[k] for k in keys], float)
    raw = raw / raw.sum() * n
    base = np.floor(raw).astype(int)
    rem = n - base.sum()
    order = np.argsort(-(raw - base) - 1e-9 * rng.random(len(keys)), kind="stable")
    base[order[:rem]] += 1
    return dict(zip(keys, base.tolist()))


def allocate(n, n_bins, rng, fault=None):
    """Hierarchical balanced allocation: composition -> phase -> shape -> size bin."""
    out = []
    comps = [c for c in COMPOSITIONS if fault != "pt_skin" or SYSTEMS[c]["M"] is not None]
    for comp, nc in largest_remainder(n, {c: 1.0 for c in comps}, rng).items():
        spec = SYSTEMS[comp]
        phases = ["A1"] if spec["M"] is None else ["A1", spec["ordered"]]
        for phase, n_p in largest_remainder(nc, {p: 1.0 for p in phases}, rng).items():
            shapes = SHAPES if phase == "A1" else list(SINGLE_CRYSTAL)
            shapes = [sh for sh in shapes if fault_allowed(fault, comp, phase, sh)]
            for shape, n_s in largest_remainder(n_p, {sh: 1.0 for sh in shapes}, rng).items():
                for b, n_b in largest_remainder(n_s, {i: 1.0 for i in range(n_bins)}, rng).items():
                    out += [(comp, phase, shape, b, fault)] * n_b
    return out


def fault_allowed(fault, comp, phase, shape):
    if fault == "pt_skin":
        return SYSTEMS[comp]["M"] is not None
    if fault == "multi_domain":
        return shape in SINGLE_CRYSTAL
    return True


def u(rng, lohi):
    return float(rng.uniform(lohi[0], lohi[1]))


def draw_instruments(rng, inst):
    xc, xs, pd, ex = inst["xrd_cu"], inst["xrd_sync"], inst["pdf"], inst["exafs"]
    return dict(
        xrd_cu=dict(wavelength_ka1=xc["wavelength_ka1"], wavelength_ka2=xc["wavelength_ka2"],
                    ka2_ratio=u(rng, xc["ka2_ratio"]), monochromator=str(rng.choice(xc["monochromator"])),
                    caglioti_U=u(rng, xc["caglioti_U"]), caglioti_V=u(rng, xc["caglioti_V"]),
                    caglioti_W=u(rng, xc["caglioti_W"]), lorentz_fraction=u(rng, xc["lorentz_fraction"]),
                    counts_max=float(math.exp(u(rng, np.log(xc["counts_max"]))))),
        xrd_sync=dict(wavelength=u(rng, xs["wavelength"]), fwhm_q0=u(rng, xs["fwhm_q0"]), fwhm_q1=u(rng, xs["fwhm_q1"]),
                      counts_max=float(math.exp(u(rng, np.log(xs["counts_max"]))))),
        pdf=dict(qmin=u(rng, pd["qmin"]), qmax=u(rng, pd["qmax"]), qdamp=u(rng, pd["qdamp"]), qbroad=u(rng, pd["qbroad"]),
                 noise_rel=float(math.exp(u(rng, np.log(pd["noise_rel"]))))),
        exafs=dict(noise_chi=float(math.exp(u(rng, np.log(ex["noise_chi"])))),
                   edge_width=u(rng, ex["mu_edge_width_ev"])),
    )


def make_plan(cfg, inst, ref):
    rng = np.random.default_rng(cfg["seed"])
    tables = shape_tables()
    smp = inst["sample"]
    bins = smp["size_bins_nm"]
    n_fault = int(round(cfg["n_total"] * cfg["fault_fraction"]))
    n_clean = cfg["n_total"] - n_fault
    rows = allocate(n_clean, len(bins) - 1, rng)  # (comp, phase, shape, bin_index, fault)
    for fault, nf in largest_remainder(n_fault, {f: 1.0 for f in ALL_FAULTS}, rng).items():
        rows += allocate(nf, len(bins) - 1, rng, fault)

    order = rng.permutation(len(rows))
    plan = []
    for idx, j in enumerate(order):
        comp, phase, shape, b, fault = rows[j]
        spec_sys = SYSTEMS[comp]
        prm = phase_params(ref, comp, phase)
        d = u(rng, (bins[b], bins[b + 1]))
        sp = dict(sample_id=f"{cfg['id_prefix']}{idx:06d}", composition=comp, M=spec_sys["M"], x_M=spec_sys["x"],
                  phase=phase, shape=shape, size_bin=b, d_target_nm=d,
                  strain=u(rng, smp["strain"]), vacancy_fraction=u(rng, smp["vacancy_fraction"]),
                  surface_relax=u(rng, smp["surface_relax"]), static_sigma=u(rng, smp["static_sigma"]),
                  u_iso_pt=u(rng, smp["u_iso_pt"]), u_ratio_m=u(rng, smp["u_ratio_m"]), u_iso_o=u(rng, smp["u_iso_o"]),
                  delta2=u(rng, smp["delta2"]), corr_floor=smp["correlation_floor"],
                  fault=fault, seed=int(rng.integers(0, 2 ** 62)), instruments=draw_instruments(rng, inst))
        if shape != "sphere":
            sp["shape_params"] = choose_shape_params(shape, d, prm["a_mean"], rng, tables)
            fam_param = json.dumps(sp["shape_params"], sort_keys=True)
        else:
            fam_param = f"d~{round(d / cfg['family_sphere_bin_nm']) * cfg['family_sphere_bin_nm']:.2f}"
        sp["family"] = f"{comp}|{phase}|{shape}|{fam_param}"
        # metadata as an experimenter would report it; faults of the metadata kind corrupt it
        meta = dict(nominal_composition=comp, candidate_phases=[phase])
        if fault == "mislabeled_metadata":
            meta["nominal_composition"] = str(rng.choice([c for c in COMPOSITIONS if c != comp]))
        if fault == "wrong_phase_candidate":
            wrong = [p for p in ("A1", "L1_2", "L1_0", "L1_1") if p != phase]
            meta["candidate_phases"] = [str(rng.choice(wrong))]
        sp["metadata"] = meta
        plan.append(sp)
    assign_splits(plan, cfg, rng)
    return plan


def assign_splits(plan, cfg, rng, restarts=400):
    """Family-level split assignment (families never cross splits).

    Per (composition, phase, shape) stratum, random-restart greedy search: families are visited in a random order and
    each goes to the split with the largest remaining deficit; the assignment with the smallest squared deviation
    from the target counts (clean samples, plus fault samples weighted 3x because they are fewer) is kept.
    """
    targets = cfg["splits"]
    names = list(targets)
    frac = np.array([targets[k] for k in names])
    n_c = Counter(s["family"] for s in plan if not s["fault"])
    n_f = Counter(s["family"] for s in plan if s["fault"])
    strata = defaultdict(list)
    for fam in sorted(set(n_c) | set(n_f)):
        strata[tuple(fam.split("|", 3)[:3])].append(fam)
    fam_split = {}
    for key in sorted(strata):
        fams = strata[key]
        cc = np.array([n_c[f] for f in fams], float)
        ff = np.array([n_f[f] for f in fams], float)
        tc, tf = frac * cc.sum(), frac * ff.sum()
        best, best_obj = None, np.inf
        for _ in range(restarts):
            order = rng.permutation(len(fams))
            gc, gf = np.zeros(len(names)), np.zeros(len(names))
            assign = np.zeros(len(fams), int)
            for j in order:
                deficit = (tc - gc) / max(cc.sum(), 1) * (cc[j] > 0) + (tf - gf) / max(ff.sum(), 1) * (ff[j] > 0)
                if cc[j] == 0 and ff[j] == 0:
                    deficit = tc - gc
                sp = int(np.argmax(deficit + 1e-9 * frac))
                assign[j] = sp
                gc[sp] += cc[j]
                gf[sp] += ff[j]
            obj = (((gc - tc) / max(cc.sum(), 1)) ** 2).sum() + 3 * (((gf - tf) / max(ff.sum(), 1)) ** 2).sum()
            if obj < best_obj:
                best, best_obj = assign.copy(), obj
        for j, fam in enumerate(fams):
            fam_split[fam] = names[best[j]]
    for s in plan:
        s["split"] = fam_split[s["family"]]
        s["set"] = "fault" if s["fault"] else "clean"


def summarize(plan):
    out = dict(n=len(plan), by_split=Counter(f"{s['set']}:{s['split']}" for s in plan),
               by_comp=Counter(s["composition"] for s in plan), by_phase=Counter(s["phase"] for s in plan),
               by_shape=Counter(s["shape"] for s in plan), by_bin=Counter(s["size_bin"] for s in plan),
               by_fault=Counter(s["fault"] or "none" for s in plan), n_families=len({s["family"] for s in plan}))
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in out.items()}


def main(cfg_path):
    cfg = load_yaml(cfg_path)
    inst = load_yaml(CONFIGS / "instruments.yaml")
    ref = json.loads((STRUCT_DIR / "reference_phases.json").read_text())
    plan = make_plan(cfg, inst, ref)
    wd = Path(cfg["work_dir"])
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "plan.json").write_text(json.dumps(plan))
    print(json.dumps(summarize(plan), indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
