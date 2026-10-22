"""Phase 1b: Pt / Pt-alloy nanoparticles (ASE cluster tools) with strain, vacancies, surface relaxation and faults.

All particles are built on an fcc template. Single-crystal shapes (cuboctahedron, truncated octahedron, sphere)
keep integer fcc coordinates, so ordered phases are made by sublattice assignment (L1_2 / L1_0 / L1_1) and the
tetragonal (L1_0) or rhombohedral (L1_1) distortion of the reference phase is applied as a homogeneous strain.
Five-fold twinned shapes (icosahedron, decahedron) are made only as disordered solid solutions or pure Pt,
because L1_x order cannot be continued through their twin boundaries without extra domain boundaries.
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np
from ase.cluster import Decahedron, Icosahedron, Octahedron
from scipy.spatial import cKDTree

from .paths import STRUCT_DIR

SHAPES = ["cuboctahedron", "truncated_octahedron", "sphere", "icosahedron", "decahedron"]
SINGLE_CRYSTAL = ("cuboctahedron", "truncated_octahedron", "sphere")
DOPANTS = ["Co", "Ni", "Fe", "Cu"]
M_O_DIST = {"Pt": 2.00, "Co": 2.08, "Ni": 2.06, "Fe": 2.02, "Cu": 1.95}
PHYSICAL_FAULTS = ["pt_oxide_shell", "pt_skin", "mixed_oxidation", "low_dopant", "multi_domain"]
METADATA_FAULTS = ["wrong_phase_candidate", "mislabeled_metadata"]
ALL_FAULTS = PHYSICAL_FAULTS + METADATA_FAULTS


# ------------------------------------------------------------------------------------------ shape tables
def _n_atoms(shape, params):
    if shape in ("cuboctahedron", "truncated_octahedron"):
        return len(Octahedron("Cu", params["length"], params["cutoff"], latticeconstant=1.0))
    if shape == "icosahedron":
        n = params["noshells"]
        return (10 * n ** 3 - 15 * n ** 2 + 11 * n - 3) // 3
    if shape == "decahedron":
        return len(Decahedron("Cu", params["p"], params["q"], params["r"], latticeconstant=1.0))
    raise ValueError(shape)


def shape_tables(n_max=26000):
    """(shape, params, N) for every discrete cluster up to n_max atoms (cached)."""
    f = STRUCT_DIR / "shape_tables.json"
    if f.exists():
        return json.loads(f.read_text())
    tab = {s: [] for s in SHAPES if s != "sphere"}
    for c in range(1, 30):
        L = 2 * c + 1
        n = _n_atoms("cuboctahedron", dict(length=L, cutoff=c))
        if n > n_max:
            break
        tab["cuboctahedron"].append(dict(params=dict(length=L, cutoff=c), N=n))
    for L in range(4, 60):
        for c in range(1, (L - 1) // 2):
            if not (0.2 <= c / L <= 0.42) or 2 * c + 1 == L:
                continue
            # cheap upper bound: the full octahedron has (2L^3 + L)/3 atoms
            if (2 * L ** 3 + L) / 3 * 0.45 > n_max:
                continue
            n = _n_atoms("truncated_octahedron", dict(length=L, cutoff=c))
            if n <= n_max:
                tab["truncated_octahedron"].append(dict(params=dict(length=L, cutoff=c), N=n))
    for ns in range(2, 30):
        n = _n_atoms("icosahedron", dict(noshells=ns))
        if n > n_max:
            break
        tab["icosahedron"].append(dict(params=dict(noshells=ns), N=n))
    for p in range(2, 22):
        for q in range(1, 22):
            for r in range(0, 4):
                if r > p - 1:
                    continue
                if p * p * q * 1.5 > n_max * 1.5:
                    continue
                try:
                    n = _n_atoms("decahedron", dict(p=p, q=q, r=r))
                except Exception:
                    continue
                if 13 <= n <= n_max:
                    # keep compact (not needle/disc-like) decahedra: elongation q/p in a sensible range
                    if 0.25 <= q / p <= 2.0:
                        tab["decahedron"].append(dict(params=dict(p=p, q=q, r=r), N=n))
    f.write_text(json.dumps(tab))
    return tab


def eq_diameter_nm(n_atoms, a):
    """Equivalent-sphere diameter from atom count and the fcc atomic volume a^3/4."""
    return (6.0 * n_atoms * a ** 3 / 4.0 / math.pi) ** (1 / 3) / 10.0


def n_for_diameter(d_nm, a):
    return math.pi * (10.0 * d_nm) ** 3 / 6.0 / (a ** 3 / 4.0)


def choose_shape_params(shape, d_nm, a, rng, tables):
    """Pick the discrete cluster whose size is closest to the target (random among near-ties)."""
    nt = n_for_diameter(d_nm, a)
    entries = tables[shape]
    ns = np.array([e["N"] for e in entries], float)
    rel = np.abs(np.log(ns / nt))
    best = rel.min()
    near = np.flatnonzero(rel <= best + 0.08)
    e = entries[int(rng.choice(near))]
    return e["params"]


# --------------------------------------------------------------------------------------- fcc templates
def _octa_int(length, cutoff):
    at = Octahedron("Cu", length, cutoff, latticeconstant=2.0)  # lattice constant 2 -> integer coords
    u = at.get_positions()
    off = u[0] - np.round(u[0])
    return np.round(u - off).astype(np.int64)


def _sphere_int(radius_half_units, rng):
    """Integer fcc coordinates within a sphere around a random centre (half-lattice units)."""
    R = radius_half_units
    m = int(math.ceil(R)) + 2
    g = np.arange(-m, m + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    u = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    u = u[(u.sum(1) % 2) == 0]
    centre = rng.uniform(0, 1, 3)  # random cut position inside the cell
    keep = np.linalg.norm(u - centre, axis=1) <= R
    return u[keep]


@dataclass
class Particle:
    positions: np.ndarray                 # (N, 3) Angstrom, centred
    species: np.ndarray                   # (N,) element symbols (numpy str)
    meta: dict = field(default_factory=dict)


def sublattice_species(u, proto, M, variant, rng):
    """Species for integer fcc coords u under an ordered prototype and variant."""
    n = len(u)
    sp = np.array(["Pt"] * n, dtype="<U2")
    par = np.mod(u, 2)
    if proto == "L1_2":
        # the four simple-cubic sublattices of fcc are the four parity patterns; variant picks the M sublattice
        pats = np.unique(par, axis=0)
        sel = np.all(par == pats[variant % len(pats)], axis=1)
        sp[sel] = M
    elif proto == "L1_0":
        axis, flip = variant % 3, (variant // 3) % 2
        sel = (np.mod(u[:, axis] - u[0, axis], 2) == 0) ^ bool(flip)
        sp[sel] = M
    elif proto == "L1_1":
        signs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])[variant % 4]
        flip = (variant // 4) % 2
        w = (u @ signs - u[0] @ signs) // 2
        sel = (np.mod(w, 2) == 0) ^ bool(flip)
        sp[sel] = M
    else:
        raise ValueError(proto)
    return sp


def distortion_matrix(proto, params, variant):
    """3x3 matrix mapping half-unit integer fcc coords to Angstrom for the ordered phase (before strain)."""
    a_mean = params["a_mean"]
    if proto == "L1_0":
        axis = variant % 3
        d = np.full(3, params["a"] / 2.0)
        d[axis] = params["c"] / 2.0
        return np.diag(d)
    if proto == "L1_1":
        signs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])[variant % 4]
        nvec = signs / np.linalg.norm(signs)
        P = np.outer(nvec, nvec)
        F = params["s_perp"] * (np.eye(3) - P) + params["s_par"] * P
        return F * a_mean / 2.0
    return np.eye(3) * params["a"] / 2.0


# --------------------------------------------------------------------------------------------- builder
def build_particle(spec, ref_params, tables, rng):
    """Build one particle.

    spec: dict(composition, M, x_M, phase ('A1'|'L1_2'|'L1_0'|'L1_1'), shape, d_target_nm, strain,
               vacancy_fraction, surface_relax, static_sigma, fault (str|None), seed)
    ref_params: pseudo-cubic parameters of the phase: dict(a, c, alpha, a_mean[, s_perp, s_par]); for A1 alloys a is
                the Vegard value.
    """
    shape, phase, M = spec["shape"], spec["phase"], spec["M"]
    fault = spec.get("fault")
    a_mean = ref_params["a_mean"]
    eps = spec["strain"]
    meta = dict(shape=shape, phase=phase)
    variant = int(rng.integers(0, 8))
    u = None
    if shape == "sphere":
        R_half = (10.0 * spec["d_target_nm"] / 2.0) / (a_mean / 2.0)
        u = _sphere_int(R_half, rng)
        meta["shape_params"] = dict(radius_A=round(10.0 * spec["d_target_nm"] / 2.0, 3))
    elif shape in ("cuboctahedron", "truncated_octahedron"):
        prm = spec.get("shape_params") or choose_shape_params(shape, spec["d_target_nm"], a_mean, rng, tables)
        u = _octa_int(prm["length"], prm["cutoff"])
        meta["shape_params"] = prm
    elif shape == "icosahedron":
        prm = spec.get("shape_params") or choose_shape_params(shape, spec["d_target_nm"], a_mean, rng, tables)
        pos = Icosahedron("Cu", prm["noshells"], latticeconstant=a_mean).get_positions()
        meta["shape_params"] = prm
    elif shape == "decahedron":
        prm = spec.get("shape_params") or choose_shape_params(shape, spec["d_target_nm"], a_mean, rng, tables)
        pos = Decahedron("Cu", prm["p"], prm["q"], prm["r"], latticeconstant=a_mean).get_positions()
        meta["shape_params"] = prm
    else:
        raise ValueError(shape)

    twin_plane = None
    if u is not None:
        if fault == "multi_domain" and phase == "A1":
            u, twin_plane = _twin_int(u, rng)
        # species
        if phase == "A1":
            sp = np.array(["Pt"] * len(u), dtype="<U2")
            if M is not None:
                k = int(round(spec["x_M"] * len(u)))
                sp[rng.choice(len(u), size=k, replace=False)] = M
            pos = None
        else:
            # ordering variant (sublattice origin / ordering axis): in small particles the variants give different
            # stoichiometries; use the one closest to the nominal composition (ties broken at random)
            n_var = {"L1_2": 4, "L1_0": 6, "L1_1": 8}[phase]
            xs = np.array([np.mean(sublattice_species(u, phase, M, v, rng) == M) for v in range(n_var)])
            dev_x = np.abs(xs - spec["x_M"])
            cands = np.flatnonzero(dev_x <= dev_x.min() + 1e-9)
            variant = int(rng.choice(cands))
            sp = sublattice_species(u, phase, M, variant, rng)
            if fault == "multi_domain":
                # second ordering variant in one half (split by a random plane through the centre)
                nrm = rng.normal(size=3)
                nrm /= np.linalg.norm(nrm)
                c = u.mean(0)
                half = (u - c) @ nrm > 0
                v2 = (variant + int(rng.integers(1, 3 if phase == "L1_0" else 4))) % (3 if phase == "L1_0" else 4)
                sp2 = sublattice_species(u, phase, M, v2, rng)
                sp[half] = sp2[half]
                meta["domain_variants"] = [int(variant), int(v2)]
        D = distortion_matrix(phase, ref_params, variant)
        if twin_plane is not None:
            pos = u.astype(float) @ D.T  # twin built in integer coords (mirror is exact for {111})
        else:
            pos = u.astype(float) @ D.T
    else:
        sp = np.array(["Pt"] * len(pos), dtype="<U2")
        if M is not None:
            k = int(round(spec["x_M"] * len(pos)))
            sp[rng.choice(len(pos), size=k, replace=False)] = M

    pos = pos - pos.mean(0)
    pos = pos * (1.0 + eps)
    d_nn = a_mean * (1.0 + eps) / math.sqrt(2.0)

    # surface atoms by coordination, determined before vacancies so that atoms next to a vacancy stay "bulk"
    tree = cKDTree(pos)
    cn = tree.query_ball_point(pos, 1.2 * d_nn, return_length=True) - 1
    surf = cn < 12

    # vacancies
    nv = int(round(spec["vacancy_fraction"] * len(pos)))
    if nv > 0:
        keep = np.ones(len(pos), bool)
        keep[rng.choice(len(pos), size=nv, replace=False)] = False
        pos, sp, surf = pos[keep], sp[keep], surf[keep]

    # faults that rearrange species
    ox = {"Pt": {"0": 1.0}}
    if M is not None:
        ox[M] = {"0": 1.0}
    if fault == "pt_skin" and M is not None:
        s_idx = np.flatnonzero(surf & (sp == M))
        core_pt = np.flatnonzero(~surf & (sp == "Pt"))
        n_sw = min(len(s_idx), len(core_pt))
        if n_sw:
            sel_core = rng.choice(core_pt, size=n_sw, replace=False)
            sp[s_idx[:n_sw]], sp[sel_core] = "Pt", M
        meta["skin_swaps"] = int(n_sw)
    if fault == "low_dopant":
        choices = [d for d in DOPANTS if d != M]
        D_el = str(rng.choice(choices))
        frac = float(rng.uniform(0.01, 0.04))
        k = max(1, int(round(frac * len(pos))))
        sp[rng.choice(len(pos), size=k, replace=False)] = D_el
        meta["dopant"] = dict(element=D_el, fraction=k / len(pos))
        ox[D_el] = {"0": 1.0}

    # surface relaxation and static disorder
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    rhat = pos / np.maximum(r, 1e-9)
    pos = pos + surf[:, None] * spec["surface_relax"] * d_nn * rhat
    sig = np.where(surf, 2.0, 1.0)[:, None] * spec["static_sigma"]
    pos = pos + rng.normal(size=pos.shape) * sig

    # oxygen-containing faults (O placed radially outside surface atoms)
    o_pos = []
    if fault == "pt_oxide_shell":
        cov = float(rng.uniform(0.6, 1.0))
        idx = np.flatnonzero(surf & (sp == "Pt"))
        idx = idx[rng.random(len(idx)) < cov]
        extra = float(rng.uniform(0.05, 0.10))
        pos[idx] += rng.normal(size=(len(idx), 3)) * extra
        rh = pos[idx] / np.linalg.norm(pos[idx], axis=1, keepdims=True)
        o_pos.append(pos[idx] + rh * (M_O_DIST["Pt"] + rng.normal(scale=0.03, size=(len(idx), 1))))
        n_pt = int((sp == "Pt").sum())
        f2 = len(idx) / max(n_pt, 1)
        ox["Pt"] = {"0": 1.0 - f2, "2": f2}
        meta["oxide"] = dict(coverage=cov, n_O=int(len(idx)), surface_disorder_A=extra)
    if fault == "mixed_oxidation":
        target = M if M is not None else "Pt"
        idx = np.flatnonzero(surf & (sp == target))
        f = float(rng.uniform(0.15, 0.5) if M is not None else rng.uniform(0.1, 0.4))
        idx = idx[rng.random(len(idx)) < f]
        rh = pos[idx] / np.linalg.norm(pos[idx], axis=1, keepdims=True)
        o_pos.append(pos[idx] + rh * (M_O_DIST[target] + rng.normal(scale=0.03, size=(len(idx), 1))))
        n_t = int((sp == target).sum())
        f2 = len(idx) / max(n_t, 1)
        ox[target] = {"0": 1.0 - f2, "2": f2}
        meta["oxidized"] = dict(element=target, fraction=f2, n_O=int(len(idx)))
    if o_pos:
        o = np.concatenate(o_pos)
        pos = np.concatenate([pos, o])
        sp = np.concatenate([sp, np.array(["O"] * len(o), dtype="<U2")])
        surf = np.concatenate([surf, np.ones(len(o), bool)])

    metal = sp != "O"
    n_metal = int(metal.sum())
    rg = float(np.sqrt(((pos[metal] - pos[metal].mean(0)) ** 2).sum(1).mean()))
    meta.update(
        n_atoms=int(len(pos)), n_metal=n_metal, n_surface=int(surf[metal].sum()),
        diameter_nm=eq_diameter_nm(n_metal, a_mean * (1 + eps)),
        diameter_rg_nm=2.0 * math.sqrt(5.0 / 3.0) * rg / 10.0,
        ordering_variant=int(variant) if phase != "A1" else None,
        twin_plane=twin_plane, oxidation_states=ox,
        composition_true={str(el): float((sp == el).sum() / len(sp)) for el in np.unique(sp)},
        lattice_parameter=a_mean * (1 + eps), lattice_a=ref_params["a"] * (1 + eps),
        lattice_c=ref_params["c"] * (1 + eps), lattice_alpha=ref_params.get("alpha", 90.0),
    )
    return Particle(pos.astype(np.float32), sp, meta)


def _twin_int(u, rng):
    """Mirror half of an fcc particle across a {111} atomic plane near its centre (twin boundary)."""
    signs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])[int(rng.integers(0, 4))]
    w = (u @ signs)  # = 2 * plane index (+ const parity), planes spaced by 2 in these units
    w0 = int(np.median(w))
    w0 -= (w0 - int(w[0])) % 2
    keep = u[w >= w0]
    src = u[w > w0]
    # reflection across plane signs.x = w0: x' = x - 2 (signs.x - w0) signs / |signs|^2 ; |signs|^2 = 3
    refl = src - (2 * ((src @ signs) - w0))[:, None] * signs[None, :] / 3.0
    out = np.concatenate([keep.astype(float), refl])
    # mirrored points are on the twin lattice (thirds of half-units): keep as floats
    return out, dict(normal=signs.tolist(), plane=int(w0))


# ---------------------------------------------------------------------------------- coordination labels
def coordination_shells(pos, sp, a_latt, u_iso, delta2, corr_floor, absorbers=None, rmax=None):
    """Per absorber element: (neighbour element, shell, N, R, sigma2) for the first 4 metal shells and the O shell.

    Shell boundaries are the midpoints between ideal fcc neighbour distances a*sqrt(n/2), n = 1..4 (+ cut at 5).
    sigma2 = static variance of the distances + thermal (U_a + U_b) * max(floor, 1 - delta2 / R^2).
    """
    ideal = a_latt * np.sqrt(np.arange(1, 6) / 2.0)
    edges = np.concatenate([[0.0], 0.5 * (ideal[:-1] + ideal[1:])])  # shell n: edges[n-1] .. edges[n]
    rmax = edges[-1] if rmax is None else rmax
    tree = cKDTree(pos)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    pi, pj = np.concatenate([pairs[:, 0], pairs[:, 1]]), np.concatenate([pairs[:, 1], pairs[:, 0]])
    pd = np.linalg.norm(pos[pi] - pos[pj], axis=1)
    els = sorted(set(sp.tolist()))
    absorbers = [e for e in els if e != "O"] if absorbers is None else absorbers
    out = []
    for X in absorbers:
        idx = np.flatnonzero(sp == X)
        if len(idx) == 0:
            continue
        mx = sp[pi] == X
        d, j = pd[mx], pj[mx]
        for Y in els:
            m = sp[j] == Y
            dd = d[m]
            if Y == "O":
                sel = (dd > 1.6) & (dd < 2.5)
                shells = [("O", 1, sel)]
            else:
                shells = [(Y, s + 1, (dd >= max(edges[s], 1.6)) & (dd < edges[s + 1])) for s in range(4)]
            for (Yn, s, sel) in shells:
                if not sel.any():
                    continue
                R = float(dd[sel].mean())
                corr = max(corr_floor, 1.0 - delta2 / R ** 2)
                s2 = float(dd[sel].var() + (u_iso[X] + u_iso[Yn]) * corr)
                out.append(dict(absorber=X, neighbor=Yn, shell=s, N=float(sel.sum() / len(idx)), R=R, sigma2=s2))
    return out
