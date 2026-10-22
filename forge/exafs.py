"""EXAFS shortcut: chi(k) from absorber-centred partial distance distributions and FEFF reference tables.

chi_X(k) = S0^2 [ sum_Y sum_bins n_XY(r_b) chi_path(k; ref_XY(s), dR = r_b - reff_s, sigma2_th(r_b))     (SS, r <= rmax_ss)
                 + sum_{MS paths p of the bulk FEFF run} c_p degen_p chi_path(k; ref_p, R = alpha reff_p, sigma2_p) ]
ref_XY(s) is the FEFF single-scattering amplitude / phase / Re(p) / mean free path of the nearest reference shell s,
taken from the bulk run of the same composition and phase (pair run as fallback). MS paths are rescaled with the
lattice ratio alpha and by the
ratio of motif counts n_particle / n_bulk (forge.motifs: species and shells of the 1-2 distinct path atoms and their
mutual distance); paths with 3 distinct atoms (~5% of the MS amplitude) use (N_s / N_s,bulk)^m, s = shell of the
farthest path atom, m = number of distinct path atoms in it.
sigma2 uses the same correlated thermal model as the Debye calculations plus, for MS, (nleg / 4) x the static
first-shell variance. The chi_path formula is larch's (see forge.feff.path_chi).
"""
import math
import pickle

import numpy as np

from .feff import path_chi
from .motifs import SPECIES, finalize, merge, motif_reff, motif_stats, path_motif
from .paths import FEFF_DIR

FCC_CN = np.array([12.0, 6.0, 24.0, 12.0, 24.0])
N_SHELLS = 5
METALS = ("Pt", "Co", "Ni", "Fe", "Cu")


def path_chi_torch(k, amp, pha, rep, lam, reff, dr, sigma2, degen):
    """Torch (complex128) version of forge.feff.path_chi (larch path equation, e0 = ei = 0)."""
    import torch
    p = torch.complex(rep, 1.0 / lam)
    pp = p * p
    arg = -2 * reff * p.imag - 2 * pp * sigma2 + 1j * (2 * k * reff + pha + 2 * p * (dr - 2 * sigma2 / reff))
    return (degen * amp * torch.exp(arg) / (k * (reff + dr) ** 2)).imag


def shell_edges(a):
    ideal = a * np.sqrt(np.arange(1, N_SHELLS + 2) / 2.0)
    return np.concatenate([[1.6], 0.5 * (ideal[:-1] + ideal[1:])])  # shell s (1-based) in [e[s-1], e[s])


def path_shell_multiplicity(coords, a):
    """(farthest shell index (0-based), number of distinct path atoms in that shell) for an MS path geometry."""
    c = np.asarray(coords, float)
    rel = c[1:] - c[0]
    d = np.linalg.norm(rel, axis=1)
    uniq = {}
    for v, dist in zip(np.round(rel, 2), d):
        if dist > 0.5:
            uniq[tuple(v)] = dist
    ideal = a * np.sqrt(np.arange(1, N_SHELLS + 1) / 2.0)
    shells = [int(np.argmin(np.abs(ideal - dist))) for dist in uniq.values()]
    s_far = max(shells)
    return s_far, shells.count(s_far)


CACHE_VERSION = "exafs-model-v4"


class ExafsModel:
    def __init__(self, refs=None, rmax_ss=6.0, rmax_ms=6.0, device=None, cache=True):
        """device: torch device for the path sums (None = numpy). The processed template tables are cached in
        artifacts/data/work/feff/exafs_model_cache.pkl, keyed by the SHA-256 of refs.pkl."""
        self.device = device
        self._torch = {}
        if refs is None and cache:
            from .paths import sha256_file
            key = f"{CACHE_VERSION}:{sha256_file(FEFF_DIR / 'refs.pkl')}:{rmax_ss}:{rmax_ms}"
            cf = FEFF_DIR / "exafs_model_cache.pkl"
            if cf.exists():
                with open(cf, "rb") as fh:
                    st = pickle.load(fh)
                if st.get("_key") == key:
                    self.__dict__.update({k: v for k, v in st.items() if k != "_key"})
                    self.device, self._torch = device, {}
                    return
            self._build(None, rmax_ss, rmax_ms)
            st = {k: v for k, v in self.__dict__.items() if k not in ("device", "_torch", "refs")}
            st["_key"] = key
            tmp = cf.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(st, fh)
            import os
            os.replace(tmp, cf)
            return
        self._build(refs, rmax_ss, rmax_ms)

    def _build(self, refs, rmax_ss, rmax_ms):
        if refs is None:
            with open(FEFF_DIR / "refs.pkl", "rb") as fh:
                refs = pickle.load(fh)
        self.refs = refs
        self.k = np.asarray(refs["k"], float)
        self.ks = np.where(self.k == 0, 1e-6, self.k)
        self.rmax_ss, self.rmax_ms = rmax_ss, rmax_ms
        self.pairs = refs["pairs"]
        # single-scattering references from the bulk run of the sample's own composition / phase (alloy potentials,
        # alloy distances); shells of one scatterer type are merged across disorder configurations by distance
        self.ss_env = {}
        for key, e in refs["envs"].items():
            by_y = {}
            for p in e["paths"]:
                if p["nleg"] != 2:
                    continue
                by_y.setdefault(p["atoms"][1], []).append(p)
            tab = {}
            for Y, ps in by_y.items():
                ps = sorted(ps, key=lambda p: p["reff"])
                shells = []
                for p in ps:
                    if shells and abs(p["reff"] - shells[-1][-1]["reff"]) < 0.02:
                        shells[-1].append(p)
                    else:
                        shells.append([p])
                def mean_phase(g):
                    # FEFF phases carry arbitrary 2*pi offsets per path: align to the first before averaging
                    p0 = g[0]["pha"]
                    return np.mean([p["pha"] - 2 * np.pi * np.round(np.median(p["pha"] - p0) / (2 * np.pi)) for p in g],
                                   axis=0)

                tab[Y] = dict(reff=np.array([np.mean([p["reff"] for p in g]) for g in shells]),
                              pha=np.stack([mean_phase(g) for g in shells]),
                              **{q: np.stack([np.mean([p[q] for p in g], axis=0) for g in shells])
                                 for q in ("amp", "rep", "lam")})
            self.ss_env[key] = tab
        self.ms = {}
        for key, e in refs["envs"].items():
            a_env = refs["env_a"][key]
            n_cfg, n_fcc = e["n_cfg"], e.get("n_fcc", e["n_cfg"])
            ps = []
            for p in e["paths"]:
                if p["nleg"] < 3:
                    continue
                if path_motif(p["coords"], p["atoms"], a_env)[0] is None:
                    # paths with 3 distinct scatterers keep the occupancy-rule fallback: fcc runs only, degeneracy
                    # renormalized to those runs (twin geometries are not counted for them)
                    if p.get("twin", False):
                        continue
                    p = dict(p, degen=p["degen"] * n_cfg / max(n_fcc, 1))
                ps.append(p)
            if not ps:
                continue
            sm = [path_shell_multiplicity(p["coords"], a_env) for p in ps]
            # motif of each path and its mean count around the absorber in the bulk reference clusters
            pm = [path_motif(p["coords"], p["atoms"], a_env) for p in ps]
            st = [motif_stats(c["pos"], np.array([SPECIES.index(x) for x in c["species"]]), [c["absorber"]], a_env)
                  for c in e.get("clusters", [])]
            f_env = finalize(merge(st)) if st else None
            n_env, l_env, v_env = [], [], []
            for kk, legs in pm:
                res = motif_reff(kk, legs, f_env) if (kk is not None and f_env is not None) else None
                n_env.append(res[0] if res else 0.0)
                l_env.append(res[1] if res else np.nan)
                v_env.append(res[2] if res else 0.0)
            mt = np.array([0 if q[0] is None else int(q[0][0]) for q in pm])
            c1 = np.array([0 if q[0] is None else q[0][1] for q in pm])
            c2 = np.array([0 if (q[0] is None or q[0][0] == "1") else q[0][2] for q in pm])
            sab = np.array([0 if (q[0] is None or q[0][0] == "1") else q[0][3] for q in pm])
            lg = np.array([[0, 0, 0] if q[1] is None else [q[1]["xa"], q[1]["xb"], q[1]["ab"]] for q in pm], float)
            self.ms[key] = dict(
                motif=[q[0] for q in pm], legs=[q[1] for q in pm], n_env=np.array(n_env), l_env=np.array(l_env),
                v_env=np.array(v_env), mtype=mt, c1=c1, c2=c2, sab=sab, lg=lg,
                s_far=np.array([v[0] for v in sm]), mult=np.array([v[1] for v in sm]),
                reff=np.array([p["reff"] for p in ps]), degen=np.array([p["degen"] for p in ps]),
                nleg=np.array([p["nleg"] for p in ps]), r_far=np.array([p["r_far"] for p in ps]),
                scat=[[a for a in p["atoms"][1:]] for p in ps],
                amp=np.stack([p["amp"] for p in ps]), pha=np.stack([p["pha"] for p in ps]),
                rep=np.stack([p["rep"] for p in ps]), lam=np.stack([p["lam"] for p in ps]))
        self.env_a = refs.get("env_a", {})

    def _nearest_ref(self, X, Y, r, env_key):
        """FEFF single-scattering reference of the nearest shell (environment run first, pair run as fallback).

        Returns arrays per bin plus the reference distance; the bin distance enters through deltaR = r - reff.
        """
        p = self.ss_env.get(env_key, {}).get(Y)
        if p is None:
            p = self.pairs[f"{X}-{Y}"]
        R = p["reff"]
        i = np.argmin(np.abs(r[:, None] - R[None, :]), axis=1)
        out = {q: p[q][i] for q in ("amp", "pha", "rep", "lam")}
        out["reff"] = R[i]
        return out

    def shell_stats(self, partials, r, a):
        """Total metal CN per fcc shell (1-5) and static variance of first-shell distances."""
        e = shell_edges(a)
        metal = sum(w for Y, w in partials.items() if Y in METALS)
        cn = np.array([metal[(r >= e[s]) & (r < e[s + 1])].sum() for s in range(N_SHELLS)])
        m1 = (r >= e[0]) & (r < e[1])
        w1 = metal[m1]
        if w1.sum() > 0:
            mu = (w1 * r[m1]).sum() / w1.sum()
            var1 = (w1 * (r[m1] - mu) ** 2).sum() / w1.sum()
        else:
            var1 = 0.0
        return cn, var1

    def _sum_paths(self, amp, pha, rep, lam, reff, dr, s2, deg, chunk=4096):
        """sum over paths of the larch path equation on self.k (numpy, or torch complex128 on self.device)."""
        if self.device is None:
            return path_chi(self.ks[None, :], amp, pha, rep, lam, reff[:, None], dr[:, None], s2[:, None],
                            deg[:, None], 1.0).sum(0)
        import torch
        dev = self.device
        k = torch.as_tensor(self.ks, dtype=torch.float64, device=dev)[None, :]
        out = torch.zeros(len(self.ks), dtype=torch.float64, device=dev)
        for i in range(0, len(reff), chunk):
            t = lambda a: torch.as_tensor(np.ascontiguousarray(a[i:i + chunk]), dtype=torch.float64, device=dev)
            out += path_chi_torch(k, t(amp), t(pha), t(rep), t(lam), t(reff)[:, None], t(dr)[:, None], t(s2)[:, None],
                                  t(deg)[:, None]).sum(0)
        return out.cpu().numpy()

    def chi(self, X, partials, r, env_key, a_sample, U, delta2, floor, s02, include_ms=True, motifs=None):
        """chi(k) on self.k for absorber element X.

        partials: dict Y -> n_XY on the bin grid r (numbers of Y neighbours per X absorber per bin).
        U: dict element -> per-direction mean-square displacement (A^2).
        motifs: finalized forge.motifs statistics over the same absorbers. Each MS path then gets the particle's motif
                count (degeneracy x n_particle / n_bulk), mean path length and static path-length variance. Without it
                (or for 3-atom motifs) the lattice ratio and the shell-occupancy rule are used instead.
        """
        chi = np.zeros_like(self.ks)
        for Y, w in partials.items():
            sel = (w > 1e-9) & (r >= (1.5 if Y == "O" else 1.8)) & (r <= self.rmax_ss)
            if not sel.any():
                continue
            rb, wb = r[sel], w[sel]
            ref = self._nearest_ref(X, Y, rb, env_key)
            s2 = (U[X] + U[Y]) * np.maximum(floor, 1.0 - delta2 / rb ** 2)
            chi += self._sum_paths(ref["amp"], ref["pha"], ref["rep"], ref["lam"], ref["reff"], rb - ref["reff"], s2, wb)
        if include_ms and env_key in self.ms:
            ms = self.ms[env_key]
            alpha = a_sample / self.env_a[env_key]
            cn, var1 = self.shell_stats(partials, r, a_sample)
            n = len(ms["reff"])
            u_sc = np.array([np.mean([U.get(a, U[X]) for a in sc]) for sc in ms["scat"]])
            # defaults (3-atom motifs or no motif statistics): lattice-scaled length, shell-occupancy count rule
            reff_s = ms["reff"] * alpha
            coord = (cn[ms["s_far"]] / FCC_CN[ms["s_far"]]) ** ms["mult"]
            var_s = ms["nleg"] / 4.0 * var1
            if motifs is not None:
                f = motifs
                ok = (ms["mtype"] > 0) & (ms["n_env"] > 0)
                one, two = ok & (ms["mtype"] == 1), ok & (ms["mtype"] == 2)
                c1, c2, sab, lg = ms["c1"], ms["c2"], ms["sab"], ms["lg"]
                n_s = np.where(one, f["n1"][c1], np.where(two, f["n2"][c1, c2, sab], 0.0))
                d1, v1 = np.nan_to_num(f["d1"][c1]), np.clip(np.nan_to_num(f["v1"][c1]), 0, None)
                g = lambda q: np.nan_to_num(f[q][c1, c2, sab])
                gv = lambda q: np.clip(np.nan_to_num(f[q + "_var"][c1, c2, sab]), 0, None)
                gc = lambda q: np.nan_to_num(f[q + "_cov"][c1, c2, sab]) if (q + "_cov") in f else 0.0
                mean2 = 0.5 * (lg[:, 0] * g("xa") + lg[:, 1] * g("xb") + lg[:, 2] * g("ab"))
                # variance of the path length with the (strong) correlations between legs of the same atom pair
                var2 = 0.25 * np.clip(lg[:, 0] ** 2 * gv("xa") + lg[:, 1] ** 2 * gv("xb") + lg[:, 2] ** 2 * gv("ab")
                                      + 2 * (lg[:, 0] * lg[:, 1] * gc("xaxb") + lg[:, 0] * lg[:, 2] * gc("xaab")
                                             + lg[:, 1] * lg[:, 2] * gc("xbab")), 0, None)
                mean = np.where(one, 0.5 * lg[:, 0] * d1, mean2)
                var = np.where(one, 0.25 * lg[:, 0] ** 2 * v1, var2)
                present = ok & (n_s > 0)
                # count ratio; path length scaled by the motif's particle / bulk mean length (keeps geometries that
                # share a motif class, e.g. in-plane and out-of-plane chains in L1_0, distinct); static variance in
                # excess of the bulk motif's own spread. Motifs absent from the particle contribute nothing.
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.where(present, mean / ms["l_env"], 1.0)
                    coord = np.where(present, n_s / np.where(ok, ms["n_env"], 1.0), np.where(ok, 0.0, coord))
                reff_s = np.where(present, ms["reff"] * ratio, reff_s)
                var_s = np.where(present, np.clip(var - ms["v_env"] * ratio ** 2, 0, None), var_s)
            s2 = (U[X] + u_sc) * np.maximum(floor, 1.0 - delta2 / reff_s ** 2) + var_s
            idx = np.flatnonzero((reff_s <= self.rmax_ms) & (coord > 0))
            if len(idx):
                chi += self._sum_paths(ms["amp"][idx], ms["pha"][idx], ms["rep"][idx], ms["lam"][idx], ms["reff"][idx],
                                       (reff_s - ms["reff"])[idx], s2[idx], (ms["degen"] * coord)[idx])
        chi = s02 * chi
        chi[self.k == 0] = 0.0
        return chi


def env_key(spec, X):
    """MS / SS template set for a sample: composition, phase and absorber; icosahedra use their own templates."""
    if spec["shape"] == "icosahedron":
        return f"{spec['composition']}|A1ico|{X}"
    return f"{spec['composition']}|{spec['phase']}|{X}"


def mu_from_chi(e_rel, k, chi, p):
    """Normalized mu(E) (edge step 1) from chi(k) with an arctan edge, Lorentzian white line and linear pre-edge.

    p: dict(edge_width, white_line, wl_pos, wl_width, shift, pre_slope, post_slope)
    """
    E = e_rel - p["shift"]
    step = 0.5 + np.arctan(E / (p["edge_width"] / 2)) / math.pi
    wl = p["white_line"] / (1 + ((E - p["wl_pos"]) / (p["wl_width"] / 2)) ** 2)
    kE = np.sqrt(np.clip(E, 0, None) / 3.80998)
    chiE = np.interp(kE, k, chi) * (1 - np.exp(-(kE / 1.2) ** 2)) * (E > 0)
    post = 1.0 - p["post_slope"] * np.clip(E, 0, None) / 1000.0
    return p["pre_slope"] * e_rel / 1000.0 + step * post * (1 + chiE) + wl * step
