"""Multiple-scattering motif statistics around absorbers.

A motif is the arrangement of the distinct scattering atoms of a path relative to the absorber X:
  1-atom motif  (class(A))                                  class = shell * N_SPECIES + species
  2-atom motif  (class(A), class(B), shell(|AB|))           with class(A) <= class(B)
Shells are the fcc neighbour shells plus the hcp-stacking distances found at twin boundaries (SHELL_R1), with
windows bounded by the midpoints of the ideal distances (first window starts at 1.6 A). Averaged over the given absorbers, motif_stats returns
  n1[c], n2[c1, c2, s]           mean numbers of atoms / ordered atom pairs (A != B) per absorber
  d1[c], v1[c]                   mean and variance of |XA| for 1-atom motifs
  xa, xb, ab (+ variances and covariances)  mean |XA|, |XB|, |AB| over the matched pairs of each 2-atom motif
so that each FEFF multiple-scattering path of a bulk template can be given the particle's own count, mean path
length and static path-length variance. Implemented with batched matrix products (torch, CPU or GPU).
"""
import numpy as np
import torch

SPECIES = ["Pt", "Co", "Ni", "Fe", "Cu", "O"]
N_SPECIES = len(SPECIES)
# neighbour distances in units of the fcc nearest-neighbour distance r1 = a / sqrt(2): fcc shells sqrt(n), n = 1..6,
# plus the hcp-stacking distances sqrt(8/3) and sqrt(11/3) that occur across {111} twin boundaries (decahedra,
# icosahedra, twinned particles); the last value only closes the outermost window
SHELL_R1 = np.array([1.0, np.sqrt(2), np.sqrt(8 / 3), np.sqrt(3), np.sqrt(11 / 3), 2.0, np.sqrt(5), np.sqrt(6), np.sqrt(7)])
N_SHELL = len(SHELL_R1) - 1
N_CLASS = N_SHELL * N_SPECIES


def shell_edges(a):
    ideal = a / np.sqrt(2) * SHELL_R1
    return np.concatenate([[1.6], 0.5 * (ideal[:-1] + ideal[1:])])  # N_SHELL + 1 edges


def shell_index(d, a):
    """0-based shell index for distances d (numpy), -1 outside the shell windows."""
    e = shell_edges(a)
    i = np.searchsorted(e, d, side="right") - 1
    return np.where((d >= e[0]) & (d < e[-1]), i, -1)


@torch.no_grad()
def motif_stats(pos, codes, absorbers, a, device="cpu", chunk=256):
    """Accumulated (unnormalized by pair counts) motif statistics, averaged over the absorbers.

    For every absorber, all ordered neighbour pairs (A, B), A != B, inside the shell windows are scattered with
    index_add into flat (class(A), class(B), shell(|AB|)) bins, carrying the weights
    1, |XA|, |XA|^2, |XB|, |XB|^2, |AB|, |AB|^2, |XA||XB|, |XA||AB|, |XB||AB|.
    """
    dev = torch.device(device)
    P = torch.as_tensor(np.asarray(pos), dtype=torch.float32, device=dev)
    S = torch.as_tensor(np.asarray(codes), dtype=torch.int64, device=dev)
    e = torch.as_tensor(shell_edges(a), dtype=torch.float32, device=dev)
    rcut = float(e[-1])
    absorbers = torch.as_tensor(np.asarray(absorbers), dtype=torch.int64, device=dev)
    nb2 = N_CLASS * N_CLASS * N_SHELL
    acc2 = torch.zeros(10, nb2, dtype=torch.float64, device=dev)
    acc1 = torch.zeros(3, N_CLASS, dtype=torch.float64, device=dev)
    for s in range(0, len(absorbers), chunk):
        ab_idx = absorbers[s:s + chunk]
        d = torch.cdist(P[ab_idx], P, compute_mode="donot_use_mm_for_euclid_dist")
        valid = (d >= e[0]) & (d < rcut)
        K = int(valid.sum(1).max().item())
        if K == 0:
            continue
        dm = torch.where(valid, d, torch.full_like(d, float("inf")))
        dk, idx = torch.topk(dm, K, dim=1, largest=False)
        ok = torch.isfinite(dk)
        dk = torch.where(ok, dk, torch.zeros_like(dk))
        sh = (torch.bucketize(dk, e, right=True) - 1).clamp(0, N_SHELL - 1)
        cls = sh * N_SPECIES + S[idx]                                                             # (B, K)
        # 1-atom statistics
        okf = ok.reshape(-1)
        c1 = cls.reshape(-1)[okf]
        d1 = dk.reshape(-1)[okf].double()
        acc1.index_add_(1, c1, torch.stack([torch.ones_like(d1), d1, d1 * d1]))
        # 2-atom statistics over ordered pairs
        Pn = P[idx]
        Dab = torch.cdist(Pn, Pn, compute_mode="donot_use_mm_for_euclid_dist")                   # (B, K, K)
        okab = (Dab >= e[0]) & (Dab < rcut) & ok[:, :, None] & ok[:, None, :]
        shab = (torch.bucketize(Dab, e, right=True) - 1).clamp(0, N_SHELL - 1)
        m = okab.reshape(-1)
        ca = cls[:, :, None].expand(-1, -1, K).reshape(-1)[m]
        cb = cls[:, None, :].expand(-1, K, -1).reshape(-1)[m]
        flat = (ca * N_CLASS + cb) * N_SHELL + shab.reshape(-1)[m]
        xa = dk[:, :, None].expand(-1, -1, K).reshape(-1)[m].double()
        xb = dk[:, None, :].expand(-1, K, -1).reshape(-1)[m].double()
        dab = Dab.reshape(-1)[m].double()
        w = torch.stack([torch.ones_like(xa), xa, xa * xa, xb, xb * xb, dab, dab * dab, xa * xb, xa * dab, xb * dab])
        acc2.index_add_(1, flat, w)
        del d, Dab, okab, shab, w, flat, ca, cb, xa, xb, dab
    n = max(len(absorbers), 1)
    a2 = (acc2 / n).cpu().numpy().reshape(10, N_CLASS, N_CLASS, N_SHELL)
    a1 = (acc1 / n).cpu().numpy()
    keys = ["n2", "xa", "xa2", "xb", "xb2", "ab", "ab2", "xaxb", "xaab", "xbab"]
    out = {k: a2[i] for i, k in enumerate(keys)}
    out.update(n1=a1[0], s1=a1[1], q1=a1[2])
    return out


def finalize(st):
    """Means and variances from accumulated sums (safe for empty motifs)."""
    n1 = st["n1"]
    n2 = st["n2"]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = dict(n1=n1, n2=n2,
                 d1=np.where(n1 > 0, st["s1"] / n1, np.nan),
                 v1=np.where(n1 > 0, st["q1"] / n1 - (st["s1"] / n1) ** 2, np.nan))
        for q in ("xa", "xb", "ab"):
            m = np.where(n2 > 0, st[q] / n2, np.nan)
            f[q] = m
            f[q + "_var"] = np.where(n2 > 0, st[q + "2"] / n2 - m ** 2, np.nan)
        # covariances between the leg lengths of the same atom pair (legs of a path are strongly correlated, e.g.
        # |XB| ~ |XA| + |AB| for collinear focusing chains)
        for q, (u, v) in (("xaxb", ("xa", "xb")), ("xaab", ("xa", "ab")), ("xbab", ("xb", "ab"))):
            if q in st:
                f[q + "_cov"] = np.where(n2 > 0, st[q] / n2 - f[u] * f[v], np.nan)
    return f


def merge(stats_list):
    """Average raw (unfinalized) stats over several absorber sets with equal weight per set."""
    return {k: np.mean([s[k] for s in stats_list], axis=0) for k in stats_list[0]}


def path_motif(coords, atoms, a):
    """Motif key and leg-type multiplicities of an FEFF path (absorber first), or (None, None).

    Returns (key, legs) with key ("1", c) or ("2", ca, cb, s_ab) and legs = dict(xa=n, xb=n, ab=n) counting how many
    path legs are X-A, X-B and A-B (A is the atom of class ca).
    """
    c = np.asarray(coords, float)
    rel = c[1:] - c[0]
    uniq, order = {}, []
    for v, el in zip(rel, atoms[1:]):
        if np.linalg.norm(v) > 0.5:
            key = tuple(np.round(v, 2))
            if key not in uniq:
                uniq[key] = (v, el)
            order.append(key)
        else:
            order.append("X")
    items = list(uniq.items())
    seq = ["X"] + order + ["X"]  # absorber -> scatterers -> absorber
    if len(items) == 1:
        (kv, (v, el)), = items
        s = int(shell_index(np.array([np.linalg.norm(v)]), a)[0])
        if s < 0:
            return None, None
        return ("1", s * N_SPECIES + SPECIES.index(el)), dict(xa=len(seq) - 1, xb=0, ab=0)
    if len(items) == 2:
        (ka, (va, ea)), (kb, (vb, eb)) = items
        sa, sb, sab = shell_index(np.array([np.linalg.norm(va), np.linalg.norm(vb), np.linalg.norm(va - vb)]), a)
        if min(sa, sb, sab) < 0:
            return None, None
        ca, cb = sa * N_SPECIES + SPECIES.index(ea), sb * N_SPECIES + SPECIES.index(eb)
        if ca > cb:
            ca, cb, ka, kb = cb, ca, kb, ka
        legs = dict(xa=0, xb=0, ab=0)
        for p, q in zip(seq[:-1], seq[1:]):
            pair = {p, q}
            if pair == {"X", ka}:
                legs["xa"] += 1
            elif pair == {"X", kb}:
                legs["xb"] += 1
            elif pair == {ka, kb}:
                legs["ab"] += 1
        return ("2", int(ca), int(cb), int(sab)), legs
    return None, None


def motif_reff(key, legs, f):
    """(count, mean reff, static variance of reff) of a motif from finalized stats; None if absent."""
    if key[0] == "1":
        c = key[1]
        n = f["n1"][c]
        if not n > 0:
            return None
        return n, 0.5 * legs["xa"] * f["d1"][c], 0.25 * legs["xa"] ** 2 * max(f["v1"][c], 0.0)
    _, ca, cb, s = key
    n = f["n2"][ca, cb, s]
    if not n > 0:
        return None
    a, b, c = legs["xa"], legs["xb"], legs["ab"]
    mean = 0.5 * (a * f["xa"][ca, cb, s] + b * f["xb"][ca, cb, s] + c * f["ab"][ca, cb, s])
    var = a * a * f["xa_var"][ca, cb, s] + b * b * f["xb_var"][ca, cb, s] + c * c * f["ab_var"][ca, cb, s]
    if "xaxb_cov" in f:
        var += 2 * (a * b * f["xaxb_cov"][ca, cb, s] + a * c * f["xaab_cov"][ca, cb, s] + b * c * f["xbab_cov"][ca, cb, s])
    return n, mean, 0.25 * max(var, 0.0)
