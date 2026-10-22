"""Phase 2: GPU simulation of XRD, PDF and EXAFS from one set of element-resolved pair-distance histograms.

Per particle:
  1. Pair histograms h_ab(r) (a <= b, unordered pairs, linear two-node spreading on a dr = 0.004 A grid), on GPU.
  2. Debye scattering, with thermal damping per pair distance (correlated-motion model, same for all techniques):
        I(Q) = sum_a N_a |f_a|^2 + 2 sum_{a<=b} Re(f_a f_b*) sum_k h_ab(r_k) sinc(Q r_k) exp(-Q^2 s2_ab(r_k) / 2)
        s2_ab(r) = (U_a + U_b) * max(floor, 1 - delta2 / r^2)
     f = f0(Q) (Waasmaier-Kirfel) + f' + i f'' (Chantler) at the photon energy.
  3. XRD: Cu K-alpha1/2 on a 2theta grid (polarization, Caglioti pseudo-Voigt resolution) and a synchrotron pattern on a
     Q grid (Gaussian Q resolution) for a sampled wavelength.
  4. PDF: S(Q) - 1 = [I(Q)/N - <|f|^2>] / |<f>|^2 with an extra Qbroad term in the pair width
     (s2 -> (U_a+U_b) * (max(floor, 1 - delta2/r^2) + Qbroad^2 r^2), as in PDFfit), F(Q) = Q (S - 1) on
     [Qmin, Qmax], G(r) = 2/pi int F(Q) sin(Qr) dQ, times exp(-(Qdamp r)^2 / 2).
  5. EXAFS: absorber-centred partial distributions n_XY(r) = h_XY / N_X (x2 for X = Y) feed the FEFF-based shortcut
     (forge.exafs); mu(E) from chi(k) with a simple edge model.
"""
import math

import numpy as np
import torch
import xraydb

HC = 12.398419843320026  # keV * A
DR = 0.004


class Grids:
    def __init__(self, cfg):
        g = cfg["grids"]
        self.tth = np.linspace(g["xrd_cu"]["two_theta_min"], g["xrd_cu"]["two_theta_max"], g["xrd_cu"]["n"])
        self.q_sync = np.linspace(g["xrd_sync"]["q_min"], g["xrd_sync"]["q_max"], g["xrd_sync"]["n"])
        self.r_pdf = g["pdf"]["r_min"] + g["pdf"]["r_step"] * np.arange(g["pdf"]["n"])
        self.q_pdf = np.round(np.arange(0.40, 30.0 + 1e-9, cfg["pdf"]["q_step"]), 4)
        self.k = g["exafs_store"]["k_min"] + g["exafs_store"]["k_step"] * np.arange(g["exafs_store"]["n"])
        x = g["xafs_mu"]
        e1 = np.arange(x["e_rel_min"], x["e_rel_xanes_end"], x["xanes_step"])
        k_lo = math.sqrt(x["e_rel_xanes_end"] / 3.80998)
        k_hi = math.sqrt(x["e_rel_max"] / 3.80998)
        e2 = 3.80998 * np.arange(k_lo, k_hi + 1e-9, 0.05) ** 2
        self.e_rel = np.concatenate([e1, e2])


def form_factors(elements, q, energy_kev=None):
    """Complex form factors f0(Q) + f' + i f'' for each element on the Q grid (numpy, shape (n_el, nQ))."""
    s = np.asarray(q) / (4 * math.pi)
    out = []
    for el in elements:
        f = np.asarray(xraydb.f0(el, s), float).astype(complex)
        if energy_kev is not None:
            e_ev = energy_kev * 1000.0
            f = f + float(xraydb.f1_chantler(el, e_ev)) + 1j * float(xraydb.f2_chantler(el, e_ev))
        out.append(f)
    return np.stack(out)


class GpuSim:
    def __init__(self, cfg, device, exafs_model=None, mem_fraction=None):
        self.cfg = cfg
        self.dev = torch.device(device)
        if mem_fraction is not None and self.dev.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(mem_fraction, self.dev)
        self.g = Grids(cfg)
        self.exafs = exafs_model
        self.sin_pdf = None

    # ------------------------------------------------------------------------------------------ histograms
    @torch.no_grad()
    def histograms(self, pos, types, n_types, rmax=None, chunk=2048):
        """Unordered pair histograms per type pair (a <= b): tensor (n_types, n_types, n_bins), only a <= b filled."""
        x = torch.as_tensor(pos, dtype=torch.float32, device=self.dev)
        t = torch.as_tensor(types, dtype=torch.int64, device=self.dev)
        n = x.shape[0]
        if rmax is None:
            rmax = float(2 * torch.linalg.norm(x - x.mean(0), dim=1).max().item()) + 1.0
        nb = int(math.ceil(rmax / DR)) + 2
        H = torch.zeros(n_types * n_types * nb, dtype=torch.float64, device=self.dev)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            d = torch.cdist(x[s:e], x[s:], compute_mode="donot_use_mm_for_euclid_dist")
            ii = torch.arange(s, e, device=self.dev)[:, None]
            jj = torch.arange(s, n, device=self.dev)[None, :]
            mask = jj > ii
            d = d[mask]
            ta = t[s:e][:, None].expand(-1, n - s)[mask]
            tb = t[s:][None, :].expand(e - s, -1)[mask]
            lo = torch.minimum(ta, tb)
            hi = torch.maximum(ta, tb)
            u = d / DR
            b0 = torch.floor(u)
            w1 = (u - b0).to(torch.float64)
            b0 = b0.to(torch.int64)
            base = (lo * n_types + hi) * nb
            H.index_add_(0, base + b0, 1.0 - w1)
            H.index_add_(0, base + b0 + 1, w1)
            del d, ta, tb, u, b0, w1, base, mask
        return H.view(n_types, n_types, nb), nb

    # ------------------------------------------------------------------------------------------ Debye sums
    def _pair_sums(self, h, r, q, s2_shape, u_ab, qbroad=0.0, rchunk=4096):
        """sum_k h_k sinc(Q r_k) exp(-Q^2 U_ab g(r_k)/2) for one type pair. h, r: (nb,) tensors; q: (nQ,) tensor."""
        nz = h > 0
        h, r = h[nz].to(torch.float32), r[nz]
        g = s2_shape(r)
        if qbroad:
            g = g + (qbroad * r) ** 2
        s2 = u_ab * g
        out = torch.zeros(q.shape[0], dtype=torch.float64, device=self.dev)
        q2 = (q * q)[:, None]
        for s in range(0, r.shape[0], rchunk):
            rr, hh, ss = r[s:s + rchunk], h[s:s + rchunk], s2[s:s + rchunk]
            qr = q[:, None] * rr[None, :]
            m = torch.sinc(qr / math.pi) * torch.exp(-0.5 * q2 * ss[None, :])
            out += (m @ hh).to(torch.float64)
        return out

    def debye(self, H, nb, type_names, counts, U, delta2, floor, q, f, qbroad=0.0):
        """Coherent I(Q) (electrons^2 per particle) on q (numpy). f: complex (n_types, nQ) numpy."""
        qt = torch.as_tensor(q, dtype=torch.float32, device=self.dev)
        r = (torch.arange(nb, device=self.dev, dtype=torch.float32)) * DR
        r_safe = torch.clamp(r, min=0.5)

        def shape(rr):
            return torch.clamp(1.0 - delta2 / torch.clamp(rr, min=0.5) ** 2, min=floor)

        n_t = len(type_names)
        I = torch.zeros(len(q), dtype=torch.float64, device=self.dev)
        fr = torch.as_tensor(f.real, dtype=torch.float64, device=self.dev)
        fi = torch.as_tensor(f.imag, dtype=torch.float64, device=self.dev)
        for a in range(n_t):
            I += counts[a] * (fr[a] ** 2 + fi[a] ** 2)
            for b in range(a, n_t):
                h = H[a, b]
                if float(h.sum()) == 0.0:
                    continue
                S = self._pair_sums(h, r, qt, shape, U[type_names[a]] + U[type_names[b]], qbroad)
                I += 2.0 * (fr[a] * fr[b] + fi[a] * fi[b]) * S
        return I.cpu().numpy()

    # ------------------------------------------------------------------------------------------ XRD
    def xrd_cu(self, H, nb, names, counts, U, delta2, floor, p):
        tth = self.g.tth
        th = np.radians(tth / 2)
        lam1, lam2 = p["wavelength_ka1"], p["wavelength_ka2"]
        q1 = 4 * math.pi * np.sin(th) / lam1
        q2 = 4 * math.pi * np.sin(th) / lam2
        q = np.concatenate([q1, q2])
        f = form_factors(names, q, HC / lam1)
        I = self.debye(H, nb, names, counts, U, delta2, floor, q, f)
        n = len(tth)
        y = I[:n] + p["ka2_ratio"] * I[n:]
        c2 = np.cos(np.radians(tth)) ** 2
        if p["monochromator"] == "graphite":
            cm = math.cos(math.radians(26.57)) ** 2
            pol = (1 + c2 * cm) / (1 + cm)
        else:
            pol = (1 + c2) / 2
        y = y * pol
        H2 = p["caglioti_U"] * np.tan(th) ** 2 + p["caglioti_V"] * np.tan(th) + p["caglioti_W"]
        fwhm = np.sqrt(np.clip(H2, 1e-6, None))
        y = self._pv_convolve(tth, y, fwhm, p["lorentz_fraction"])
        return y / counts.sum()

    def xrd_sync(self, H, nb, names, counts, U, delta2, floor, p):
        q = self.g.q_sync
        lam = p["wavelength"]
        f = form_factors(names, q, HC / lam)
        I = self.debye(H, nb, names, counts, U, delta2, floor, q, f)
        tth = 2 * np.arcsin(np.clip(q * lam / (4 * math.pi), 0, 1))
        I = I * (1 + np.cos(tth) ** 2) / 2
        fwhm = np.sqrt(p["fwhm_q0"] ** 2 + (p["fwhm_q1"] * q) ** 2)
        y = self._pv_convolve(q, I, fwhm, 0.0)
        return y / counts.sum(), np.degrees(tth)

    def _pv_convolve(self, x, y, fwhm, eta):
        """Variable-width pseudo-Voigt convolution as a dense column-normalized kernel (grid <= 4096 points)."""
        xt = torch.as_tensor(x, dtype=torch.float64, device=self.dev)
        yt = torch.as_tensor(y, dtype=torch.float64, device=self.dev)
        w = torch.as_tensor(fwhm, dtype=torch.float64, device=self.dev)
        dx = xt[:, None] - xt[None, :]  # (out, in): profile of input j centred at x_j
        sg = w[None, :] / 2.3548200450309493
        gauss = torch.exp(-0.5 * (dx / sg) ** 2) / sg
        lor = (w[None, :] / 2) / (dx ** 2 + (w[None, :] / 2) ** 2)
        K = (1 - eta) * gauss + eta * lor
        K = K / K.sum(0, keepdim=True)
        return (K @ yt).cpu().numpy()

    # ------------------------------------------------------------------------------------------ PDF
    def pdf(self, H, nb, names, counts, U, delta2, floor, p):
        q = self.g.q_pdf
        f = form_factors(names, q, None).real.astype(complex)
        I = self.debye(H, nb, names, counts, U, delta2, floor, q, f, qbroad=p["qbroad"])
        N = counts.sum()
        c = counts / N
        fr = f.real
        f_mean = (c[:, None] * fr).sum(0)
        f2_mean = (c[:, None] * fr ** 2).sum(0)
        S = 1.0 + (I / N - f2_mean) / f_mean ** 2
        F = q * (S - 1.0)
        mask = (q >= p["qmin"]) & (q <= p["qmax"])
        Fm = np.where(mask, F, 0.0)
        if self.sin_pdf is None:
            qt = torch.as_tensor(q, dtype=torch.float64, device=self.dev)
            rt = torch.as_tensor(self.g.r_pdf, dtype=torch.float64, device=self.dev)
            self.sin_pdf = torch.sin(qt[:, None] * rt[None, :])
        dq = self.cfg["pdf"]["q_step"]
        G = (2 / math.pi) * (torch.as_tensor(Fm, dtype=torch.float64, device=self.dev) @ self.sin_pdf) * dq
        G = G.cpu().numpy() * np.exp(-0.5 * (p["qdamp"] * self.g.r_pdf) ** 2)
        return G, q, F

    # ------------------------------------------------------------------------------------------ EXAFS input
    @staticmethod
    def absorber_partials(H, nb, names, counts, X, rmax=6.6):
        """n_XY(r) per neighbour type Y (numpy dict), r grid = k * DR, for r < rmax."""
        kmax = int(rmax / DR)
        a = names.index(X)
        out = {}
        for b, Y in enumerate(names):
            lo, hi = min(a, b), max(a, b)
            h = H[lo, hi, :kmax].double().cpu().numpy()
            if a == b:
                h = 2.0 * h
            out[Y] = h / counts[a]
        return out, np.arange(kmax) * DR
