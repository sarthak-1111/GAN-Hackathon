"""Conditional residual GAN realism model (one per technique).

Generator (label-preserving by construction): a 1D U-Net (4 levels, 64 -> 512 channels, FiLM conditioning on the
condition vector and z) predicts three per-position raw fields that are projected onto smooth cubic B-spline bases:
    background   12 coefficients
    gain         8 coefficients, exp() and clipped to [0.9, 1.1]
    noise scale  16 coefficients through softplus, multiplied by coloured noise (white noise through a learned,
                 unit-norm 1D convolution)
    glitches     EXAFS only: up to 2 Gaussian spikes with sigmoid gates, positions and tanh amplitudes
    output = sim * gain + background + noise_scale * coloured_noise (+ glitches)
The smoothest field (gain, 8 coefficients over the whole grid) cannot shift a peak. The background basis (knot spacing
XRD 6.7 deg, PDF 3.3 A, EXAFS 1.42 A^-1) can form bumps no narrower than 5.5 deg, 2.7 A and 1.15 A^-1 FWHM: >= 5x and
>= 9x the narrowest XRD and PDF peaks, but comparable to the EXAFS first-shell period, so first-shell preservation for
EXAFS relies on the preservation loss and is checked by tests (tests/test_label_preservation.py).

Discriminator: input = (spectrum, first derivative); 1D ResNet with 6 blocks (64 -> 512 channels), spectral norm on
every layer, projection conditioning on the condition vector. Losses: hinge + R1 on real samples; preservation losses
(XRD / PDF: cross-correlation of smoothed derivatives at zero lag; EXAFS: Fourier-filtered first-shell amplitude and
phase); penalty on artifact magnitude.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import BSpline
from torch.nn.utils.parametrizations import spectral_norm


def bspline_basis(L, n_coef, degree=3):
    """(L, n_coef) clamped uniform cubic B-spline design matrix on [0, 1]."""
    n_inner = n_coef - degree + 1
    inner = np.linspace(0, 1, n_inner)
    t = np.concatenate([[0.0] * degree, inner, [1.0] * degree])
    x = np.linspace(0, 1, L)
    B = BSpline.design_matrix(x, t, degree).toarray()
    return torch.tensor(B, dtype=torch.float32)


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout, cdim, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.film = nn.Linear(cdim, 4 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        g1, b1, g2, b2 = self.film(c)[:, :, None].chunk(4, dim=1)
        h = F.silu(self.n1(self.c1(x)) * (1 + g1) + b1)
        h = F.silu(self.n2(self.c2(h)) * (1 + g2) + b2)
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, cin, cdim, chans=(64, 128, 256, 512)):
        super().__init__()
        self.inp = nn.Conv1d(cin, chans[0], 7, padding=3)
        self.enc = nn.ModuleList()
        prev = chans[0]
        for c in chans:
            self.enc.append(FiLMBlock(prev, c, cdim))
            prev = c
        self.mid = FiLMBlock(chans[-1], chans[-1], cdim)
        self.dec = nn.ModuleList()
        for c in reversed(chans[:-1]):
            self.dec.append(FiLMBlock(prev + c, c, cdim))
            prev = c
        self.out_ch = prev

    def forward(self, x, c):
        h = self.inp(x)
        skips = []
        for i, blk in enumerate(self.enc):
            h = blk(h, c)
            if i < len(self.enc) - 1:
                skips.append(h)
                h = F.avg_pool1d(h, 2)
        h = self.mid(h, c)
        bott = h
        for blk in self.dec:
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-1], mode="linear", align_corners=False)
            h = blk(torch.cat([h, s], 1), c)
        return h, bott


class Generator(nn.Module):
    def __init__(self, L, cond_dim, z_dim=64, glitches=False, n_bg=12, n_gain=8, n_noise=16, noise_kernel=31,
                 max_glitch=0.5):
        super().__init__()
        self.L, self.z_dim, self.glitches = L, z_dim, glitches
        cdim = 128
        self.cond = nn.Sequential(nn.Linear(cond_dim + z_dim, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.unet = UNet1D(2, cdim)
        self.head = nn.Conv1d(self.unet.out_ch, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        for name, n in (("B_bg", n_bg), ("B_gain", n_gain), ("B_noise", n_noise)):
            B = bspline_basis(L, n)
            self.register_buffer(name, B)
            self.register_buffer(name + "_pinv", torch.linalg.pinv(B))
        self.noise_kernel = nn.Parameter(torch.randn(1, 1, noise_kernel) * 0.05 + F.one_hot(
            torch.tensor(noise_kernel // 2), noise_kernel).float()[None, None])
        self.noise_bias = nn.Parameter(torch.tensor(-4.0))  # softplus(-4) ~ 0.018: start with little noise
        if glitches:
            self.glitch = nn.Sequential(nn.Linear(512 + cdim, 128), nn.SiLU(), nn.Linear(128, 2 * 4))
            nn.init.zeros_(self.glitch[-1].weight)
            nn.init.constant_(self.glitch[-1].bias, 0.0)
            with torch.no_grad():
                self.glitch[-1].bias[0::4] = -4.0  # gates start closed
            self.max_glitch = max_glitch
        self.register_buffer("grid", torch.linspace(0, 1, L))

    def coloured_noise(self, B, device, gen=None):
        w = torch.randn(B, 1, self.L + self.noise_kernel.shape[-1] - 1, device=device, generator=gen)
        k = self.noise_kernel / self.noise_kernel.norm()
        return F.conv1d(w, k)[:, 0]

    def forward(self, sim, cond, z=None, gen=None, return_parts=False):
        B = sim.shape[0]
        if z is None:
            z = torch.randn(B, self.z_dim, device=sim.device, generator=gen)
        c = self.cond(torch.cat([cond, z], 1))
        x = torch.stack([sim, self.grid.expand(B, -1)], 1)
        h, bott = self.unet(x, c)
        raw = self.head(h)                                            # (B, 3, L)
        c_bg = raw[:, 0] @ self.B_bg_pinv.T                           # least-squares projection onto the bases
        c_gain = raw[:, 1] @ self.B_gain_pinv.T
        c_noise = raw[:, 2] @ self.B_noise_pinv.T
        bg = c_bg @ self.B_bg.T
        gain = torch.clamp(torch.exp(c_gain @ self.B_gain.T), 0.9, 1.1)
        nscale = F.softplus(c_noise @ self.B_noise.T + self.noise_bias)
        noise = nscale * self.coloured_noise(B, sim.device, gen)
        out = sim * gain + bg + noise
        parts = dict(bg=bg, gain=gain, nscale=nscale, noise=noise)
        if self.glitches:
            gp = self.glitch(torch.cat([bott.mean(-1), c], 1)).view(B, 2, 4)
            gate = torch.sigmoid(gp[..., 0])
            pos = torch.sigmoid(gp[..., 1])
            amp = torch.tanh(gp[..., 2]) * self.max_glitch
            width = 0.002 + 0.006 * torch.sigmoid(gp[..., 3])
            spikes = (gate * amp)[..., None] * torch.exp(-0.5 * ((self.grid[None, None] - pos[..., None]) / width[..., None]) ** 2)
            gl = spikes.sum(1)
            out = out + gl
            parts.update(glitch=gl, gate=gate, gamp=amp)
        return (out, parts) if return_parts else out


class SNResBlock(nn.Module):
    def __init__(self, cin, cout, down=True):
        super().__init__()
        self.c1 = spectral_norm(nn.Conv1d(cin, cout, 3, padding=1))
        self.c2 = spectral_norm(nn.Conv1d(cout, cout, 3, padding=1))
        self.sk = spectral_norm(nn.Conv1d(cin, cout, 1))
        self.down = down

    def forward(self, x):
        h = self.c2(F.leaky_relu(self.c1(F.leaky_relu(x, 0.2)), 0.2))
        s = self.sk(x)
        if self.down:
            h, s = F.avg_pool1d(h, 2), F.avg_pool1d(s, 2)
        return h + s


class Discriminator(nn.Module):
    def __init__(self, cond_dim, chans=(64, 128, 256, 512, 512, 512)):
        super().__init__()
        self.inp = spectral_norm(nn.Conv1d(2, chans[0], 7, padding=3))
        blocks, prev = [], chans[0]
        for c in chans:
            blocks.append(SNResBlock(prev, c, down=True))
            prev = c
        self.blocks = nn.Sequential(*blocks)
        self.lin = spectral_norm(nn.Linear(prev, 1))
        self.embed = spectral_norm(nn.Linear(cond_dim, prev))

    @staticmethod
    def with_derivative(x):
        d = torch.diff(x, dim=1, prepend=x[:, :1])
        # RMS with the epsilon inside the square root: finite gradients for flat inputs (needed by R1)
        d = d / torch.sqrt(d.pow(2).mean(dim=1, keepdim=True) + 1e-10)
        return torch.stack([x, d], 1)

    def forward(self, x, cond):
        h = self.blocks(self.inp(self.with_derivative(x)))
        h = F.leaky_relu(h, 0.2).sum(-1)
        return self.lin(h)[:, 0] + (self.embed(cond) * h).sum(1)


# --------------------------------------------------------------------------------------------- losses
def d_hinge(real_logits, fake_logits):
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def g_hinge(fake_logits):
    return -fake_logits.mean()


def r1_penalty(D, real, cond):
    real = real.detach().requires_grad_(True)
    out = D(real, cond)
    g, = torch.autograd.grad(out.sum(), real, create_graph=True)
    return g.pow(2).sum(1).mean()


def _smooth(x, sigma=3.0):
    half = int(3 * sigma) + 1
    t = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum())[None, None]
    return F.conv1d(F.pad(x[:, None], (half, half), mode="replicate"), k)[:, 0]


def xcorr_preservation(sim, gen, max_lag=8, sigma=3.0):
    """1 - normalized cross-correlation of smoothed derivatives at zero lag, plus a penalty on the soft-argmax lag."""
    ds = torch.diff(_smooth(sim, sigma), dim=1)
    dg = torch.diff(_smooth(gen, sigma), dim=1)
    ds = ds - ds.mean(1, keepdim=True)
    dg = dg - dg.mean(1, keepdim=True)
    ns, ng = ds.norm(dim=1) + 1e-8, dg.norm(dim=1) + 1e-8
    lags = range(-max_lag, max_lag + 1)
    cc = torch.stack([(ds * torch.roll(dg, l, 1)).sum(1) / (ns * ng) for l in lags], 1)
    w = torch.softmax(cc * 20, 1)
    lag = (w * torch.tensor(list(lags), device=sim.device, dtype=sim.dtype)).sum(1)
    return (1 - cc[:, max_lag]).mean() + 0.1 * (lag ** 2).mean()


class FirstShellFilter(nn.Module):
    """Fourier filter of k^2 chi(k): forward FT (Hanning window) and back-transform of the R window [r1, r2]."""

    def __init__(self, k, r1=1.8, r2=3.2, nr=64):
        super().__init__()
        k = torch.as_tensor(k, dtype=torch.float32)
        r = torch.linspace(r1, r2, nr)
        win = torch.hann_window(len(k), periodic=False)
        dk = float(k[1] - k[0])
        self.register_buffer("Ef", torch.exp(2j * k[:, None] * r[None, :]) * win[:, None] * dk)   # (K, R)
        self.register_buffer("Eb", torch.exp(-2j * r[:, None] * k[None, :]) * (r2 - r1) / nr)    # (R, K)

    def forward(self, y):
        yr = y.to(torch.complex64) @ self.Ef
        return yr @ self.Eb  # complex back-transform on k


def exafs_preservation(filt, sim, gen):
    a, b = filt(sim), filt(gen)
    amp_s, amp_g = a.abs(), b.abs()
    w = amp_s / (amp_s.mean(1, keepdim=True) + 1e-8)
    amp_term = ((amp_g - amp_s).abs().mean(1) / (amp_s.mean(1) + 1e-8)).mean()
    dphi = torch.angle(b * torch.conj(a))
    return amp_term + (w * dphi.abs()).mean()


def magnitude_penalty(parts, scale):
    pen = ((parts["gain"] - 1) ** 2).mean() / 0.05 ** 2 + (parts["bg"] ** 2).mean() / scale ** 2 + \
          (parts["nscale"] ** 2).mean() / scale ** 2
    if "gamp" in parts:
        pen = pen + ((parts["gate"] * parts["gamp"]) ** 2).mean() / scale ** 2
    return pen
