"""Classical (baseline) augmentation of model-view spectra, batched in torch.

Per the spec: Poisson and Gaussian noise, polynomial background, Gaussian broadening and normalization jitter. The
xca aberration code (maffettone/xca) was studied: it adds max-normalized Gaussian noise, polynomial / inverse-power
backgrounds, Caglioti pseudo-Voigt profiles, March-Dollase texture and a sample-displacement 2theta shift. Peak shifts
and texture change label-relevant features (lattice parameter, intensity ratios), so they are not used here - the
same constraint the GAN realism layer works under.

Two strengths: "train" (augmentation for predictor training) and "harsh" (stress-test set; ~3x noise, 2x background).
"""
import math

import torch
import torch.nn.functional as F

RANGES = {
    "train": dict(xrd_counts=(5e2, 1e5), xrd_gauss=(1e-3, 1e-2), xrd_bg=0.12, xrd_broad=1.5,
                  pdf_gauss=(2e-3, 3e-2), pdf_bg=0.02, pdf_broad=1.0,
                  ex_sigma=(3e-4, 4e-3), ex_bg=0.08, jitter=0.1),
    "harsh": dict(xrd_counts=(1e2, 1e4), xrd_gauss=(3e-3, 3e-2), xrd_bg=0.25, xrd_broad=3.0,
                  pdf_gauss=(1e-2, 9e-2), pdf_bg=0.04, pdf_broad=2.0,
                  ex_sigma=(1e-3, 1.2e-2), ex_bg=0.16, jitter=0.15),
}


def _logu(lohi, n, g, dev):
    lo, hi = math.log(lohi[0]), math.log(lohi[1])
    return torch.exp(lo + (hi - lo) * torch.rand(n, 1, generator=g, device=dev))


def _u(lo, hi, n, g, dev):
    return lo + (hi - lo) * torch.rand(n, 1, generator=g, device=dev)


def gaussian_broaden(x, sigma_pts):
    """Per-sample Gaussian convolution with width sigma_pts (B, 1) in grid points (0 = identity)."""
    B, L = x.shape
    half = int(math.ceil(3 * float(sigma_pts.max()))) + 1
    t = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)[None, :]
    s = sigma_pts.clamp(min=1e-3)
    ker = torch.exp(-0.5 * (t / s) ** 2)
    ker = ker / ker.sum(1, keepdim=True)
    xp = F.pad(x[:, None, :], (half, half), mode="replicate")
    out = F.conv1d(xp.transpose(0, 1), ker[:, None, :], groups=B).transpose(0, 1)[:, 0]
    return out


def poly_background(B, L, amp, g, dev, positive=False, order=3):
    t = torch.linspace(0, 1, L, device=dev)[None, :]
    c = (torch.rand(B, order + 1, generator=g, device=dev) * 2 - 1) * amp
    if positive:
        c[:, 0] = c[:, 0].abs()
    bg = sum(c[:, j:j + 1] * (t ** j) for j in range(order + 1))
    if positive:
        bg = bg - torch.clamp(bg.min(1, keepdim=True).values, max=0.0)
    return bg


def classical(x, technique, g, strength="train"):
    """Augment a batch of model-view spectra (B, L). Returns a new tensor."""
    R = RANGES[strength]
    B, L = x.shape
    dev = x.device
    if technique in ("xrd", "xrd_sync"):
        y = gaussian_broaden(x, _u(0.0, R["xrd_broad"], B, g, dev))
        y = y + poly_background(B, L, R["xrd_bg"], g, dev, positive=True)
        y = y * _u(1 - R["jitter"], 1 + R["jitter"], B, g, dev)
        counts = _logu(R["xrd_counts"], B, g, dev)
        y = torch.poisson(torch.clamp(y, min=0) * counts, generator=g) / counts
        y = y + torch.randn(B, L, generator=g, device=dev) * _logu(R["xrd_gauss"], B, g, dev)
        return y / torch.clamp(y.max(1, keepdim=True).values, min=1e-6)
    if technique == "pdf":
        y = gaussian_broaden(x, _u(0.0, R["pdf_broad"], B, g, dev))
        y = y + poly_background(B, L, R["pdf_bg"], g, dev)
        y = y * _u(1 - R["jitter"], 1 + R["jitter"], B, g, dev)
        noise = torch.randn(B, L, generator=g, device=dev)
        noise = gaussian_broaden(noise, torch.full((B, 1), 1.5, device=dev)) * 2.0  # r-correlated (Q-white) noise
        y = y + noise * _logu(R["pdf_gauss"], B, g, dev)
        return y / torch.clamp(y.abs().max(1, keepdim=True).values, min=1e-6)
    if technique.startswith("exafs"):
        k = torch.linspace(2.0, 14.75, L, device=dev)[None, :]
        y = x * _u(1 - R["jitter"], 1 + R["jitter"], B, g, dev)
        y = y + poly_background(B, L, R["ex_bg"], g, dev, order=2)
        sig = _logu(R["ex_sigma"], B, g, dev)
        return y + torch.randn(B, L, generator=g, device=dev) * sig * k ** 2
    raise ValueError(technique)
