"""Multi-technique predictor: one 1D ResNet-18-style encoder per technique + transformer fusion.

Tokens: [CLS] + one per present technique (xrd, xrd_sync, pdf, exafs_pt, exafs_m) + optional metadata token
(nominal composition and candidate phases, as an experimenter would report them). Absent techniques are masked, and
training drops techniques at random (p = 0.3 each, at least one spectrum kept; metadata dropped with p = 0.5), so the
model works with any subset.
Heads on [CLS]: composition (7 classes), ordered/disordered (alloys), size, lattice parameter, first-shell N and R for
the Pt and the M absorber, fault flags (7, multi-label).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import EDGES, TECHNIQUES

REG_TARGETS = ["size", "lattice", "n1_pt", "r1_pt", "n1_m", "r1_m"]


class BasicBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sk = None if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + (x if self.sk is None else self.sk(x)))


class ResNet18_1D(nn.Module):
    def __init__(self, cin=2, widths=(64, 128, 256, 512)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, widths[0], 7, 2, 3, bias=False), nn.BatchNorm1d(widths[0]), nn.ReLU(),
                                  nn.MaxPool1d(3, 2, 1))
        layers, prev = [], widths[0]
        for i, w in enumerate(widths):
            layers += [BasicBlock1D(prev, w, 1 if i == 0 else 2), BasicBlock1D(w, w)]
            prev = w
        self.layers = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.layers(self.stem(x)).mean(-1)


def with_derivative(x):
    d = torch.diff(x, dim=1, prepend=x[:, :1])
    return torch.stack([x, d / torch.sqrt(d.pow(2).mean(1, keepdim=True) + 1e-10)], 1)


class Predictor(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, meta_dim=11, n_comp=7, n_faults=7):
        super().__init__()
        self.enc = nn.ModuleDict({t: ResNet18_1D(2) for t in TECHNIQUES})
        self.proj = nn.ModuleDict({t: nn.Linear(512, d) for t in TECHNIQUES})
        self.tech_emb = nn.Parameter(torch.randn(len(TECHNIQUES) + 2, d) * 0.02)  # techniques + meta + cls
        self.edge_emb = nn.Embedding(len(EDGES), d)
        self.meta = nn.Sequential(nn.Linear(meta_dim, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.h_comp = nn.Linear(d, n_comp)
        self.h_ord = nn.Linear(d, 1)
        self.h_reg = nn.Linear(d, len(REG_TARGETS))
        self.h_fault = nn.Linear(d, n_faults)

    def embed(self, specs):
        """Encoder embeddings (B, 512) per technique - used for Frechet distances."""
        return {t: self.enc[t](with_derivative(x)) for t, x in specs.items()}

    def forward(self, specs, present, meta=None, meta_present=None, m_edge=None):
        """specs: dict technique -> (B, L); present: dict technique -> (B,) bool; meta: (B, 11)."""
        B = next(iter(specs.values())).shape[0]
        dev = next(iter(specs.values())).device
        toks, masks = [self.tech_emb[-1].expand(B, -1)], [torch.ones(B, dtype=torch.bool, device=dev)]
        for i, t in enumerate(TECHNIQUES):
            if t not in specs:
                continue
            e = self.proj[t](self.enc[t](with_derivative(specs[t]))) + self.tech_emb[i]
            if t == "exafs_m" and m_edge is not None:
                e = e + self.edge_emb(m_edge)
            toks.append(e)
            masks.append(present[t])
        if meta is not None:
            toks.append(self.meta(meta) + self.tech_emb[len(TECHNIQUES)])
            masks.append(meta_present if meta_present is not None else torch.ones(B, dtype=torch.bool, device=dev))
        x = torch.stack(toks, 1)
        keep = torch.stack(masks, 1)
        h = self.fusion(x, src_key_padding_mask=~keep)
        c = self.norm(h[:, 0])
        return dict(composition=self.h_comp(c), ordered=self.h_ord(c)[:, 0], reg=self.h_reg(c), faults=self.h_fault(c))


def technique_dropout(present, gen, p=0.3):
    """Randomly drop techniques (in place on copies), keeping at least one present spectrum per sample."""
    out = {t: v.clone() for t, v in present.items()}
    B = next(iter(out.values())).shape[0]
    dev = next(iter(out.values())).device
    for t in out:
        out[t] &= torch.rand(B, generator=gen, device=dev) >= p
    any_left = torch.stack(list(out.values()), 1).any(1)
    if (~any_left).any():
        # restore one originally present technique at random for samples that lost all
        orig = torch.stack([present[t] for t in out], 1).float()
        pick = torch.multinomial(orig[~any_left] + 1e-9, 1, generator=gen)[:, 0]
        idx = torch.nonzero(~any_left)[:, 0]
        keys = list(out)
        for j, i in zip(pick.tolist(), idx.tolist()):
            out[keys[j]][i] = True
    return out


def losses(out, tgt, stats):
    """Masked multi-task loss. tgt: dict of tensors; stats: mean/std per regression target."""
    l_comp = F.cross_entropy(out["composition"], tgt["composition"])
    ordm = tgt["ordered"] >= 0
    l_ord = F.binary_cross_entropy_with_logits(out["ordered"][ordm], tgt["ordered"][ordm]) if ordm.any() else out["ordered"].sum() * 0
    regs = []
    for j, k in enumerate(REG_TARGETS):
        y = tgt[k]
        m = torch.isfinite(y)
        if m.any():
            yz = (y[m] - stats[k][0]) / stats[k][1]
            regs.append(F.smooth_l1_loss(out["reg"][m, j], yz, beta=0.5))
    l_reg = torch.stack(regs).mean()
    l_fault = F.binary_cross_entropy_with_logits(out["faults"], tgt["faults"])
    total = l_comp + l_ord + l_reg + l_fault
    return total, dict(comp=float(l_comp), ord=float(l_ord), reg=float(l_reg), fault=float(l_fault))


def decode(out, stats):
    """Network outputs -> predictions in physical units."""
    pred = dict(composition=out["composition"].argmax(1), composition_prob=out["composition"].softmax(1),
                ordered_prob=torch.sigmoid(out["ordered"]), fault_prob=torch.sigmoid(out["faults"]))
    for j, k in enumerate(REG_TARGETS):
        pred[k] = out["reg"][:, j] * stats[k][1] + stats[k][0]
    return pred
