"""Data-parallel training helpers for Windows (one process per GPU).

NCCL is not available on Windows and gloo crashes on CUDA tensors here (artifacts/reports/hardware.md), so gradients
are averaged through host memory. The fast path is a shared-memory all-reduce: every rank copies its flattened
gradients into its row of a shared (world, n) tensor, each rank then averages one contiguous chunk across rows into
a shared result buffer (double-buffered), and all ranks copy the result back to their GPU. gloo is only used for
barriers and for broadcasting the initial weights.
"""
import os

os.environ.setdefault("USE_LIBUV", "0")

import torch
import torch.distributed as dist


def init(rank, world, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)


def barrier():
    if dist.is_initialized():
        dist.barrier()


class SharedReducer:
    """Average a flat float32 vector over ranks through shared host memory."""

    def __init__(self, n, world):
        self.n, self.world = n, world
        self.inbuf = torch.zeros(world, n).share_memory_()
        self.outbuf = torch.zeros(2, n).share_memory_()
        self.flip = 0

    def allreduce_(self, vec, rank):
        """In-place average of a CUDA (or CPU) float32 vector of length n."""
        self.inbuf[rank].copy_(vec)
        dist.barrier()
        lo = (self.n * rank) // self.world
        hi = (self.n * (rank + 1)) // self.world
        torch.sum(self.inbuf[:, lo:hi], dim=0, out=self.outbuf[self.flip, lo:hi])
        self.outbuf[self.flip, lo:hi].div_(self.world)
        dist.barrier()
        vec.copy_(self.outbuf[self.flip])
        self.flip ^= 1
        return vec


def flat_grads(params):
    gs = [p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=p.device) for p in params]
    return torch.cat(gs)


def set_flat_grads(params, vec):
    i = 0
    for p in params:
        n = p.numel()
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.copy_(vec[i:i + n].view_as(p))
        i += n


def sync_grads(params, reducer, rank):
    if reducer is None or reducer.world == 1:
        return
    v = flat_grads(params).float()
    reducer.allreduce_(v, rank)
    set_flat_grads(params, v)


def broadcast_module(module, src=0):
    """Broadcast parameters and buffers from rank src via CPU copies."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for t in list(module.parameters()) + list(module.buffers()):
        c = t.detach().cpu()
        dist.broadcast(c, src)
        t.data.copy_(c)


def n_params(module):
    return sum(p.numel() for p in module.parameters())
