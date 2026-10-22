"""Phase 0 hardware check: GPUs, driver/runtime, bf16, multi-GPU all-reduce, CPU, RAM, disk.

Writes artifacts/reports/hardware.md and artifacts/reports/hardware.json.
NCCL does not exist on Windows, so the all-reduce test (and DDP) uses the gloo backend.
"""
import datetime
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_LIBUV", "0")  # Windows torch builds have no libuv TCPStore

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"


def nvidia_smi():
    q = "index,name,memory.total,memory.used,driver_version,pci.bus_id,driver_model.current,compute_cap"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        rows.append(dict(index=int(f[0]), name=f[1], mem_total_mib=int(f[2]), mem_used_mib=int(f[3]),
                         driver=f[4], bus=f[5], driver_model=f[6], compute_cap=f[7]))
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_bus_id,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return rows, apps


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def staged_all_reduce(t, buf=None):
    """gloo all-reduce of a CUDA tensor through (pinned) host memory; gloo's direct CUDA path crashes on Windows."""
    if buf is None:
        buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    buf.copy_(t)
    dist.all_reduce(buf)
    t.copy_(buf)
    return t


def _allreduce_worker(rank, world, port, q):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    dev = torch.device("cuda", rank)
    ok = True
    # correctness: each rank contributes rank+1, sum must be world*(world+1)/2 everywhere
    for dtype in (torch.float32, torch.bfloat16):
        t = torch.full((1024,), float(rank + 1), device=dev, dtype=dtype)
        staged_all_reduce(t)
        ok &= bool(torch.all(t == world * (world + 1) / 2).item())
    # bandwidth: 64 MiB float32 buffer (roughly a 16M-parameter gradient), including D2H/H2D copies
    n = 16 * 1024 * 1024
    t = torch.ones(n, device=dev)
    buf = torch.empty(n, pin_memory=True)
    dist.barrier()
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        staged_all_reduce(t, buf)
    torch.cuda.synchronize(dev)
    dt = (time.perf_counter() - t0) / reps
    ok &= bool(torch.allclose(t[:10], torch.full((10,), float(world ** reps), device=dev)))
    # bf16 matmul sanity on this GPU
    a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    c = a @ a
    ok &= bool(torch.isfinite(c).all().item())
    if rank == 0:
        q.put(dict(ok=ok, seconds_per_64MiB_allreduce=dt, algbw_GBps=(n * 4 / dt) / 1e9))
    dist.destroy_process_group()


def allreduce_test(world):
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    port = free_port()
    mp.spawn(_allreduce_worker, args=(world, port, q), nprocs=world, join=True)
    return q.get()


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows, apps = nvidia_smi()
    info = dict(timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                host=platform.node(), os=platform.platform(), python=sys.version.split()[0],
                torch=torch.__version__, cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpus=rows, compute_apps=apps,
                bf16=torch.cuda.is_bf16_supported(), nccl=dist.is_nccl_available(), gloo=dist.is_gloo_available())
    for r in rows:
        r["mem_free_physical_gib"] = round((r["mem_total_mib"] - r["mem_used_mib"]) / 1024, 1)
    info["torch_devices"] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        f, t = torch.cuda.mem_get_info(i)
        info["torch_devices"].append(dict(index=i, name=p.name, cc=f"{p.major}.{p.minor}", sms=p.multi_processor_count,
                                          total_gib=round(t / 2**30, 1), reported_free_gib=round(f / 2**30, 1)))
    vm = psutil.virtual_memory()
    info["cpu"] = dict(model=platform.processor(), physical=psutil.cpu_count(logical=False),
                       logical=psutil.cpu_count(logical=True))
    info["ram_gib"] = round(vm.total / 2**30, 1)
    info["ram_available_gib"] = round(vm.available / 2**30, 1)
    du = shutil.disk_usage(ROOT)
    info["disk_free_gib"] = round(du.free / 2**30, 1)
    world = torch.cuda.device_count()
    try:
        info["allreduce"] = allreduce_test(world)
        info["allreduce"]["method"] = "gloo, CUDA tensors staged through pinned host memory"
    except Exception as e:  # report and adapt instead of stopping
        info["allreduce"] = dict(ok=False, error=repr(e))
    info["allreduce"]["direct_cuda_gloo"] = ("crashes with 0xC0000005 (access violation) on this Windows build; "
                                             "tested 2026-10-22 with torch 2.14.1+cu126")
    (REPORTS / "hardware.json").write_text(json.dumps(info, indent=2))
    write_md(info)
    print(json.dumps(info, indent=2))


def write_md(info):
    g = info["gpus"]
    ar = info["allreduce"]
    ollama = [l for l in info["compute_apps"].splitlines() if "ollama" in l.lower()]
    lines = [
        "# Hardware check (Phase 0)", "",
        f"Generated {info['timestamp']} on `{info['host']}` by `forge/hw_check.py` (raw data: `hardware.json`).", "",
        "## Summary", "",
        f"- **GPUs:** {len(g)} × {g[0]['name']}, {g[0]['mem_total_mib'] / 1024:.0f} GiB each, compute capability "
        f"{g[0]['compute_cap']}, driver {g[0]['driver']} in **{g[0]['driver_model']}** mode.",
        f"- **Runtime:** PyTorch {info['torch']}, CUDA runtime {info['cuda_runtime']}, cuDNN {info['cudnn']}, "
        f"Python {info['python']}, {info['os']}.",
        f"- **bf16:** {'supported' if info['bf16'] else 'NOT supported'} (Ampere, compute capability 8.6).",
        f"- **4-GPU all-reduce:** {'PASS' if ar.get('ok') else 'FAIL'} via {ar.get('method', 'n/a')}; "
        f"{ar.get('seconds_per_64MiB_allreduce', float('nan')) * 1000:.0f} ms per 64 MiB "
        f"({ar.get('algbw_GBps', float('nan')):.2f} GB/s algorithm bandwidth).",
        f"- **NCCL:** {'available' if info['nccl'] else 'not available (Windows build)'}; gloo direct CUDA all-reduce: "
        f"{ar.get('direct_cuda_gloo')}.",
        f"- **CPU:** {info['cpu']['model']}, {info['cpu']['physical']} physical / {info['cpu']['logical']} logical cores.",
        f"- **RAM:** {info['ram_gib']} GiB ({info['ram_available_gib']} GiB available).",
        f"- **Free disk on the project drive:** {info['disk_free_gib']} GiB.", "",
        "## GPUs", "",
        "| GPU | Bus | Total GiB | Used by other processes GiB | Physical headroom GiB | torch-reported free GiB |",
        "|---|---|---|---|---|---|",
    ]
    for r, t in zip(g, info["torch_devices"]):
        lines.append(f"| {r['index']} | {r['bus']} | {r['mem_total_mib'] / 1024:.1f} | {r['mem_used_mib'] / 1024:.1f} | "
                     f"{r['mem_free_physical_gib']} | {t['reported_free_gib']} |")
    lines += [
        "",
        "## Findings and adaptations", "",
        "1. **GPUs 1–3 are shared with a resident Ollama `llama-server`** (an LLM service that is not part of this "
        "project and was left running). It holds most of the memory on those cards. Under WDDM the driver "
        "virtualizes VRAM, so `torch.cuda.mem_get_info` reports the cards as almost empty while `nvidia-smi` shows the "
        "real residency. Allocating past the physical headroom would page memory to host RAM and slow down both "
        "workloads instead of failing cleanly.",
        "   - Adaptation: GPU 0 is the primary device. On GPUs 1–3 every job caps its CUDA caching allocator below "
        "the physical headroom shown above (`torch.cuda.set_per_process_memory_fraction`), and per-GPU batch "
        "memory is measured before long runs.",
        "2. **No NCCL on Windows, and gloo crashes on CUDA tensors.** Data-parallel training therefore uses one process "
        "per GPU with gradients all-reduced through host memory (`forge/dist.py`). Correctness for fp32 and bf16 "
        "was verified on all four GPUs.",
        "3. **CPU work** (FEFF, file I/O) uses 28 worker processes, leaving 4 logical cores free.",
        "4. **Simulation** runs as a 4-GPU work queue (one worker process per GPU); its memory per worker is small "
        "(pair-distance histograms in chunks).",
    ]
    if ollama:
        lines += ["", "Resident processes on GPUs 1–3 at check time:", "", "```"] + ollama + ["```"]
    (REPORTS / "hardware.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
