# Hardware check (Phase 0)

Generated 2026-10-22T02:38:57 on `NEDOGPU-WS3` by `forge/hw_check.py` (raw data: `hardware.json`).

## Summary

- **GPUs:** 4 × NVIDIA RTX A6000, 48 GiB each, compute capability 8.6, driver 566.03 in **WDDM** mode.
- **Runtime:** PyTorch 2.14.1+cu126, CUDA runtime 12.6, cuDNN 91002, Python 3.12.15, Windows-11-10.0.22631-SP0.
- **bf16:** supported (Ampere, compute capability 8.6).
- **4-GPU all-reduce:** PASS via gloo, CUDA tensors staged through pinned host memory; 294 ms per 64 MiB (0.23 GB/s algorithm bandwidth).
- **NCCL:** not available (Windows build); gloo direct CUDA all-reduce: crashes with 0xC0000005 (access violation) on this Windows build; tested 2026-10-22 with torch 2.14.1+cu126.
- **CPU:** Intel64 Family 6 Model 143 Stepping 8, GenuineIntel, 16 physical / 32 logical cores.
- **RAM:** 495.6 GiB (432.3 GiB available).
- **Free disk on the project drive:** 2320.9 GiB.

## GPUs

| GPU | Bus | Total GiB | Used by other processes GiB | Physical headroom GiB | torch-reported free GiB |
|---|---|---|---|---|---|
| 0 | 00000000:47:00.0 | 48.0 | 1.1 | 46.9 | 46.4 |
| 1 | 00000000:5E:00.0 | 48.0 | 41.5 | 6.5 | 46.4 |
| 2 | 00000000:75:00.0 | 48.0 | 41.1 | 6.9 | 46.4 |
| 3 | 00000000:A3:00.0 | 48.0 | 36.6 | 11.4 | 46.4 |

## Findings and adaptations

1. **GPUs 1–3 are shared with a resident Ollama `llama-server`** (an LLM service that is not part of this project and was left running). It holds most of the memory on those cards. Under WDDM the driver virtualizes VRAM, so `torch.cuda.mem_get_info` reports the cards as almost empty while `nvidia-smi` shows the real residency. Allocating past the physical headroom would page memory to host RAM and slow down both workloads instead of failing cleanly.
   - Adaptation: GPU 0 is the primary device. On GPUs 1–3 every job caps its CUDA caching allocator below the physical headroom shown above (`torch.cuda.set_per_process_memory_fraction`), and per-GPU batch memory is measured before long runs.
2. **No NCCL on Windows, and gloo crashes on CUDA tensors.** Data-parallel training therefore uses one process per GPU with gradients all-reduced through host memory (`forge/dist.py`). Correctness for fp32 and bf16 was verified on all four GPUs.
3. **CPU work** (FEFF, file I/O) uses 28 worker processes, leaving 4 logical cores free.
4. **Simulation** runs as a 4-GPU work queue (one worker process per GPU); its memory per worker is small (pair-distance histograms in chunks).

Resident processes on GPUs 1–3 at check time:

```
31596, C:\Users\NEDO-WS3\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe, 00000000:5E:00.0, [N/A]
31596, C:\Users\NEDO-WS3\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe, 00000000:75:00.0, [N/A]
31596, C:\Users\NEDO-WS3\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe, 00000000:A3:00.0, [N/A]
```

## How the GPUs were used (adaptation, recorded during the build)

- **Simulation:** a lock-file work queue with one worker process per GPU, plus extra workers on GPU 0 and GPU 3
  where memory allowed. Workers hold no shared state, so the GPUs run independently. A worker that hits a CUDA
  error exits and is respawned instead of failing the rest of the queue. This happened once: an "illegal memory
  access" on GPU 0 while it was oversubscribed by five processes.
- **Training: one model per GPU, not 4-GPU data parallelism.**
  - Gradient all-reduce through host memory measured 0.29 s per 64 MiB (gloo, pinned staging). That is as long as
    a whole GAN step on an uncontended GPU. GPUs 1–3 also have only 4–11 GB of free physical memory next to the
    resident Ollama model.
  - The realism GANs and the predictor runs were therefore trained concurrently, one model per GPU, each with the
    specified per-GPU batch of 128. Every GPU stays busy, with no communication overhead.
  - The gradient all-reduce itself (staged gloo, and the shared-memory reducer in `forge/dist.py`) was verified on
    four GPUs. The multi-GPU path of `forge/train_predictor.py` (several entries in `gpus`) is implemented but was
    not exercised end to end and was not used for the final runs.

  | GPU | Physical headroom | Training jobs (in order) |
  |---|---|---|
  | 0 | 46.9 GiB | XRD realism GAN (26,000 steps, early stop); GAN diagnostics; predictor (b) seed 1 |
  | 1 | 6.5 GiB | predictor (a) seeds 1 and 2 side by side (2.5 GiB cap each, 69 min each); EXAFS realism run 2 (R1 γ = 10) |
  | 2 | 6.9 GiB | EXAFS realism run 1 (R1 γ = 1). A PDF run capped at 6 GiB ran out of memory here and was moved to GPU 3 |
  | 3 | 11.4 GiB | PDF realism run 1 (R1 γ = 1), then run 2 (R1 γ = 10); predictor (b) seed 2 |

- **GAN throughput is limited by the CPU, not the GPU.** The GANs ran at 1.3–1.9 steps/s with GPU utilization of
  25–60 %. A profile of one XRD training step showed:
  - about 6,600 kernel launches at about 40 µs each;
  - five blocking host-to-device copies;
  - roughly 0.15–0.3 s of GPU work, against a 0.6 s wall-clock step.

  The loop is bound by Python dispatch and kernel launches under WDDM. CUDA graphs could roughly double the rate.
  That was not attempted on running jobs, because the step budget fit the time plan.
- **Sharing a GPU between several heavy processes** under WDDM costs 2–3× per process (time-slicing whole contexts).
  Long runs were scheduled so that each GPU carries at most one heavy job.
