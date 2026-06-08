---
title: "GPU, CUDA, and the Ecosystem: A Progressive Learning Path"
date: 2026-06-09T00:00:00Z
draft: false
tags: ["gpu", "cuda", "learning", "system-design"]
showToc: true
---

## Tier 1: GPU Architecture (deep understanding)

### Stanford CS149 Lecture 7 — Kayvon Fatahalian

The best lecture on SIMT execution model, warp scheduling, memory hierarchy, and occupancy. Start here. Videos on YouTube.

**URL:** <https://cs149.stanford.edu>

### "Programming Massively Parallel Processors" — Kirk, Hwu, El Hajj (4th ed.)

The canonical textbook. Chapters 1-4 = architecture fundamentals. 4th edition covers tensor cores and Hopper.

**URL:** <https://www.sciencedirect.com/book/9780124159921/programming-massively-parallel-processors>

### Brendan Lynskey — NVIDIA GPU Architectures Series

Deep dives on warp scheduling internals, SIMT vs SIMD nuance, memory latency per architecture, TMA on Hopper. Diagrams + assembly snippets.

**URL:** <https://brendanjameslynskey.github.io/>

---

## Tier 2: CUDA Instructions (PTX → SASS)

### Philip Fabianek — A Gentle Introduction to CUDA PTX

Step-by-step walkthrough of a hand-written PTX kernel. Covers state spaces, special registers, control flow. Runnable repo included. Start here if you've never seen PTX.

**URL:** <https://philipfabianek.com/posts/cuda-ptx-introduction>

### Brendan Lynskey — PTX & SASS: The Real GPU Instruction Sets

CUDA → PTX → SASS pipeline with annotated kernels. Covers FFMA/WGMMA/LDMATRIX, predication, barrier primitives, ptxas diagnostics. Side-by-side sm_75 vs sm_90 SASS comparison. This is your primary instruction-level resource.

**URL:** <https://brendanjameslynskey.github.io/NVIDIA_GPU_21_PTX_and_SASS/>

### yuninxia/nvidia-sass-lessons (GitHub)

Interactive CLI with 9 progressive lessons. Compare -O0 vs -O3 SASS. Supports sm_75 through sm_90. Hands-on practice after theory.

**URL:** <https://github.com/yuninxia/nvidia-sass-lessons>

### Reference: Instructions Latencies for NVIDIA GPGPUs (arXiv)

Precise cycle counts per instruction.

**URL:** <https://arxiv.org/pdf/1905.08778v1>

---

## Tier 3: CUDA Ecosystem (Runtime → Driver → Libraries)

### Kunwar — Hardware-aware kernel design: CUDA, CUTLASS, Triton, TVM

Read first for the landscape overview. Maps every layer with real usage (vLLM → Triton, TensorRT → CUTLASS). 2025 snapshot.

**URL:** <https://www.kunwar.page/chapter/038-hardware-aware-kernel-design-cuda-cutlass-triton-tvm>

### Brendan Lynskey — CUDA Libraries & Ecosystem

Full stack map: Runtime API vs Driver API → cuBLAS/cuDNN → CUTLASS → NCCL → TensorRT. Explains when to use each and how cuBLAS/cuDNN are implemented on top of CUTLASS.

**URL:** <https://brendanjameslynskey.github.io/CUDA_09_Libraries/>

### GPU MODE (YouTube lecture series)

Community reading group (Andreas Kopf, Mark Saroufim). Covers PTX/SASS deep dives, CUTLASS (CuTe layout algebra), Triton internals, NCCL. Discussion format with practitioner insights.

**URL:** <https://www.youtube.com/playlist?list=PLhm9wZhiumoLuu6RG5xGZdh_O0CP9M5xh>

**Notes:** <https://christianjmills.com/series/notes/cuda-mode-notes.html>

---

## Suggested Reading Order

```
CS149 Lecture 7 + PMPP Ch.1-4          ─→ Architecture foundations
         ↓
Lynskey GPU Arch Series                 ─→ Warp/tensor-core deep dives
         ↓
Philip Fabianek PTX Intro              ─→ First contact with instructions
         ↓
Lynskey PTX/SASS talk                  ─→ Full pipeline understanding
         ↓
nvidia-sass-lessons                     ─→ Hands-on SASS reading
         ↓
Kunwar ecosystem overview               ─→ Map the ecosystem
         ↓
Lynskey CUDA 09 Libraries              ─→ Technical detail per library
         ↓
GPU MODE lectures                      ─→ Everything connected in practice
```
