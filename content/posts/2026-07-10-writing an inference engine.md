+++
title = "Building an LLM Inference Engine from Scratch"
author = ["Harisudarsan"]
tags = ["inference", "llm", "vllm", "sglang", "tokenizer", "odin", "systems-programming"]
draft = false
+++

## Building an LLM Inference Engine from Scratch {#building-an-llm-inference-engine-from-scratch}

I recently started exploring machine learning and AI from a systems engineering perspective. While most people begin by focusing on model training, I found myself much more interested in the inference side of the stack.

I wanted to understand how deep learning models and neural networks work under the hood, but what fascinated me even more were the engineering breakthroughs that made modern LLM serving possible. Everywhere I looked, people were discussing innovations such as PagedAttention in vLLM, RadixAttention in SGLang, continuous batching, speculative decoding, fused attention kernels, and distributed inference.

As someone who enjoys optimizing software and understanding performance bottlenecks, inference quickly became the most interesting area for me to explore.

Rather than simply reading papers and blog posts, I decided to learn by building. My goal is to write an inference engine from scratch and gradually implement the optimizations used by modern serving systems one by one.


### Choosing the Right Language {#choosing-the-right-language}

Before writing any code, I spent a considerable amount of time deciding which language to use.

My default choice was Go. I've been working with Go for a long time, and I also have some experience with Rust. In fact, my first experiment was a simple CPU-based inference engine written in Go. It was heavily inspired by Andrej Karpathy's llama2.c project and was primarily intended as a learning exercise for understanding transformers and the inference pipeline.

The implementation worked, but it was intentionally simple. There was no scheduler, no KV cache management, no batching optimizations, and no serving layer. It was essentially a static inference engine that could run a model and generate tokens.

As I continued learning about modern inference systems, I realized that the host-side logic is often just as important as the GPU kernels themselves. Scheduling requests, managing KV caches, handling batching, and minimizing memory overhead are all critical components of a high-performance inference engine.

This led me to explore different programming languages.

Initially, Python seemed like the obvious choice. Most of the AI ecosystem is built around Python, and frameworks like Triton make writing custom GPU kernels significantly easier. However, I generally prefer statically typed languages for large systems projects, and I wasn't particularly excited about building a long-term project around Python.

For a while, OCaml became my primary candidate.

I was impressed by its type system, which allowed me to express scheduling and batching constraints elegantly. The language also provides extremely efficient FFI support, making integration with CUDA libraries feasible. I enjoyed the concurrency model provided by Eio, which offers the flexibility of lightweight fibers while still allowing the use of operating system threads when necessary.

I spent some time learning OCaml and even started implementing parts of the project.

Then, during one of my late-night rabbit holes researching programming languages, I came across several newer systems languages including Nim, Odin, Jai, and C3. I spent a few weeks experimenting with all of them.

Eventually, I settled on Odin.

The language immediately felt familiar because of its similarities to Go, but it also provided a number of features that aligned well with the kind of system I wanted to build.

Some of the reasons I chose Odin were:

1.  Familiar Go-like syntax and package system.
2.  Strong emphasis on data-oriented programming.
3.  Excellent interoperability with C and C++.
4.  Custom memory allocators and explicit memory management.
5.  Native SIMD support and numerical primitives.
6.  Vendor libraries for Metal and other graphics APIs.
7.  A relatively stable ecosystem with a clear roadmap.
8.  Language design choices that I personally appreciate, such as the absence of macros, traits, and a complex dependency management system.

However, the most important factor was its focus on data-oriented design.


### Why Data-Oriented Design Matters for Inference {#why-data-oriented-design-matters-for-inference}

At a high level, an inference engine performs a surprisingly simple task.

It accepts text, tokenizes it, schedules work, performs model computation, predicts the next token, and streams the generated response back to the user.

The complexity comes from doing this efficiently for many users simultaneously.

Once I started studying inference systems more closely, I realized that many performance bottlenecks are fundamentally memory problems rather than compute problems.

Modern CPUs are incredibly fast. The challenge is keeping them supplied with data efficiently.

To maximize throughput, we want:

-   Minimal memory copies.
-   Cache-friendly data layouts.
-   Reduced pointer chasing.
-   Predictable memory access patterns.
-   Efficient SIMD utilization.
-   Reduced allocation overhead.

Many of these concerns apply to both the CPU and GPU sides of the inference stack.

This is where data-oriented design becomes valuable.

Languages such as C, Zig, and Odin give developers direct control over memory layout and allocation strategies. Odin stood out because it combines low-level control with language features specifically designed around data-oriented programming.

Some of the features that attracted me were:

-   Native Structure of Arrays (SoA) support.
-   SIMD-friendly abstractions.
-   Custom allocators.
-   Built-in matrix and numerical types.

Interestingly, Odin was originally designed with game development in mind, but many of the same principles apply to inference systems. Both domains involve processing large amounts of data while spending significant effort optimizing memory access patterns and minimizing unnecessary work.


### Talk Is Cheap {#talk-is-cheap}

Everything I've discussed so far is mostly theory.

The real question is whether these ideas actually translate into measurable performance gains.

As an experiment, I built a tokenizer in Odin and benchmarked it against Hugging Face Tokenizers and Fastoken. To my surprise, the implementation performed significantly better on my benchmark workload.


#### Single-Threaded Benchmarks {#single-threaded-benchmarks}

| Implementation          | Tokens    | ns/token | Throughput | Relative Performance |
|-------------------------|-----------|----------|------------|----------------------|
| Odin                    | 2,020,000 | 305.07   | 14.64 MB/s | Baseline             |
| Hugging Face Tokenizers | 2,020,000 | 619.77   | 7.20 MB/s  | 2.03× slower         |
| Fastoken                | 2,020,000 | 824.00   | 5.42 MB/s  | 2.70× slower         |


#### Multi-Threaded Benchmarks {#multi-threaded-benchmarks}

| Implementation                                  | Tokens     | ns/token | Throughput  | Relative Performance |
|-------------------------------------------------|------------|----------|-------------|----------------------|
| Odin (12-thread, no sharing)                    | 60,600,000 | 36.42    | 122.62 MB/s | Baseline             |
| Hugging Face Tokenizers (12-thread, no sharing) | 60,600,000 | 89.93    | 49.65 MB/s  | 2.47× slower         |
| Fastoken (12-thread Rayon batch)                | 60,600,000 | 164.98   | 27.07 MB/s  | 4.53× slower         |

For the full benchmark report, see
[Benchmark Report](https://github.com/harisudarsan1/odin_tokenizer/blob/main/docs/public-benchmark.md).

The complete source code is available in the
[GitHub repository](https://github.com/harisudarsan1/odin_tokenizer).

I genuinely did not expect a custom tokenizer written as a side experiment to outperform established implementations.

What makes the result particularly interesting is that the tokenizer was never intended to be the final goal. It was simply the first step toward building a complete inference engine.

Many of the optimizations came from applying systems programming principles:

-   Aggressive use of Structure of Arrays (SoA).
-   SIMD acceleration.
-   Avoiding pointer chasing.
-   Keeping data contiguous in memory.
-   Maximizing cache locality.
-   Reducing unnecessary allocations.

In hindsight, these are exactly the kinds of optimizations that data-oriented programming encourages.

That said, benchmark numbers alone are not enough. In a future post, I will discuss the benchmark setup in detail, including the dataset, vocabulary, hardware, implementation details, and the methodology used to ensure a fair comparison.


### Roadmap {#roadmap}

The tokenizer is only the first component of this project.

My long-term goal is to build a complete inference engine from scratch and progressively implement the techniques used by modern systems such as vLLM, SGLang, and TensorRT-LLM.

Some of the topics I plan to explore include:

-   Tokenization pipelines.
-   Request scheduling.
-   Continuous batching.
-   KV cache management.
-   PagedAttention.
-   RadixAttention.
-   Speculative decoding.
-   CUDA kernels.
-   Distributed inference.
-   Memory management strategies.

This project started as a way to learn how modern LLM inference works. Along the way, it has evolved into a deep exploration of systems design, performance engineering, memory optimization, and data-oriented programming.

In the next post, I'll dive into the tokenizer architecture itself and explain the optimizations that enabled it to outperform the implementations I benchmarked.
