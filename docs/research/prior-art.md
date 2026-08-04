# Prior Art Research

**Date:** 2026-08-04

This document captures the competitive landscape and technical research that shaped the `llm-server` product thesis.

## 1. The Graveyard (Why Cortex.cpp died)
The most cautionary tale in this space is `cortex.cpp` (archived Jul 2025 at 2.8k stars). It was built by a funded team to be exactly what we originally envisioned: an Ollama alternative wrapping `llama.cpp` with zero-config pulls and an OpenAI API. 

**Cause of death:** Absorbed by upstream. Upstream `ggml-org/llama.cpp` shipped a built-in router (`server-models.cpp`), unified the CLI (`llama serve -hf <repo>`), and added model switching, TTL unloads, and an HTTP API. The cortex.cpp maintainers archived their repo and told users to "contribute directly to llama.cpp". 

Projects that position as "local AI platforms" wrapping llama.cpp die (cortex.cpp, gpt4all, llama-gpt). Projects that position as narrow tools *riding* llama.cpp survive (llama-swap, llama-cpp-python, koboldcpp).

## 2. The Incumbents
- **Ollama:** Now a pure supervisor over `llama-server`. It dropped its native Go engine. It relies heavily on its own OCI-inspired blob storage and refuses to scan directories for loose `.gguf` files.
- **LocalAI (48k stars):** The heavyweight. It already does what we originally pitched: zero-config directory scanning, dual Ollama/OpenAI APIs, and GGUF metadata parsing. 
- **llama-swap (5k stars):** The lightweight proxy. However, the maintainer explicitly refuses to add auto-discovery or an Ollama API, preferring hand-written YAML configs per model.

## 3. The Technical Wedge: Empirical VRAM Tuning
The remaining defensible gap in the ecosystem is **hardware fit**. 

Static GGUF metadata is notoriously brittle for estimating VRAM requirements. Every new architecture (MoE, multimodal, MTP, sliding windows) breaks the static math. 

**Ollama's Retreat:** As of 0.32, Ollama deleted its static memory estimator (`llm/memory.go`). Instead, it relies on upstream llama.cpp's auto-fit, uses a crude file-size guess for placement, and **regex-scrapes llama-server's stdout** to catch OOM strings and retry with downgraded settings. 

**LocalAI's Approach:** Uses `gpustack/gguf-parser-go`, but wraps the estimator in panic handlers and applies a flat 20% VRAM headroom because the math cannot account for the compute buffer or allocator fragmentation.

**Our Thesis:** `llm-server` will not guess. It will use **Empirical Tuning**. On the first request for a model, the supervisor will spin up a test `llama-server` process, binary-search the boundary flags (e.g., `-ngl`), measure the actual allocation, and cache the result against a hardware fingerprint. 

We win by acknowledging the math is impossible and building an elegant supervisor loop to measure reality instead.
