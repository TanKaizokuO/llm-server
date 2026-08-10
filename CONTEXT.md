# Domain Vocabulary: `llm-server`

This document defines the ubiquitous language for the `llm-server` project. Use these exact terms in code, issues, and documentation to avoid the terminology collisions common in this space.

## Core Concepts

- **Model**: A physical `.gguf` file on disk. It is *not* a running process, nor is it a configuration entry. Models are discovered by scanning directories.
- **Instance** (or **Runner**): A spawned `llama-server` subprocess executing a Model. 
- **Supervisor**: The `llm-server` Go daemon itself. It manages Instances, proxies the API, and handles discovery and tuning.
- **Slot**: A concurrent generation sequence within an Instance. This maps 1:1 to upstream `llama-server`'s `-np` (parallel slots) concept. It is the unit of concurrency.
- **Tuning**: The empirical, binary-search process of launching a Model in a test Instance to discover the optimal hardware flags (e.g., `-ngl`) before serving it.
- **Tunable Flags**: The context length and GPU offload pair (`-c` and `-ngl`) that Tuning searches over, a Pin overrides, and a Tuning result caches. It is only these two dimensions — arbitrary extra launch flags an operator appends are not Tunable Flags.
- **Fingerprint**: The hardware signature (GPU VRAM, System RAM) used as the cache key for Tuning results. A Model tuned on one Fingerprint does not need retuning unless the hardware changes.
- **Pin**: A set of Tunable Flags supplied by the operator and applied verbatim at every launch, bypassing Tuning. Unlike a Tuning result, a Pin is never cached and is not scoped to a Fingerprint, so it survives a hardware change unchanged. _Avoid_: calling a Pin "tuned" — a Pin is precisely the value that was **not** measured.
- **API Shim**: The translation layer in the Supervisor that converts Ollama `/api/*` requests into upstream OpenAI `/v1/*` or native `llama-server` HTTP requests.
