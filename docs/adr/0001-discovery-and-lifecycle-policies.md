# 1. Discovery and Lifecycle Policies

Date: 2026-08-06

## Status

Accepted

## Context

During the code review of the "Override without enumerating" feature (#19), several ambiguities and edge cases were discovered in how the Supervisor discovers models, manages its conventional directory scanning, and handles model lifecycles (specifically when models are removed). We needed to standardize these behaviors to ensure predictability and prevent orphaned resources.

## Decisions

### 1. Conventional Directory Scan Policy
**Decision**: Conventional directories (HuggingFace, LM Studio, GPT4All, `llama.cpp`) are scanned **only if no directories are explicitly given on the command line**.
**Rationale**: This aligns with the documented behavior in the README. Appending them unconditionally would surprise operators who intentionally restrict the Supervisor to a specific directory.

### 2. Ollama Blob Store Support
**Decision**: Skip scanning the `~/.ollama/models/blobs` directory.
**Rationale**: Ollama stores its blobs extensionless and content-addressed (`sha256-...`). Scanning them with relaxed extension filtering produces models named after hashes, which breaks the `name:tag` identity mapping the Ollama surface depends on. Proper support requires parsing Ollama manifests, which is deferred to a separate feature issue.

### 3. Orphaned Instances Lifecycle
**Decision**: When a model is deleted from disk and discovered missing during a `Rescan`, its resident instance is immediately drain-stopped (`Evict`).
**Rationale**: Failing to evict the instance results in ghost processes that consume VRAM indefinitely but are unroutable and invisible to `/api/tags` and `/api/ps`.

### 4. Transient-Empty Heuristic
**Decision**: Remove the heuristic that preserved the old model list if a scan returned exactly zero models.
**Rationale**: The daemon legitimately needs to support a zero-model state (e.g., an empty directory at startup, or all models being deleted). The heuristic prevented the daemon from ever gracefully transitioning to an empty state. Transient mount failures are an operator infrastructure problem, not a daemon responsibility.

### 5. TTL Map Separation
**Decision**: Maintain separate maps for runtime TTL overrides (`modelTTLs`) and config file overrides (`configTTL`), abandoning a proposed map merge.
**Rationale**: Merging the maps would change precedence from *source-based* (any runtime override beats any config override) to *key-based* (an ID match from config beats a Path match from runtime). Strict source precedence is the correct semantic behavior.