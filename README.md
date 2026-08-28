# llm-server

A lightweight Supervisor for local LLMs: point it at a folder of GGUF files and
it will serve them over the Ollama and OpenAI HTTP APIs, running `llama-server` as a
child process. It never performs inference itself.

What makes it different: it does not *estimate* how much VRAM a Model needs. It
**measures** — launching a throwaway Instance, bisecting the offload flags
against real failure, and caching the answer against a hardware Fingerprint.

See `CONTEXT.md` for the project's vocabulary and `docs/research/prior-art.md`
for why this shape was chosen.

## Architecture

This is a concurrency and process-supervision exercise as much as an LLM
wrapper. Concretely:

- **`llama-server` process supervision** — each Model is a child process the
  Supervisor starts, health-checks, and stops; `internal/host` abstracts the
  real OS boundary behind `Host`/`Instance` interfaces, with a `FakeHost` used
  for fast, deterministic unit tests.
- **`sync.Cond`-gated slot admission** — concurrent requests against a Model
  block on a `sync.Cond` until a slot frees up or the instance is draining,
  rather than busy-polling (`internal/supervisor/supervisor.go`).
- **Idle-TTL draining** — an instance with no active requests past its `ttl`
  is drained and the underlying process stopped, freeing VRAM without an
  explicit shutdown call.
- **Reverse proxy** — `net/http/httputil.ReverseProxy` forwards
  completion/embedding requests to the resident `llama-server` instance once
  it is admitted.
- **Race-detected tests** — `go test -race ./...` is the default suite; real
  GPU hardware tests are isolated behind an `integration` build tag.
- **ADRs** — architectural decisions (discovery and lifecycle policy) are
  recorded under `docs/adr/`.

## Status

**In active development.** Tickets #3–#19 (Supervisor bootstrap, GGUF metadata reader, model discovery, Model resolution, real Host supervision, Tuning convergence loop, Tuning result persistence, completion/embedding API surfaces, and configuration overrides) are complete.

## Build and run

```sh
CGO_ENABLED=0 go build -o llm-server ./cmd/llm-server
./llm-server -addr 127.0.0.1:11434 -tuning-cache tuning.json /path/to/models
```

With no directories given on the command line, the Supervisor still scans
the conventional cache and data locations other local-LLM tools use: HuggingFace,
LM Studio, GPT4All, and `llama.cpp`'s default cache. (It no longer scans the
current directory `.` by default — point it at nothing, and it discovers what
those tools downloaded; point it at `.` explicitly, and it scans your working
directory).

- `GET /health` reports readiness of the daemon itself.
- `GET /api/tags` lists discovered Models in Ollama format.
- `GET /v1/models` lists discovered Models in OpenAI format.
- `GET /v1/tuning` (or `GET /api/tuning`) exposes tuning state and cache contents.
- `POST /v1/tuning/reset` (or `DELETE /v1/tuning`) forces re-measurement of a model (`?model=ref`) or all models.
- `POST /v1/rescan` (or `POST /api/rescan`) re-scans the configured directories on demand; a newly dropped GGUF also becomes servable automatically on the `-rescan-interval` timer, with no restart required either way.
- `llm-server inspect [-json] <file.gguf>` displays GGUF metadata header fields.

### Configuration file

Configuration exists to override, never to enumerate: every Model is served
from discovery alone, with or without a config file. Pass `-config
path/to/config.json` to override specific Models by ID (`name:tag`), bare
name, path, or filename:

```json
{
  "models": {
    "llama-3-8b-instruct:q4_k_m": {
      "ttl": "30m",
      "slots": 2,
      "argv": ["--flash-attn", "on"],
      "tuned": { "ctx_len": 8192, "offload": 33 }
    }
  }
}
```

- `ttl` overrides the idle TTL for that Model.
- `slots` overrides the Slot count for that Model.
- `argv` is appended to the launch command the Supervisor computes.
- `tuned` declares a **Pin**: it fixes the context length and offload verbatim,
  skipping Tuning entirely. A Pin is never written to the Tuning cache and is
  not scoped to a Fingerprint, so it is re-applied unchanged after a hardware
  change. (The key is spelled `tuned` for compatibility; see `CONTEXT.md`.)

**Validation:** The Supervisor fails at startup if:
- The config file is unreadable or contains malformed JSON.
- A `ttl` string cannot be parsed as a duration.
- `slots` is less than 1.
- `tuned` is present but missing either `ctx_len` or `offload`.
- `tuned.ctx_len` is 0.
- A Model's `argv` array contains any of the reserved flags (`-m`, `--model`, `-c`, `--ctx-size`, `-ngl`, `--n-gpu-layers`, `-np`, `--parallel`), in either the separate-value form (`--ctx-size 4096`) or the `=` form (`--ctx-size=4096`).

An entry keyed to a Model that does not exist on disk is ignored and is not an error.

## Tests

```sh
go test -race ./...
```

### Real-Hardware Integration Tests

Real-hardware tests in `internal/host/real_integration_test.go` exercise the real `Host` boundary against a physical GPU and a live `llama-server` process. They are build-tagged with `//go:build integration` and excluded from default test runs. This exclusion is a deliberate gap: unit test suites rely on `FakeHost` or helper subprocess mocks for fast, deterministic CI execution, while physical hardware suites are executed explicitly on GPU test nodes:

```sh
TEST_MODEL_PATH=/path/to/model.gguf go test -tags integration -race -v ./internal/host/...
```
