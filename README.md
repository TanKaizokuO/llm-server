# llm-server

A lightweight Supervisor for local LLMs: point it at a folder of GGUF files and
it will serve them over the Ollama and OpenAI HTTP APIs, running `llama-server` as a
child process. It never performs inference itself.

What makes it different: it does not *estimate* how much VRAM a Model needs. It
**measures** — launching a throwaway Instance, bisecting the offload flags
against real failure, and caching the answer against a hardware Fingerprint.

See `CONTEXT.md` for the project's vocabulary and `docs/research/prior-art.md`
for why this shape was chosen.

## Status

**In active development.** Tickets #3–#19 (Supervisor bootstrap, GGUF metadata reader, model discovery, Model resolution, real Host supervision, Tuning convergence loop, Tuning result persistence, completion/embedding API surfaces, and configuration overrides) are complete.

## Build and run

```sh
CGO_ENABLED=0 go build -o llm-server ./cmd/llm-server
./llm-server -addr 127.0.0.1:11434 -tuning-cache tuning.json /path/to/models
```

With no directories given on the command line, the Supervisor still scans
the conventional cache and data locations other local-LLM tools use (e.g.
LM Studio's and GPT4All's model directories) — pointing it at nothing is a
valid way to run it if a supported tool already manages your GGUF files.

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
- `tuned` pins the context length and offload verbatim, skipping empirical
  measurement entirely; it is never written to the Tuning cache.

## Tests

```sh
go test -race ./...
```

### Real-Hardware Integration Tests

Real-hardware tests in `internal/host/real_integration_test.go` exercise the real `Host` boundary against a physical GPU and a live `llama-server` process. They are build-tagged with `//go:build integration` and excluded from default test runs. This exclusion is a deliberate gap: unit test suites rely on `FakeHost` or helper subprocess mocks for fast, deterministic CI execution, while physical hardware suites are executed explicitly on GPU test nodes:

```sh
TEST_MODEL_PATH=/path/to/model.gguf go test -tags integration -race -v ./internal/host/...
```
