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

**In active development.** Tickets #3, #4, #5, #6, #7, and #8 (Supervisor bootstrap, GGUF metadata reader, model discovery, Model resolution, and real Host accelerator enumeration/fingerprinting/supervision) are complete. Empirical tuning and completion API surfaces are in progress across tracked issues (#9–#19).

## Build and run

```sh
CGO_ENABLED=0 go build -o llm-server ./cmd/llm-server
./llm-server -addr 127.0.0.1:11434 /path/to/models
```

- `GET /health` reports readiness of the daemon itself.
- `GET /api/tags` lists discovered Models in Ollama format.
- `GET /v1/models` lists discovered Models in OpenAI format.
- `llm-server inspect [-json] <file.gguf>` displays GGUF metadata header fields.

## Tests

```sh
go test -race ./...
```

### Real-Hardware Integration Tests

Real-hardware tests in `internal/host/real_integration_test.go` exercise the real `Host` boundary against a physical GPU and a live `llama-server` process. They are build-tagged with `//go:build integration` and excluded from default test runs. This exclusion is a deliberate gap: unit test suites rely on `FakeHost` or helper subprocess mocks for fast, deterministic CI execution, while physical hardware suites are executed explicitly on GPU test nodes:

```sh
TEST_MODEL_PATH=/path/to/model.gguf go test -tags integration -race -v ./internal/host/...
```
