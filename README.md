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

**In active development.** Ticket #3 (Supervisor binary bootstrap and health endpoint) is complete. Subprocess supervision, discovery, empirical tuning, and HTTP API surfaces are in progress across tracked issues (#4–#19).

## Build and run

```sh
CGO_ENABLED=0 go build -o llm-server ./cmd/llm-server
./llm-server -addr 127.0.0.1:11434
```

`GET /health` reports readiness of the daemon itself, independently of whether
any Model is loaded.

## Tests

```sh
go test ./...
```
