// Package supervisor implements the llm-server Supervisor: the daemon that
// discovers Models, supervises llama-server Instances, and proxies the Ollama
// and OpenAI HTTP surfaces onto them.
//
// The Supervisor never performs inference. It links no inference engine and
// executes no model graph; every generation is proxied to a child
// llama-server Instance. That constraint is what keeps this a narrow tool
// rather than a platform, and it is not negotiable.
package supervisor

import (
	"encoding/json"
	"net/http"
)

// Supervisor is the daemon's root object. It owns the HTTP surface and, in
// later work, Model discovery, the Instance registry, and the Tuning cache.
type Supervisor struct{}

// New builds a Supervisor.
func New() *Supervisor {
	return &Supervisor{}
}

// Handler returns the Supervisor's HTTP router. This is the single router the
// binary serves and the one tests drive; there is no separate test wiring.
func (s *Supervisor) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	return mux
}

// handleHealth reports that the Supervisor process itself is up and able to
// accept requests. It is deliberately independent of any Model: a process
// supervisor managing llm-server needs to know that the daemon is alive, not
// that some Model happens to be resident.
func (s *Supervisor) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
