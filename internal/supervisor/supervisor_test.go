package supervisor_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

// newTestServer starts the Supervisor's real router on an in-process test
// server. Every test in this package observes the Supervisor the way a client
// does: over HTTP, through the router that main wires up. Nothing reaches into
// Supervisor state.
func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(supervisor.New().Handler())
	t.Cleanup(srv.Close)
	return srv
}

func TestHealthReportsReadyWithNoModelLoaded(t *testing.T) {
	srv := newTestServer(t)

	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	var got struct {
		Status string `json:"status"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if got.Status != "ready" {
		t.Errorf("status field = %q, want %q", got.Status, "ready")
	}
}
