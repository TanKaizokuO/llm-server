package supervisor_test

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

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

func TestServerServesOnConfiguredAddressAndReleasesItOnShutdown(t *testing.T) {
	srv, err := supervisor.Listen("127.0.0.1:0", supervisor.New().Handler())
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	addr := srv.Addr()

	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- srv.Serve(ctx) }()

	resp, err := http.Get("http://" + addr + "/health")
	if err != nil {
		t.Fatalf("GET /health on %s: %v", addr, err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	cancel()
	select {
	case err := <-served:
		if err != nil {
			t.Errorf("Serve returned %v, want nil after graceful shutdown", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Serve did not return within 5s of context cancellation")
	}

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatalf("address %s still held after shutdown: %v", addr, err)
	}
	ln.Close()
}

func TestListenReportsAnUnusableAddress(t *testing.T) {
	held, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserving address: %v", err)
	}
	defer held.Close()

	if _, err := supervisor.Listen(held.Addr().String(), supervisor.New().Handler()); err == nil {
		t.Fatal("Listen on an occupied address returned no error")
	}
}
