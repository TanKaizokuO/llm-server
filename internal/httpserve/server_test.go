package httpserve_test

import (
	"context"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/httpserve"
)

func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}

func TestServerServesOnConfiguredAddressAndReleasesItOnShutdown(t *testing.T) {
	srv, err := httpserve.Listen("127.0.0.1:0", okHandler())
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	addr := srv.Addr()

	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- srv.Serve(ctx) }()

	resp, err := http.Get("http://" + addr + "/")
	if err != nil {
		t.Fatalf("GET on %s: %v", addr, err)
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

func TestServeWaitsForAnInFlightRequestBeforeReleasingTheAddress(t *testing.T) {
	release := make(chan struct{})
	arrived := make(chan struct{})
	srv, err := httpserve.Listen("127.0.0.1:0", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(arrived)
		<-release
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("finished"))
	}))
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- srv.Serve(ctx) }()

	body := make(chan int, 1)
	go func() {
		resp, err := http.Get("http://" + srv.Addr() + "/")
		if err != nil {
			body <- 0
			return
		}
		defer resp.Body.Close()
		body <- resp.StatusCode
	}()

	<-arrived
	cancel() // shutdown begins while the request is still in the handler

	select {
	case <-served:
		t.Fatal("Serve returned before the in-flight request finished")
	case <-time.After(100 * time.Millisecond):
	}

	close(release)
	if got := <-body; got != http.StatusOK {
		t.Errorf("in-flight request got status %d, want %d", got, http.StatusOK)
	}
	select {
	case err := <-served:
		if err != nil {
			t.Errorf("Serve returned %v, want nil", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Serve did not return after the in-flight request finished")
	}
}

func TestListenReportsAnUnusableAddress(t *testing.T) {
	held, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserving address: %v", err)
	}
	defer held.Close()

	if _, err := httpserve.Listen(held.Addr().String(), okHandler()); err == nil {
		t.Fatal("Listen on an occupied address returned no error")
	}
}
