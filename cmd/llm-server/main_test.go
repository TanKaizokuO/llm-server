package main

import (
	"context"
	"os"
	"syscall"
	"testing"
	"time"
)

func TestRunGracefulShutdownOnSignal(t *testing.T) {
	errCh := make(chan error, 1)
	go func() {
		errCh <- run(context.Background(), "127.0.0.1:0")
	}()

	time.Sleep(50 * time.Millisecond)

	// Send real SIGINT signal to current process to test signal.NotifyContext
	if err := syscall.Kill(os.Getpid(), syscall.SIGINT); err != nil {
		t.Fatalf("sending SIGINT: %v", err)
	}

	select {
	case err := <-errCh:
		if err != nil {
			t.Errorf("run returned error on SIGINT: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("run did not exit within 5s of SIGINT")
	}
}

func TestRunFailsOnInvalidAddress(t *testing.T) {
	err := run(context.Background(), "invalid-address-format:99999999")
	if err == nil {
		t.Fatal("expected error on invalid address, got nil")
	}
}
