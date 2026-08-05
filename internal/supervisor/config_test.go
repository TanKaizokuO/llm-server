package supervisor_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/host"
	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

func writeTestConfig(t *testing.T, dir, filename, contents string) string {
	t.Helper()
	path := filepath.Join(dir, filename)
	if err := os.WriteFile(path, []byte(contents), 0644); err != nil {
		t.Fatalf("writing config file %s: %v", filename, err)
	}
	return path
}

// TestConfig_TunedOverridePinsFlagsAndSkipsMeasurement covers acceptance
// criteria: "A pinned tuned value bypasses measurement entirely and is used
// verbatim." A single Launch call with exactly the pinned flags proves no
// bisection ran, and the tuning cache stays empty since the pin is never
// persisted there.
func TestConfig_TunedOverridePinsFlagsAndSkipsMeasurement(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "pinned-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"pinned-model:q4_k_m": {"tuned": {"ctx_len": 3072, "offload": 12}}
		}
	}`)

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir}, supervisor.WithConfigFile(configPath))
	if err != nil {
		t.Fatalf("NewWithOpts failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	body := []byte(`{"model":"pinned-model:q4_k_m","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200. body: %s", rr.Code, rr.Body.String())
	}

	launches := fakeHost.Launches()
	if len(launches) != 1 {
		t.Fatalf("launches = %d, want 1 (pin must bypass measurement bisection): %v", len(launches), launches)
	}

	argv := launches[0]
	assertArgvFlag(t, argv, "-c", "3072")
	assertArgvFlag(t, argv, "-ngl", "12")

	tuningReq := httptest.NewRequest("GET", "/v1/tuning", nil)
	tuningRR := httptest.NewRecorder()
	sup.Handler().ServeHTTP(tuningRR, tuningReq)
	var tuningResp struct {
		Entries map[string]json.RawMessage `json:"entries"`
	}
	if err := json.Unmarshal(tuningRR.Body.Bytes(), &tuningResp); err != nil {
		t.Fatalf("decoding /v1/tuning response: %v", err)
	}
	if len(tuningResp.Entries) != 0 {
		t.Errorf("tuning cache entries = %v, want empty (pin must not be persisted to the cache)", tuningResp.Entries)
	}
}

// TestConfig_ArgvOverrideAppendsExtraFlags covers the argv override: extra
// flags an operator supplies are appended to every launch of that Model,
// including tuning probes.
func TestConfig_ArgvOverrideAppendsExtraFlags(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "argv-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"argv-model:q4_k_m": {"argv": ["--flash-attn", "on"]}
		}
	}`)

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir}, supervisor.WithConfigFile(configPath))
	if err != nil {
		t.Fatalf("NewWithOpts failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	body := []byte(`{"model":"argv-model:q4_k_m","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200. body: %s", rr.Code, rr.Body.String())
	}

	launches := fakeHost.Launches()
	if len(launches) == 0 {
		t.Fatal("expected at least one launch")
	}
	for _, argv := range launches {
		assertArgvFlag(t, argv, "--flash-attn", "on")
	}
}

// TestConfig_SlotsOverride covers the per-Model Slot count override,
// checking both the launched -np flag and the Supervisor's own occupancy
// accounting agree.
func TestConfig_SlotsOverride(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "slots-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"slots-model:q4_k_m": {"tuned": {"ctx_len": 2048, "offload": 10}, "slots": 4}
		}
	}`)

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir}, supervisor.WithConfigFile(configPath), supervisor.WithSlotsPerInstance(1))
	if err != nil {
		t.Fatalf("NewWithOpts failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	body := []byte(`{"model":"slots-model:q4_k_m","messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	sup.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200. body: %s", rr.Code, rr.Body.String())
	}

	launches := fakeHost.Launches()
	if len(launches) != 1 {
		t.Fatalf("launches = %d, want 1", len(launches))
	}
	assertArgvFlag(t, launches[0], "-np", "4")

	_, maxSlots, ok := sup.InstanceOccupancy("slots-model:q4_k_m")
	if !ok {
		t.Fatal("expected resident instance to be tracked")
	}
	if maxSlots != 4 {
		t.Errorf("InstanceOccupancy max = %d, want 4", maxSlots)
	}
}

// TestConfig_TTLOverride covers the per-Model TTL override.
func TestConfig_TTLOverride(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "ttl-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"ttl-model:q4_k_m": {"ttl": "42s"}
		}
	}`)

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir}, supervisor.WithConfigFile(configPath), supervisor.WithDefaultTTL(5*time.Minute))
	if err != nil {
		t.Fatalf("NewWithOpts failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	ttl, err := sup.GetModelTTL("ttl-model:q4_k_m")
	if err != nil {
		t.Fatalf("GetModelTTL failed: %v", err)
	}
	if ttl != 42*time.Second {
		t.Errorf("GetModelTTL = %v, want 42s", ttl)
	}
}

// TestConfig_InvalidTTLFailsStartup ensures a malformed config surfaces a
// clear startup error rather than silently falling back to the default.
func TestConfig_InvalidTTLFailsStartup(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "bad-ttl-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"bad-ttl-model:q4_k_m": {"ttl": "not-a-duration"}
		}
	}`)

	_, err := supervisor.NewWithOpts(host.NewFakeHost(), []string{tmpDir}, supervisor.WithConfigFile(configPath))
	if err == nil {
		t.Fatal("expected NewWithOpts to fail on invalid ttl, got nil")
	}
	if !strings.Contains(err.Error(), "invalid ttl") {
		t.Errorf("error = %v, want containing 'invalid ttl'", err)
	}
}

// TestConfig_MissingFileFailsStartup ensures an explicitly configured but
// unreadable config path is a startup error, distinct from the file simply
// being absent by default (which is the documented zero-config path).
func TestConfig_MissingFileFailsStartup(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "model.gguf", "llama", "Q4_K_M")

	_, err := supervisor.NewWithOpts(host.NewFakeHost(), []string{tmpDir}, supervisor.WithConfigFile(filepath.Join(tmpDir, "does-not-exist.json")))
	if err == nil {
		t.Fatal("expected NewWithOpts to fail when the configured file is missing, got nil")
	}
}

// TestConfig_NoModelRequiresConfigEntry ensures Models with no matching
// config entry serve normally with global defaults, so configuration
// remains strictly optional per Model.
func TestConfig_NoModelRequiresConfigEntry(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "unconfigured-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"some-other-model:q4_k_m": {"ttl": "1s"}
		}
	}`)

	sup, err := supervisor.NewWithOpts(host.NewFakeHost(), []string{tmpDir}, supervisor.WithConfigFile(configPath), supervisor.WithDefaultTTL(9*time.Minute))
	if err != nil {
		t.Fatalf("NewWithOpts failed: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	ttl, err := sup.GetModelTTL("unconfigured-model:q4_k_m")
	if err != nil {
		t.Fatalf("GetModelTTL failed: %v", err)
	}
	if ttl != 9*time.Minute {
		t.Errorf("GetModelTTL = %v, want default 9m", ttl)
	}
}

func assertArgvFlag(t *testing.T, argv []string, flag, value string) {
	t.Helper()
	for i, arg := range argv {
		if arg == flag && i+1 < len(argv) {
			if argv[i+1] != value {
				t.Errorf("argv flag %s = %s, want %s (argv: %v)", flag, argv[i+1], value, argv)
			}
			return
		}
	}
	t.Errorf("argv missing flag %s (argv: %v)", flag, argv)
}
