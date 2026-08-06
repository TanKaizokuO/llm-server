package supervisor_test

import (
	"bytes"
	"encoding/json"
	"fmt"
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

// TestConfig_PartialTunedOverrideFailsStartup ensures a pin missing either
// field is rejected at load time rather than silently launching with the
// other flag at its zero value (CPU-only offload, or an unusable
// zero-length context) with no measurement left to correct it.
func TestConfig_PartialTunedOverrideFailsStartup(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "partial-pin-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"partial-pin-model:q4_k_m": {"tuned": {"ctx_len": 4096}}
		}
	}`)

	_, err := supervisor.NewWithOpts(host.NewFakeHost(), []string{tmpDir}, supervisor.WithConfigFile(configPath))
	if err == nil {
		t.Fatal("expected NewWithOpts to fail on a tuned override missing offload, got nil")
	}
	if !strings.Contains(err.Error(), "tuned requires both ctx_len and offload") {
		t.Errorf("error = %v, want containing 'tuned requires both ctx_len and offload'", err)
	}
}

// TestConfig_ZeroCtxLenTunedOverrideFailsStartup ensures a pin cannot set an
// unusable zero-length context, even when both fields are present.
func TestConfig_ZeroCtxLenTunedOverrideFailsStartup(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "zero-ctx-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"zero-ctx-model:q4_k_m": {"tuned": {"ctx_len": 0, "offload": 10}}
		}
	}`)

	_, err := supervisor.NewWithOpts(host.NewFakeHost(), []string{tmpDir}, supervisor.WithConfigFile(configPath))
	if err == nil {
		t.Fatal("expected NewWithOpts to fail on a zero ctx_len pin, got nil")
	}
	if !strings.Contains(err.Error(), "ctx_len must be positive") {
		t.Errorf("error = %v, want containing 'ctx_len must be positive'", err)
	}
}

// TestConfig_TunedOverrideAllowsZeroOffload ensures a legitimate CPU-only
// pin (offload explicitly 0, with both fields present) is accepted — only a
// missing field or a zero context length is rejected.
func TestConfig_TunedOverrideAllowsZeroOffload(t *testing.T) {
	tmpDir := t.TempDir()
	writeTestGGUF(t, tmpDir, "cpu-only-model.gguf", "llama", "Q4_K_M")
	configPath := writeTestConfig(t, tmpDir, "config.json", `{
		"models": {
			"cpu-only-model:q4_k_m": {"tuned": {"ctx_len": 2048, "offload": 0}}
		}
	}`)

	fakeHost := host.NewFakeHost()
	sup, err := supervisor.NewWithOpts(fakeHost, []string{tmpDir}, supervisor.WithConfigFile(configPath))
	if err != nil {
		t.Fatalf("NewWithOpts failed on a legitimate zero-offload pin: %v", err)
	}
	t.Cleanup(func() { _ = sup.Close() })

	body := []byte(`{"model":"cpu-only-model:q4_k_m","messages":[{"role":"user","content":"hi"}]}`)
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
	assertArgvFlag(t, launches[0], "-ngl", "0")
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

// TestConfig_ArgvCannotOverrideReservedFlags ensures model config argv cannot
// contain flags managed by the supervisor (-m, -c, -ngl, -np).
func TestConfig_ArgvCannotOverrideReservedFlags(t *testing.T) {
	reservedFlags := []string{
		"-m", "-m=foo",
		"--model", "--model=foo",
		"-c", "-c=4096",
		"--ctx-size", "--ctx-size=4096",
		"-ngl", "-ngl=99",
		"--n-gpu-layers", "--n-gpu-layers=99",
		"-np", "-np=4",
		"--parallel", "--parallel=4",
	}
	for _, flag := range reservedFlags {
		t.Run(flag, func(t *testing.T) {
			tmpDir := t.TempDir()
			writeTestGGUF(t, tmpDir, "model.gguf", "llama", "Q4_K_M")
			configPath := writeTestConfig(t, tmpDir, "config.json", fmt.Sprintf(`{
				"models": {
					"model:q4_k_m": {"argv": ["%s"]}
				}
			}`, flag))

			_, err := supervisor.NewWithOpts(host.NewFakeHost(), []string{tmpDir}, supervisor.WithConfigFile(configPath))
			if err == nil {
				t.Fatalf("expected NewWithOpts to fail when argv contains reserved flag %s, got nil error", flag)
			}
		})
	}
}

// TestConfig_ArgvAllowedFlags ensures similarly-named but unreserved flags
// (e.g. --model-draft, --parallel-slots) are accepted and reach the launch argv.
func TestConfig_ArgvAllowedFlags(t *testing.T) {
	allowedFlags := []string{
		"--model-draft",
		"--model-url",
		"--models",
		"--parallel-slots",
		"--flash-attn",
	}
	for _, flag := range allowedFlags {
		t.Run(flag, func(t *testing.T) {
			tmpDir := t.TempDir()
			writeTestGGUF(t, tmpDir, "model.gguf", "llama", "Q4_K_M")
			configPath := writeTestConfig(t, tmpDir, "config.json", fmt.Sprintf(`{
				"models": {
					"model:q4_k_m": {"argv": ["%s", "value"]}
				}
			}`, flag))

			fake := host.NewFakeHost()
			sup, err := supervisor.NewWithOpts(fake, []string{tmpDir}, supervisor.WithConfigFile(configPath))
			if err != nil {
				t.Fatalf("expected NewWithOpts to succeed with allowed flag %s, got err: %v", flag, err)
			}
			t.Cleanup(func() { _ = sup.Close() })
			
			body := []byte(`{"model":"model:q4_k_m","messages":[{"role":"user","content":"hi"}]}`)
			req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
			rr := httptest.NewRecorder()
			sup.Handler().ServeHTTP(rr, req)

			if rr.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200. body: %s", rr.Code, rr.Body.String())
			}

			launches := fake.Launches()
			if len(launches) == 0 {
				t.Fatal("expected at least one launch")
			}
			
			found := false
			for _, argv := range launches {
				for _, a := range argv {
					if a == flag {
						found = true
						break
					}
				}
			}
			if !found {
				t.Errorf("expected flag %q to be in launched argv. Launches: %v", flag, launches)
			}
		})
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
