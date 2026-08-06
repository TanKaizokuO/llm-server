package supervisor_test

import (
	"strings"
	"testing"

	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

// TestConventionalModelDirs_UsesExplicitRoots tests that all conventional
// locations are rooted in discoverable user configuration paths, and never
// hardcoded absolute paths, satisfying portability across platforms.
func TestConventionalModelDirs_UsesExplicitRoots(t *testing.T) {
	fakeHome := t.TempDir()
	fakeLocalApp := t.TempDir()
	fakeXdgCache := t.TempDir()
	fakeXdgData := t.TempDir()
	fakeHfHome := t.TempDir()

	t.Setenv("HOME", fakeHome)
	t.Setenv("LOCALAPPDATA", fakeLocalApp)
	t.Setenv("XDG_CACHE_HOME", fakeXdgCache)
	t.Setenv("XDG_DATA_HOME", fakeXdgData)
	t.Setenv("HF_HOME", fakeHfHome)

	dirs := supervisor.ConventionalModelDirs()
	if len(dirs) == 0 {
		t.Fatal("expected at least one conventional directory")
	}

	roots := []string{fakeHome, fakeLocalApp, fakeXdgCache, fakeXdgData, fakeHfHome}

	for _, d := range dirs {
		found := false
		for _, root := range roots {
			if strings.HasPrefix(d, root) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("conventional dir %q is not under any explicitly configured user root", d)
		}
	}
}

func TestConventionalModelDirs_XDGAndHFOverrides(t *testing.T) {
	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)

	// 1. With XDG_CACHE_HOME and HF_HOME unset, it falls back to $HOME/.cache
	t.Setenv("XDG_CACHE_HOME", "")
	t.Setenv("HF_HOME", "")
	dirs := supervisor.ConventionalModelDirs()
	foundHfCache := false
	for _, d := range dirs {
		if strings.Contains(d, "huggingface") && strings.HasPrefix(d, fakeHome+"/.cache") {
			foundHfCache = true
			break
		}
	}
	if !foundHfCache {
		t.Errorf("expected huggingface dir under $HOME/.cache when overrides are unset, got: %v", dirs)
	}

	// 2. With overrides set, they are honored
	fakeXdgCache := t.TempDir()
	fakeHfHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", fakeXdgCache)
	t.Setenv("HF_HOME", fakeHfHome)
	dirs = supervisor.ConventionalModelDirs()

	foundHfHome := false
	foundXdgLmStudio := false
	for _, d := range dirs {
		if strings.HasPrefix(d, fakeHfHome) {
			foundHfHome = true
		}
		if strings.HasPrefix(d, fakeXdgCache) && strings.Contains(d, "lm-studio") {
			foundXdgLmStudio = true
		}
	}
	if !foundHfHome {
		t.Errorf("expected huggingface dir under %q, got: %v", fakeHfHome, dirs)
	}
	if !foundXdgLmStudio {
		t.Errorf("expected lm-studio dir under XDG cache %q, got: %v", fakeXdgCache, dirs)
	}
}

func TestConventionalModelDirs_Deterministic(t *testing.T) {
	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)
	
	dirs1 := supervisor.ConventionalModelDirs()
	dirs2 := supervisor.ConventionalModelDirs()
	
	if len(dirs1) != len(dirs2) {
		t.Fatalf("length mismatch: %d != %d", len(dirs1), len(dirs2))
	}
	for i := range dirs1 {
		if dirs1[i] != dirs2[i] {
			t.Errorf("order mismatch at index %d: %q != %q", i, dirs1[i], dirs2[i])
		}
	}
	
	// Check for duplicates
	seen := make(map[string]bool)
	for _, d := range dirs1 {
		if seen[d] {
			t.Errorf("duplicate directory returned: %q", d)
		}
		seen[d] = true
	}
}
