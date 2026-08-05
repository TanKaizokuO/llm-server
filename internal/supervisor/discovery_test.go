package supervisor_test

import (
	"strings"
	"testing"

	"github.com/TanKaizokuO/llm-server/internal/supervisor"
)

// TestConventionalModelDirs_UsesHomeDir covers: "With no configuration file
// the Supervisor scans conventional cache and data locations plus any
// directories given on the command line." Every returned directory must sit
// under the operator's home directory, since that's what makes a "zero
// config" scan possible without knowing anything about the machine.
func TestConventionalModelDirs_UsesHomeDir(t *testing.T) {
	fakeHome := t.TempDir()
	t.Setenv("HOME", fakeHome)

	dirs := supervisor.ConventionalModelDirs()
	if len(dirs) == 0 {
		t.Fatal("expected at least one conventional directory")
	}
	for _, d := range dirs {
		if !strings.HasPrefix(d, fakeHome) {
			t.Errorf("conventional dir %q is not under HOME %q", d, fakeHome)
		}
	}
}
