package supervisor

import (
	"encoding/json"
	"time"
)

// ExportParseKeepAlive exports parseKeepAlive for package supervisor_test.
func ExportParseKeepAlive(raw json.RawMessage) (time.Duration, bool) {
	return parseKeepAlive(raw)
}
