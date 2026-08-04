package supervisor

import (
	"bufio"
	"crypto/sha256"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/TanKaizokuO/llm-server/internal/gguf"
)

// Model represents a physical .gguf file discovered on disk.
type Model struct {
	ID            string    `json:"id"`             // Address format name:tag (e.g. "llama-3-8b-instruct:q4_k_m")
	Name          string    `json:"name"`           // Sanitised stem (e.g. "llama-3-8b-instruct")
	Tag           string    `json:"tag"`            // Quantisation derived from metadata (e.g. "q4_k_m")
	Path          string    `json:"path"`           // File system path to the model (or first shard)
	Size          int64     `json:"size"`           // File size in bytes
	ModTime       time.Time `json:"mod_time"`       // File modification time
	Digest        string    `json:"digest"`         // Pre-computed sha256 digest
	Architecture  string    `json:"architecture"`   // GGUF general.architecture
	ContextLength uint64    `json:"context_length"` // GGUF context length
	Quantization  string    `json:"quantization"`   // Raw GGUF quantization string (e.g. "Q4_K_M")
}

var (
	shardRegex1 = regexp.MustCompile(`(?i)^(.+?)[._-]([0-9]{5})-of-([0-9]{5})$`)
	shardRegex2 = regexp.MustCompile(`(?i)^(.+?)[._-]([0-9]{5})$`)
	shardRegex3 = regexp.MustCompile(`(?i)^(.+?)[._-]part-?([0-9]+)(?:-of-([0-9]+))?$`)

	genericQuantRegex = regexp.MustCompile(`(?i)[._-]([i]?q[0-9]_[a-z0-9_]+|q[0-9]_[0-9]|f16|f32)$`)
)

// discoverModels recursively scans the given directories for valid GGUF models.
// It skips projectors, corrupt/unparseable files, and non-first shards.
func discoverModels(dirs []string) ([]Model, error) {
	discovered := make(map[string]Model)

	for _, dir := range dirs {
		if dir == "" {
			continue
		}

		err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				slog.Warn("failed to access path during model scan", "path", path, "err", err)
				return nil
			}

			if d.IsDir() {
				return nil
			}

			if !strings.EqualFold(filepath.Ext(path), ".gguf") {
				return nil
			}

			model, ok, scanErr := inspectFile(path, d)
			if scanErr != nil {
				slog.Warn("skipping corrupt or unparseable GGUF file", "path", path, "err", scanErr)
				return nil
			}
			if !ok {
				return nil
			}

			if existing, exists := discovered[model.ID]; exists {
				slog.Warn("duplicate model ID discovered, keeping first", "id", model.ID, "existing", existing.Path, "duplicate", model.Path)
			} else {
				discovered[model.ID] = model
			}
			return nil
		})

		if err != nil {
			slog.Warn("failed walking directory", "dir", dir, "err", err)
		}
	}

	models := make([]Model, 0, len(discovered))
	for _, m := range discovered {
		models = append(models, m)
	}

	sort.Slice(models, func(i, j int) bool {
		return models[i].ID < models[j].ID
	})

	return models, nil
}

func inspectFile(path string, d fs.DirEntry) (Model, bool, error) {
	base := d.Name()
	baseLower := strings.ToLower(base)

	// 1. Filename-based Projector Filter
	if strings.Contains(baseLower, "mmproj") || strings.Contains(baseLower, "projector") {
		return Model{}, false, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return Model{}, false, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return Model{}, false, err
	}

	reader := bufio.NewReaderSize(f, 64*1024)
	hdr, err := gguf.ReadHeader(reader)
	if err != nil {
		return Model{}, false, err
	}

	// 2. Metadata-based Projector Filter
	archLower := strings.ToLower(hdr.Metadata.Architecture)
	if isProjectorArch(archLower, hdr.KV) {
		return Model{}, false, nil
	}

	// 3. Shard Filter
	isShard, shardNum, stemWithoutShard := parseShardInfo(base, hdr.KV)
	if isShard && shardNum > 1 {
		// Skip subsequent shards
		return Model{}, false, nil
	}

	// 4. Derive Name & Tag
	name, tag := deriveNameAndTag(stemWithoutShard, hdr.Metadata.Quantization)
	id := name + ":" + tag

	m := Model{
		ID:            id,
		Name:          name,
		Tag:           tag,
		Path:          path,
		Size:          info.Size(),
		ModTime:       info.ModTime(),
		Architecture:  hdr.Metadata.Architecture,
		ContextLength: hdr.Metadata.ContextLength,
		Quantization:  hdr.Metadata.Quantization,
	}
	m.Digest = computeDigest(m)

	return m, true, nil
}

func isProjectorArch(arch string, kv map[string]any) bool {
	switch arch {
	case "clip", "siglip", "mmproj", "projector", "adapter":
		return true
	}
	if strings.HasSuffix(arch, "-proj") {
		return true
	}
	if genType, ok := kv["general.type"].(string); ok {
		tLower := strings.ToLower(genType)
		if tLower == "projector" || tLower == "mmproj" || tLower == "adapter" {
			return true
		}
	}
	return false
}

func parseShardInfo(filename string, kv map[string]any) (bool, int, string) {
	stem := strings.TrimSuffix(filename, filepath.Ext(filename))

	var isShard bool
	var shardNum int
	cleanStem := stem

	// Try pattern matching on stem to extract cleanStem and shardNum
	if m := shardRegex1.FindStringSubmatch(stem); len(m) == 4 {
		shardNum, _ = strconv.Atoi(m[2])
		cleanStem = m[1]
		isShard = true
	} else if m := shardRegex3.FindStringSubmatch(stem); len(m) >= 3 {
		shardNum, _ = strconv.Atoi(m[2])
		cleanStem = m[1]
		isShard = true
	}

	// Check metadata KV
	metaShardNo := -1
	if val, ok := kv["split.no"]; ok {
		if no, ok := toInt(val); ok {
			metaShardNo = no + 1
		}
	} else if val, ok := kv["general.split.no"]; ok {
		if no, ok := toInt(val); ok {
			metaShardNo = no + 1
		}
	}

	if metaShardNo > 0 {
		isShard = true
		shardNum = metaShardNo
		// Also trim bare digits if pattern match didn't catch it
		if cleanStem == stem {
			if m := shardRegex2.FindStringSubmatch(stem); len(m) == 3 {
				cleanStem = m[1]
			}
		}
	}

	return isShard, shardNum, cleanStem
}

func deriveNameAndTag(stemWithoutShard string, metaQuant string) (string, string) {
	tag := strings.ToLower(metaQuant)
	name := stemWithoutShard

	// 1. Strip metaQuant from end of stem if present
	if metaQuant != "" && metaQuant != "unknown" {
		qLen := len(metaQuant)
		nLen := len(name)
		if nLen > qLen+1 {
			sep := name[nLen-qLen-1]
			if (sep == '-' || sep == '.' || sep == '_') && strings.EqualFold(name[nLen-qLen:], metaQuant) {
				name = name[:nLen-qLen-1]
			}
		}
	}

	// 2. Strip generic quant pattern if still present
	if loc := genericQuantRegex.FindStringIndex(name); loc != nil {
		if tag == "" || tag == "unknown" {
			// Extract tag from filename if metadata didn't provide one
			extractedTag := strings.TrimLeft(name[loc[0]:loc[1]], ".-_")
			if extractedTag != "" {
				tag = strings.ToLower(extractedTag)
			}
		}
		name = name[:loc[0]]
	}

	if tag == "" || tag == "unknown" {
		tag = "unknown"
	}

	name = strings.ToLower(name)
	name = strings.TrimRight(name, ".-_")
	if name == "" {
		name = "model"
	}

	return name, tag
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case uint8:
		return int(n), true
	case uint16:
		return int(n), true
	case uint32:
		return int(n), true
	case uint64:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	}
	return 0, false
}

func computeDigest(m Model) string {
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "%s:%s:%d", m.ID, m.Path, m.Size)
	return fmt.Sprintf("sha256:%x", h.Sum(nil))
}
