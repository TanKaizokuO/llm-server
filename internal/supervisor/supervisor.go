// Package supervisor implements the llm-server Supervisor: the daemon that
// discovers Models, supervises llama-server Instances, and proxies the Ollama
// and OpenAI HTTP surfaces onto them.
//
// The Supervisor never performs inference. It links no inference engine and
// executes no model graph; every generation is proxied to a child
// llama-server Instance. That constraint is what keeps this a narrow tool
// rather than a platform, and it is not negotiable.
package supervisor

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Supervisor is the daemon's root object. It owns Model discovery, the HTTP surface,
// and, in later work, the Instance registry and the Tuning cache.
type Supervisor struct {
	mu         sync.RWMutex
	models     map[string]Model
	modelsList []Model
}

// New builds a Supervisor by scanning the configured directories for Models.
// It returns an error if no valid Models are found in any scan directory.
func New(dirs ...string) (*Supervisor, error) {
	models, err := discoverModels(dirs)
	if err != nil {
		return nil, err
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("no models found in scanned directories: %v", dirs)
	}

	modelMap := make(map[string]Model, len(models))
	for _, m := range models {
		modelMap[m.ID] = m
	}

	return &Supervisor{
		models:     modelMap,
		modelsList: models,
	}, nil
}

// Handler returns the Supervisor's HTTP router. This is the single router the
// binary serves and the one tests drive; there is no separate test wiring.
func (s *Supervisor) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /api/tags", s.handleAPITags)
	mux.HandleFunc("GET /v1/models", s.handleV1Models)
	return mux
}

// handleHealth reports that the Supervisor process itself is up and able to
// accept requests. It is deliberately independent of any Model: a process
// supervisor managing llm-server needs to know that the daemon is alive, not
// that some Model happens to be resident.
func (s *Supervisor) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

type ollamaModelDetails struct {
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families,omitempty"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type ollamaModel struct {
	Name       string             `json:"name"`
	Model      string             `json:"model"`
	ModifiedAt string             `json:"modified_at"`
	Size       int64              `json:"size"`
	Digest     string             `json:"digest"`
	Details    ollamaModelDetails `json:"details"`
}

type ollamaTagsResponse struct {
	Models []ollamaModel `json:"models"`
}

func (s *Supervisor) handleAPITags(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	models := s.modelsList
	s.mu.RUnlock()

	list := make([]ollamaModel, 0, len(models))
	for _, m := range models {
		list = append(list, ollamaModel{
			Name:       m.ID,
			Model:      m.ID,
			ModifiedAt: m.ModTime.UTC().Format(time.RFC3339),
			Size:       m.Size,
			Digest:     m.Digest,
			Details: ollamaModelDetails{
				Format:            "gguf",
				Family:            m.Architecture,
				Families:          []string{m.Architecture},
				ParameterSize:     "",
				QuantizationLevel: m.Quantization,
			},
		})
	}

	writeJSON(w, http.StatusOK, ollamaTagsResponse{Models: list})
}

type openAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type openAIModelsResponse struct {
	Object string        `json:"object"`
	Data   []openAIModel `json:"data"`
}

func (s *Supervisor) handleV1Models(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	models := s.modelsList
	s.mu.RUnlock()

	list := make([]openAIModel, 0, len(models))
	for _, m := range models {
		list = append(list, openAIModel{
			ID:      m.ID,
			Object:  "model",
			Created: m.ModTime.Unix(),
			OwnedBy: "llm-server",
		})
	}

	writeJSON(w, http.StatusOK, openAIModelsResponse{
		Object: "list",
		Data:   list,
	})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
