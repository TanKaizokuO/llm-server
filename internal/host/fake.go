package host

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"time"
)

// FakeHost implements Host for unit and integration testing.
type FakeHost struct {
	mu           sync.Mutex
	vramBudget   int64
	ramBudget    int64
	layerCost    int64
	fingerprint  string
	accelerators []Accelerator
	launches     [][]string
	instances    []Instance
	onLaunch     func(argv []string) (http.Handler, error)
}

// NewFakeHost constructs a FakeHost with synthetic hardware budget defaults.
func NewFakeHost() *FakeHost {
	f := &FakeHost{
		vramBudget:  24 * 1024 * 1024 * 1024, // 24 GiB VRAM default
		ramBudget:   64 * 1024 * 1024 * 1024, // 64 GiB RAM default
		layerCost:   500 * 1024 * 1024,       // 500 MiB per layer default
		fingerprint: "fake-host-fingerprint",
	}
	f.accelerators = []Accelerator{
		{ID: "gpu:0", Name: "Fake GPU 0", TotalMemory: f.vramBudget},
	}
	return f
}

// Accelerators returns the configured list of fake compute devices.
func (f *FakeHost) Accelerators() []Accelerator {
	f.mu.Lock()
	defer f.mu.Unlock()
	cp := make([]Accelerator, len(f.accelerators))
	copy(cp, f.accelerators)
	return cp
}

// SetAccelerators sets a custom list of fake compute devices.
func (f *FakeHost) SetAccelerators(accs []Accelerator) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.accelerators = make([]Accelerator, len(accs))
	copy(f.accelerators, accs)
}

// SetBudget configures the synthetic hardware budget and per-layer cost.
func (f *FakeHost) SetBudget(vram, ram, layerCost int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.vramBudget = vram
	f.ramBudget = ram
	f.layerCost = layerCost
	if len(f.accelerators) == 1 && f.accelerators[0].ID == "gpu:0" {
		f.accelerators[0].TotalMemory = vram
	}
}

// VRAMBudget returns the synthetic VRAM budget.
func (f *FakeHost) VRAMBudget() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.vramBudget
}

// RAMBudget returns the synthetic RAM budget.
func (f *FakeHost) RAMBudget() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.ramBudget
}

// LayerCost returns the synthetic per-layer cost.
func (f *FakeHost) LayerCost() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.layerCost
}

// Fingerprint returns the configured fake hardware fingerprint.
func (f *FakeHost) Fingerprint() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.fingerprint
}

// SetFingerprint sets a custom fingerprint for the fake host.
func (f *FakeHost) SetFingerprint(fp string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fingerprint = fp
}

// SetOnLaunch overrides how launched fake instances respond to HTTP requests.
func (f *FakeHost) SetOnLaunch(fn func(argv []string) (http.Handler, error)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.onLaunch = fn
}

// Launches returns deep copies of all argv slices passed to Launch.
func (f *FakeHost) Launches() [][]string {
	f.mu.Lock()
	defer f.mu.Unlock()
	result := make([][]string, len(f.launches))
	for i, l := range f.launches {
		cp := make([]string, len(l))
		copy(cp, l)
		result[i] = cp
	}
	return result
}

// Instances returns all currently active and stopped fake instances.
func (f *FakeHost) Instances() []Instance {
	f.mu.Lock()
	defer f.mu.Unlock()
	cp := make([]Instance, len(f.instances))
	copy(cp, f.instances)
	return cp
}

// LastLaunch returns a deep copy of the most recent argv passed to Launch, or nil if none.
func (f *FakeHost) LastLaunch() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.launches) == 0 {
		return nil
	}
	l := f.launches[len(f.launches)-1]
	cp := make([]string, len(l))
	copy(cp, l)
	return cp
}

// Launch starts an in-process httptest.Server representing a llama-server Instance.
func (f *FakeHost) Launch(ctx context.Context, argv []string) (Instance, error) {
	f.mu.Lock()
	cp := make([]string, len(argv))
	copy(cp, argv)
	f.launches = append(f.launches, cp)
	onLaunch := f.onLaunch
	f.mu.Unlock()

	var handler http.Handler
	if onLaunch != nil {
		h, err := onLaunch(argv)
		if err != nil {
			return nil, err
		}
		handler = h
	} else {
		handler = http.HandlerFunc(DefaultMockHandler)
	}

	server := httptest.NewServer(handler)
	parsedURL, err := url.Parse(server.URL)
	if err != nil {
		server.Close()
		return nil, fmt.Errorf("parsing fake server url: %w", err)
	}

	done := make(chan struct{})
	inst := &fakeInstance{
		server: server,
		url:    parsedURL,
		done:   done,
	}

	f.mu.Lock()
	f.instances = append(f.instances, inst)
	f.mu.Unlock()

	return inst, nil
}

// DefaultMockHandler simulates llama-server HTTP endpoints.
func DefaultMockHandler(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/health":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	case "/v1/chat/completions":
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming unsupported!", http.StatusInternalServerError)
			return
		}

		chunks := []string{
			`data: {"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1700000000,"model":"fake","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1700000000,"model":"fake","choices":[{"index":0,"delta":{"content":" world!"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1700000000,"model":"fake","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
			`data: [DONE]` + "\n\n",
		}

		for _, chunk := range chunks {
			select {
			case <-r.Context().Done():
				return
			default:
				_, _ = w.Write([]byte(chunk))
				flusher.Flush()
				time.Sleep(5 * time.Millisecond)
			}
		}
	case "/v1/completions":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"cmpl-fake","object":"text_completion","created":1700000000,"model":"fake","choices":[{"text":"Hello world!","index":0,"finish_reason":"stop"}]}`))
	case "/v1/embeddings":
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],"model":"fake","usage":{"prompt_tokens":2,"total_tokens":2}}`))
	default:
		http.NotFound(w, r)
	}
}

type fakeInstance struct {
	server *httptest.Server
	url    *url.URL
	done   chan struct{}
	once   sync.Once
	err    error
}

func (fi *fakeInstance) WaitReady(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}

func (fi *fakeInstance) URL() *url.URL {
	return fi.url
}

func (fi *fakeInstance) ObservedAllocation() Allocation {
	return Allocation{
		VRAM: 1024 * 1024 * 1024,
		RAM:  512 * 1024 * 1024,
	}
}

func (fi *fakeInstance) Stop(ctx context.Context) error {
	fi.once.Do(func() {
		fi.server.Close()
		close(fi.done)
	})
	return nil
}

func (fi *fakeInstance) Done() <-chan struct{} {
	return fi.done
}

func (fi *fakeInstance) Err() error {
	return fi.err
}
