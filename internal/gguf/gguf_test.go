package gguf_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"testing"

	"github.com/TanKaizokuO/llm-server/internal/gguf"
)

func TestReadHeader_Valid(t *testing.T) {
	tests := []struct {
		name          string
		params        gguf.FixtureParams
		expectedArch  string
		expectedCtx   uint64
		expectedQuant string
	}{
		{
			name:          "Default Llama Q4_K_M",
			params:        gguf.FixtureParams{Architecture: "llama", ContextLength: 4096, Quantization: "Q4_K_M"},
			expectedArch:  "llama",
			expectedCtx:   4096,
			expectedQuant: "Q4_K_M",
		},
		{
			name:          "Qwen2 Q8_0",
			params:        gguf.FixtureParams{Architecture: "qwen2", ContextLength: 32768, Quantization: "Q8_0"},
			expectedArch:  "qwen2",
			expectedCtx:   32768,
			expectedQuant: "Q8_0",
		},
		{
			name:          "Gemma2 F16",
			params:        gguf.FixtureParams{Architecture: "gemma2", ContextLength: 8192, Quantization: "F16"},
			expectedArch:  "gemma2",
			expectedCtx:   8192,
			expectedQuant: "F16",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := gguf.CreateTestGGUF(tt.params)
			hdr, err := gguf.ReadHeader(bytes.NewReader(buf))
			if err != nil {
				t.Fatalf("unexpected error reading header: %v", err)
			}

			if hdr.Metadata.Architecture != tt.expectedArch {
				t.Errorf("expected architecture %q, got %q", tt.expectedArch, hdr.Metadata.Architecture)
			}
			if hdr.Metadata.ContextLength != tt.expectedCtx {
				t.Errorf("expected context length %d, got %d", tt.expectedCtx, hdr.Metadata.ContextLength)
			}
			if hdr.Metadata.Quantization != tt.expectedQuant {
				t.Errorf("expected quantization %q, got %q", tt.expectedQuant, hdr.Metadata.Quantization)
			}
		})
	}
}

func TestReadHeader_InvalidMagic(t *testing.T) {
	invalidInputs := []struct {
		name string
		data []byte
	}{
		{"Empty", []byte{}},
		{"Too Short", []byte{'G', 'G'}},
		{"Wrong Magic", []byte{'G', 'G', 'J', 'T'}},
		{"PNG Magic", []byte{0x89, 'P', 'N', 'G'}},
	}

	for _, tt := range invalidInputs {
		t.Run(tt.name, func(t *testing.T) {
			_, err := gguf.ReadHeader(bytes.NewReader(tt.data))
			if err == nil {
				t.Fatal("expected error for invalid magic, got nil")
			}
			if !errors.Is(err, gguf.ErrInvalidMagic) && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Errorf("expected ErrInvalidMagic or EOF, got: %v", err)
			}
		})
	}
}

func TestReadHeader_UnsupportedVersion(t *testing.T) {
	buf := new(bytes.Buffer)
	buf.Write([]byte{'G', 'G', 'U', 'F'})
	_ = binary.Write(buf, binary.LittleEndian, uint32(99)) // unsupported version

	_, err := gguf.ReadHeader(buf)
	if err == nil {
		t.Fatal("expected error for unsupported version, got nil")
	}
	if !errors.Is(err, gguf.ErrUnsupportedVersion) {
		t.Errorf("expected ErrUnsupportedVersion, got: %v", err)
	}
}

func TestReadHeader_TruncatedInputs(t *testing.T) {
	validBuf := gguf.CreateTestGGUF(gguf.FixtureParams{
		Architecture:  "llama",
		ContextLength: 4096,
		Quantization:  "Q4_K_M",
	})

	for i := 0; i < len(validBuf)-1; i++ {
		truncated := validBuf[:i]
		_, err := gguf.ReadHeader(bytes.NewReader(truncated))
		if err == nil {
			t.Fatalf("expected error for truncated header at length %d, got nil", i)
		}
	}
}

func TestReadHeader_HeaderTooLarge(t *testing.T) {
	// Create header claiming key length of 1GB
	buf := new(bytes.Buffer)
	buf.Write([]byte{'G', 'G', 'U', 'F'})
	_ = binary.Write(buf, binary.LittleEndian, uint32(3))     // version 3
	_ = binary.Write(buf, binary.LittleEndian, uint64(0))     // tensor count
	_ = binary.Write(buf, binary.LittleEndian, uint64(1))     // 1 KV pair
	_ = binary.Write(buf, binary.LittleEndian, uint64(1<<30)) // 1GB key length

	_, err := gguf.ReadHeader(buf)
	if err == nil {
		t.Fatal("expected ErrHeaderTooLarge, got nil")
	}
	if !errors.Is(err, gguf.ErrHeaderTooLarge) {
		t.Errorf("expected ErrHeaderTooLarge, got: %v", err)
	}
}

func TestReadHeader_UnknownValueType(t *testing.T) {
	buf := new(bytes.Buffer)
	buf.Write([]byte{'G', 'G', 'U', 'F'})
	_ = binary.Write(buf, binary.LittleEndian, uint32(3)) // version 3
	_ = binary.Write(buf, binary.LittleEndian, uint64(0)) // tensor count
	_ = binary.Write(buf, binary.LittleEndian, uint64(1)) // 1 KV pair

	// Key: "test"
	key := "test"
	_ = binary.Write(buf, binary.LittleEndian, uint64(len(key)))
	buf.WriteString(key)
	// Invalid value type: 255
	_ = binary.Write(buf, binary.LittleEndian, uint32(255))

	_, err := gguf.ReadHeader(buf)
	if err == nil {
		t.Fatal("expected error for unknown value type, got nil")
	}
	if !errors.Is(err, gguf.ErrCorruptHeader) {
		t.Errorf("expected ErrCorruptHeader, got: %v", err)
	}
}

func TestReadHeader_DeeplyNestedArray(t *testing.T) {
	buf := new(bytes.Buffer)
	buf.Write([]byte{'G', 'G', 'U', 'F'})
	_ = binary.Write(buf, binary.LittleEndian, uint32(3)) // version 3
	_ = binary.Write(buf, binary.LittleEndian, uint64(0)) // tensor count
	_ = binary.Write(buf, binary.LittleEndian, uint64(1)) // 1 KV pair

	key := "deep"
	_ = binary.Write(buf, binary.LittleEndian, uint64(len(key)))
	buf.WriteString(key)

	// Nest arrays 10 levels deep (TypeArray = 9)
	for range 10 {
		_ = binary.Write(buf, binary.LittleEndian, uint32(gguf.TypeArray))
		_ = binary.Write(buf, binary.LittleEndian, uint32(gguf.TypeArray)) // elemType = TypeArray
		_ = binary.Write(buf, binary.LittleEndian, uint64(1))              // count = 1
	}

	_, err := gguf.ReadHeader(buf)
	if err == nil {
		t.Fatal("expected error for deeply nested array, got nil")
	}
	if !errors.Is(err, gguf.ErrCorruptHeader) {
		t.Errorf("expected ErrCorruptHeader, got: %v", err)
	}
}

func TestReadHeader_HeaderOnlyRead(t *testing.T) {
	hdrBuf := gguf.CreateTestGGUF(gguf.FixtureParams{
		Architecture:  "llama",
		ContextLength: 4096,
		Quantization:  "Q4_K_M",
	})
	hdrLen := len(hdrBuf)

	// Append 1MB of dummy tensor data
	dummyTensorData := make([]byte, 1024*1024)
	fullData := append(hdrBuf, dummyTensorData...)

	reader := bytes.NewReader(fullData)
	_, err := gguf.ReadHeader(reader)
	if err != nil {
		t.Fatalf("unexpected error reading header: %v", err)
	}

	bytesRead := int64(len(fullData)) - int64(reader.Len())
	if bytesRead != int64(hdrLen) {
		t.Errorf("expected to read exactly %d header bytes, but read %d bytes", hdrLen, bytesRead)
	}
}
