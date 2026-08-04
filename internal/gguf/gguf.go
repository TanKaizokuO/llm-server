package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

var (
	// ErrInvalidMagic is returned when magic bytes do not match "GGUF".
	ErrInvalidMagic = errors.New("invalid GGUF magic bytes")

	// ErrUnsupportedVersion is returned when GGUF version is not supported (only 1, 2, 3 supported).
	ErrUnsupportedVersion = errors.New("unsupported GGUF version")

	// ErrHeaderTooLarge is returned when string, array, or KV count exceeds security bounds.
	ErrHeaderTooLarge = errors.New("GGUF header field exceeds size limit")

	// ErrCorruptHeader is returned when GGUF header is malformed or truncated unexpectedly.
	ErrCorruptHeader = errors.New("corrupt GGUF header")
)

// Magic bytes for GGUF ("GGUF" in ASCII).
var Magic = [4]byte{'G', 'G', 'U', 'F'}

// GGUF Value Type constants.
const (
	TypeUint8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUint16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUint32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUint64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12
)

const (
	maxStringLength = 16 * 1024 * 1024 // 16 MB max string length
	maxArrayLength  = 1_000_000        // max array element count
	maxKVCount      = 100_000          // max key-value pair count
	maxArrayDepth   = 8                // max array nesting depth
)

var FileTypeNames = map[uint32]string{
	0:  "F32",
	1:  "F16",
	2:  "Q4_0",
	3:  "Q4_1",
	7:  "Q8_0",
	8:  "Q5_0",
	9:  "Q5_1",
	10: "Q2_K",
	11: "Q3_K_S",
	12: "Q3_K_M",
	13: "Q3_K_L",
	14: "Q4_K_S",
	15: "Q4_K_M",
	16: "Q5_K_S",
	17: "Q5_K_M",
	18: "Q6_K",
	19: "IQ2_XXS",
	20: "IQ2_XS",
	21: "Q2_K_S",
	22: "IQ3_XS",
	23: "IQ3_S",
	24: "IQ2_S",
	25: "IQ2_M",
	26: "IQ1_S",
	27: "IQ4_NL",
	28: "IQ3_M",
	29: "IQ4_XS",
	30: "IQ1_M",
	31: "BF16",
	32: "Q4_0_4_4",
	33: "Q4_0_4_8",
	34: "Q4_0_8_8",
	35: "TQ1_0",
	36: "TQ2_0",
}

// FileTypeNameToType converts a quantization string (e.g. "Q4_K_M") to its GGUF file_type uint32 enum value.
func FileTypeNameToType(name string) (uint32, bool) {
	nameUpper := strings.ToUpper(name)
	for code, strName := range FileTypeNames {
		if strings.ToUpper(strName) == nameUpper {
			return code, true
		}
	}
	return 0, false
}

// Metadata contains high-level attributes extracted from a GGUF header.
type Metadata struct {
	Architecture  string `json:"architecture"`
	ContextLength uint64 `json:"context_length"`
	Quantization  string `json:"quantization"`
}

// Header contains full GGUF header information including metadata and raw KV pairs.
type Header struct {
	Version     uint32         `json:"version"`
	TensorCount uint64         `json:"tensor_count"`
	KVCount     uint64         `json:"kv_count"`
	Metadata    Metadata       `json:"metadata"`
	KV          map[string]any `json:"kv"`
}

// ReadHeader parses the GGUF binary header from r without loading tensor payload.
func ReadHeader(r io.Reader) (*Header, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, fmt.Errorf("%w: %v", ErrInvalidMagic, err)
		}
		return nil, err
	}
	if magic != Magic {
		return nil, ErrInvalidMagic
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("%w: reading version: %v", ErrCorruptHeader, err)
	}

	if version != 1 && version != 2 && version != 3 {
		return nil, fmt.Errorf("%w: %d", ErrUnsupportedVersion, version)
	}

	var tensorCount, kvCount uint64
	if version == 1 {
		var tc, kvc uint32
		if err := binary.Read(r, binary.LittleEndian, &tc); err != nil {
			return nil, fmt.Errorf("%w: reading tensor count: %v", ErrCorruptHeader, err)
		}
		if err := binary.Read(r, binary.LittleEndian, &kvc); err != nil {
			return nil, fmt.Errorf("%w: reading KV count: %v", ErrCorruptHeader, err)
		}
		tensorCount, kvCount = uint64(tc), uint64(kvc)
	} else {
		if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
			return nil, fmt.Errorf("%w: reading tensor count: %v", ErrCorruptHeader, err)
		}
		if err := binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
			return nil, fmt.Errorf("%w: reading KV count: %v", ErrCorruptHeader, err)
		}
	}

	if kvCount > maxKVCount {
		return nil, fmt.Errorf("%w: KV count %d exceeds limit %d", ErrHeaderTooLarge, kvCount, maxKVCount)
	}

	kv := make(map[string]any, kvCount)
	v1 := version == 1

	for range int(kvCount) {
		key, err := readString(r, v1)
		if err != nil {
			if errors.Is(err, ErrHeaderTooLarge) {
				return nil, err
			}
			return nil, fmt.Errorf("%w: reading KV key: %v", ErrCorruptHeader, err)
		}

		var valType uint32
		if err := binary.Read(r, binary.LittleEndian, &valType); err != nil {
			return nil, fmt.Errorf("%w: reading KV value type for key %q: %v", ErrCorruptHeader, key, err)
		}

		val, err := readValue(r, valType, v1, 0)
		if err != nil {
			if errors.Is(err, ErrHeaderTooLarge) {
				return nil, err
			}
			return nil, fmt.Errorf("%w: reading KV value for key %q: %v", ErrCorruptHeader, key, err)
		}

		kv[key] = val
	}

	meta := extractMetadata(kv)

	return &Header{
		Version:     version,
		TensorCount: tensorCount,
		KVCount:     kvCount,
		Metadata:    meta,
		KV:          kv,
	}, nil
}

func readString(r io.Reader, v1 bool) (string, error) {
	var strLen uint64
	if v1 {
		var l uint32
		if err := binary.Read(r, binary.LittleEndian, &l); err != nil {
			return "", err
		}
		strLen = uint64(l)
	} else {
		if err := binary.Read(r, binary.LittleEndian, &strLen); err != nil {
			return "", err
		}
	}

	if strLen > maxStringLength {
		return "", fmt.Errorf("%w: string length %d exceeds limit", ErrHeaderTooLarge, strLen)
	}

	buf := make([]byte, strLen)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}

	return string(buf), nil
}

func readValue(r io.Reader, valType uint32, v1 bool, depth int) (any, error) {
	switch valType {
	case TypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeBool:
		var v uint8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return false, err
		}
		return v != 0, nil
	case TypeString:
		return readString(r, v1)
	case TypeArray:
		if depth >= maxArrayDepth {
			return nil, fmt.Errorf("%w: array nesting depth %d exceeds limit %d", ErrCorruptHeader, depth, maxArrayDepth)
		}
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, fmt.Errorf("%w: reading array element type: %v", ErrCorruptHeader, err)
		}

		var count uint64
		if v1 {
			var c uint32
			if err := binary.Read(r, binary.LittleEndian, &c); err != nil {
				return nil, fmt.Errorf("%w: reading array length: %v", ErrCorruptHeader, err)
			}
			count = uint64(c)
		} else {
			if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
				return nil, fmt.Errorf("%w: reading array length: %v", ErrCorruptHeader, err)
			}
		}

		if count > maxArrayLength {
			return nil, fmt.Errorf("%w: array count %d exceeds limit", ErrHeaderTooLarge, count)
		}

		arr := make([]any, count)
		for i := range int(count) {
			v, err := readValue(r, elemType, v1, depth+1)
			if err != nil {
				return nil, fmt.Errorf("%w: reading array element %d: %v", ErrCorruptHeader, i, err)
			}
			arr[i] = v
		}
		return arr, nil
	case TypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case TypeFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	default:
		return nil, fmt.Errorf("%w: unknown value type %d", ErrCorruptHeader, valType)
	}
}

func extractMetadata(kv map[string]any) Metadata {
	var meta Metadata

	// 1. Architecture
	if v, ok := kv["general.architecture"].(string); ok {
		meta.Architecture = v
	} else {
		meta.Architecture = "unknown"
	}

	// 2. Context Length
	ctxKeys := []string{
		meta.Architecture + ".context_length",
		"general.context_length",
		"llm.context_length",
	}

	for _, k := range ctxKeys {
		if val, ok := kv[k]; ok {
			if num, ok := toUint64(val); ok {
				meta.ContextLength = num
				break
			}
		}
	}

	// 3. Quantization
	if val, ok := kv["general.file_type"]; ok {
		if code, ok := toUint64(val); ok {
			if name, exists := FileTypeNames[uint32(code)]; exists {
				meta.Quantization = name
			} else {
				meta.Quantization = fmt.Sprintf("TYPE_%d", code)
			}
		} else if str, ok := val.(string); ok {
			meta.Quantization = str
		}
	}

	if meta.Quantization == "" {
		if val, ok := kv["general.quantization"].(string); ok {
			meta.Quantization = val
		} else {
			meta.Quantization = "unknown"
		}
	}

	return meta
}

func toUint64(v any) (uint64, bool) {
	switch n := v.(type) {
	case uint64:
		return n, true
	case uint32:
		return uint64(n), true
	case uint16:
		return uint64(n), true
	case uint8:
		return uint64(n), true
	case int64:
		if n >= 0 {
			return uint64(n), true
		}
	case int32:
		if n >= 0 {
			return uint64(n), true
		}
	case int16:
		if n >= 0 {
			return uint64(n), true
		}
	case int8:
		if n >= 0 {
			return uint64(n), true
		}
	case float64:
		if n >= 0 && n <= math.MaxUint64 {
			return uint64(n), true
		}
	case float32:
		if n >= 0 && n <= math.MaxUint64 {
			return uint64(n), true
		}
	}
	return 0, false
}
