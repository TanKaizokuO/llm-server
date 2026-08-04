package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

// FixtureParams configures the generated test GGUF header.
type FixtureParams struct {
	Architecture  string
	ContextLength uint64
	Quantization  string
	ExtraKV       map[string]any
}

// CreateTestGGUF generates a minimal valid binary GGUF header byte slice.
func CreateTestGGUF(params FixtureParams) []byte {
	arch := params.Architecture
	if arch == "" {
		arch = "llama"
	}
	ctxLen := params.ContextLength
	if ctxLen == 0 {
		ctxLen = 4096
	}
	quant := params.Quantization
	if quant == "" {
		quant = "Q4_K_M"
	}

	kv := make(map[string]any)
	kv["general.architecture"] = arch
	kv[arch+".context_length"] = ctxLen

	if ftype, ok := FileTypeNameToType(quant); ok {
		kv["general.file_type"] = ftype
	} else {
		kv["general.quantization"] = quant
	}

	for k, v := range params.ExtraKV {
		kv[k] = v
	}

	buf := new(bytes.Buffer)
	// Write magic
	buf.Write(Magic[:])
	// Write version = 3
	_ = binary.Write(buf, binary.LittleEndian, uint32(3))
	// Write tensor count = 0
	_ = binary.Write(buf, binary.LittleEndian, uint64(0))
	// Write KV count
	_ = binary.Write(buf, binary.LittleEndian, uint64(len(kv)))

	for k, v := range kv {
		writeString(buf, k)
		writeValue(buf, v)
	}

	return buf.Bytes()
}

func writeString(buf *bytes.Buffer, s string) {
	_ = binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
}

func writeValue(buf *bytes.Buffer, v any) {
	switch val := v.(type) {
	case uint8:
		_ = binary.Write(buf, binary.LittleEndian, TypeUint8)
		buf.WriteByte(val)
	case int8:
		_ = binary.Write(buf, binary.LittleEndian, TypeInt8)
		_ = binary.Write(buf, binary.LittleEndian, val)
	case uint16:
		_ = binary.Write(buf, binary.LittleEndian, TypeUint16)
		_ = binary.Write(buf, binary.LittleEndian, val)
	case int16:
		_ = binary.Write(buf, binary.LittleEndian, TypeInt16)
		_ = binary.Write(buf, binary.LittleEndian, val)
	case uint32:
		_ = binary.Write(buf, binary.LittleEndian, TypeUint32)
		_ = binary.Write(buf, binary.LittleEndian, val)
	case int32:
		_ = binary.Write(buf, binary.LittleEndian, TypeInt32)
		_ = binary.Write(buf, binary.LittleEndian, val)
	case int:
		_ = binary.Write(buf, binary.LittleEndian, TypeUint32)
		_ = binary.Write(buf, binary.LittleEndian, uint32(val))
	case uint:
		_ = binary.Write(buf, binary.LittleEndian, TypeUint64)
		_ = binary.Write(buf, binary.LittleEndian, uint64(val))
	case float32:
		_ = binary.Write(buf, binary.LittleEndian, TypeFloat32)
		_ = binary.Write(buf, binary.LittleEndian, val)
	case bool:
		_ = binary.Write(buf, binary.LittleEndian, TypeBool)
		if val {
			buf.WriteByte(1)
		} else {
			buf.WriteByte(0)
		}
	case string:
		_ = binary.Write(buf, binary.LittleEndian, TypeString)
		writeString(buf, val)
	case uint64:
		_ = binary.Write(buf, binary.LittleEndian, TypeUint64)
		_ = binary.Write(buf, binary.LittleEndian, val)
	case int64:
		_ = binary.Write(buf, binary.LittleEndian, TypeInt64)
		_ = binary.Write(buf, binary.LittleEndian, val)
	case float64:
		_ = binary.Write(buf, binary.LittleEndian, TypeFloat64)
		_ = binary.Write(buf, binary.LittleEndian, val)
	default:
		panic(fmt.Sprintf("gguf: CreateTestGGUF cannot encode value of type %T", v))
	}
}
