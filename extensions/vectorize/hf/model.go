package hf

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// initORT initializes the process-wide ONNX Runtime environment. The
// environment stays alive for the life of the process; individual sessions
// are created and destroyed per vectorizer. A failed initialization (e.g. a
// wrong library path) can be retried with a corrected path.
var (
	ortMu    sync.Mutex
	ortReady bool
)

func initORT(libPath string) error {
	ortMu.Lock()
	defer ortMu.Unlock()
	if ortReady {
		return nil
	}
	if libPath == "" {
		libPath = os.Getenv("ONNXRUNTIME_LIB_PATH")
	}
	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf(
			"hf: initializing ONNX Runtime failed: %w (install the onnxruntime "+
				"shared library from https://onnxruntime.ai/docs/install/ and point "+
				"Config.ONNXRuntimePath or ONNXRUNTIME_LIB_PATH at it)", err)
	}
	ortReady = true
	return nil
}

// knownInputs are the transformer input tensors we can construct from a
// tokenizer encoding. Models discovered to need anything else are rejected.
const (
	inputIDs      = "input_ids"
	attentionMask = "attention_mask"
	tokenTypeIDs  = "token_type_ids"
)

// embeddingOutputs are the preferred embedding-model outputs, in order.
// "sentence_embedding" is produced by sentence-transformers ONNX exports
// and is already pooled; the other two are per-token outputs that we pool
// ourselves. Cross-encoder (sequence classification) exports produce
// "logits" instead.
var embeddingOutputs = []string{"sentence_embedding", "last_hidden_state", "token_embeddings"}

// onnxModel wraps an ONNX Runtime session for a transformer encoder.
type onnxModel struct {
	session    *ort.DynamicAdvancedSession
	inputNames []string // subset of knownInputs, in the model's order
	outputName string
	pooled     bool // output is [batch, hidden] rather than [batch, seq, hidden]
}

func loadONNXModel(path, libPath string, preferredOutputs []string) (*onnxModel, error) {
	if err := initORT(libPath); err != nil {
		return nil, err
	}

	inputs, outputs, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, fmt.Errorf("hf: inspecting ONNX model %s: %w", path, err)
	}

	m := &onnxModel{}
	for _, in := range inputs {
		switch in.Name {
		case inputIDs, attentionMask, tokenTypeIDs:
			m.inputNames = append(m.inputNames, in.Name)
		default:
			return nil, fmt.Errorf("hf: unsupported ONNX model input %q", in.Name)
		}
	}
	if len(m.inputNames) == 0 {
		return nil, fmt.Errorf("hf: ONNX model has no recognized inputs")
	}

	available := make(map[string]bool, len(outputs))
	for _, out := range outputs {
		available[out.Name] = true
	}
	for _, name := range preferredOutputs {
		if available[name] {
			m.outputName = name
			break
		}
	}
	if m.outputName == "" && len(outputs) > 0 {
		// Fall back to the model's first output and infer its kind from rank.
		m.outputName = outputs[0].Name
	}
	if m.outputName == "" {
		return nil, fmt.Errorf("hf: ONNX model has no outputs")
	}
	m.pooled = m.outputName == "sentence_embedding"

	session, err := ort.NewDynamicAdvancedSession(path,
		m.inputNames, []string{m.outputName}, nil)
	if err != nil {
		return nil, fmt.Errorf("hf: creating ONNX session: %w", err)
	}
	m.session = session
	return m, nil
}

// batchInput is the padded, tokenized form of one batch: three parallel
// [batch x seqLen] matrices flattened row-major.
type batchInput struct {
	batch, seqLen int
	ids           []int64
	mask          []int64
	typeIDs       []int64
	// per-row attention masks, used for pooling
	rowMasks [][]int
}

// buildInputs creates the int64 input tensors for a tokenized batch, in the
// model's input order. The returned cleanup function destroys them and is
// safe to call after partial construction.
func (m *onnxModel) buildInputs(in *batchInput) ([]ort.Value, func(), error) {
	shape := ort.NewShape(int64(in.batch), int64(in.seqLen))

	var inputs []ort.Value
	destroy := func() {
		for _, t := range inputs {
			if t != nil {
				_ = t.Destroy()
			}
		}
	}
	for _, name := range m.inputNames {
		var data []int64
		switch name {
		case inputIDs:
			data = in.ids
		case attentionMask:
			data = in.mask
		case tokenTypeIDs:
			data = in.typeIDs
		}
		t, err := ort.NewTensor(shape, data)
		if err != nil {
			destroy()
			return nil, nil, fmt.Errorf("hf: creating %s tensor: %w", name, err)
		}
		inputs = append(inputs, t)
	}
	return inputs, destroy, nil
}

// runRaw executes the model and returns the ORT-allocated float32 output
// tensor; the caller must Destroy it.
func (m *onnxModel) runRaw(in *batchInput) (*ort.Tensor[float32], error) {
	inputs, destroy, err := m.buildInputs(in)
	if err != nil {
		return nil, err
	}
	defer destroy()

	// Let ONNX Runtime allocate the output: its shape depends on the model
	// kind ([batch, hidden], [batch, seq, hidden], or [batch, labels]).
	outputs := []ort.Value{nil}
	if err := m.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("hf: running ONNX model: %w", err)
	}
	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		if outputs[0] != nil {
			_ = outputs[0].Destroy()
		}
		return nil, fmt.Errorf("hf: model output %q is not a float32 tensor", m.outputName)
	}
	return out, nil
}

// run executes the model on a tokenized batch and returns one embedding per
// row, pooling and normalizing per the model settings.
func (m *onnxModel) run(in *batchInput, settings *modelSettings) ([][]float64, error) {
	out, err := m.runRaw(in)
	if err != nil {
		return nil, err
	}
	defer func() { _ = out.Destroy() }()

	outShape := out.GetShape()
	data := out.GetData()

	embeddings := make([][]float64, in.batch)
	switch {
	case m.pooled || len(outShape) == 2:
		hidden := int(outShape[len(outShape)-1])
		if len(data) < in.batch*hidden {
			return nil, fmt.Errorf("hf: output tensor smaller than expected")
		}
		for b := 0; b < in.batch; b++ {
			row := data[b*hidden : (b+1)*hidden]
			emb := make([]float64, hidden)
			for j, v := range row {
				emb[j] = float64(v)
			}
			embeddings[b] = emb
		}
	case len(outShape) == 3:
		seqLen := int(outShape[1])
		hidden := int(outShape[2])
		if len(data) < in.batch*seqLen*hidden {
			return nil, fmt.Errorf("hf: output tensor smaller than expected")
		}
		for b := 0; b < in.batch; b++ {
			tokens := data[b*seqLen*hidden : (b+1)*seqLen*hidden]
			embeddings[b] = poolTokens(tokens, seqLen, hidden, in.rowMasks[b], settings.pooling)
		}
	default:
		return nil, fmt.Errorf("hf: unexpected model output shape %v", outShape)
	}

	if settings.normalize {
		for _, emb := range embeddings {
			l2Normalize(emb)
		}
	}
	return embeddings, nil
}

// runLogits executes a sequence-classification model and returns one score
// per row. The output must be [batch] or [batch, 1] (single-label models).
func (m *onnxModel) runLogits(in *batchInput) ([]float64, error) {
	out, err := m.runRaw(in)
	if err != nil {
		return nil, err
	}
	defer func() { _ = out.Destroy() }()

	outShape := out.GetShape()
	data := out.GetData()

	switch {
	case len(outShape) == 1,
		len(outShape) == 2 && outShape[1] == 1:
		if len(data) < in.batch {
			return nil, fmt.Errorf("hf: output tensor smaller than expected")
		}
		scores := make([]float64, in.batch)
		for i := range scores {
			scores[i] = float64(data[i])
		}
		return scores, nil
	case len(outShape) == 2:
		return nil, fmt.Errorf("hf: model has %d labels; only single-label cross-encoders are supported", outShape[1])
	default:
		return nil, fmt.Errorf("hf: unexpected model output shape %v", outShape)
	}
}

func (m *onnxModel) close() error {
	if m.session != nil {
		err := m.session.Destroy()
		m.session = nil
		return err
	}
	return nil
}
