package hf

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// poolingMode selects how token embeddings are reduced to one sentence
// embedding, following sentence-transformers' Pooling module options.
type poolingMode int

const (
	poolMean poolingMode = iota
	poolCLS
	poolMax
)

// modelSettings aggregates everything read from the model repository's
// config files that influences inference.
type modelSettings struct {
	hiddenSize   int
	padTokenID   int
	maxSeqLength int
	pooling      poolingMode
	normalize    bool
	// cross-encoder specifics
	numLabels int
	sigmoid   bool // apply sigmoid to logits (sentence-transformers activation rule)
}

// hfConfig mirrors the fields we need from config.json.
type hfConfig struct {
	HiddenSize            int               `json:"hidden_size"`
	PadTokenID            *int              `json:"pad_token_id"`
	MaxPositionEmbeddings int               `json:"max_position_embeddings"`
	ModelType             string            `json:"model_type"`
	ID2Label              map[string]string `json:"id2label"`
	// sentence-transformers records the CrossEncoder activation override
	// here (e.g. "torch.nn.modules.linear.Identity").
	SbertCEActivation string `json:"sbert_ce_default_activation_function"`
}

// stPoolingConfig mirrors 1_Pooling/config.json from sentence-transformers.
type stPoolingConfig struct {
	CLSToken         bool `json:"pooling_mode_cls_token"`
	MeanTokens       bool `json:"pooling_mode_mean_tokens"`
	MaxTokens        bool `json:"pooling_mode_max_tokens"`
	MeanSqrtLenToken bool `json:"pooling_mode_mean_sqrt_len_tokens"`
	LastToken        bool `json:"pooling_mode_lasttoken"`
}

// stModule mirrors one entry of modules.json.
type stModule struct {
	Type string `json:"type"`
}

// stSentenceBertConfig mirrors sentence_bert_config.json.
type stSentenceBertConfig struct {
	MaxSeqLength int `json:"max_seq_length"`
}

// loadModelSettings reads the downloaded config files and resolves the
// inference settings, matching sentence-transformers behavior: pooling mode
// from 1_Pooling/config.json (mean when absent), L2 normalization only when
// modules.json declares a Normalize module, and truncation length from
// sentence_bert_config.json (falling back to max_position_embeddings).
func loadModelSettings(files *modelFiles) (*modelSettings, error) {
	raw, err := os.ReadFile(files.config)
	if err != nil {
		return nil, err
	}
	var cfg hfConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("hf: parsing config.json: %w", err)
	}
	if cfg.HiddenSize <= 0 {
		return nil, fmt.Errorf("hf: config.json has no hidden_size; not a supported transformer encoder")
	}

	s := &modelSettings{
		hiddenSize: cfg.HiddenSize,
		pooling:    poolMean,
		numLabels:  len(cfg.ID2Label),
	}
	if s.numLabels == 0 {
		s.numLabels = 1
	}
	if cfg.PadTokenID != nil {
		s.padTokenID = *cfg.PadTokenID
	}

	// CrossEncoder score activation, following sentence-transformers: an
	// explicit config override wins; otherwise sigmoid for single-label
	// models, identity for multi-label.
	switch {
	case strings.HasSuffix(cfg.SbertCEActivation, ".Identity"):
		s.sigmoid = false
	case strings.HasSuffix(cfg.SbertCEActivation, ".Sigmoid"):
		s.sigmoid = true
	case cfg.SbertCEActivation != "":
		return nil, fmt.Errorf("hf: unsupported sbert_ce_default_activation_function %q", cfg.SbertCEActivation)
	default:
		s.sigmoid = s.numLabels == 1
	}

	if files.pooling != "" {
		raw, err := os.ReadFile(files.pooling)
		if err != nil {
			return nil, err
		}
		var pc stPoolingConfig
		if err := json.Unmarshal(raw, &pc); err != nil {
			return nil, fmt.Errorf("hf: parsing 1_Pooling/config.json: %w", err)
		}
		if pc.MeanSqrtLenToken || pc.LastToken {
			return nil, fmt.Errorf("hf: unsupported pooling mode in 1_Pooling/config.json")
		}
		enabled := 0
		for _, on := range []bool{pc.MeanTokens, pc.CLSToken, pc.MaxTokens} {
			if on {
				enabled++
			}
		}
		if enabled > 1 {
			// concatenated pooling modes would silently diverge from Python
			return nil, fmt.Errorf("hf: multiple pooling modes enabled in 1_Pooling/config.json; not supported")
		}
		switch {
		case pc.MeanTokens:
			s.pooling = poolMean
		case pc.CLSToken:
			s.pooling = poolCLS
		case pc.MaxTokens:
			s.pooling = poolMax
		}
	}

	if files.modules != "" {
		raw, err := os.ReadFile(files.modules)
		if err != nil {
			return nil, err
		}
		var modules []stModule
		if err := json.Unmarshal(raw, &modules); err != nil {
			return nil, fmt.Errorf("hf: parsing modules.json: %w", err)
		}
		for _, m := range modules {
			if strings.HasSuffix(m.Type, ".Normalize") {
				s.normalize = true
			}
			if strings.HasSuffix(m.Type, ".Dense") {
				return nil, fmt.Errorf("hf: models with a Dense projection module are not supported")
			}
		}
	}

	if files.sentenceBert != "" {
		raw, err := os.ReadFile(files.sentenceBert)
		if err != nil {
			return nil, err
		}
		var sb stSentenceBertConfig
		if err := json.Unmarshal(raw, &sb); err != nil {
			return nil, fmt.Errorf("hf: parsing sentence_bert_config.json: %w", err)
		}
		s.maxSeqLength = sb.MaxSeqLength
	}
	if s.maxSeqLength <= 0 {
		s.maxSeqLength = cfg.MaxPositionEmbeddings
	}
	if s.maxSeqLength <= 0 {
		s.maxSeqLength = 512
	}

	return s, nil
}
