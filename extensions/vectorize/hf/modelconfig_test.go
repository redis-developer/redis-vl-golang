package hf

import (
	"os"
	"path/filepath"
	"testing"
)

// writeFiles materializes a fake downloaded model dir and returns the
// corresponding modelFiles (empty path = file absent).
func writeFiles(t *testing.T, contents map[string]string) *modelFiles {
	t.Helper()
	dir := t.TempDir()
	files := &modelFiles{}
	dests := map[string]*string{
		"config.json":               &files.config,
		"tokenizer.json":            &files.tokenizer,
		"modules.json":              &files.modules,
		"1_Pooling/config.json":     &files.pooling,
		"sentence_bert_config.json": &files.sentenceBert,
	}
	for name, body := range contents {
		path := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if dest, ok := dests[name]; ok {
			*dest = path
		}
	}
	return files
}

func TestLoadModelSettingsMiniLMStyle(t *testing.T) {
	// mirrors sentence-transformers/all-MiniLM-L6-v2: mean pooling +
	// Normalize module + max_seq_length 256
	files := writeFiles(t, map[string]string{
		"config.json": `{"hidden_size":384,"pad_token_id":0,"max_position_embeddings":512,"model_type":"bert"}`,
		"1_Pooling/config.json": `{"word_embedding_dimension":384,"pooling_mode_cls_token":false,
			"pooling_mode_mean_tokens":true,"pooling_mode_max_tokens":false}`,
		"modules.json": `[{"type":"sentence_transformers.models.Transformer"},
			{"type":"sentence_transformers.models.Pooling"},
			{"type":"sentence_transformers.models.Normalize"}]`,
		"sentence_bert_config.json": `{"max_seq_length":256,"do_lower_case":false}`,
	})

	s, err := loadModelSettings(files)
	if err != nil {
		t.Fatal(err)
	}
	if s.hiddenSize != 384 || s.padTokenID != 0 {
		t.Errorf("hidden/pad = %d/%d", s.hiddenSize, s.padTokenID)
	}
	if s.pooling != poolMean {
		t.Errorf("pooling = %v, want mean", s.pooling)
	}
	if !s.normalize {
		t.Error("normalize = false, want true (Normalize module present)")
	}
	if s.maxSeqLength != 256 {
		t.Errorf("maxSeqLength = %d, want 256", s.maxSeqLength)
	}
}

func TestLoadModelSettingsLangCacheStyle(t *testing.T) {
	// mirrors redis/langcache-embed-v1: ModernBERT, mean pooling, NO
	// Normalize module, pad_token_id 50283
	files := writeFiles(t, map[string]string{
		"config.json":           `{"hidden_size":768,"pad_token_id":50283,"max_position_embeddings":8192,"model_type":"modernbert"}`,
		"1_Pooling/config.json": `{"pooling_mode_mean_tokens":true}`,
		"modules.json": `[{"type":"sentence_transformers.models.Transformer"},
			{"type":"sentence_transformers.models.Pooling"}]`,
	})

	s, err := loadModelSettings(files)
	if err != nil {
		t.Fatal(err)
	}
	if s.padTokenID != 50283 {
		t.Errorf("padTokenID = %d, want 50283", s.padTokenID)
	}
	if s.normalize {
		t.Error("normalize = true, want false (no Normalize module)")
	}
	if s.maxSeqLength != 8192 {
		t.Errorf("maxSeqLength = %d, want 8192 (max_position_embeddings fallback)", s.maxSeqLength)
	}
}

func TestLoadModelSettingsDefaults(t *testing.T) {
	// plain HF model: no sentence-transformers files at all
	files := writeFiles(t, map[string]string{
		"config.json": `{"hidden_size":768}`,
	})
	s, err := loadModelSettings(files)
	if err != nil {
		t.Fatal(err)
	}
	if s.pooling != poolMean || s.normalize {
		t.Errorf("defaults = pooling %v normalize %v, want mean/false", s.pooling, s.normalize)
	}
	if s.maxSeqLength != 512 {
		t.Errorf("maxSeqLength = %d, want 512 fallback", s.maxSeqLength)
	}
}

func TestLoadModelSettingsRejects(t *testing.T) {
	// no hidden_size -> not an encoder we can run
	files := writeFiles(t, map[string]string{"config.json": `{"model_type":"gpt2"}`})
	if _, err := loadModelSettings(files); err == nil {
		t.Error("expected error for config without hidden_size")
	}

	// Dense projection module unsupported
	files = writeFiles(t, map[string]string{
		"config.json":  `{"hidden_size":768}`,
		"modules.json": `[{"type":"sentence_transformers.models.Dense"}]`,
	})
	if _, err := loadModelSettings(files); err == nil {
		t.Error("expected error for Dense module")
	}

	// multiple simultaneous pooling modes unsupported
	files = writeFiles(t, map[string]string{
		"config.json":           `{"hidden_size":768}`,
		"1_Pooling/config.json": `{"pooling_mode_mean_tokens":true,"pooling_mode_cls_token":true}`,
	})
	if _, err := loadModelSettings(files); err == nil {
		t.Error("expected error for multiple pooling modes")
	}

	// CLS pooling is supported
	files = writeFiles(t, map[string]string{
		"config.json":           `{"hidden_size":768}`,
		"1_Pooling/config.json": `{"pooling_mode_cls_token":true}`,
	})
	s, err := loadModelSettings(files)
	if err != nil {
		t.Fatal(err)
	}
	if s.pooling != poolCLS {
		t.Errorf("pooling = %v, want cls", s.pooling)
	}
}

func TestLoadModelSettingsCrossEncoderActivation(t *testing.T) {
	cases := []struct {
		name        string
		config      string
		wantSigmoid bool
		wantLabels  int
		wantErr     bool
	}{
		{
			// mirrors cross-encoder/ms-marco-MiniLM-L-6-v2: explicit
			// Identity override -> raw logits
			name:        "explicit identity",
			config:      `{"hidden_size":384,"pad_token_id":0,"id2label":{"0":"LABEL_0"},"sbert_ce_default_activation_function":"torch.nn.modules.linear.Identity"}`,
			wantSigmoid: false,
			wantLabels:  1,
		},
		{
			name:        "explicit sigmoid",
			config:      `{"hidden_size":384,"id2label":{"0":"LABEL_0"},"sbert_ce_default_activation_function":"torch.nn.modules.activation.Sigmoid"}`,
			wantSigmoid: true,
			wantLabels:  1,
		},
		{
			// sentence-transformers default: sigmoid when single-label
			name:        "default single label",
			config:      `{"hidden_size":384,"id2label":{"0":"LABEL_0"}}`,
			wantSigmoid: true,
			wantLabels:  1,
		},
		{
			// sentence-transformers default: identity when multi-label
			name:        "default multi label",
			config:      `{"hidden_size":384,"id2label":{"0":"A","1":"B","2":"C"}}`,
			wantSigmoid: false,
			wantLabels:  3,
		},
		{
			name:    "unknown activation",
			config:  `{"hidden_size":384,"sbert_ce_default_activation_function":"torch.nn.modules.activation.Tanh"}`,
			wantErr: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			files := writeFiles(t, map[string]string{"config.json": tc.config})
			s, err := loadModelSettings(files)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if s.sigmoid != tc.wantSigmoid || s.numLabels != tc.wantLabels {
				t.Errorf("sigmoid/labels = %v/%d, want %v/%d",
					s.sigmoid, s.numLabels, tc.wantSigmoid, tc.wantLabels)
			}
		})
	}
}
