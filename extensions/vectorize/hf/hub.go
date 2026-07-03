package hf

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// defaultEndpoint is the Hugging Face Hub base URL. Override with
// Config.Endpoint or the HF_ENDPOINT environment variable (mirror sites).
const defaultEndpoint = "https://huggingface.co"

// modelFiles lists the files fetched from a model repository. Only
// tokenizer.json, config.json and the ONNX export are required; the
// sentence-transformers files are optional and drive pooling/normalization
// when present.
type modelFiles struct {
	tokenizer    string // tokenizer.json (required)
	config       string // config.json (required)
	onnx         string // ONNX export (required)
	modules      string // modules.json ("" when absent)
	pooling      string // 1_Pooling/config.json ("" when absent)
	sentenceBert string // sentence_bert_config.json ("" when absent)
}

// hubClient downloads model files from the Hugging Face Hub with a local
// cache, the same lazy download-on-first-use behavior the Python
// sentence-transformers library provides.
type hubClient struct {
	endpoint string
	token    string
	cacheDir string
	client   *http.Client
}

func newHubClient(cfg Config) (*hubClient, error) {
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = os.Getenv("HF_ENDPOINT")
	}
	if endpoint == "" {
		endpoint = defaultEndpoint
	}
	token := cfg.Token
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}
	cacheDir := cfg.CacheDir
	if cacheDir == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return nil, fmt.Errorf("hf: resolving user cache dir: %w (set Config.CacheDir)", err)
		}
		cacheDir = filepath.Join(base, "redisvl-go", "models")
	}
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &hubClient{
		endpoint: strings.TrimRight(endpoint, "/"),
		token:    token,
		cacheDir: cacheDir,
		client:   client,
	}, nil
}

// fetchModel ensures all files for the model are present locally,
// downloading any that are missing, and returns their paths.
func (h *hubClient) fetchModel(ctx context.Context, model, onnxFile string) (*modelFiles, error) {
	if err := validateRepoPath(model); err != nil {
		return nil, err
	}
	if err := validateRepoPath(onnxFile); err != nil {
		return nil, err
	}

	files := &modelFiles{}

	required := []struct {
		remote string
		dest   *string
	}{
		{"tokenizer.json", &files.tokenizer},
		{"config.json", &files.config},
		{onnxFile, &files.onnx},
	}
	for _, f := range required {
		path, err := h.fetchFile(ctx, model, f.remote)
		if err != nil {
			return nil, err
		}
		*f.dest = path
	}

	optional := []struct {
		remote string
		dest   *string
	}{
		{"modules.json", &files.modules},
		{"1_Pooling/config.json", &files.pooling},
		{"sentence_bert_config.json", &files.sentenceBert},
	}
	for _, f := range optional {
		path, err := h.fetchFile(ctx, model, f.remote)
		if err != nil {
			if isNotFound(err) {
				continue
			}
			return nil, err
		}
		*f.dest = path
	}

	return files, nil
}

// fetchFile returns the local path of one repository file, downloading it
// if it is not cached yet.
func (h *hubClient) fetchFile(ctx context.Context, model, remote string) (string, error) {
	local := filepath.Join(h.cacheDir, filepath.FromSlash(model), filepath.FromSlash(remote))
	if info, err := os.Stat(local); err == nil && info.Size() > 0 {
		return local, nil
	}

	url := fmt.Sprintf("%s/%s/resolve/main/%s", h.endpoint, model, remote)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	if h.token != "" {
		req.Header.Set("Authorization", "Bearer "+h.token)
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("hf: downloading %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == http.StatusNotFound {
		return "", &notFoundError{url: url}
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("hf: downloading %s: HTTP %d: %s", url, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	if err := os.MkdirAll(filepath.Dir(local), 0o755); err != nil {
		return "", err
	}

	// Download to a temp file and rename, so a partial download never
	// masquerades as a cached file.
	tmp, err := os.CreateTemp(filepath.Dir(local), ".download-*")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return "", fmt.Errorf("hf: downloading %s: %w", url, err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return "", err
	}
	if err := os.Rename(tmpName, local); err != nil {
		_ = os.Remove(tmpName)
		return "", err
	}
	return local, nil
}

// validateRepoPath rejects path segments that could escape the cache
// directory when joined to a local path.
func validateRepoPath(p string) error {
	if p == "" {
		return fmt.Errorf("hf: empty model or file path")
	}
	for _, seg := range strings.Split(p, "/") {
		if seg == "" || seg == "." || seg == ".." || strings.ContainsAny(seg, `\:`) {
			return fmt.Errorf("hf: invalid path segment in %q", p)
		}
	}
	return nil
}

type notFoundError struct{ url string }

func (e *notFoundError) Error() string { return "hf: not found: " + e.url }

func isNotFound(err error) bool {
	_, ok := err.(*notFoundError)
	return ok
}
