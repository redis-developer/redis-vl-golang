package hf

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

// fakeHub serves a minimal model repository and counts requests per path.
func fakeHub(t *testing.T, files map[string]string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		body, ok := files[r.URL.Path]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &requests
}

func minimalRepo() map[string]string {
	return map[string]string{
		"/acme/tiny-model/resolve/main/tokenizer.json":        `{"fake": true}`,
		"/acme/tiny-model/resolve/main/config.json":           `{"hidden_size": 8, "pad_token_id": 0}`,
		"/acme/tiny-model/resolve/main/onnx/model.onnx":       "onnx-bytes",
		"/acme/tiny-model/resolve/main/modules.json":          `[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"}]`,
		"/acme/tiny-model/resolve/main/1_Pooling/config.json": `{"word_embedding_dimension":8,"pooling_mode_mean_tokens":true}`,
	}
}

func TestFetchModelDownloadsAndCaches(t *testing.T) {
	srv, requests := fakeHub(t, minimalRepo())

	hub, err := newHubClient(Config{Endpoint: srv.URL, CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}

	files, err := hub.fetchModel(context.Background(), "acme/tiny-model", "onnx/model.onnx")
	if err != nil {
		t.Fatal(err)
	}

	for name, path := range map[string]string{
		"tokenizer": files.tokenizer,
		"config":    files.config,
		"onnx":      files.onnx,
		"modules":   files.modules,
		"pooling":   files.pooling,
	} {
		if path == "" {
			t.Fatalf("%s file path is empty", name)
		}
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("%s file missing on disk: %v", name, err)
		}
	}
	// optional files absent on the server stay empty
	if files.sentenceBert != "" {
		t.Errorf("sentence_bert path = %q, want empty (404 on server)", files.sentenceBert)
	}

	// files live under cacheDir/model
	wantPrefix := filepath.Join(hub.cacheDir, "acme", "tiny-model")
	if filepath.Dir(files.config) != wantPrefix {
		t.Errorf("config stored at %s, want under %s", files.config, wantPrefix)
	}

	firstRun := requests.Load()

	// Second fetch: everything cached except re-probing the optional 404s.
	if _, err := hub.fetchModel(context.Background(), "acme/tiny-model", "onnx/model.onnx"); err != nil {
		t.Fatal(err)
	}
	// Only the missing optional file (sentence_bert_config.json) should be
	// re-requested; cached files short-circuit without a request.
	if got := requests.Load() - firstRun; got != 1 {
		t.Errorf("second fetch made %d requests, want 1 (optional 404 re-probe)", got)
	}
}

func TestFetchFileSendsAuthToken(t *testing.T) {
	authCh := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authCh <- r.Header.Get("Authorization")
		_, _ = w.Write([]byte("data"))
	}))
	defer srv.Close()

	hub, err := newHubClient(Config{Endpoint: srv.URL, CacheDir: t.TempDir(), Token: "hf_secret"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := hub.fetchFile(context.Background(), "acme/tiny-model", "config.json"); err != nil {
		t.Fatal(err)
	}
	if gotAuth := <-authCh; gotAuth != "Bearer hf_secret" {
		t.Errorf("Authorization = %q", gotAuth)
	}
}

func TestFetchModelRejectsTraversal(t *testing.T) {
	hub, err := newHubClient(Config{Endpoint: "http://unused", CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"../evil", "a//b", ".", ""} {
		if _, err := hub.fetchModel(context.Background(), bad, "onnx/model.onnx"); err == nil {
			t.Errorf("model path %q accepted, want error", bad)
		}
		if _, err := hub.fetchModel(context.Background(), "acme/ok", bad); err == nil {
			t.Errorf("onnx path %q accepted, want error", bad)
		}
	}
}

func TestFetchModelRequiredMissing(t *testing.T) {
	repo := minimalRepo()
	delete(repo, "/acme/tiny-model/resolve/main/onnx/model.onnx")
	srv, _ := fakeHub(t, repo)

	hub, err := newHubClient(Config{Endpoint: srv.URL, CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := hub.fetchModel(context.Background(), "acme/tiny-model", "onnx/model.onnx"); err == nil {
		t.Fatal("expected error when the ONNX export is missing")
	}
}
