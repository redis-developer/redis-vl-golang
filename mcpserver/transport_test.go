package mcpserver

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	redisvl "github.com/redis/redis-vl-golang"
)

// transportTestServer builds a Server with a live MCP core but no Redis
// (transport and auth boundaries only).
func transportTestServer(t *testing.T, authCfg *AuthConfig) *Server {
	t.Helper()
	if authCfg != nil {
		if err := validateAuth(authCfg); err != nil {
			t.Fatal(err)
		}
	}
	return &Server{
		cfg: &Config{Server: ServerConfig{Auth: authCfg}},
		mcp: mcp.NewServer(&mcp.Implementation{
			Name:    "redisvl-mcp-test",
			Version: redisvl.Version,
		}, nil),
	}
}

const initializeBody = `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}`

func postInitialize(t *testing.T, url, bearer string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(initializeBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestStreamableHTTPWithoutAuth(t *testing.T) {
	s := transportTestServer(t, nil)
	handler, err := s.streamableHandler()
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(handler)
	defer srv.Close()

	resp := postInitialize(t, srv.URL, "")
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		t.Fatalf("unauthenticated server rejected request: %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "redisvl-mcp-test") {
		t.Errorf("initialize response missing server info: %s", body)
	}
}

func TestSSEWithoutAuth(t *testing.T) {
	s := transportTestServer(t, nil)
	handler, err := s.sseHandler()
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(handler)
	defer srv.Close()

	req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "text/event-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("SSE GET status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
}

func TestHTTPTransportsRequireAuth(t *testing.T) {
	key, pemKey := testKeyPair(t)
	authCfg := &AuthConfig{
		Type:      "jwt",
		PublicKey: pemKey,
		Issuer:    "https://issuer.test",
		Audience:  "redisvl-mcp",
		// The SDK advertises this in WWW-Authenticate challenges.
		BaseURL: "https://mcp.test/.well-known/oauth-protected-resource",
	}
	s := transportTestServer(t, authCfg)

	streamable, err := s.streamableHandler()
	if err != nil {
		t.Fatal(err)
	}
	sse, err := s.sseHandler()
	if err != nil {
		t.Fatal(err)
	}

	for name, handler := range map[string]http.Handler{
		"streamable-http": streamable,
		"sse":             sse,
	} {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(handler)
			defer srv.Close()

			// No token: 401 with a WWW-Authenticate challenge.
			resp := postInitialize(t, srv.URL, "")
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusUnauthorized {
				t.Fatalf("tokenless request status = %d, want 401", resp.StatusCode)
			}
			// The SDK includes the resource-metadata challenge only when a
			// metadata URL is configured (BaseURL above).
			if resp.Header.Get("WWW-Authenticate") == "" {
				t.Error("401 response missing WWW-Authenticate header")
			}

			// Garbage token: still 401.
			resp = postInitialize(t, srv.URL, "not-a-jwt")
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusUnauthorized {
				t.Fatalf("invalid-token status = %d, want 401", resp.StatusCode)
			}

			// Valid token: passes the auth boundary.
			resp = postInitialize(t, srv.URL, signToken(t, key, "", baseClaims()))
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
				t.Fatalf("valid token rejected: %d", resp.StatusCode)
			}
		})
	}
}
