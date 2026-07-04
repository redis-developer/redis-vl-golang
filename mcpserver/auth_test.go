package mcpserver

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/modelcontextprotocol/go-sdk/auth"
)

func testKeyPair(t *testing.T) (*rsa.PrivateKey, string) {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKIXPublicKey(&key.PublicKey)
	if err != nil {
		t.Fatal(err)
	}
	pemKey := string(pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: der}))
	return key, pemKey
}

func signToken(t *testing.T, key *rsa.PrivateKey, kid string, claims jwt.MapClaims) string {
	t.Helper()
	token := jwt.NewWithClaims(jwt.SigningMethodRS256, claims)
	if kid != "" {
		token.Header["kid"] = kid
	}
	signed, err := token.SignedString(key)
	if err != nil {
		t.Fatal(err)
	}
	return signed
}

func baseClaims() jwt.MapClaims {
	now := time.Now()
	return jwt.MapClaims{
		"iss": "https://issuer.test",
		"aud": "redisvl-mcp",
		"sub": "user-1",
		"iat": now.Unix(),
		"exp": now.Add(time.Hour).Unix(),
		"scp": "search:read search:write",
	}
}

func testAuthConfig(pemKey string) *AuthConfig {
	cfg := &AuthConfig{
		Type:      "jwt",
		PublicKey: pemKey,
		Issuer:    "https://issuer.test",
		Audience:  "redisvl-mcp",
	}
	// apply defaults the way Config.Validate does
	if err := validateAuth(cfg); err != nil {
		panic(err)
	}
	return cfg
}

func TestVerifierAcceptsValidToken(t *testing.T) {
	key, pemKey := testKeyPair(t)
	verifier, err := buildTokenVerifier(testAuthConfig(pemKey))
	if err != nil {
		t.Fatal(err)
	}

	info, err := verifier(context.Background(), signToken(t, key, "", baseClaims()), nil)
	if err != nil {
		t.Fatal(err)
	}
	if info.UserID != "user-1" {
		t.Errorf("UserID = %q", info.UserID)
	}
	if len(info.Scopes) != 2 || info.Scopes[0] != "search:read" {
		t.Errorf("Scopes = %v", info.Scopes)
	}
	if info.Expiration.Before(time.Now()) {
		t.Errorf("Expiration in the past: %v", info.Expiration)
	}
}

func TestVerifierRejects(t *testing.T) {
	key, pemKey := testKeyPair(t)
	otherKey, _ := testKeyPair(t)
	verifier, err := buildTokenVerifier(testAuthConfig(pemKey))
	if err != nil {
		t.Fatal(err)
	}

	cases := map[string]string{}

	wrongIssuer := baseClaims()
	wrongIssuer["iss"] = "https://evil.test"
	cases["wrong issuer"] = signToken(t, key, "", wrongIssuer)

	wrongAudience := baseClaims()
	wrongAudience["aud"] = "other-service"
	cases["wrong audience"] = signToken(t, key, "", wrongAudience)

	expired := baseClaims()
	expired["exp"] = time.Now().Add(-time.Hour).Unix()
	cases["expired"] = signToken(t, key, "", expired)

	noExp := baseClaims()
	delete(noExp, "exp")
	cases["missing exp claim"] = signToken(t, key, "", noExp)

	noIat := baseClaims()
	delete(noIat, "iat")
	cases["missing iat claim"] = signToken(t, key, "", noIat)

	cases["wrong key"] = signToken(t, otherKey, "", baseClaims())

	for name, token := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := verifier(context.Background(), token, nil)
			if err == nil {
				t.Fatal("token accepted, want rejection")
			}
			if !errors.Is(err, auth.ErrInvalidToken) {
				t.Errorf("error does not unwrap to ErrInvalidToken: %v", err)
			}
		})
	}
}

func TestVerifierRejectsSymmetricAlgorithm(t *testing.T) {
	_, pemKey := testKeyPair(t)
	verifier, err := buildTokenVerifier(testAuthConfig(pemKey))
	if err != nil {
		t.Fatal(err)
	}
	// alg confusion attack: HS256 token "signed" with the public key bytes
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, baseClaims())
	signed, err := token.SignedString([]byte(pemKey))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := verifier(context.Background(), signed, nil); err == nil {
		t.Fatal("HS256 token accepted, want rejection")
	}
}

func TestVerifierWithJWKS(t *testing.T) {
	key, _ := testKeyPair(t)
	b64 := base64.RawURLEncoding
	jwks := map[string]any{
		"keys": []map[string]any{{
			"kty": "RSA",
			"kid": "key-1",
			"n":   b64.EncodeToString(key.N.Bytes()),
			"e":   b64.EncodeToString(big.NewInt(int64(key.E)).Bytes()),
		}},
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(jwks)
	}))
	defer srv.Close()

	cfg := &AuthConfig{
		Type:     "jwt",
		JWKSURI:  srv.URL,
		Issuer:   "https://issuer.test",
		Audience: "redisvl-mcp",
	}
	if err := validateAuth(cfg); err != nil {
		t.Fatal(err)
	}
	verifier, err := buildTokenVerifier(cfg)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := verifier(context.Background(), signToken(t, key, "key-1", baseClaims()), nil); err != nil {
		t.Fatalf("kid lookup failed: %v", err)
	}
	// kid-less token against a single-key JWKS also verifies
	if _, err := verifier(context.Background(), signToken(t, key, "", baseClaims()), nil); err != nil {
		t.Fatalf("single-key fallback failed: %v", err)
	}
}

func TestRequireScope(t *testing.T) {
	_, pemKey := testKeyPair(t)
	authCfg := testAuthConfig(pemKey)
	authCfg.ReadScope = "search:read"
	s := &Server{cfg: &Config{Server: ServerConfig{Auth: authCfg}}}

	// no token in context (stdio): gate does not apply
	if err := s.requireScope(context.Background(), "search:read"); err != nil {
		t.Errorf("stdio context rejected: %v", err)
	}

	withToken := func(scopes []string, extra map[string]any) context.Context {
		// TokenInfo reaches handlers via the SDK middleware; simulate it.
		info := &auth.TokenInfo{Scopes: scopes, Extra: extra, Expiration: time.Now().Add(time.Hour)}
		mw := auth.RequireBearerToken(
			func(context.Context, string, *http.Request) (*auth.TokenInfo, error) { return info, nil }, nil)
		var captured context.Context
		h := mw(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			captured = r.Context()
		}))
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.Header.Set("Authorization", "Bearer x")
		h.ServeHTTP(httptest.NewRecorder(), req)
		if captured == nil {
			t.Fatal("middleware did not invoke handler")
		}
		return captured
	}

	if err := s.requireScope(withToken([]string{"search:read"}, nil), "search:read"); err != nil {
		t.Errorf("token with scope rejected: %v", err)
	}
	if err := s.requireScope(withToken([]string{"other"}, nil), "search:read"); err == nil {
		t.Error("token without scope accepted")
	}

	// non-standard authorization claim (e.g. roles)
	authCfg.AuthorizationClaim = "roles"
	if err := s.requireScope(withToken(nil, map[string]any{"roles": []any{"admin"}}), "admin"); err != nil {
		t.Errorf("roles claim rejected: %v", err)
	}
	if err := s.requireScope(withToken(nil, map[string]any{"roles": "user viewer"}), "admin"); err == nil {
		t.Error("missing role accepted")
	}
}

func TestAuthConfigValidation(t *testing.T) {
	cases := []struct {
		name    string
		cfg     *AuthConfig
		wantErr bool
	}{
		{"nil config", nil, false},
		{"explicit none", &AuthConfig{Type: "none"}, false},
		{"jwt fields without type", &AuthConfig{Issuer: "x"}, true},
		{"jwt missing key source", &AuthConfig{Type: "jwt", Issuer: "x", Audience: "y"}, true},
		{"jwt both key sources", &AuthConfig{Type: "jwt", JWKSURI: "u", PublicKey: "k", Issuer: "x", Audience: "y"}, true},
		{"jwt missing issuer", &AuthConfig{Type: "jwt", JWKSURI: "u", Audience: "y"}, true},
		{"jwt missing audience", &AuthConfig{Type: "jwt", JWKSURI: "u", Issuer: "x"}, true},
		{"jwt valid", &AuthConfig{Type: "jwt", JWKSURI: "u", Issuer: "x", Audience: "y"}, false},
		{"unknown type", &AuthConfig{Type: "basic"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAuth(tc.cfg)
			if (err != nil) != tc.wantErr {
				t.Errorf("err = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}

	// defaults applied on valid jwt config
	cfg := &AuthConfig{Type: "jwt", JWKSURI: "u", Issuer: "x", Audience: "y"}
	if err := validateAuth(cfg); err != nil {
		t.Fatal(err)
	}
	if len(cfg.RequiredClaims) != 2 || cfg.RequiredClaims[0] != "exp" || cfg.RequiredClaims[1] != "iat" {
		t.Errorf("RequiredClaims default = %v", cfg.RequiredClaims)
	}
	if cfg.AuthorizationClaim != "scp" {
		t.Errorf("AuthorizationClaim default = %q", cfg.AuthorizationClaim)
	}
}

func TestAuthEnvOverrides(t *testing.T) {
	t.Setenv("REDISVL_MCP_AUTH_TYPE", "jwt")
	t.Setenv("REDISVL_MCP_AUTH_JWKS_URI", "https://idp.test/jwks")
	t.Setenv("REDISVL_MCP_AUTH_ISSUER", "https://idp.test")
	t.Setenv("REDISVL_MCP_AUTH_AUDIENCE", "redisvl")
	t.Setenv("REDISVL_MCP_AUTH_REQUIRED_SCOPES", "a, b")

	// env wins over a conflicting YAML block
	got := authEnvOverrides(&AuthConfig{Type: "none", Issuer: "yaml-issuer"})
	if got == nil || got.Type != "jwt" || got.Issuer != "https://idp.test" {
		t.Fatalf("overrides = %+v", got)
	}
	if len(got.RequiredScopes) != 2 || got.RequiredScopes[1] != "b" {
		t.Errorf("RequiredScopes = %v", got.RequiredScopes)
	}

	// explicit type=none disables YAML auth entirely
	t.Setenv("REDISVL_MCP_AUTH_TYPE", "none")
	if got := authEnvOverrides(&AuthConfig{Type: "jwt", Issuer: "x"}); got != nil {
		t.Errorf("type=none did not disable auth: %+v", got)
	}
}
