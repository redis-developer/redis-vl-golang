package mcpserver

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/modelcontextprotocol/go-sdk/auth"
)

// defaultAllowedAlgorithms are the accepted asymmetric signing algorithms
// when auth.algorithm is not configured. Symmetric algorithms (HS*) are
// deliberately excluded: a shared secret makes every client a token
// minter.
var defaultAllowedAlgorithms = []string{
	"RS256", "RS384", "RS512",
	"PS256", "PS384", "PS512",
	"ES256", "ES384", "ES512",
	"EdDSA",
}

// buildTokenVerifier turns an AuthConfig into the SDK's TokenVerifier,
// mirroring the semantics of the Python server's JWTVerifier: signature,
// issuer, audience, allowed algorithm, and required-claims checks, with
// scopes extracted from the scp/scope claim.
func buildTokenVerifier(cfg *AuthConfig) (auth.TokenVerifier, error) {
	keys, err := newKeySource(cfg)
	if err != nil {
		return nil, err
	}

	algorithms := defaultAllowedAlgorithms
	if cfg.Algorithm != "" {
		algorithms = []string{cfg.Algorithm}
	}

	parser := jwt.NewParser(
		jwt.WithValidMethods(algorithms),
		jwt.WithIssuer(cfg.Issuer),
		jwt.WithAudience(cfg.Audience),
	)

	return func(ctx context.Context, token string, _ *http.Request) (*auth.TokenInfo, error) {
		claims := jwt.MapClaims{}
		parsed, err := parser.ParseWithClaims(token, claims, func(t *jwt.Token) (any, error) {
			kid, _ := t.Header["kid"].(string)
			return keys.key(ctx, kid)
		})
		if err != nil || !parsed.Valid {
			return nil, fmt.Errorf("%w: %v", auth.ErrInvalidToken, err)
		}

		for _, required := range cfg.RequiredClaims {
			if _, ok := claims[required]; !ok {
				return nil, fmt.Errorf("%w: missing required claim %q", auth.ErrInvalidToken, required)
			}
		}

		info := &auth.TokenInfo{
			Scopes: claimValues(claims, "scp", "scope"),
			Extra:  map[string]any(claims),
		}
		if sub, err := claims.GetSubject(); err == nil {
			info.UserID = sub
		}
		if exp, err := claims.GetExpirationTime(); err == nil && exp != nil {
			info.Expiration = exp.Time
		}
		return info, nil
	}, nil
}

// claimValues returns the authorization values carried by the first
// present claim, accepting either a list or a space-delimited string
// (matching the Python server's authorization_values).
func claimValues(claims jwt.MapClaims, names ...string) []string {
	for _, name := range names {
		raw, ok := claims[name]
		if !ok {
			continue
		}
		switch v := raw.(type) {
		case string:
			return strings.Fields(v)
		case []any:
			out := make([]string, 0, len(v))
			for _, item := range v {
				out = append(out, fmt.Sprint(item))
			}
			return out
		case []string:
			return v
		}
	}
	return nil
}

// authorizationValues resolves the values used by the read/write scope
// gates: standard scp/scope claims come from the verifier-parsed scopes;
// any other claim (for example "roles") is read from the raw claims.
func authorizationValues(info *auth.TokenInfo, claim string) []string {
	if claim == "scp" || claim == "scope" {
		return info.Scopes
	}
	return claimValues(jwt.MapClaims(info.Extra), claim)
}

// requireScope enforces a per-tool scope gate. It no-ops when auth is
// disabled, no scope is configured, or there is no authenticated request
// context (the stdio transport is never authenticated).
func (s *Server) requireScope(ctx context.Context, scope string) error {
	if !s.cfg.Server.Auth.Enabled() || scope == "" {
		return nil
	}
	info := auth.TokenInfoFromContext(ctx)
	if info == nil {
		return nil
	}
	for _, v := range authorizationValues(info, s.cfg.Server.Auth.AuthorizationClaim) {
		if v == scope {
			return nil
		}
	}
	return fmt.Errorf("token is missing the required scope %q", scope)
}

// keySource resolves signing keys either from a static PEM public key or
// from a JWKS endpoint (with kid lookup and periodic refresh).
type keySource struct {
	static crypto.PublicKey // non-nil for public_key configs

	jwksURI string
	client  *http.Client

	mu      sync.Mutex
	keys    map[string]crypto.PublicKey // kid -> key
	fetched time.Time
}

const jwksRefreshInterval = 5 * time.Minute

func newKeySource(cfg *AuthConfig) (*keySource, error) {
	if cfg.PublicKey != "" {
		key, err := parsePublicKeyPEM([]byte(cfg.PublicKey))
		if err != nil {
			return nil, fmt.Errorf("mcp config: auth.public_key: %w", err)
		}
		return &keySource{static: key}, nil
	}
	return &keySource{
		jwksURI: cfg.JWKSURI,
		client:  &http.Client{Timeout: 10 * time.Second},
		keys:    map[string]crypto.PublicKey{},
	}, nil
}

func (k *keySource) key(ctx context.Context, kid string) (crypto.PublicKey, error) {
	if k.static != nil {
		return k.static, nil
	}

	k.mu.Lock()
	defer k.mu.Unlock()

	// Refresh on a fixed interval only — never per unknown/absent kid, so
	// attacker-supplied kids cannot force a JWKS fetch per request.
	if time.Since(k.fetched) >= jwksRefreshInterval || len(k.keys) == 0 {
		if err := k.refreshLocked(ctx); err != nil && len(k.keys) == 0 {
			return nil, err
		}
	}
	if key, ok := k.keys[kid]; ok {
		return key, nil
	}
	// A token without a kid can still verify when the JWKS has exactly
	// one key.
	if kid == "" && len(k.keys) == 1 {
		for _, key := range k.keys {
			return key, nil
		}
	}
	return nil, fmt.Errorf("no JWKS key found for kid %q", kid)
}

func (k *keySource) refreshLocked(ctx context.Context) error {
	// Record the attempt time regardless of outcome so failures back off
	// for a full interval instead of retrying on every request.
	k.fetched = time.Now()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, k.jwksURI, nil)
	if err != nil {
		return err
	}
	resp, err := k.client.Do(req)
	if err != nil {
		return fmt.Errorf("fetching JWKS %s: %w", k.jwksURI, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fetching JWKS %s: HTTP %d", k.jwksURI, resp.StatusCode)
	}

	var doc struct {
		Keys []jwk `json:"keys"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		return fmt.Errorf("parsing JWKS %s: %w", k.jwksURI, err)
	}

	keys := map[string]crypto.PublicKey{}
	for _, key := range doc.Keys {
		pub, err := key.publicKey()
		if err != nil {
			continue // skip unsupported key types
		}
		keys[key.Kid] = pub
	}
	if len(keys) == 0 {
		return fmt.Errorf("JWKS %s contains no usable keys", k.jwksURI)
	}
	k.keys = keys
	k.fetched = time.Now()
	return nil
}

// jwk is the subset of RFC 7517 needed to build RSA, EC, and Ed25519
// public keys.
type jwk struct {
	Kty string `json:"kty"`
	Kid string `json:"kid"`
	// RSA
	N string `json:"n"`
	E string `json:"e"`
	// EC / OKP
	Crv string `json:"crv"`
	X   string `json:"x"`
	Y   string `json:"y"`
}

func (j *jwk) publicKey() (crypto.PublicKey, error) {
	b64 := func(s string) ([]byte, error) { return base64.RawURLEncoding.DecodeString(s) }
	switch j.Kty {
	case "RSA":
		n, err := b64(j.N)
		if err != nil {
			return nil, err
		}
		e, err := b64(j.E)
		if err != nil {
			return nil, err
		}
		return &rsa.PublicKey{
			N: new(big.Int).SetBytes(n),
			E: int(new(big.Int).SetBytes(e).Int64()),
		}, nil
	case "EC":
		var curve elliptic.Curve
		switch j.Crv {
		case "P-256":
			curve = elliptic.P256()
		case "P-384":
			curve = elliptic.P384()
		case "P-521":
			curve = elliptic.P521()
		default:
			return nil, fmt.Errorf("unsupported EC curve %q", j.Crv)
		}
		x, err := b64(j.X)
		if err != nil {
			return nil, err
		}
		y, err := b64(j.Y)
		if err != nil {
			return nil, err
		}
		return &ecdsa.PublicKey{
			Curve: curve,
			X:     new(big.Int).SetBytes(x),
			Y:     new(big.Int).SetBytes(y),
		}, nil
	case "OKP":
		if j.Crv != "Ed25519" {
			return nil, fmt.Errorf("unsupported OKP curve %q", j.Crv)
		}
		x, err := b64(j.X)
		if err != nil {
			return nil, err
		}
		return ed25519.PublicKey(x), nil
	}
	return nil, fmt.Errorf("unsupported JWK key type %q", j.Kty)
}

// parsePublicKeyPEM parses an RSA, EC, or Ed25519 public key from PEM
// (PKIX "PUBLIC KEY" or PKCS#1 "RSA PUBLIC KEY" blocks).
func parsePublicKeyPEM(data []byte) (crypto.PublicKey, error) {
	block, _ := pem.Decode(data)
	if block == nil {
		return nil, fmt.Errorf("no PEM block found")
	}
	switch block.Type {
	case "RSA PUBLIC KEY":
		return x509.ParsePKCS1PublicKey(block.Bytes)
	case "PUBLIC KEY":
		return x509.ParsePKIXPublicKey(block.Bytes)
	case "CERTIFICATE":
		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return nil, err
		}
		return cert.PublicKey, nil
	}
	return nil, fmt.Errorf("unsupported PEM block type %q", block.Type)
}
