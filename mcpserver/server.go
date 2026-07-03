package mcpserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/redis/go-redis/v9"

	redisvl "github.com/redis-developer/redis-vl-golang"
	"github.com/redis-developer/redis-vl-golang/extensions/vectorize"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// Server exposes one existing Redis search index to MCP clients.
type Server struct {
	cfg        *Config
	client     redis.UniversalClient
	index      *redisvl.SearchIndex
	vectorizer vectorize.Vectorizer // nil for text-only search
	dtype      vectors.DataType
	readOnly   bool
	mcp        *mcp.Server
}

// Options tweak server construction.
type Options struct {
	// ReadOnly disables the upsert tool regardless of the config value.
	ReadOnly bool
}

// New connects to Redis, verifies the index exists, builds the configured
// vectorizer, and registers the MCP tools.
func New(ctx context.Context, cfg *Config, opts ...Options) (*Server, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var o Options
	if len(opts) > 0 {
		o = opts[0]
	}

	ropts, err := redis.ParseURL(cfg.Server.RedisURL)
	if err != nil {
		return nil, fmt.Errorf("invalid redis url: %w", err)
	}
	client := redis.NewClient(ropts)

	index, err := redisvl.FromExisting(ctx, cfg.Index.RedisName, client)
	if err != nil {
		_ = client.Close()
		return nil, err
	}

	var vec vectorize.Vectorizer
	if vc := cfg.Index.Vectorizer; vc != nil {
		vec, err = buildVectorizer(ctx, vc)
		if err != nil {
			_ = client.Close()
			return nil, err
		}
	}

	dt, _ := vectors.Parse(cfg.Runtime.Dtype) // validated in cfg.Validate

	s := &Server{
		cfg:        cfg,
		client:     client,
		index:      index,
		vectorizer: vec,
		dtype:      dt,
		readOnly:   cfg.ReadOnly || o.ReadOnly,
	}

	s.mcp = mcp.NewServer(&mcp.Implementation{
		Name:    "redisvl-mcp",
		Version: redisvl.Version,
	}, nil)

	mcp.AddTool(s.mcp, &mcp.Tool{
		Name: "search-records",
		Description: fmt.Sprintf(
			"Search the %q Redis index. Returns matching records with relevance scores.",
			cfg.Index.RedisName),
	}, s.searchTool)

	if !s.readOnly {
		mcp.AddTool(s.mcp, &mcp.Tool{
			Name: "upsert-records",
			Description: fmt.Sprintf(
				"Insert or update records in the %q Redis index. Text in the %q field is embedded automatically.",
				cfg.Index.RedisName, cfg.Runtime.DefaultEmbedTextField),
		}, s.upsertTool)
	}
	return s, nil
}

func buildVectorizer(ctx context.Context, vc *VectorizerConfig) (vectorize.Vectorizer, error) {
	switch strings.ToLower(vc.Class) {
	case "openai":
		return vectorize.NewOpenAIVectorizer(ctx, vectorize.OpenAIConfig{Model: vc.Model, Dims: vc.Dims})
	case "azure_openai":
		return vectorize.NewAzureOpenAIVectorizer(ctx, vectorize.AzureOpenAIConfig{Model: vc.Model, Dims: vc.Dims})
	case "cohere":
		return vectorize.NewCohereVectorizer(ctx, vectorize.CohereConfig{Model: vc.Model, Dims: vc.Dims})
	case "mistral":
		return vectorize.NewMistralVectorizer(ctx, vectorize.MistralConfig{Model: vc.Model, Dims: vc.Dims})
	case "voyageai":
		return vectorize.NewVoyageAIVectorizer(ctx, vectorize.VoyageAIConfig{Model: vc.Model, Dims: vc.Dims})
	case "ollama":
		return vectorize.NewOllamaVectorizer(ctx, vectorize.OllamaConfig{Model: vc.Model, Dims: vc.Dims})
	}
	return nil, fmt.Errorf("unsupported vectorizer class %q", vc.Class)
}

// Close releases the Redis connection.
func (s *Server) Close() error { return s.client.Close() }

// ReadOnly reports whether the upsert tool is disabled.
func (s *Server) ReadOnly() bool { return s.readOnly }

// Run serves MCP over stdio until the context is canceled.
func (s *Server) Run(ctx context.Context) error {
	return s.mcp.Run(ctx, &mcp.StdioTransport{})
}

// RunHTTP serves MCP over Streamable HTTP on addr (host:port) until the
// context is canceled.
func (s *Server) RunHTTP(ctx context.Context, addr string) error {
	handler := mcp.NewStreamableHTTPHandler(
		func(*http.Request) *mcp.Server { return s.mcp }, nil)
	httpServer := &http.Server{Addr: addr, Handler: handler}

	errCh := make(chan error, 1)
	go func() { errCh <- httpServer.ListenAndServe() }()

	select {
	case <-ctx.Done():
		_ = httpServer.Shutdown(context.Background())
		return nil
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}
