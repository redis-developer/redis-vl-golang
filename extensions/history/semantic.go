package history

import (
	"context"
	"fmt"

	"github.com/redis/go-redis/v9"

	redisvl "github.com/redis/redis-vl-golang"
	"github.com/redis/redis-vl-golang/extensions/vectorize"
	"github.com/redis/redis-vl-golang/query"
	"github.com/redis/redis-vl-golang/schema"
	"github.com/redis/redis-vl-golang/vectors"
)

// SemanticMessageHistoryOptions configure a SemanticMessageHistory.
type SemanticMessageHistoryOptions struct {
	// SessionTag links entries to a conversation session (default: new ULID).
	SessionTag string
	// Prefix for Redis keys (default: the history name).
	Prefix string
	// DistanceThreshold is the maximum semantic distance for relevance
	// retrieval (default 0.3).
	DistanceThreshold float64
	// Overwrite recreates the index schema if it already exists.
	Overwrite bool
}

// SemanticMessageHistory stores chat messages with content embeddings and
// supports retrieval by semantic relevance (port of
// redisvl.extensions.message_history.SemanticMessageHistory).
type SemanticMessageHistory struct {
	MessageHistory
	vectorizer        vectorize.Vectorizer
	distanceThreshold float64
}

// NewSemanticMessageHistory creates (and if necessary FT.CREATEs) a
// semantic message history.
func NewSemanticMessageHistory(ctx context.Context, client redis.UniversalClient, name string, vectorizer vectorize.Vectorizer, opts ...SemanticMessageHistoryOptions) (*SemanticMessageHistory, error) {
	var o SemanticMessageHistoryOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	if vectorizer == nil {
		return nil, fmt.Errorf("a vectorizer is required (no default local model in the Go port)")
	}
	if o.SessionTag == "" {
		o.SessionTag = redisvl.NewULID()
	}
	if o.Prefix == "" {
		o.Prefix = name
	}
	if o.DistanceThreshold == 0 {
		o.DistanceThreshold = 0.3
	}

	vf, err := schema.NewVectorField(vectorField, schema.VectorAttrs{
		Dims:           vectorizer.Dims(),
		Algorithm:      schema.Flat,
		Datatype:       string(vectorizer.Dtype()),
		DistanceMetric: schema.Cosine,
	})
	if err != nil {
		return nil, err
	}
	s, err := schema.NewIndexSchema(
		schema.IndexInfo{Name: name, Prefixes: []string{o.Prefix}},
		schema.NewTagField(roleField),
		schema.NewTextField(contentField),
		schema.NewTagField(toolField),
		schema.NewNumericField(timestampField),
		schema.NewTagField(sessionField),
		schema.NewTextField(metadataField),
		vf,
	)
	if err != nil {
		return nil, err
	}

	h := &SemanticMessageHistory{
		MessageHistory: MessageHistory{
			name:       name,
			sessionTag: o.SessionTag,
			index:      redisvl.NewSearchIndex(s, client),
			client:     client,
		},
		vectorizer:        vectorizer,
		distanceThreshold: o.DistanceThreshold,
	}
	if err := h.index.Create(ctx, redisvl.CreateOptions{Overwrite: o.Overwrite}); err != nil {
		return nil, err
	}
	return h, nil
}

// DistanceThreshold returns the relevance distance threshold.
func (h *SemanticMessageHistory) DistanceThreshold() float64 { return h.distanceThreshold }

// SetDistanceThreshold updates the relevance distance threshold.
func (h *SemanticMessageHistory) SetDistanceThreshold(t float64) { h.distanceThreshold = t }

// AddMessages inserts messages, embedding their content for semantic
// retrieval.
func (h *SemanticMessageHistory) AddMessages(ctx context.Context, messages []Message, sessionTag ...string) error {
	tag := h.sessionTag
	if len(sessionTag) > 0 && sessionTag[0] != "" {
		tag = sessionTag[0]
	}

	contents := make([]string, len(messages))
	for i, m := range messages {
		contents[i] = m.Content
	}
	embeddings, err := h.vectorizer.EmbedMany(ctx, contents)
	if err != nil {
		return fmt.Errorf("vectorizing messages: %w", err)
	}
	if len(embeddings) != len(messages) {
		return fmt.Errorf("vectorizer returned %d embeddings for %d messages", len(embeddings), len(messages))
	}

	data := make([]map[string]any, 0, len(messages))
	for i, m := range messages {
		fields, err := messageFields(m, tag)
		if err != nil {
			return err
		}
		blob, err := vectors.ToBuffer(embeddings[i], h.vectorizer.Dtype())
		if err != nil {
			return err
		}
		fields[vectorField] = blob
		data = append(data, fields)
	}
	_, err = h.index.Load(ctx, data, redisvl.LoadOptions{IDField: idField})
	return err
}

// AddMessage inserts a single message with its embedding.
func (h *SemanticMessageHistory) AddMessage(ctx context.Context, message Message, sessionTag ...string) error {
	return h.AddMessages(ctx, []Message{message}, sessionTag...)
}

// Store inserts a prompt/response exchange as "user" and "llm" messages
// with embeddings.
func (h *SemanticMessageHistory) Store(ctx context.Context, prompt, response string, sessionTag ...string) error {
	return h.AddMessages(ctx, []Message{
		{Role: "user", Content: prompt},
		{Role: "llm", Content: response},
	}, sessionTag...)
}

// GetRelevantOptions customize GetRelevant.
type GetRelevantOptions struct {
	// TopK is the maximum number of messages to return (default 5).
	TopK int
	// SessionTag overrides the instance default.
	SessionTag string
	// Roles filters messages by role.
	Roles []string
	// DistanceThreshold overrides the instance default.
	DistanceThreshold float64
	// FallBack returns the most recent messages when nothing relevant is
	// found.
	FallBack bool
}

// GetRelevant retrieves messages semantically related to the prompt.
func (h *SemanticMessageHistory) GetRelevant(ctx context.Context, prompt string, opts ...GetRelevantOptions) ([]Message, error) {
	var o GetRelevantOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	if o.TopK == 0 {
		o.TopK = 5
	}
	if o.TopK < 0 {
		return nil, fmt.Errorf("top_k must be an integer greater than or equal to 0")
	}
	if err := validateRoles(o.Roles); err != nil {
		return nil, err
	}
	threshold := o.DistanceThreshold
	if threshold == 0 {
		threshold = h.distanceThreshold
	}

	vector, err := h.vectorizer.Embed(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("vectorizing prompt: %w", err)
	}

	returnFields := []string{
		idField, sessionField, roleField, contentField,
		timestampField, toolField, metadataField,
	}
	fe := rolesFilter(h.sessionFilter(o.SessionTag), o.Roles)

	q := query.NewVectorRangeQuery(vectorField, vector).
		DistanceThreshold(threshold).
		NumResults(o.TopK).
		Dtype(h.vectorizer.Dtype()).
		ReturnFields(returnFields...).
		Filter(fe)

	docs, err := h.index.Query(ctx, q)
	if err != nil {
		return nil, err
	}

	if len(docs) == 0 && o.FallBack {
		return h.GetRecent(ctx, GetRecentOptions{
			TopK: o.TopK, SessionTag: o.SessionTag, Roles: o.Roles,
		})
	}
	return docsToMessages(docs), nil
}
