// Package history provides Redis-backed chat message history with recency
// (MessageHistory) and semantic relevance (SemanticMessageHistory)
// retrieval. Port of redisvl.extensions.message_history.
package history

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	redisvl "github.com/redis/redis-vl-golang"
	"github.com/redis/redis-vl-golang/filter"
	"github.com/redis/redis-vl-golang/query"
	"github.com/redis/redis-vl-golang/schema"
)

// Field names shared with the Python implementation
// (redisvl.extensions.constants).
const (
	idField        = "entry_id"
	roleField      = "role"
	contentField   = "content"
	toolField      = "tool_call_id"
	timestampField = "timestamp"
	sessionField   = "session_tag"
	metadataField  = "metadata"
	vectorField    = "vector_field"
)

// Valid chat roles ("llm" is accepted for backward compatibility with the
// Python library).
var validRoles = map[string]bool{
	"user": true, "assistant": true, "system": true, "tool": true, "llm": true,
}

// Message is a single chat message.
type Message struct {
	EntryID    string
	Role       string
	Content    string
	SessionTag string
	Timestamp  float64
	ToolCallID string
	Metadata   any
}

// MessageHistoryOptions configure a MessageHistory.
type MessageHistoryOptions struct {
	// SessionTag links entries to a conversation session (default: new ULID).
	SessionTag string
	// Prefix for Redis keys (default: the history name).
	Prefix string
}

// MessageHistory stores chat messages and retrieves them by recency
// (port of redisvl.extensions.message_history.MessageHistory).
type MessageHistory struct {
	name       string
	sessionTag string
	index      *redisvl.SearchIndex
	client     redis.UniversalClient
}

// NewMessageHistory creates (and if necessary FT.CREATEs) a message history.
func NewMessageHistory(ctx context.Context, client redis.UniversalClient, name string, opts ...MessageHistoryOptions) (*MessageHistory, error) {
	var o MessageHistoryOptions
	if len(opts) > 0 {
		o = opts[0]
	}
	if o.SessionTag == "" {
		o.SessionTag = redisvl.NewULID()
	}
	if o.Prefix == "" {
		o.Prefix = name
	}

	s, err := schema.NewIndexSchema(
		schema.IndexInfo{Name: name, Prefixes: []string{o.Prefix}},
		schema.NewTagField(roleField),
		schema.NewTextField(contentField),
		schema.NewTagField(toolField),
		schema.NewNumericField(timestampField),
		schema.NewTagField(sessionField),
		schema.NewTextField(metadataField),
	)
	if err != nil {
		return nil, err
	}
	h := &MessageHistory{
		name:       name,
		sessionTag: o.SessionTag,
		index:      redisvl.NewSearchIndex(s, client),
		client:     client,
	}
	if err := h.index.Create(ctx); err != nil {
		return nil, err
	}
	return h, nil
}

// Index exposes the underlying SearchIndex.
func (h *MessageHistory) Index() *redisvl.SearchIndex { return h.index }

// SessionTag returns the instance's default session tag.
func (h *MessageHistory) SessionTag() string { return h.sessionTag }

func (h *MessageHistory) sessionFilter(sessionTag string) *filter.Expression {
	if sessionTag == "" {
		sessionTag = h.sessionTag
	}
	return filter.Tag(sessionField).Eq(sessionTag)
}

func validateRoles(roles []string) error {
	for _, r := range roles {
		if !validRoles[r] {
			return fmt.Errorf("invalid role %q; valid roles: user, assistant, system, tool", r)
		}
	}
	return nil
}

func rolesFilter(base *filter.Expression, roles []string) *filter.Expression {
	if len(roles) == 0 {
		return base
	}
	roleFilter := filter.Tag(roleField).Eq(roles[0])
	for _, r := range roles[1:] {
		roleFilter = roleFilter.Or(filter.Tag(roleField).Eq(r))
	}
	return base.And(roleFilter)
}

func nowTS() float64 { return float64(time.Now().UnixNano()) / 1e9 }

func formatTS(ts float64) string { return strconv.FormatFloat(ts, 'f', -1, 64) }

func randSuffix() string {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(fmt.Sprintf("history: crypto/rand failed: %v", err))
	}
	return hex.EncodeToString(b[:])
}

// makeEntryID mirrors Python: "{session}:{timestamp}:{8-char suffix}".
func makeEntryID(sessionTag string, ts float64) string {
	return sessionTag + ":" + formatTS(ts) + ":" + randSuffix()
}

// messageFields builds the stored hash fields for a message.
func messageFields(m Message, sessionTag string) (map[string]any, error) {
	if !validRoles[m.Role] {
		return nil, fmt.Errorf("invalid role %q; valid roles: user, assistant, system, tool", m.Role)
	}
	ts := m.Timestamp
	if ts == 0 {
		ts = nowTS()
	}
	entryID := m.EntryID
	if entryID == "" {
		entryID = makeEntryID(sessionTag, ts)
	}
	fields := map[string]any{
		idField:        entryID,
		roleField:      m.Role,
		contentField:   m.Content,
		sessionField:   sessionTag,
		timestampField: formatTS(ts),
	}
	if m.ToolCallID != "" {
		fields[toolField] = m.ToolCallID
	}
	if m.Metadata != nil {
		metaJSON, err := json.Marshal(m.Metadata)
		if err != nil {
			return nil, err
		}
		fields[metadataField] = string(metaJSON)
	}
	return fields, nil
}

// AddMessages inserts messages into the history, timestamped for later
// sequential retrieval. sessionTag overrides the instance default when
// non-empty.
func (h *MessageHistory) AddMessages(ctx context.Context, messages []Message, sessionTag ...string) error {
	tag := h.sessionTag
	if len(sessionTag) > 0 && sessionTag[0] != "" {
		tag = sessionTag[0]
	}
	data := make([]map[string]any, 0, len(messages))
	for _, m := range messages {
		fields, err := messageFields(m, tag)
		if err != nil {
			return err
		}
		data = append(data, fields)
	}
	_, err := h.index.Load(ctx, data, redisvl.LoadOptions{IDField: idField})
	return err
}

// AddMessage inserts a single message.
func (h *MessageHistory) AddMessage(ctx context.Context, message Message, sessionTag ...string) error {
	return h.AddMessages(ctx, []Message{message}, sessionTag...)
}

// Store inserts a prompt/response exchange as a "user" and an "llm" message.
func (h *MessageHistory) Store(ctx context.Context, prompt, response string, sessionTag ...string) error {
	return h.AddMessages(ctx, []Message{
		{Role: "user", Content: prompt},
		{Role: "llm", Content: response},
	}, sessionTag...)
}

// GetRecentOptions customize GetRecent.
type GetRecentOptions struct {
	// TopK is the number of messages to return (default 5).
	TopK int
	// SessionTag overrides the instance default.
	SessionTag string
	// Roles filters messages by role.
	Roles []string
}

var historyReturnFields = []string{
	idField, sessionField, roleField, contentField,
	toolField, timestampField, metadataField,
}

// GetRecent retrieves the most recent messages in chronological order.
func (h *MessageHistory) GetRecent(ctx context.Context, opts ...GetRecentOptions) ([]Message, error) {
	var o GetRecentOptions
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

	fe := rolesFilter(h.sessionFilter(o.SessionTag), o.Roles)
	q := query.NewFilterQuery(fe).
		ReturnFields(historyReturnFields...).
		NumResults(o.TopK).
		SortByField(timestampField, false)

	docs, err := h.index.Query(ctx, q)
	if err != nil {
		return nil, err
	}
	msgs := docsToMessages(docs)
	reverse(msgs)
	return msgs, nil
}

// Messages returns the full history for the session in chronological order.
func (h *MessageHistory) Messages(ctx context.Context) ([]Message, error) {
	q := query.NewFilterQuery(h.sessionFilter("")).
		ReturnFields(historyReturnFields...).
		NumResults(1000).
		SortByField(timestampField, true)
	docs, err := h.index.Query(ctx, q)
	if err != nil {
		return nil, err
	}
	return docsToMessages(docs), nil
}

// Count returns the number of messages for the session.
func (h *MessageHistory) Count(ctx context.Context, sessionTag ...string) (int64, error) {
	tag := ""
	if len(sessionTag) > 0 {
		tag = sessionTag[0]
	}
	return h.index.Count(ctx, query.NewCountQuery(h.sessionFilter(tag)))
}

// Drop removes a message by entry ID; with an empty id the most recent
// message is removed.
func (h *MessageHistory) Drop(ctx context.Context, id string) error {
	if id == "" {
		recent, err := h.GetRecent(ctx, GetRecentOptions{TopK: 1})
		if err != nil {
			return err
		}
		if len(recent) == 0 {
			return nil
		}
		id = recent[0].EntryID
	}
	_, err := h.index.DropDocuments(ctx, id)
	return err
}

// Clear removes all messages but keeps the index.
func (h *MessageHistory) Clear(ctx context.Context) error {
	_, err := h.index.Clear(ctx)
	return err
}

// Delete removes all messages and the index.
func (h *MessageHistory) Delete(ctx context.Context) error {
	return h.index.Delete(ctx, true)
}

func docsToMessages(docs []map[string]any) []Message {
	msgs := make([]Message, 0, len(docs))
	for _, d := range docs {
		m := Message{
			EntryID:    str(d[idField]),
			Role:       str(d[roleField]),
			Content:    str(d[contentField]),
			SessionTag: str(d[sessionField]),
			ToolCallID: str(d[toolField]),
		}
		m.Timestamp, _ = strconv.ParseFloat(str(d[timestampField]), 64)
		if raw := str(d[metadataField]); raw != "" {
			var meta any
			if err := json.Unmarshal([]byte(raw), &meta); err == nil {
				m.Metadata = meta
			} else {
				m.Metadata = raw
			}
		}
		msgs = append(msgs, m)
	}
	return msgs
}

func str(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

func reverse(msgs []Message) {
	for i, j := 0, len(msgs)-1; i < j; i, j = i+1, j-1 {
		msgs[i], msgs[j] = msgs[j], msgs[i]
	}
}
