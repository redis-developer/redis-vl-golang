package history

import (
	"strings"
	"testing"
)

func TestMessageFields(t *testing.T) {
	fields, err := messageFields(Message{Role: "user", Content: "hi"}, "sess1")
	if err != nil {
		t.Fatal(err)
	}
	if fields[roleField] != "user" || fields[contentField] != "hi" || fields[sessionField] != "sess1" {
		t.Errorf("fields = %v", fields)
	}
	id := fields[idField].(string)
	// entry_id format: {session}:{timestamp}:{8-char suffix}
	parts := strings.Split(id, ":")
	if len(parts) != 3 || parts[0] != "sess1" || len(parts[2]) != 8 {
		t.Errorf("entry_id = %q", id)
	}
	if _, ok := fields[toolField]; ok {
		t.Error("tool_call_id should be omitted when empty")
	}

	// metadata serialization
	fields, err = messageFields(Message{
		Role: "tool", Content: "x", ToolCallID: "t1",
		Metadata: map[string]any{"k": "v"},
	}, "s")
	if err != nil {
		t.Fatal(err)
	}
	if fields[toolField] != "t1" || fields[metadataField] != `{"k":"v"}` {
		t.Errorf("fields = %v", fields)
	}

	// invalid role
	if _, err := messageFields(Message{Role: "robot", Content: "x"}, "s"); err == nil {
		t.Error("expected invalid role error")
	}
}

func TestRolesFilter(t *testing.T) {
	base := (&MessageHistory{sessionTag: "s1"}).sessionFilter("")
	if got := base.String(); got != "@session_tag:{s1}" {
		t.Errorf("session filter = %q", got)
	}
	combined := rolesFilter(base, []string{"user", "llm"})
	want := "(@session_tag:{s1} (@role:{user} | @role:{llm}))"
	if got := combined.String(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestValidateRoles(t *testing.T) {
	if err := validateRoles([]string{"user", "assistant", "system", "tool", "llm"}); err != nil {
		t.Error(err)
	}
	if err := validateRoles([]string{"alien"}); err == nil {
		t.Error("expected error for invalid role")
	}
}

func TestDocsToMessages(t *testing.T) {
	docs := []map[string]any{{
		idField:        "s:1:abc",
		roleField:      "user",
		contentField:   "hello",
		sessionField:   "s",
		timestampField: "1705320000.5",
		metadataField:  `{"a":1}`,
	}}
	msgs := docsToMessages(docs)
	if len(msgs) != 1 {
		t.Fatal("no messages")
	}
	m := msgs[0]
	if m.Role != "user" || m.Content != "hello" || m.Timestamp != 1705320000.5 {
		t.Errorf("message = %+v", m)
	}
	meta, ok := m.Metadata.(map[string]any)
	if !ok || meta["a"] != float64(1) {
		t.Errorf("metadata = %v", m.Metadata)
	}
}
