package mcpserver

import (
	"strings"
	"testing"
)

// resp2Fields mirrors the FT.INFO "attributes" shape on RESP2 for a hash
// index: flat pair arrays.
func resp2Fields() map[string]*inspectedField {
	fields := map[string]*inspectedField{}
	for _, f := range []*inspectedField{
		{Name: "title", Type: "text", Attrs: map[string]any{}},
		{Name: "genre", Type: "tag", Attrs: map[string]any{}},
		{Name: "embedding", Type: "vector", Attrs: map[string]any{}},
	} {
		fields[f.Name] = f
	}
	return fields
}

func TestApplySchemaOverrides(t *testing.T) {
	fields := resp2Fields()
	overrides := &SchemaOverrides{Fields: []SchemaOverrideField{{
		Name: "embedding",
		Type: "vector",
		Attrs: map[string]any{
			"dims":      1536,
			"Data_Type": "float32", // keys are lowercased on merge
		},
	}}}
	if err := applySchemaOverrides(fields, overrides); err != nil {
		t.Fatal(err)
	}
	if fields["embedding"].Attrs["dims"] != 1536 {
		t.Errorf("dims = %v", fields["embedding"].Attrs["dims"])
	}
	if fields["embedding"].Attrs["data_type"] != "float32" {
		t.Errorf("data_type = %v", fields["embedding"].Attrs["data_type"])
	}
}

func TestApplySchemaOverridesRejects(t *testing.T) {
	cases := []struct {
		name     string
		override SchemaOverrideField
		wantMsg  string
	}{
		{
			"unknown field",
			SchemaOverrideField{Name: "missing", Type: "text"},
			"not found in inspected schema",
		},
		{
			"type change",
			SchemaOverrideField{Name: "genre", Type: "text"},
			"cannot change discovered field type",
		},
		{
			"path change",
			SchemaOverrideField{Name: "title", Type: "text", Path: "$.title"},
			"cannot change discovered field path",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := applySchemaOverrides(resp2Fields(), &SchemaOverrides{
				Fields: []SchemaOverrideField{tc.override},
			})
			if err == nil {
				t.Fatal("override accepted, want error")
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Errorf("error %q does not contain %q", err, tc.wantMsg)
			}
		})
	}
}

func TestValidateRuntimeMapping(t *testing.T) {
	fields := resp2Fields()

	ok := &Config{Runtime: RuntimeConfig{
		TextFieldName:         "title",
		VectorFieldName:       "embedding",
		DefaultEmbedTextField: "title",
	}}
	if err := validateRuntimeMapping(fields, ok); err != nil {
		t.Errorf("valid mapping rejected: %v", err)
	}

	missing := &Config{Runtime: RuntimeConfig{TextFieldName: "nope"}}
	if err := validateRuntimeMapping(fields, missing); err == nil {
		t.Error("missing text field accepted")
	}

	notVector := &Config{Runtime: RuntimeConfig{VectorFieldName: "genre"}}
	if err := validateRuntimeMapping(fields, notVector); err == nil {
		t.Error("non-vector field accepted as vector_field_name")
	}
}

func TestAsStringMap(t *testing.T) {
	// RESP2 flat pairs
	m, ok := asStringMap([]any{"identifier", "title", "type", "TEXT"})
	if !ok || m["identifier"] != "title" || m["type"] != "TEXT" {
		t.Errorf("RESP2 pairs: %v %v", m, ok)
	}
	// RESP3 map
	m, ok = asStringMap(map[any]any{"identifier": "title"})
	if !ok || m["identifier"] != "title" {
		t.Errorf("RESP3 map: %v %v", m, ok)
	}
	// odd-length array is not a map
	if _, ok := asStringMap([]any{"a"}); ok {
		t.Error("odd-length array accepted")
	}
}
