package mcpserver

import (
	"context"
	"fmt"
	"strings"

	"github.com/redis/go-redis/v9"
)

// inspectedField is one field discovered from FT.INFO, with any configured
// schema overrides merged in (port of the Python server's inspected-schema
// handling).
type inspectedField struct {
	Name  string
	Type  string // lowercase: text, tag, numeric, geo, vector
	Path  string // JSON path ("" for hash storage)
	Attrs map[string]any
}

// inspectSchema reads FT.INFO for the index and returns the discovered
// fields, preserving field identity even when attrs are incomplete (as
// older Redis versions omit vector attrs).
func inspectSchema(ctx context.Context, client redis.UniversalClient, indexName string) (map[string]*inspectedField, error) {
	reply, err := client.Do(ctx, "FT.INFO", indexName).Result()
	if err != nil {
		return nil, fmt.Errorf("inspecting index %q: %w", indexName, err)
	}
	info, ok := asStringMap(reply)
	if !ok {
		return nil, fmt.Errorf("inspecting index %q: unexpected FT.INFO reply shape", indexName)
	}

	storage := "hash"
	if def, ok := asStringMap(info["index_definition"]); ok {
		if kt, ok := def["key_type"]; ok && strings.EqualFold(fmt.Sprint(kt), "JSON") {
			storage = "json"
		}
	}

	attributes, _ := info["attributes"].([]any)
	fields := map[string]*inspectedField{}
	for _, raw := range attributes {
		attr, ok := asAttributeMap(raw)
		if !ok {
			continue
		}
		field := &inspectedField{
			Type:  strings.ToLower(fmt.Sprint(attr["type"])),
			Attrs: map[string]any{},
		}
		identifier := fmt.Sprint(attr["identifier"])
		alias := identifier
		if a, ok := attr["attribute"]; ok {
			alias = fmt.Sprint(a)
		}
		// Python parity: hash indexes are addressed by the stored field
		// (identifier); JSON indexes by the alias, with the JSON path kept
		// separately.
		field.Name = identifier
		if storage == "json" {
			field.Name = alias
			field.Path = identifier
		}
		// Preserve any extra discovered attributes (dims, distance metric,
		// data type, ...) for override merging and validation.
		for key, value := range attr {
			switch key {
			case "identifier", "attribute", "type":
				continue
			}
			field.Attrs[key] = value
		}
		if field.Name != "" && field.Name != "<nil>" {
			fields[field.Name] = field
		}
	}
	return fields, nil
}

// attributeFlags are the standalone (valueless) tokens FT.INFO may emit in
// a field's attribute array on RESP2.
var attributeFlags = map[string]bool{
	"sortable": true, "unf": true, "nostem": true, "noindex": true,
	"casesensitive": true, "withsuffixtrie": true,
	"indexempty": true, "indexmissing": true,
}

// asAttributeMap parses one FT.INFO attribute entry. Unlike the strict
// pair arrays handled by asStringMap, RESP2 attribute entries may contain
// bare flag tokens (SORTABLE, NOINDEX, ...) between key/value pairs.
func asAttributeMap(v any) (map[string]any, bool) {
	switch m := v.(type) {
	case map[string]any, map[any]any:
		out, ok := asStringMap(m)
		if !ok {
			return nil, false
		}
		lowered := make(map[string]any, len(out))
		for key, value := range out {
			lowered[strings.ToLower(key)] = value
		}
		return lowered, true
	case []any:
		out := map[string]any{}
		for i := 0; i < len(m); {
			key := strings.ToLower(fmt.Sprint(m[i]))
			if attributeFlags[key] || i+1 == len(m) {
				out[key] = true
				i++
				continue
			}
			out[key] = m[i+1]
			i += 2
		}
		return out, true
	}
	return nil, false
}

// asStringMap normalizes RESP2 (flat pair arrays) and RESP3 (maps) reply
// shapes into a string-keyed map.
func asStringMap(v any) (map[string]any, bool) {
	switch m := v.(type) {
	case map[string]any:
		return m, true
	case map[any]any:
		out := make(map[string]any, len(m))
		for key, value := range m {
			out[fmt.Sprint(key)] = value
		}
		return out, true
	case []any:
		if len(m)%2 != 0 {
			return nil, false
		}
		out := make(map[string]any, len(m)/2)
		for i := 0; i+1 < len(m); i += 2 {
			out[fmt.Sprint(m[i])] = m[i+1]
		}
		return out, true
	}
	return nil, false
}

// applySchemaOverrides merges configured overrides into the inspected
// fields, enforcing the Python server's identity rules: overrides may only
// patch attrs of discovered fields, never add fields or change a field's
// type or path.
func applySchemaOverrides(fields map[string]*inspectedField, overrides *SchemaOverrides) error {
	if overrides == nil {
		return nil
	}
	for _, override := range overrides.Fields {
		discovered, ok := fields[override.Name]
		if !ok {
			return fmt.Errorf("schema_overrides.fields %q not found in inspected schema", override.Name)
		}
		if !strings.EqualFold(override.Type, discovered.Type) {
			return fmt.Errorf("schema_overrides.fields %q cannot change discovered field type", override.Name)
		}
		if override.Path != "" && override.Path != discovered.Path {
			return fmt.Errorf("schema_overrides.fields %q cannot change discovered field path", override.Name)
		}
		for key, value := range override.Attrs {
			discovered.Attrs[strings.ToLower(key)] = value
		}
	}
	return nil
}

// validateRuntimeMapping ensures the configured runtime field names point
// at fields of the right type in the effective schema (port of the Python
// validate_runtime_mapping).
func validateRuntimeMapping(fields map[string]*inspectedField, cfg *Config) error {
	check := func(name, option string) error {
		if name == "" {
			return nil
		}
		if _, ok := fields[name]; !ok {
			return fmt.Errorf("runtime.%s %q not found in schema", option, name)
		}
		return nil
	}
	if err := check(cfg.Runtime.TextFieldName, "text_field_name"); err != nil {
		return err
	}
	if err := check(cfg.Runtime.DefaultEmbedTextField, "default_embed_text_field"); err != nil {
		return err
	}
	if name := cfg.Runtime.VectorFieldName; name != "" {
		field, ok := fields[name]
		if !ok {
			return fmt.Errorf("runtime.vector_field_name %q not found in schema", name)
		}
		if field.Type != "vector" {
			return fmt.Errorf("runtime.vector_field_name %q must reference a vector field", name)
		}
	}
	return nil
}
