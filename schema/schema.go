package schema

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// StorageType is how documents are stored in Redis.
type StorageType string

// Supported storage types.
const (
	Hash StorageType = "hash"
	JSON StorageType = "json"
)

// IndexInfo holds the essential index settings (port of
// redisvl.schema.IndexInfo).
type IndexInfo struct {
	// Name is the unique index name.
	Name string
	// Prefixes are the Redis key prefixes covered by this index
	// (default ["rvl"]). Multiple prefixes are supported but discouraged.
	Prefixes []string
	// KeySeparator joins prefix and id in keys (default ":").
	KeySeparator string
	// StorageType is hash or json (default hash).
	StorageType StorageType
	// Stopwords: nil uses Redis defaults, an empty non-nil slice disables
	// stopwords (STOPWORDS 0), otherwise a custom list.
	Stopwords *[]string
}

func (i *IndexInfo) applyDefaults() error {
	if i.Name == "" {
		return fmt.Errorf("index name is required")
	}
	if len(i.Prefixes) == 0 {
		i.Prefixes = []string{"rvl"}
	}
	if i.KeySeparator == "" {
		i.KeySeparator = ":"
	}
	if i.StorageType == "" {
		i.StorageType = Hash
	}
	switch i.StorageType {
	case Hash, JSON:
	default:
		return fmt.Errorf("invalid storage type: %s", i.StorageType)
	}
	return nil
}

// IndexSchema is a schema definition for a Redis search index (port of
// redisvl.schema.IndexSchema).
type IndexSchema struct {
	Index   IndexInfo
	Version string

	fields     map[string]*Field
	fieldOrder []string
}

// NewIndexSchema creates a schema from index info and fields.
func NewIndexSchema(info IndexInfo, fields ...Field) (*IndexSchema, error) {
	if err := info.applyDefaults(); err != nil {
		return nil, err
	}
	s := &IndexSchema{Index: info, Version: "0.1.0", fields: map[string]*Field{}}
	for _, f := range fields {
		if err := s.AddField(f); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// AddField adds a field, enforcing unique names and defaulting the JSON path
// for JSON storage.
func (s *IndexSchema) AddField(f Field) error {
	if f.Name == "" {
		return fmt.Errorf("field name is required")
	}
	if _, exists := s.fields[f.Name]; exists {
		return fmt.Errorf("duplicate field name: %s; field names must be unique across all fields", f.Name)
	}
	if s.Index.StorageType == JSON {
		if f.Path == "" {
			f.Path = "$." + f.Name
		}
	} else {
		f.Path = "" // path is ignored for hash storage
	}
	if f.Type == TypeVector && f.Vector != nil {
		if err := f.Vector.Validate(); err != nil {
			return fmt.Errorf("vector field %q: %w", f.Name, err)
		}
	}
	s.fields[f.Name] = &f
	s.fieldOrder = append(s.fieldOrder, f.Name)
	return nil
}

// RemoveField removes a field by name (no-op if absent).
func (s *IndexSchema) RemoveField(name string) {
	if _, ok := s.fields[name]; !ok {
		return
	}
	delete(s.fields, name)
	for i, n := range s.fieldOrder {
		if n == name {
			s.fieldOrder = append(s.fieldOrder[:i], s.fieldOrder[i+1:]...)
			break
		}
	}
}

// Field returns the field with the given name, or nil.
func (s *IndexSchema) Field(name string) *Field {
	return s.fields[name]
}

// FieldNames lists field names in insertion order.
func (s *IndexSchema) FieldNames() []string {
	out := make([]string, len(s.fieldOrder))
	copy(out, s.fieldOrder)
	return out
}

// Fields lists fields in insertion order.
func (s *IndexSchema) Fields() []*Field {
	out := make([]*Field, 0, len(s.fieldOrder))
	for _, n := range s.fieldOrder {
		out = append(out, s.fields[n])
	}
	return out
}

// SchemaArgs renders the SCHEMA portion of the FT.CREATE command.
func (s *IndexSchema) SchemaArgs() ([]any, error) {
	if len(s.fieldOrder) == 0 {
		return nil, fmt.Errorf("no fields defined for index")
	}
	var args []any
	for _, name := range s.fieldOrder {
		fa, err := s.fields[name].RedisArgs()
		if err != nil {
			return nil, err
		}
		args = append(args, fa...)
	}
	return args, nil
}

// ------------------------------------------------------------- YAML / maps

type rawIndex struct {
	Name         string    `yaml:"name" json:"name"`
	Prefix       any       `yaml:"prefix" json:"prefix"`
	KeySeparator string    `yaml:"key_separator" json:"key_separator"`
	StorageType  string    `yaml:"storage_type" json:"storage_type"`
	Stopwords    *[]string `yaml:"stopwords" json:"stopwords"`
}

type rawField struct {
	Name  string         `yaml:"name" json:"name"`
	Type  string         `yaml:"type" json:"type"`
	Path  string         `yaml:"path" json:"path"`
	Attrs map[string]any `yaml:"attrs" json:"attrs"`
}

type rawSchema struct {
	Version string     `yaml:"version" json:"version"`
	Index   rawIndex   `yaml:"index" json:"index"`
	Fields  []rawField `yaml:"fields" json:"fields"`
}

// FromYAML parses a schema from YAML bytes (same format as the Python
// library's schema.yaml files).
func FromYAML(data []byte) (*IndexSchema, error) {
	var raw rawSchema
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("invalid schema yaml: %w", err)
	}
	return fromRaw(&raw)
}

// FromYAMLFile reads and parses a schema YAML file.
func FromYAMLFile(path string) (*IndexSchema, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("schema file %s: %w", path, err)
	}
	return FromYAML(data)
}

func fromRaw(raw *rawSchema) (*IndexSchema, error) {
	info := IndexInfo{
		Name:         raw.Index.Name,
		KeySeparator: raw.Index.KeySeparator,
		StorageType:  StorageType(strings.ToLower(raw.Index.StorageType)),
		Stopwords:    raw.Index.Stopwords,
	}
	switch p := raw.Index.Prefix.(type) {
	case string:
		info.Prefixes = []string{p}
	case []any:
		for _, v := range p {
			info.Prefixes = append(info.Prefixes, fmt.Sprint(v))
		}
	case nil:
	default:
		return nil, fmt.Errorf("index prefix must be a string or list of strings")
	}

	s, err := NewIndexSchema(info)
	if err != nil {
		return nil, err
	}
	if raw.Version != "" {
		s.Version = raw.Version
	}
	for _, rf := range raw.Fields {
		f, err := fieldFromRaw(rf)
		if err != nil {
			return nil, err
		}
		if rf.Path != "" {
			f.Path = rf.Path
		}
		if err := s.AddField(f); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func fieldFromRaw(rf rawField) (Field, error) {
	attrs := rf.Attrs
	switch FieldType(strings.ToLower(rf.Type)) {
	case TypeText:
		a := TextAttrs{Weight: 1}
		a.Sortable = getBool(attrs, "sortable")
		a.IndexMissing = getBool(attrs, "index_missing")
		a.NoIndex = getBool(attrs, "no_index")
		if w, ok := getFloat(attrs, "weight"); ok {
			a.Weight = w
		}
		a.NoStem = getBool(attrs, "no_stem")
		a.WithSuffixTrie = getBool(attrs, "withsuffixtrie")
		a.PhoneticMatcher = getString(attrs, "phonetic_matcher")
		a.IndexEmpty = getBool(attrs, "index_empty")
		a.UNF = getBool(attrs, "unf")
		return Field{Name: rf.Name, Type: TypeText, Text: &a}, nil

	case TypeTag:
		a := TagAttrs{Separator: ","}
		a.Sortable = getBool(attrs, "sortable")
		a.IndexMissing = getBool(attrs, "index_missing")
		a.NoIndex = getBool(attrs, "no_index")
		if sep := getString(attrs, "separator"); sep != "" {
			a.Separator = sep
		}
		a.CaseSensitive = getBool(attrs, "case_sensitive")
		a.WithSuffixTrie = getBool(attrs, "withsuffixtrie")
		a.IndexEmpty = getBool(attrs, "index_empty")
		return Field{Name: rf.Name, Type: TypeTag, Tag: &a}, nil

	case TypeNumeric:
		a := NumericAttrs{}
		a.Sortable = getBool(attrs, "sortable")
		a.IndexMissing = getBool(attrs, "index_missing")
		a.NoIndex = getBool(attrs, "no_index")
		a.UNF = getBool(attrs, "unf")
		return Field{Name: rf.Name, Type: TypeNumeric, Numeric: &a}, nil

	case TypeGeo:
		a := GeoAttrs{}
		a.Sortable = getBool(attrs, "sortable")
		a.IndexMissing = getBool(attrs, "index_missing")
		a.NoIndex = getBool(attrs, "no_index")
		return Field{Name: rf.Name, Type: TypeGeo, Geo: &a}, nil

	case TypeVector:
		if attrs == nil {
			return Field{}, fmt.Errorf("vector field %q requires attrs with algorithm and dims", rf.Name)
		}
		a := VectorAttrs{}
		if d, ok := getInt(attrs, "dims"); ok {
			a.Dims = d
		} else {
			return Field{}, fmt.Errorf("must provide dims param for the vector field %q", rf.Name)
		}
		algo := getString(attrs, "algorithm")
		if algo == "" {
			return Field{}, fmt.Errorf("must provide algorithm param for the vector field %q", rf.Name)
		}
		a.Algorithm = Algorithm(strings.ToUpper(algo))
		a.Datatype = getString(attrs, "datatype")
		a.DistanceMetric = DistanceMetric(strings.ToUpper(getString(attrs, "distance_metric")))
		a.IndexMissing = getBool(attrs, "index_missing")
		if v, ok := getInt(attrs, "initial_cap"); ok {
			a.InitialCap = &v
		}
		if v, ok := getInt(attrs, "block_size"); ok {
			a.BlockSize = &v
		}
		if v, ok := getInt(attrs, "m"); ok {
			a.M = &v
		}
		if v, ok := getInt(attrs, "ef_construction"); ok {
			a.EfConstruction = &v
		}
		if v, ok := getInt(attrs, "ef_runtime"); ok {
			a.EfRuntime = &v
		}
		if v, ok := getFloat(attrs, "epsilon"); ok {
			a.Epsilon = &v
		}
		if v, ok := getInt(attrs, "graph_max_degree"); ok {
			a.GraphMaxDegree = &v
		}
		if v, ok := getInt(attrs, "construction_window_size"); ok {
			a.ConstructionWindowSize = &v
		}
		if v, ok := getInt(attrs, "search_window_size"); ok {
			a.SearchWindowSize = &v
		}
		if c := getString(attrs, "compression"); c != "" {
			a.Compression = Compression(c)
		}
		if v, ok := getInt(attrs, "reduce"); ok {
			a.Reduce = &v
		}
		if v, ok := getInt(attrs, "training_threshold"); ok {
			a.TrainingThreshold = &v
		}
		if err := a.Validate(); err != nil {
			return Field{}, fmt.Errorf("vector field %q: %w", rf.Name, err)
		}
		return Field{Name: rf.Name, Type: TypeVector, Vector: &a}, nil
	}
	return Field{}, fmt.Errorf("unknown field type: %s", rf.Type)
}

// ToYAML serializes the schema to YAML in the same layout as the Python
// library.
func (s *IndexSchema) ToYAML() ([]byte, error) {
	raw := s.toRaw()
	return yaml.Marshal(raw)
}

// ToYAMLFile writes the schema to a YAML file.
func (s *IndexSchema) ToYAMLFile(path string, overwrite bool) error {
	if !overwrite {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("schema file %s already exists", path)
		}
	}
	data, err := s.ToYAML()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func (s *IndexSchema) toRaw() *rawSchema {
	var prefix any
	if len(s.Index.Prefixes) == 1 {
		prefix = s.Index.Prefixes[0]
	} else {
		prefix = s.Index.Prefixes
	}
	raw := &rawSchema{
		Version: s.Version,
		Index: rawIndex{
			Name:         s.Index.Name,
			Prefix:       prefix,
			KeySeparator: s.Index.KeySeparator,
			StorageType:  string(s.Index.StorageType),
			Stopwords:    s.Index.Stopwords,
		},
	}
	for _, f := range s.Fields() {
		rf := rawField{Name: f.Name, Type: string(f.Type), Path: f.Path, Attrs: map[string]any{}}
		switch f.Type {
		case TypeText:
			a := f.Text
			rf.Attrs = map[string]any{
				"sortable": a.Sortable, "index_missing": a.IndexMissing, "no_index": a.NoIndex,
				"weight": a.Weight, "no_stem": a.NoStem, "withsuffixtrie": a.WithSuffixTrie,
				"index_empty": a.IndexEmpty, "unf": a.UNF,
			}
			if a.PhoneticMatcher != "" {
				rf.Attrs["phonetic_matcher"] = a.PhoneticMatcher
			}
		case TypeTag:
			a := f.Tag
			rf.Attrs = map[string]any{
				"sortable": a.Sortable, "index_missing": a.IndexMissing, "no_index": a.NoIndex,
				"separator": a.Separator, "case_sensitive": a.CaseSensitive,
				"withsuffixtrie": a.WithSuffixTrie, "index_empty": a.IndexEmpty,
			}
		case TypeNumeric:
			a := f.Numeric
			rf.Attrs = map[string]any{
				"sortable": a.Sortable, "index_missing": a.IndexMissing, "no_index": a.NoIndex,
				"unf": a.UNF,
			}
		case TypeGeo:
			a := f.Geo
			rf.Attrs = map[string]any{
				"sortable": a.Sortable, "index_missing": a.IndexMissing, "no_index": a.NoIndex,
			}
		case TypeVector:
			a := f.Vector
			rf.Attrs = map[string]any{
				"dims": a.Dims, "algorithm": string(a.Algorithm), "datatype": a.Datatype,
				"distance_metric": string(a.DistanceMetric), "index_missing": a.IndexMissing,
			}
			if a.InitialCap != nil {
				rf.Attrs["initial_cap"] = *a.InitialCap
			}
			if a.BlockSize != nil {
				rf.Attrs["block_size"] = *a.BlockSize
			}
			if a.M != nil {
				rf.Attrs["m"] = *a.M
			}
			if a.EfConstruction != nil {
				rf.Attrs["ef_construction"] = *a.EfConstruction
			}
			if a.EfRuntime != nil {
				rf.Attrs["ef_runtime"] = *a.EfRuntime
			}
			if a.Epsilon != nil {
				rf.Attrs["epsilon"] = *a.Epsilon
			}
			if a.GraphMaxDegree != nil {
				rf.Attrs["graph_max_degree"] = *a.GraphMaxDegree
			}
			if a.ConstructionWindowSize != nil {
				rf.Attrs["construction_window_size"] = *a.ConstructionWindowSize
			}
			if a.SearchWindowSize != nil {
				rf.Attrs["search_window_size"] = *a.SearchWindowSize
			}
			if a.Compression != "" {
				rf.Attrs["compression"] = string(a.Compression)
			}
			if a.Reduce != nil {
				rf.Attrs["reduce"] = *a.Reduce
			}
			if a.TrainingThreshold != nil {
				rf.Attrs["training_threshold"] = *a.TrainingThreshold
			}
		}
		raw.Fields = append(raw.Fields, rf)
	}
	return raw
}

// ---------------------------------------------------------------- helpers

func getBool(m map[string]any, key string) bool {
	if m == nil {
		return false
	}
	if v, ok := m[key].(bool); ok {
		return v
	}
	return false
}

func getString(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getInt(m map[string]any, key string) (int, bool) {
	if m == nil {
		return 0, false
	}
	switch v := m[key].(type) {
	case int:
		return v, true
	case int64:
		return int(v), true
	case float64:
		return int(v), true
	}
	return 0, false
}

func getFloat(m map[string]any, key string) (float64, bool) {
	if m == nil {
		return 0, false
	}
	switch v := m[key].(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	}
	return 0, false
}
