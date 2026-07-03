// Package schema defines Redis search index schemas: index settings and
// typed fields (text, tag, numeric, geo, vector). Port of redisvl.schema.
package schema

import (
	"fmt"
	"strings"
)

// FieldType is the kind of a search field.
type FieldType string

// Supported field types.
const (
	TypeTag     FieldType = "tag"
	TypeText    FieldType = "text"
	TypeNumeric FieldType = "numeric"
	TypeGeo     FieldType = "geo"
	TypeVector  FieldType = "vector"
)

// DistanceMetric is the similarity metric for vector fields.
type DistanceMetric string

// Supported distance metrics.
const (
	Cosine DistanceMetric = "COSINE"
	L2     DistanceMetric = "L2"
	IP     DistanceMetric = "IP"
)

// Algorithm is the vector indexing algorithm.
type Algorithm string

// Supported vector indexing algorithms.
const (
	Flat      Algorithm = "FLAT"
	HNSW      Algorithm = "HNSW"
	SVSVamana Algorithm = "SVS-VAMANA"
)

// Compression is the SVS-VAMANA vector compression type.
type Compression string

// Supported SVS-VAMANA compression types.
const (
	LVQ4       Compression = "LVQ4"
	LVQ4x4     Compression = "LVQ4x4"
	LVQ4x8     Compression = "LVQ4x8"
	LVQ8       Compression = "LVQ8"
	LeanVec4x8 Compression = "LeanVec4x8"
	LeanVec8x8 Compression = "LeanVec8x8"
)

// BaseAttrs are attributes shared by all lexical (non-vector) fields.
type BaseAttrs struct {
	Sortable     bool `json:"sortable" yaml:"sortable"`
	IndexMissing bool `json:"index_missing" yaml:"index_missing"`
	NoIndex      bool `json:"no_index" yaml:"no_index"`
}

// TextAttrs are TEXT field attributes.
type TextAttrs struct {
	BaseAttrs       `yaml:",inline"`
	Weight          float64 `json:"weight" yaml:"weight"` // default 1
	NoStem          bool    `json:"no_stem" yaml:"no_stem"`
	WithSuffixTrie  bool    `json:"withsuffixtrie" yaml:"withsuffixtrie"`
	PhoneticMatcher string  `json:"phonetic_matcher,omitempty" yaml:"phonetic_matcher,omitempty"`
	IndexEmpty      bool    `json:"index_empty" yaml:"index_empty"`
	UNF             bool    `json:"unf" yaml:"unf"`
}

// TagAttrs are TAG field attributes.
type TagAttrs struct {
	BaseAttrs      `yaml:",inline"`
	Separator      string `json:"separator" yaml:"separator"` // default ","
	CaseSensitive  bool   `json:"case_sensitive" yaml:"case_sensitive"`
	WithSuffixTrie bool   `json:"withsuffixtrie" yaml:"withsuffixtrie"`
	IndexEmpty     bool   `json:"index_empty" yaml:"index_empty"`
}

// NumericAttrs are NUMERIC field attributes.
type NumericAttrs struct {
	BaseAttrs `yaml:",inline"`
	UNF       bool `json:"unf" yaml:"unf"`
}

// GeoAttrs are GEO field attributes.
type GeoAttrs struct {
	BaseAttrs `yaml:",inline"`
}

// VectorAttrs are VECTOR field attributes for all algorithms. Algorithm-
// specific parameters are pointers; nil means "use the Redis default".
type VectorAttrs struct {
	Dims           int            `json:"dims" yaml:"dims"`
	Algorithm      Algorithm      `json:"algorithm" yaml:"algorithm"`
	Datatype       string         `json:"datatype" yaml:"datatype"` // default float32
	DistanceMetric DistanceMetric `json:"distance_metric" yaml:"distance_metric"`
	InitialCap     *int           `json:"initial_cap,omitempty" yaml:"initial_cap,omitempty"`
	IndexMissing   bool           `json:"index_missing" yaml:"index_missing"`

	// FLAT
	BlockSize *int `json:"block_size,omitempty" yaml:"block_size,omitempty"`

	// HNSW
	M              *int     `json:"m,omitempty" yaml:"m,omitempty"`
	EfConstruction *int     `json:"ef_construction,omitempty" yaml:"ef_construction,omitempty"`
	EfRuntime      *int     `json:"ef_runtime,omitempty" yaml:"ef_runtime,omitempty"`
	Epsilon        *float64 `json:"epsilon,omitempty" yaml:"epsilon,omitempty"`

	// SVS-VAMANA
	GraphMaxDegree         *int        `json:"graph_max_degree,omitempty" yaml:"graph_max_degree,omitempty"`
	ConstructionWindowSize *int        `json:"construction_window_size,omitempty" yaml:"construction_window_size,omitempty"`
	SearchWindowSize       *int        `json:"search_window_size,omitempty" yaml:"search_window_size,omitempty"`
	Compression            Compression `json:"compression,omitempty" yaml:"compression,omitempty"`
	Reduce                 *int        `json:"reduce,omitempty" yaml:"reduce,omitempty"`
	TrainingThreshold      *int        `json:"training_threshold,omitempty" yaml:"training_threshold,omitempty"`
}

// Validate checks vector attribute constraints (port of the pydantic
// validators in redisvl.schema.fields).
func (v *VectorAttrs) Validate() error {
	if v.Dims <= 0 {
		return fmt.Errorf("vector field requires dims > 0")
	}
	v.Datatype = strings.ToLower(v.Datatype)
	if v.Datatype == "" {
		v.Datatype = "float32"
	}
	switch v.Datatype {
	case "bfloat16", "float16", "float32", "float64", "int8", "uint8":
	default:
		return fmt.Errorf("invalid vector datatype: %s", v.Datatype)
	}
	if v.DistanceMetric == "" {
		v.DistanceMetric = Cosine
	}
	v.DistanceMetric = DistanceMetric(strings.ToUpper(string(v.DistanceMetric)))
	switch v.DistanceMetric {
	case Cosine, L2, IP:
	default:
		return fmt.Errorf("invalid distance metric: %s", v.DistanceMetric)
	}
	v.Algorithm = Algorithm(strings.ToUpper(string(v.Algorithm)))
	switch v.Algorithm {
	case Flat, HNSW, SVSVamana:
	default:
		return fmt.Errorf("unknown vector field algorithm: %s", v.Algorithm)
	}

	if v.Algorithm == SVSVamana {
		if v.Datatype != "float16" && v.Datatype != "float32" {
			return fmt.Errorf(
				"SVS-VAMANA only supports FLOAT16 and FLOAT32 datatypes, got: %s",
				strings.ToUpper(v.Datatype))
		}
		if v.Reduce != nil {
			if *v.Reduce >= v.Dims {
				return fmt.Errorf("reduce (%d) must be less than dims (%d)", *v.Reduce, v.Dims)
			}
			if v.Compression == "" {
				return fmt.Errorf("reduce parameter requires compression to be set (LeanVec4x8 or LeanVec8x8)")
			}
			if !strings.HasPrefix(string(v.Compression), "LeanVec") {
				return fmt.Errorf(
					"reduce parameter is only supported with LeanVec compression types, got compression=%s",
					v.Compression)
			}
		}
	}
	return nil
}

// Field is a single schema field. Exactly one of the attrs pointers matching
// Type should be set; the New*Field constructors take care of this.
type Field struct {
	Name string    `json:"name" yaml:"name"`
	Type FieldType `json:"type" yaml:"type"`
	// Path is the JSONPath to the field for JSON storage (default "$.<name>").
	Path string `json:"path,omitempty" yaml:"path,omitempty"`

	Text    *TextAttrs    `json:"-" yaml:"-"`
	Tag     *TagAttrs     `json:"-" yaml:"-"`
	Numeric *NumericAttrs `json:"-" yaml:"-"`
	Geo     *GeoAttrs     `json:"-" yaml:"-"`
	Vector  *VectorAttrs  `json:"-" yaml:"-"`
}

// NewTextField creates a TEXT field with default attributes; pass attrs to
// override.
func NewTextField(name string, attrs ...TextAttrs) Field {
	a := TextAttrs{Weight: 1}
	if len(attrs) > 0 {
		a = attrs[0]
		if a.Weight == 0 {
			a.Weight = 1
		}
	}
	return Field{Name: name, Type: TypeText, Text: &a}
}

// NewTagField creates a TAG field with default attributes.
func NewTagField(name string, attrs ...TagAttrs) Field {
	a := TagAttrs{Separator: ","}
	if len(attrs) > 0 {
		a = attrs[0]
		if a.Separator == "" {
			a.Separator = ","
		}
	}
	return Field{Name: name, Type: TypeTag, Tag: &a}
}

// NewNumericField creates a NUMERIC field with default attributes.
func NewNumericField(name string, attrs ...NumericAttrs) Field {
	a := NumericAttrs{}
	if len(attrs) > 0 {
		a = attrs[0]
	}
	return Field{Name: name, Type: TypeNumeric, Numeric: &a}
}

// NewGeoField creates a GEO field with default attributes.
func NewGeoField(name string, attrs ...GeoAttrs) Field {
	a := GeoAttrs{}
	if len(attrs) > 0 {
		a = attrs[0]
	}
	return Field{Name: name, Type: TypeGeo, Geo: &a}
}

// NewVectorField creates a VECTOR field. attrs.Dims and attrs.Algorithm are
// required; the attrs are validated.
func NewVectorField(name string, attrs VectorAttrs) (Field, error) {
	if err := attrs.Validate(); err != nil {
		return Field{}, fmt.Errorf("vector field %q: %w", name, err)
	}
	return Field{Name: name, Type: TypeVector, Vector: &attrs}, nil
}

// IntPtr is a convenience for building VectorAttrs literals.
func IntPtr(v int) *int { return &v }

// FloatPtr is a convenience for building VectorAttrs literals.
func FloatPtr(v float64) *float64 { return &v }

// RedisArgs renders the field definition for the SCHEMA section of
// FT.CREATE. Modifier ordering follows the canonical order required by the
// Redis Search parser: [INDEXEMPTY] [INDEXMISSING] [SORTABLE [UNF]] [NOINDEX].
func (f *Field) RedisArgs() ([]any, error) {
	name := f.Name
	var alias string
	if f.Path != "" {
		name = f.Path
		alias = f.Name
	}
	args := []any{name}
	if alias != "" {
		args = append(args, "AS", alias)
	}

	switch f.Type {
	case TypeText:
		a := f.Text
		if a == nil {
			a = &TextAttrs{Weight: 1}
		}
		args = append(args, "TEXT")
		if a.NoStem {
			args = append(args, "NOSTEM")
		}
		if a.Weight != 0 && a.Weight != 1 {
			args = append(args, "WEIGHT", a.Weight)
		}
		if a.PhoneticMatcher != "" {
			args = append(args, "PHONETIC", a.PhoneticMatcher)
		}
		if a.WithSuffixTrie {
			args = append(args, "WITHSUFFIXTRIE")
		}
		if a.IndexEmpty {
			args = append(args, "INDEXEMPTY")
		}
		if a.IndexMissing {
			args = append(args, "INDEXMISSING")
		}
		if a.Sortable {
			args = append(args, "SORTABLE")
			if a.UNF {
				args = append(args, "UNF")
			}
		}
		if a.NoIndex {
			args = append(args, "NOINDEX")
		}

	case TypeTag:
		a := f.Tag
		if a == nil {
			a = &TagAttrs{Separator: ","}
		}
		args = append(args, "TAG")
		sep := a.Separator
		if sep == "" {
			sep = ","
		}
		args = append(args, "SEPARATOR", sep)
		if a.CaseSensitive {
			args = append(args, "CASESENSITIVE")
		}
		if a.WithSuffixTrie {
			args = append(args, "WITHSUFFIXTRIE")
		}
		if a.IndexEmpty {
			args = append(args, "INDEXEMPTY")
		}
		if a.IndexMissing {
			args = append(args, "INDEXMISSING")
		}
		if a.Sortable {
			args = append(args, "SORTABLE")
		}
		if a.NoIndex {
			args = append(args, "NOINDEX")
		}

	case TypeNumeric:
		a := f.Numeric
		if a == nil {
			a = &NumericAttrs{}
		}
		args = append(args, "NUMERIC")
		if a.IndexMissing {
			args = append(args, "INDEXMISSING")
		}
		if a.Sortable {
			args = append(args, "SORTABLE")
			if a.UNF {
				args = append(args, "UNF")
			}
		}
		if a.NoIndex {
			args = append(args, "NOINDEX")
		}

	case TypeGeo:
		a := f.Geo
		if a == nil {
			a = &GeoAttrs{}
		}
		args = append(args, "GEO")
		if a.IndexMissing {
			args = append(args, "INDEXMISSING")
		}
		if a.Sortable {
			args = append(args, "SORTABLE")
		}
		if a.NoIndex {
			args = append(args, "NOINDEX")
		}

	case TypeVector:
		a := f.Vector
		if a == nil {
			return nil, fmt.Errorf("vector field %q has no attributes", f.Name)
		}
		if err := a.Validate(); err != nil {
			return nil, err
		}
		// Attribute key/value pairs preceded by their count.
		kv := []any{
			"TYPE", strings.ToUpper(a.Datatype),
			"DIM", a.Dims,
			"DISTANCE_METRIC", string(a.DistanceMetric),
		}
		if a.InitialCap != nil {
			kv = append(kv, "INITIAL_CAP", *a.InitialCap)
		}
		switch a.Algorithm {
		case Flat:
			if a.BlockSize != nil {
				kv = append(kv, "BLOCK_SIZE", *a.BlockSize)
			}
		case HNSW:
			kv = append(kv,
				"M", orDefault(a.M, 16),
				"EF_CONSTRUCTION", orDefault(a.EfConstruction, 200),
				"EF_RUNTIME", orDefault(a.EfRuntime, 10),
				"EPSILON", orDefaultF(a.Epsilon, 0.01),
			)
		case SVSVamana:
			kv = append(kv,
				"GRAPH_MAX_DEGREE", orDefault(a.GraphMaxDegree, 40),
				"CONSTRUCTION_WINDOW_SIZE", orDefault(a.ConstructionWindowSize, 250),
				"SEARCH_WINDOW_SIZE", orDefault(a.SearchWindowSize, 20),
				"EPSILON", orDefaultF(a.Epsilon, 0.01),
			)
			if a.Compression != "" {
				kv = append(kv, "COMPRESSION", string(a.Compression))
			}
			if a.Reduce != nil {
				kv = append(kv, "REDUCE", *a.Reduce)
			}
			if a.TrainingThreshold != nil {
				kv = append(kv, "TRAINING_THRESHOLD", *a.TrainingThreshold)
			}
		}
		args = append(args, "VECTOR", string(a.Algorithm), len(kv))
		args = append(args, kv...)
		if a.IndexMissing {
			args = append(args, "INDEXMISSING")
		}

	default:
		return nil, fmt.Errorf("unknown field type: %s", f.Type)
	}
	return args, nil
}

func orDefault(p *int, def int) int {
	if p != nil {
		return *p
	}
	return def
}

func orDefaultF(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}
