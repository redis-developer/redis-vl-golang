package redisvl

import (
	"fmt"
	"strconv"

	"github.com/redis-developer/redis-vl-golang/schema"
	"github.com/redis-developer/redis-vl-golang/vectors"
)

// validateRecord checks a record's values against the index schema (light
// port of Python's schema validation: vector dimensions/encodings are
// enforced strictly, numeric fields must be numeric; other field types are
// checked leniently). Fields absent from the record are skipped.
func (i *SearchIndex) validateRecord(record map[string]any) error {
	isJSON := i.Schema.Index.StorageType == schema.JSON

	for _, f := range i.Schema.Fields() {
		value, ok := record[f.Name]
		if !ok || value == nil {
			continue
		}
		switch f.Type {
		case schema.TypeVector:
			if err := validateVectorValue(f, value, isJSON); err != nil {
				return fmt.Errorf("field %q: %w", f.Name, err)
			}
		case schema.TypeNumeric:
			if !isNumericValue(value) {
				return fmt.Errorf("field %q: expected a numeric value, got %T", f.Name, value)
			}
		case schema.TypeText, schema.TypeGeo:
			if _, ok := value.(string); !ok {
				return fmt.Errorf("field %q: expected a string, got %T", f.Name, value)
			}
		case schema.TypeTag:
			switch value.(type) {
			case string, []string, []any:
			default:
				return fmt.Errorf("field %q: expected a string or string slice, got %T", f.Name, value)
			}
		}
	}
	return nil
}

func validateVectorValue(f *schema.Field, value any, isJSON bool) error {
	attrs := f.Vector
	if attrs == nil {
		return nil
	}

	if isJSON {
		n, ok := floatSliceLen(value)
		if !ok {
			return fmt.Errorf("JSON storage expects a float slice, got %T", value)
		}
		if n != attrs.Dims {
			return fmt.Errorf("expected %d dimensions, got %d", attrs.Dims, n)
		}
		return nil
	}

	blob, ok := value.([]byte)
	if !ok {
		return fmt.Errorf("hash storage expects []byte (use vectors.ToBuffer), got %T", value)
	}
	elemSize, err := vectors.Size(vectors.DataType(attrs.Datatype))
	if err != nil {
		return err
	}
	want := attrs.Dims * elemSize
	if len(blob) != want {
		return fmt.Errorf("expected %d bytes (%d dims x %d-byte %s), got %d",
			want, attrs.Dims, elemSize, attrs.Datatype, len(blob))
	}
	return nil
}

// floatSliceLen reports the length of a numeric slice in any of the shapes
// JSON-bound records commonly use.
func floatSliceLen(value any) (int, bool) {
	switch v := value.(type) {
	case []float64:
		return len(v), true
	case []float32:
		return len(v), true
	case []any:
		for _, e := range v {
			if !isNumericValue(e) {
				return 0, false
			}
		}
		return len(v), true
	}
	return 0, false
}

func isNumericValue(value any) bool {
	switch v := value.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64,
		float32, float64:
		return true
	case string:
		_, err := strconv.ParseFloat(v, 64)
		return err == nil
	}
	return false
}
