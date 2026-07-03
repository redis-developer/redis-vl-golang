package redisvl

import "testing"

func TestParseAggregateRowsRESP2(t *testing.T) {
	reply := []any{
		int64(2),
		[]any{"description", "doc one", "hybrid_score", "0.9", "__score", "1.2"},
		[]any{"description", "doc two", "hybrid_score", "0.4"},
	}
	rows, err := parseAggregateRows(reply)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0]["description"] != "doc one" || rows[0]["hybrid_score"] != "0.9" {
		t.Errorf("rows = %v", rows)
	}
	if _, ok := rows[0]["__score"]; ok {
		t.Error("__score should be dropped")
	}
}

func TestParseAggregateRowsRESP3(t *testing.T) {
	reply := map[any]any{
		"total_results": int64(1),
		"results": []any{
			map[any]any{
				"extra_attributes": map[any]any{
					"description":    "doc one",
					"combined_score": "0.7",
				},
			},
		},
	}
	rows, err := parseAggregateRows(reply)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0]["combined_score"] != "0.7" {
		t.Errorf("rows = %v", rows)
	}
}
