package redisvl

import (
	"testing"

	"github.com/redis-developer/redis-vl-golang/query"
)

func TestParseSearchReplyRESP2(t *testing.T) {
	reply := []any{
		int64(2),
		"doc:1", []any{"title", "hello", "price", "10"},
		"doc:2", []any{"title", "world", "price", "20"},
	}
	res, err := parseSearchReply(reply, &query.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 2 || len(res.Docs) != 2 {
		t.Fatalf("total=%d docs=%d", res.Total, len(res.Docs))
	}
	if res.Docs[0].ID != "doc:1" || res.Docs[0].Fields["title"] != "hello" {
		t.Errorf("doc0 = %+v", res.Docs[0])
	}
}

func TestParseSearchReplyRESP2NoContent(t *testing.T) {
	reply := []any{int64(3), "doc:1", "doc:2", "doc:3"}
	res, err := parseSearchReply(reply, &query.Options{NoContent: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 3 || len(res.Docs) != 3 || res.Docs[2].ID != "doc:3" {
		t.Errorf("res = %+v", res)
	}
}

func TestParseSearchReplyRESP2WithScores(t *testing.T) {
	reply := []any{
		int64(1),
		"doc:1", "1.5", []any{"title", "hello"},
	}
	res, err := parseSearchReply(reply, &query.Options{WithScores: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Docs[0].Score != 1.5 || res.Docs[0].Fields["title"] != "hello" {
		t.Errorf("doc = %+v", res.Docs[0])
	}
}

func TestParseSearchReplyRESP3(t *testing.T) {
	reply := map[any]any{
		"total_results": int64(1),
		"results": []any{
			map[any]any{
				"id": "doc:1",
				"extra_attributes": map[any]any{
					"title": "hello",
				},
			},
		},
	}
	res, err := parseSearchReply(reply, &query.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 1 || res.Docs[0].ID != "doc:1" || res.Docs[0].Fields["title"] != "hello" {
		t.Errorf("res = %+v", res)
	}
}

func TestParseInfoReply(t *testing.T) {
	reply := []any{"index_name", "idx", "num_docs", int64(5)}
	info, err := parseInfoReply(reply)
	if err != nil {
		t.Fatal(err)
	}
	if toString(info["index_name"]) != "idx" {
		t.Errorf("info = %+v", info)
	}
}

func TestUnpackJSONDoc(t *testing.T) {
	doc := Document{
		ID:     "doc:1",
		Fields: map[string]string{"$": `{"title":"hello","price":10}`},
	}
	m := unpackJSONDoc(doc)
	if m["id"] != "doc:1" || m["title"] != "hello" {
		t.Errorf("unpacked = %+v", m)
	}
}

func TestNewULID(t *testing.T) {
	a, b := NewULID(), NewULID()
	if len(a) != 26 || len(b) != 26 {
		t.Errorf("ulid lengths: %d %d", len(a), len(b))
	}
	if a == b {
		t.Error("ulids should be unique")
	}
	for _, c := range a {
		found := false
		for _, v := range crockford {
			if c == v {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("invalid ulid char %q in %s", c, a)
		}
	}
}

func TestHashify(t *testing.T) {
	h1 := Hashify("hello", nil)
	h2 := Hashify("hello", nil)
	if h1 != h2 || len(h1) != 64 {
		t.Errorf("hashify not deterministic or wrong length: %s", h1)
	}
	h3 := Hashify("hello", map[string]any{"model": "gpt"})
	if h3 == h1 {
		t.Error("extras should change the hash")
	}
}
