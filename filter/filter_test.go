package filter

import (
	"testing"
	"time"
)

// Expected strings mirror the outputs of the Python redisvl filter DSL.

func TestTag(t *testing.T) {
	cases := []struct {
		got  *Expression
		want string
	}{
		{Tag("brand").Eq("nike"), "@brand:{nike}"},
		{Tag("brand").Ne("nike"), "(-@brand:{nike})"},
		{Tag("brand").Eq("nike", "adidas"), "@brand:{nike|adidas}"},
		{Tag("brand").Eq("nike-shoes"), `@brand:{nike\-shoes}`},
		{Tag("category").Like("tech*"), "@category:{tech*}"},
		{Tag("category").Like("*tech*"), "@category:{*tech*}"},
		{Tag("brand").Eq(), "*"},
		{Tag("brand").Eq(""), "*"},
		{Tag("brand").IsMissing(), "ismissing(@brand)"},
	}
	for _, c := range cases {
		if s := c.got.String(); s != c.want {
			t.Errorf("got %q, want %q", s, c.want)
		}
	}
}

func TestNum(t *testing.T) {
	cases := []struct {
		got  *Expression
		want string
	}{
		{Num("zipcode").Eq(90210), "@zipcode:[90210 90210]"},
		{Num("zipcode").Ne(90210), "(-@zipcode:[90210 90210])"},
		{Num("age").Gt(18), "@age:[(18 +inf]"},
		{Num("age").Lt(18), "@age:[-inf (18]"},
		{Num("age").Ge(18), "@age:[18 +inf]"},
		{Num("age").Le(18), "@age:[-inf 18]"},
		{Num("price").Lt(99.5), "@price:[-inf (99.5]"},
		{Num("x").Between(5, 10, Both), "@x:[5 10]"},
		{Num("x").Between(5, 10, Neither), "@x:[(5 (10]"},
		{Num("x").Between(5, 10, Left), "@x:[5 (10]"},
		{Num("x").Between(5, 10, Right), "@x:[(5 10]"},
		{Num("age").IsMissing(), "ismissing(@age)"},
	}
	for _, c := range cases {
		if s := c.got.String(); s != c.want {
			t.Errorf("got %q, want %q", s, c.want)
		}
	}
}

func TestText(t *testing.T) {
	cases := []struct {
		got  *Expression
		want string
	}{
		{Text("job").Eq("engineer"), `@job:("engineer")`},
		{Text("job").Ne("engineer"), `(-@job:"engineer")`},
		{Text("job").Like("engine*"), "@job:(engine*)"},
		{Text("job").Like("engineer|doctor"), "@job:(engineer|doctor)"},
		{Text("job").Eq(""), "*"},
	}
	for _, c := range cases {
		if s := c.got.String(); s != c.want {
			t.Errorf("got %q, want %q", s, c.want)
		}
	}
}

func TestGeo(t *testing.T) {
	r := GeoRadius{Longitude: -122.4194, Latitude: 37.7749, Radius: 1, Unit: "m"}
	if s := Geo("location").Within(r).String(); s != "@location:[-122.4194 37.7749 1 m]" {
		t.Errorf("got %q", s)
	}
	if s := Geo("location").NotWithin(r).String(); s != "(-@location:[-122.4194 37.7749 1 m])" {
		t.Errorf("got %q", s)
	}
}

func TestCombinations(t *testing.T) {
	nike := Tag("brand").Eq("nike")
	cheap := Num("price").Lt(100)

	if s := nike.And(cheap).String(); s != "(@brand:{nike} @price:[-inf (100])" {
		t.Errorf("AND got %q", s)
	}
	if s := nike.Or(cheap).String(); s != "(@brand:{nike} | @price:[-inf (100])" {
		t.Errorf("OR got %q", s)
	}
	// wildcard collapsing
	if s := All().And(nike).String(); s != "@brand:{nike}" {
		t.Errorf("collapse got %q", s)
	}
	if s := All().Or(All()).String(); s != "*" {
		t.Errorf("collapse got %q", s)
	}
	// nested
	nested := nike.And(cheap.Or(Tag("brand").Eq("adidas")))
	want := "(@brand:{nike} (@price:[-inf (100] | @brand:{adidas}))"
	if s := nested.String(); s != want {
		t.Errorf("nested got %q, want %q", s, want)
	}
}

func TestTimestamp(t *testing.T) {
	ts := time.Date(2024, 1, 15, 12, 0, 0, 0, time.UTC)
	if s := Timestamp("created_at").Gt(ts).String(); s != "@created_at:[(1705320000 +inf]" {
		t.Errorf("got %q", s)
	}
	day := Timestamp("created_at").OnDate(ts).String()
	want := "@created_at:[1705276800 1705363199.999999]"
	if day != want {
		t.Errorf("got %q, want %q", day, want)
	}
}

func TestEscaper(t *testing.T) {
	if got := EscapeToken("hello-world.com", false); got != `hello\-world\.com` {
		t.Errorf("got %q", got)
	}
	// wildcards preserved
	if got := EscapeToken("tech*", true); got != "tech*" {
		t.Errorf("got %q", got)
	}
	// wildcards escaped by default
	if got := EscapeToken("tech*", false); got != `tech\*` {
		t.Errorf("got %q", got)
	}
}
