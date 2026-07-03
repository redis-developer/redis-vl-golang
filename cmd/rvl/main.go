// rvl is the RedisVL command line tool (Go port of the Python rvl CLI).
//
// Usage:
//
//	rvl index create  -s schema.yaml [-u redis://localhost:6379] [--overwrite]
//	rvl index info    -i <name> [--json]
//	rvl index listall [--json]
//	rvl index delete  -i <name>          # delete index, keep data
//	rvl index destroy -i <name>          # delete index and its data
//	rvl stats         -i <name> | -s schema.yaml
//	rvl mcp           --config mcp.yaml [--read-only] [--transport stdio|streamable-http]
//	rvl version       [-s]
//
// The Redis URL defaults to the REDIS_URL environment variable, then
// redis://localhost:6379.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/redis/go-redis/v9"

	redisvl "github.com/redis-developer/redis-vl-golang"
	"github.com/redis-developer/redis-vl-golang/schema"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	var err error
	switch os.Args[1] {
	case "index":
		err = indexCmd(os.Args[2:])
	case "stats":
		err = statsCmd(os.Args[2:])
	case "mcp":
		err = mcpCmd(os.Args[2:])
	case "version":
		err = versionCmd(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "rvl: unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "rvl:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Print(`rvl - RedisVL command line tool

Commands:
  index create  -s schema.yaml [--overwrite]   Create a new index from a schema file
  index info    -i <name> [--json]             Show details about an index
  index listall [--json]                       List all indexes
  index delete  -i <name>                      Delete an index, keep its data
  index destroy -i <name>                      Delete an index and drop its data
  stats         -i <name> | -s schema.yaml     Display index statistics
  mcp           --config mcp.yaml              Run the RedisVL MCP server
  version       [-s]                           Print the library version

Options:
  -u, --url    Redis connection URL (default: $REDIS_URL or redis://localhost:6379)
`)
}

func defaultURL() string {
	if u := os.Getenv("REDIS_URL"); u != "" {
		return u
	}
	return "redis://localhost:6379"
}

func connect(url string) (redis.UniversalClient, error) {
	opts, err := redis.ParseURL(url)
	if err != nil {
		return nil, fmt.Errorf("invalid redis url %q: %w", url, err)
	}
	return redis.NewClient(opts), nil
}

type commonFlags struct {
	fs     *flag.FlagSet
	url    string
	index  string
	schema string
	asJSON bool
}

func newFlags(name string) *commonFlags {
	c := &commonFlags{fs: flag.NewFlagSet(name, flag.ContinueOnError)}
	c.fs.StringVar(&c.url, "u", defaultURL(), "Redis connection URL")
	c.fs.StringVar(&c.url, "url", defaultURL(), "Redis connection URL")
	c.fs.StringVar(&c.index, "i", "", "index name")
	c.fs.StringVar(&c.index, "index", "", "index name")
	c.fs.StringVar(&c.schema, "s", "", "path to schema YAML file")
	c.fs.StringVar(&c.schema, "schema", "", "path to schema YAML file")
	c.fs.BoolVar(&c.asJSON, "json", false, "machine-readable JSON output")
	return c
}

// indexFor builds a SearchIndex from --schema (preferred) or --index.
func (c *commonFlags) indexFor(ctx context.Context, client redis.UniversalClient) (*redisvl.SearchIndex, error) {
	switch {
	case c.schema != "":
		s, err := schema.FromYAMLFile(c.schema)
		if err != nil {
			return nil, err
		}
		return redisvl.NewSearchIndex(s, client), nil
	case c.index != "":
		return redisvl.FromExisting(ctx, c.index, client)
	}
	return nil, fmt.Errorf("provide an index name (-i) or a schema file (-s)")
}

func indexCmd(args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: rvl index <create|info|listall|delete|destroy> [options]")
	}
	sub := args[0]
	c := newFlags("index " + sub)
	overwrite := c.fs.Bool("overwrite", false, "overwrite the index if it exists")
	if err := c.fs.Parse(args[1:]); err != nil {
		return err
	}

	ctx := context.Background()
	client, err := connect(c.url)
	if err != nil {
		return err
	}
	defer client.Close()

	switch sub {
	case "create":
		if c.schema == "" {
			return fmt.Errorf("index create requires a schema file (-s schema.yaml)")
		}
		idx, err := c.indexFor(ctx, client)
		if err != nil {
			return err
		}
		if err := idx.Create(ctx, redisvl.CreateOptions{Overwrite: *overwrite}); err != nil {
			return err
		}
		fmt.Printf("Index created successfully: %s\n", idx.Name())
		return nil

	case "info":
		idx, err := c.indexFor(ctx, client)
		if err != nil {
			return err
		}
		info, err := idx.Info(ctx)
		if err != nil {
			return err
		}
		if c.asJSON {
			return printJSON(info)
		}
		printInfoTable(idx.Name(), info)
		return nil

	case "listall":
		idx := redisvl.NewSearchIndex(mustEmptySchema(), client)
		names, err := idx.ListAll(ctx)
		if err != nil {
			return err
		}
		if c.asJSON {
			return printJSON(map[string]any{"indices": names})
		}
		fmt.Printf("Indices (%d):\n", len(names))
		for _, n := range names {
			fmt.Println("  " + n)
		}
		return nil

	case "delete", "destroy":
		if c.index == "" {
			return fmt.Errorf("index %s requires an index name (-i)", sub)
		}
		idx, err := redisvl.FromExisting(ctx, c.index, client)
		if err != nil {
			return err
		}
		drop := sub == "destroy"
		if err := idx.Delete(ctx, drop); err != nil {
			return err
		}
		if drop {
			fmt.Printf("Index destroyed (index and data deleted): %s\n", c.index)
		} else {
			fmt.Printf("Index deleted (data preserved): %s\n", c.index)
		}
		return nil
	}
	return fmt.Errorf("unknown index subcommand %q", sub)
}

// statsKeys mirrors the rows shown by the Python `rvl stats` table.
var statsKeys = []string{
	"num_docs", "num_terms", "max_doc_id", "num_records", "percent_indexed",
	"hash_indexing_failures", "number_of_uses", "bytes_per_record_avg",
	"doc_table_size_mb", "inverted_sz_mb", "key_table_size_mb",
	"offset_bits_per_record_avg", "offset_vectors_sz_mb",
	"offsets_per_term_avg", "records_per_doc_avg", "sortable_values_size_mb",
	"total_indexing_time", "total_inverted_index_blocks",
	"vector_index_sz_mb",
}

func statsCmd(args []string) error {
	c := newFlags("stats")
	if err := c.fs.Parse(args); err != nil {
		return err
	}
	ctx := context.Background()
	client, err := connect(c.url)
	if err != nil {
		return err
	}
	defer client.Close()

	idx, err := c.indexFor(ctx, client)
	if err != nil {
		return err
	}
	info, err := idx.Info(ctx)
	if err != nil {
		return err
	}
	if c.asJSON {
		out := map[string]any{}
		for _, k := range statsKeys {
			if v, ok := info[k]; ok {
				out[k] = fmt.Sprint(v)
			}
		}
		return printJSON(out)
	}

	fmt.Println("Statistics:")
	fmt.Printf("%-32s %s\n", "Stat Key", "Value")
	fmt.Println(strings.Repeat("-", 48))
	for _, k := range statsKeys {
		if v, ok := info[k]; ok {
			fmt.Printf("%-32s %v\n", k, v)
		}
	}
	return nil
}

func versionCmd(args []string) error {
	fs := flag.NewFlagSet("version", flag.ContinueOnError)
	short := fs.Bool("s", false, "print only the version number")
	fs.BoolVar(short, "short", *short, "print only the version number")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *short {
		fmt.Println(redisvl.Version)
	} else {
		fmt.Printf("RedisVL for Golang version %s\n", redisvl.Version)
	}
	return nil
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	// RESP3 replies contain map[any]any values, which encoding/json
	// cannot marshal — normalize the whole tree first.
	return enc.Encode(normalizeJSON(v))
}

func printInfoTable(name string, info map[string]any) {
	fmt.Printf("Index Information for %q:\n", name)
	keys := make([]string, 0, len(info))
	for k := range info {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := info[k]
		// keep nested structures readable
		switch v.(type) {
		case []any, map[string]any, map[any]any:
			fmt.Printf("  %-32s %s\n", k, compactJSON(v))
		default:
			fmt.Printf("  %-32s %v\n", k, v)
		}
	}
}

func compactJSON(v any) string {
	b, err := json.Marshal(normalizeJSON(v))
	if err != nil {
		return fmt.Sprint(v)
	}
	if len(b) > 120 {
		return string(b[:117]) + "..."
	}
	return string(b)
}

// normalizeJSON makes RESP3 replies encodable: map[any]any keys become
// strings, and non-finite floats (RediSearch reports "nan" averages on
// empty indexes) become their string forms since JSON cannot express them.
func normalizeJSON(v any) any {
	switch m := v.(type) {
	case float64:
		if math.IsNaN(m) || math.IsInf(m, 0) {
			return fmt.Sprint(m) // "NaN", "+Inf", "-Inf"
		}
		return m
	case map[any]any:
		out := make(map[string]any, len(m))
		for k, val := range m {
			out[fmt.Sprint(k)] = normalizeJSON(val)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(m))
		for k, val := range m {
			out[k] = normalizeJSON(val)
		}
		return out
	case []any:
		out := make([]any, len(m))
		for i, val := range m {
			out[i] = normalizeJSON(val)
		}
		return out
	}
	return v
}

// mustEmptySchema builds a placeholder schema for operations that don't
// need one (listall).
func mustEmptySchema() *schema.IndexSchema {
	s, err := schema.NewIndexSchema(schema.IndexInfo{Name: "_rvl_placeholder"})
	if err != nil {
		panic(err) // static input; cannot fail
	}
	return s
}
