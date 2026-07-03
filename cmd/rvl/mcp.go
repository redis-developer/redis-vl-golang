package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/redis-developer/redis-vl-golang/mcpserver"
)

var loopbackHosts = map[string]bool{
	"127.0.0.1": true, "::1": true, "localhost": true,
}

// mcpCmd runs the RedisVL MCP server (Go port of `rvl mcp`).
func mcpCmd(args []string) error {
	fs := flag.NewFlagSet("mcp", flag.ContinueOnError)
	configPath := fs.String("config", "", "path to MCP config file (required)")
	readOnly := fs.Bool("read-only", false, "disable the upsert tool")
	transport := fs.String("transport", "stdio", "transport protocol: stdio or streamable-http")
	host := fs.String("host", "127.0.0.1", "host to bind to for HTTP transport")
	port := fs.Int("port", 8000, "port to bind to for HTTP transport")
	allowUnauthenticated := fs.Bool("allow-unauthenticated",
		false, "allow binding the HTTP transport to a non-loopback host")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *configPath == "" {
		return fmt.Errorf("usage: rvl mcp --config <path> [--read-only] [--transport stdio|streamable-http] [--host <host>] [--port <port>]")
	}

	cfg, err := mcpserver.LoadConfig(*configPath)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	server, err := mcpserver.New(ctx, cfg, mcpserver.Options{ReadOnly: *readOnly})
	if err != nil {
		return err
	}
	defer server.Close()

	switch *transport {
	case "stdio":
		return server.Run(ctx)
	case "streamable-http":
		// The Go server has no built-in auth: fail closed on non-loopback
		// binds unless explicitly allowed (mirrors the Python CLI check).
		if !loopbackHosts[*host] && !*allowUnauthenticated {
			return fmt.Errorf(
				"refusing to bind an unauthenticated MCP server to non-loopback host %q; "+
					"pass --allow-unauthenticated to override", *host)
		}
		addr := fmt.Sprintf("%s:%d", *host, *port)
		fmt.Fprintf(os.Stderr,
			"WARNING: serving MCP over streamable-http on %s without authentication. "+
				"Any client that can reach this address has full access.\n", addr)
		return server.RunHTTP(ctx, addr)
	}
	return fmt.Errorf("unknown transport %q (supported: stdio, streamable-http)", *transport)
}
