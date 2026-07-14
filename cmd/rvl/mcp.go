package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/redis/redis-vl-golang/mcpserver"
)

var loopbackHosts = map[string]bool{
	"127.0.0.1": true, "::1": true, "localhost": true,
}

// checkHTTPBind decides whether an HTTP transport may bind to host and
// whether to print the unauthenticated-exposure warning (mirrors the
// Python CLI check): configured auth permits any bind silently;
// unauthenticated non-loopback binds require the explicit override flag;
// every unauthenticated bind warns.
func checkHTTPBind(host string, authEnabled, allowUnauthenticated bool) (warn bool, err error) {
	if authEnabled {
		return false, nil
	}
	if !loopbackHosts[host] && !allowUnauthenticated {
		return false, fmt.Errorf(
			"refusing to bind an unauthenticated MCP server to non-loopback host %q; "+
				"configure server.auth or pass --allow-unauthenticated to override", host)
	}
	return true, nil
}

// mcpCmd runs the RedisVL MCP server (Go port of `rvl mcp`).
func mcpCmd(args []string) error {
	fs := flag.NewFlagSet("mcp", flag.ContinueOnError)
	configPath := fs.String("config", "", "path to MCP config file (required)")
	readOnly := fs.Bool("read-only", false, "disable the upsert tool")
	transport := fs.String("transport", "stdio", "transport protocol: stdio, sse, or streamable-http")
	host := fs.String("host", "127.0.0.1", "host to bind to for HTTP transports")
	port := fs.Int("port", 8000, "port to bind to for HTTP transports")
	allowUnauthenticated := fs.Bool("allow-unauthenticated",
		false, "allow binding an HTTP transport to a non-loopback host without auth")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *configPath == "" {
		return fmt.Errorf("usage: rvl mcp --config <path> [--read-only] [--transport stdio|sse|streamable-http] [--host <host>] [--port <port>] [--allow-unauthenticated]")
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

	if *transport == "stdio" {
		return server.Run(ctx)
	}
	if *transport != "sse" && *transport != "streamable-http" {
		return fmt.Errorf("unknown transport %q (supported: stdio, sse, streamable-http)", *transport)
	}

	warn, err := checkHTTPBind(*host, server.AuthEnabled(), *allowUnauthenticated)
	if err != nil {
		return err
	}
	if warn {
		fmt.Fprintf(os.Stderr,
			"WARNING: serving MCP over %s on %s:%d without authentication. "+
				"Any client that can reach this address has full access.\n",
			*transport, *host, *port)
	}

	addr := fmt.Sprintf("%s:%d", *host, *port)
	if *transport == "sse" {
		return server.RunSSE(ctx, addr)
	}
	return server.RunHTTP(ctx, addr)
}
