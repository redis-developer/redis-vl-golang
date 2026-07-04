# Security Policy

## Supported versions

Security fixes are applied to the latest released minor version of both
modules in this repository (the core module and
`extensions/vectorize/hf`).

## Reporting a vulnerability

Please **do not** report security vulnerabilities through public GitHub
issues.

Report them privately via
[GitHub Security Advisories](https://github.com/redis-developer/redis-vl-golang/security/advisories/new)
("Report a vulnerability" on the repository's Security tab). You should
receive a response within a few business days.

For vulnerabilities in Redis itself rather than this client library, follow
the [Redis security policy](https://redis.io/docs/latest/operate/rc/security/).

## Scope notes

- The MCP server's HTTP transports (`sse`, `streamable-http`) support JWT
  bearer authentication (`server.auth` in the config or
  `REDISVL_MCP_AUTH_*` environment variables). With authentication
  configured, non-loopback binds are permitted. Without it, the server
  refuses to bind to non-loopback addresses unless explicitly started
  with `--allow-unauthenticated`; running it exposed that way without an
  authenticating reverse proxy is an unsupported configuration.
- Symmetric JWT algorithms (`HS*`) are rejected at configuration time, and
  tokens without `exp`/`iat` claims are rejected by default.
- The `stdio` transport is a local subprocess and is never authenticated.
- API keys for embedding providers are read from environment variables and
  are never logged or persisted by the library.
