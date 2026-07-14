# Contributing

## Introduction

First off, thank you for considering contributions to RedisVL! We value community contributions and appreciate your interest in helping make this project better.

## Types of Contributions We Need

You may already know what you want to contribute -- a fix for a bug you encountered, or a new feature your team wants to use.

If you don't know what to contribute, keep an open mind! Here are some valuable ways to contribute:

- **Bug fixes**: Help us identify and resolve issues
- **Feature development**: Add new functionality that benefits the community
- **Documentation improvements**: Enhance clarity, add examples, or fix typos
- **Bug triaging**: Help categorize and prioritize issues
- **Writing tutorials**: Create guides that help others use RedisVL
- **Testing**: Write tests or help improve test coverage

## Getting Started

Here's how to get started with your code contribution:

1. **Fork the repository**: Create your own fork of this repo
2. **Set up your development environment**: Follow the setup instructions below
3. **Make your changes**: Apply the changes in your forked codebase
4. **Test your changes**: Ensure your changes work and don't break existing functionality
5. **Submit a pull request**: If you like the change and think the project could use it, send us a pull request

## Development Environment Setup

### Prerequisites

- **Go**: RedisVL for Golang requires Go 1.23 or above ([install Go](https://go.dev/doc/install))
- **Docker**: Required for running Redis and integration tests
- **golangci-lint** (optional): For the full lint suite ([install guide](https://golangci-lint.run/welcome/install/))

### Project Setup

Once Go is installed, fetch the project dependencies:

```bash
go mod download
```

That's it — no virtual environments or package manager setup required.

## Using the Makefile

We provide a Makefile to streamline common development tasks. Here are the available commands:

| Command | Description |
|---------|-------------|
| `make deps` | Tidies and downloads module dependencies |
| `make fmt` | Formats all Go source files with gofmt |
| `make vet` | Runs go vet static analysis |
| `make lint` | Runs the golangci-lint suite |
| `make test` | Runs all tests (integration tests use a Redis testcontainer, or skip without Docker) |
| `make test-integration` | Runs only the integration tests |
| `make test-integration-external` | Runs integration tests against an external Redis (`REDIS_URL`) |
| `make build-rvl` | Builds the `rvl` CLI binary into `bin/` |
| `make check` | Runs formatting, vet, and tests |
| `make work` | Creates a Go workspace linking the core and `hf` modules for local development |
| `make vet-hf` / `make test-hf` | Vet / test the `extensions/vectorize/hf` module |

**Quick Start Example:**
```bash
# Fetch dependencies
make deps

# Run linting and tests (Docker running for integration coverage)
make check
```

### Working across modules

This repository contains two Go modules: the core library (repository
root) and `extensions/vectorize/hf` (separate because it uses cgo / ONNX
Runtime). The hf module depends on the core module *by released version*,
so to develop both against the same checkout, create a workspace first:

```bash
make work        # creates a gitignored go.work
make test-hf
```

CI does the same, so cross-module changes are always tested together.

### Releasing

The core module is tagged first — the hf module's `go.mod` references the
core version by tag, and Go must be able to resolve that tag (even in
workspace mode) before anything depending on it builds:

```bash
# 1. tag and push the core module
git tag vX.Y.Z && git push origin vX.Y.Z

# 2. bump the core requirement in extensions/vectorize/hf/go.mod to vX.Y.Z,
#    refresh go.sum, and validate — including standalone (no-workspace)
#    resolution, which is how external consumers build the module
GOPRIVATE=github.com/redis/redis-vl-golang make deps-hf
GOPRIVATE=github.com/redis/redis-vl-golang make check test-hf
cd extensions/vectorize/hf && GOWORK=off go test ./... && cd ../..
#    commit and push (go.mod + go.sum)

# 3. tag and push the hf module on that commit
git tag extensions/vectorize/hf/vX.Y.Z
git push origin extensions/vectorize/hf/vX.Y.Z
```

Also keep `Version` in `version.go` in sync with the release tag, and add
a CHANGELOG.md entry for the release.

Pushing the core `vX.Y.Z` tag also triggers the Release workflow
(GoReleaser), which cross-compiles the `rvl` CLI for macOS, Linux, and
Windows and attaches the binaries to the GitHub Release. Test the
pipeline locally with `make release-snapshot` (requires goreleaser).

## Code Quality and Testing

### Linting and Formatting

We maintain high code quality standards. Before submitting your changes, ensure they pass our quality checks:

```bash
# Format code
make fmt

# Static analysis
make vet

# Full lint suite (requires golangci-lint)
make lint
```

You can also run these commands directly:
```bash
gofmt -w .
go vet ./...
golangci-lint run
```

### Running Tests

#### TestContainers

RedisVL uses [Testcontainers for Go](https://golang.testcontainers.org/) for integration tests. Testcontainers provisions throwaway, on-demand containers for development and testing — with Docker running, `go test ./...` starts a `redis:8.8.0` container automatically.

**Requirements:**
- Local Docker installation such as:
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - [Docker Engine on Linux](https://docs.docker.com/engine/install/)

#### Test Commands

```bash
# Run all tests (unit + integration via testcontainer)
make test

# Run tests against an external Redis server instead
REDIS_URL=redis://localhost:6379 go test ./...

# Test against a different Redis image
REDIS_IMAGE=redis:8.2 go test -run Integration ./...

# Run tests in a specific package
go test ./query/ -v

# Run tests with coverage
go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out
```

**Note:** Live vectorizer tests (e.g. `TestLiveOpenAIEmbed`) require the appropriate API keys set as environment variables and are skipped otherwise.

## Documentation

API documentation is generated from doc comments and served on [pkg.go.dev](https://pkg.go.dev/github.com/redis/redis-vl-golang) once published. Preview locally with:

```bash
# Serve godoc locally at http://localhost:6060
go run golang.org/x/tools/cmd/godoc@latest -http=:6060
```

All exported identifiers must carry doc comments (enforced by lint).

## Redis Setup

To develop and test RedisVL applications, you need Redis with the Query Engine (Redis 8+). You have several options:

### Option 1: Testcontainers (Recommended for Development)

Nothing to set up — the integration tests start and clean up their own `redis:8.8.0` container when Docker is running.

### Option 2: Redis with Docker

```bash
docker run -d --name redis -p 6379:6379 redis:8.8.0
```

### Option 3: Redis Cloud

For production-like testing, use [Redis Cloud](https://redis.io/cloud/) which provides managed Redis instances with Redis Query Engine capabilities.

## Reporting Issues

### Security Vulnerabilities

**⚠️ IMPORTANT**: If you find a security vulnerability, do NOT open a public issue. Email [Redis OSS](mailto:oss@redis.com) instead.

**Questions to determine if it's a security issue:**
- Can I access something that's not mine, or something I shouldn't have access to?
- Can I disable something for other people?

If you answer *yes* to either question, it's likely a security issue.

### Bug Reports

When filing a bug report, please include:

1. **Go version**: Output of `go version`
2. **Package versions**: What versions of `redis-vl-golang` and `go-redis` are you using?
3. **Redis version**: Output of `redis-cli INFO server | grep redis_version`
4. **Steps to reproduce**: What did you do?
5. **Expected behavior**: What did you expect to see?
6. **Actual behavior**: What did you see instead?
7. **Environment**: Operating system, Docker version (if applicable)
8. **Code sample**: Minimal code that reproduces the issue

## Suggesting Features

Before suggesting a new feature:

1. **Check existing issues**: Search our [issue list](https://github.com/redis/redis-vl-golang/issues) to see if someone has already proposed it
2. **Consider the scope**: Ensure the feature aligns with RedisVL's goals
3. **Provide details**: If you don't see anything similar, open a new issue that describes:
   - The feature you would like
   - How it should work
   - Why it would be beneficial
   - Any implementation ideas you have

## Pull Request Process

1. **Fork and create a branch**: Create a descriptive branch name (e.g., `fix-search-bug` or `add-vector-similarity`)
2. **Make your changes**: Follow our coding standards and include tests
3. **Test thoroughly**: Ensure your changes work and don't break existing functionality
4. **Update documentation**: Add or update doc comments and README sections as needed
5. **Submit your PR**: Include a clear description of what your changes do

### Review Process

- The core team reviews Pull Requests regularly
- We provide feedback as soon as possible
- After feedback, we expect a response within two weeks
- PRs may be closed if they show no activity after this period

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code is gofmt-clean and passes static analysis (`make fmt vet` produces no changes or warnings)
- [ ] Tests pass (`make test` passes)
- [ ] New exported identifiers have doc comments
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains what changes were made and why

## Getting Help

If you need help or have questions:

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check the [README](README.md) and the [Python RedisVL docs](https://docs.redisvl.com/) — the concepts translate directly

Thank you for contributing to RedisVL! 🚀
