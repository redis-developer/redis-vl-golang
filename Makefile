REDIS_URL ?= redis://localhost:6379

# Prefer `python3` (present on all modern systems), fall back to `python`.
# pip is always invoked through the interpreter (`-m pip`) to avoid broken
# or missing pip shims on PATH.
PYTHON ?= $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)

.PHONY: deps check test test-integration fmt vet lint lint-hf build-rvl release-snapshot work deps-hf vet-hf test-hf test-hf-live docs-deps docs-build docs-serve bench-go bench-py bench-py-deps

build-rvl:
	go build -o bin/rvl ./cmd/rvl

# Local dry run of the release pipeline (requires goreleaser:
# brew install goreleaser). Builds all platform binaries into dist/
# without publishing anything.
release-snapshot:
	goreleaser release --snapshot --clean

lint:
	golangci-lint run

lint-hf:
	cd extensions/vectorize/hf && golangci-lint run

deps:
	go mod tidy

fmt:
	gofmt -w .

vet:
	go vet ./...

test:
	go test ./...

# Integration tests start a Redis 8 testcontainer automatically when
# REDIS_URL is not set (requires Docker).
test-integration:
	go test -run Integration ./...

test-integration-external:
	REDIS_URL=$(REDIS_URL) go test -run Integration ./...

check: fmt vet test

# --- extensions/vectorize/hf is a separate module (cgo / ONNX Runtime) ---

# Creates a Go workspace (gitignored) so the hf module builds against the
# core module from this checkout instead of the released version. Run this
# once before working on both modules together.
work:
	@test -f go.work || go work init . ./extensions/vectorize/hf
	@echo "go.work ready"

# Note: `go mod tidy` resolves the released core module version, so it
# needs the corresponding tag to exist on GitHub. GOWORK=off ensures the
# go.sum entries for the core module are written even when a local
# go.work workspace is active (standalone consumers need them).
deps-hf:
	cd extensions/vectorize/hf && GOWORK=off go mod tidy

vet-hf:
	cd extensions/vectorize/hf && go vet ./...

test-hf:
	cd extensions/vectorize/hf && go test ./...

# Downloads all-MiniLM-L6-v2 (~90MB, cached) and runs real inference.
# Requires the onnxruntime shared library; set ONNXRUNTIME_LIB_PATH.
test-hf-live:
	cd extensions/vectorize/hf && RUN_HF_LIVE_TESTS=1 go test -run TestLive -v ./...

# --- documentation site (Antora; see docs/) ---

docs-deps:
	cd docs && npm install --no-audit --no-fund

docs-build:
	cd docs && npx antora --stacktrace antora-playbook.yml

# Port 5001: macOS AirPlay Receiver occupies 5000.
docs-serve: docs-build
	cd docs && npx http-server build/site -c-1 -p 5001

# --- benchmarks (see benchmarks/README.md) ---

bench-go:
	go run ./benchmarks/gobench

# Requires the redisvl Python package in the active environment; run
# `make bench-py-deps` (or pip/pip3 install redisvl) once first.
bench-py:
	@if [ -x benchmarks/pybench/.venv/bin/python ]; then \
		benchmarks/pybench/.venv/bin/python benchmarks/pybench/bench.py; \
	else \
		$(PYTHON) benchmarks/pybench/bench.py; \
	fi

# Installs redisvl into a local virtualenv (benchmarks/pybench/.venv) so it
# works on PEP 668 externally-managed Pythons (Homebrew, Debian, ...).
bench-py-deps:
	$(PYTHON) -m venv benchmarks/pybench/.venv
	benchmarks/pybench/.venv/bin/python -m pip install --upgrade pip redisvl
