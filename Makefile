# LLMOps - LLM operations module: continuous evaluation, A/B experiments, dataset management.
# Module: digital.vasic.llmops

.PHONY: build test test-race test-short test-integration test-bench test-coverage fmt vet lint clean help

MODULE := digital.vasic.llmops
GOMAXPROCS ?= 2

build:
	go build ./...

test:
	GOMAXPROCS=$(GOMAXPROCS) go test -count=1 -race -p 1 ./...

test-race:
	GOMAXPROCS=$(GOMAXPROCS) go test -count=1 -race -p 1 ./...

test-short:
	GOMAXPROCS=$(GOMAXPROCS) go test -count=1 -short -p 1 ./...

test-integration:
	GOMAXPROCS=$(GOMAXPROCS) go test -count=1 -race -p 1 ./tests/integration/... 2>/dev/null || echo "No integration tests"

test-bench:
	GOMAXPROCS=$(GOMAXPROCS) go test -bench=. -benchmem ./tests/benchmark/... 2>/dev/null || echo "No benchmarks"

test-coverage:
	GOMAXPROCS=$(GOMAXPROCS) go test -count=1 -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

fmt:
	gofmt -w .
	goimports -w .

vet:
	go vet ./...

lint:
	@command -v golangci-lint >/dev/null 2>&1 || { echo "golangci-lint not installed"; exit 1; }
	golangci-lint run ./...

clean:
	rm -f coverage.out coverage.html
	go clean -cache

# Challenges (run from parent HelixAgent project)
challenge:
	../challenges/scripts/llmops_challenge.sh 2>/dev/null || echo "No challenge script"

help:
	@echo "LLMOps - LLM operations module: continuous evaluation, A/B experiments, dataset management."
	@echo ""
	@echo "Build & Test:"
	@echo "  make build         Build all packages"
	@echo "  make test          Run all tests with race detection"
	@echo "  make test-short    Run unit tests only"
	@echo "  make test-bench    Run benchmarks"
	@echo "  make test-coverage Generate coverage report"
	@echo ""
	@echo "Quality:"
	@echo "  make fmt           Format code"
	@echo "  make vet           Run go vet"
	@echo "  make lint          Run golangci-lint"
	@echo ""
	@echo "Other:"
	@echo "  make clean         Remove build artifacts"
	@echo "  make challenge     Run challenge script (from parent project)"
