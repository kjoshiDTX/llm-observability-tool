.PHONY: build run test test-go test-py clean load-test lint

BIN := bin
GOFLAGS ?=

build:
	mkdir -p $(BIN)
	go build $(GOFLAGS) -o $(BIN)/llm-obs-collector ./collector/cmd/collector
	go build $(GOFLAGS) -o $(BIN)/llm-obs ./collector/cmd/llm-obs

run: build
	./$(BIN)/llm-obs-collector

test: test-go test-py

test-go:
	go vet ./...
	go test -race -count=1 ./...

test-py:
	cd client && python3 -m pytest -q

install-client:
	pip install -e client/

load-test: build
	go run ./test/load -duration 10m

clean:
	rm -rf $(BIN) *.db
