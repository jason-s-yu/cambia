.PHONY: libcambia service test-engine test-service test-cfr test clean

# Build the shared library for Python FFI
libcambia:
	go build -buildmode=c-shared -o cfr/libcambia.so ./engine/cgo/

# Run the game server
service:
	cd service && go run cmd/server/main.go

# Tests
test-engine:
	cd engine && go test ./...

test-service:
	cd service && go test ./...

test-cfr:
	cd cfr && python -m pytest tests/

test: test-engine test-service test-cfr

# Clean build artifacts
clean:
	rm -f cfr/libcambia.so cfr/libcambia.h
