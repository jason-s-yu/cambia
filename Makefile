.PHONY: libcambia service test-engine test-service test-cfr test parity-gate clean

# Absolute path to this checkout, resolved from the Makefile's own location.
# The parity gate uses it to pin the interpreter at THIS tree: cfr is an
# editable install whose .pth finder otherwise resolves `src` to the main
# checkout, so a worktree run would silently test the wrong code.
ROOT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

PYTHON ?= python

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

# ---------------------------------------------------------------------------
# Go vs Python cross-engine parity gate (cambia-1234)
# ---------------------------------------------------------------------------
# Migration acceptance for retiring the Python reference engine (cambia-1424):
# while both engines exist, they must produce identical numbers. Go is the
# RULES.md reference. Seed breadth is the single declared constant in
# cfr/tests/parity_seeds.py (default 40); override without editing any test:
#
#   CAMBIA_PARITY_SEEDS=200 make parity-gate
#
# This target retires with cambia-1430.
PARITY_TESTS := \
	tests/test_parity_gate.py \
	tests/test_token_stream_parity.py \
	tests/test_cross_engine_samples.py \
	tests/test_cross_validation.py \
	tests/test_snap_race_cross_engine.py

parity-gate: libcambia
	@echo "== parity gate: pinning the interpreter to $(ROOT_DIR)/cfr =="
	@cd $(ROOT_DIR)/cfr && PYTHONPATH=$(ROOT_DIR)/cfr $(PYTHON) -c \
		"import pathlib, sys; import src.encoding as m; import src.game.engine as g; \
		root = pathlib.Path('$(ROOT_DIR)').resolve(); \
		bad = [x.__file__ for x in (m, g) if not pathlib.Path(x.__file__).resolve().is_relative_to(root)]; \
		sys.exit('parity gate ABORTED: the Python engine imported from OUTSIDE this checkout: %s. The cfr editable install won over PYTHONPATH, so the gate would test the wrong code.' % bad) if bad else print('  src pinned:', m.__file__, '/', g.__file__)"
	@echo "== parity gate: Go vs Python lockstep, 2 and 4 seats =="
	cd $(ROOT_DIR)/cfr && \
		PYTHONPATH=$(ROOT_DIR)/cfr \
		LIBCAMBIA_PATH=$(ROOT_DIR)/cfr/libcambia.so \
		$(PYTHON) -m pytest $(PARITY_TESTS) -q -s -rs

# Clean build artifacts
clean:
	rm -f cfr/libcambia.so cfr/libcambia.h
